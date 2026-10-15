"""
Evaluate DehazeFormer-T models, DCP baseline, and Hazy Input on REVIDE test set.
Computes PSNR, SSIM, and temporal flicker metrics.
Saves results to results/results.md and results/results.csv.

Usage:
    python eval.py --data data/REVIDE --weights pretrained=checkpoints/dehazeformer-t.pth
    python eval.py --data data/REVIDE --weights pretrained=checkpoints/dehazeformer-t.pth finetuned=checkpoints/best.pth
"""

import os, sys, math, time, argparse, csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

try:
    from pytorch_msssim import ssim
except ImportError:
    def ssim(img1, img2, data_range=1.0, size_average=True):
        mu1 = F.avg_pool2d(img1, kernel_size=11, stride=1, padding=5)
        mu2 = F.avg_pool2d(img2, kernel_size=11, stride=1, padding=5)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        sigma1_sq = F.avg_pool2d(img1 * img1, 11, 1, 5) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, 11, 1, 5) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, 11, 1, 5) - mu1_mu2
        C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        res = ssim_map.mean(dim=[-3, -2, -1])
        return res.mean() if size_average else res

from model import build_dehazeformer_t
from train import REVIDEValDataset


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device(device_str=None):
    if device_str and device_str.lower() != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Fast Dark Channel Prior (DCP) + Guided Filter Baseline
# ---------------------------------------------------------------------------

def dark_channel(img, patch=9):
    """img: HxWx3 float [0,1]. Fast vectorized min filter via cv2.erode."""
    dc = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    return cv2.erode(dc, kernel, borderType=cv2.BORDER_REPLICATE)


def atm_light(img, dc):
    """Estimate atmospheric light A from top 0.1% bright pixels in dark channel."""
    flat_dc = dc.ravel()
    n_pixels = max(1, flat_dc.size // 1000)
    idx = np.argpartition(flat_dc, -n_pixels)[-n_pixels:]
    flat_img = img.reshape(-1, 3)
    A = flat_img[idx].max(axis=0)
    return np.clip(A, 0.0, 1.0)


def transmission(img, A, patch=9, omega=0.95):
    norm = img / (A + 1e-6)
    dc = dark_channel(norm, patch)
    return 1.0 - omega * dc


def guided_filter(I, p, r=20, eps=1e-3):
    """Fast box-filter guided filter (He et al.)."""
    mean_I  = cv2.boxFilter(I, -1, (r, r))
    mean_p  = cv2.boxFilter(p, -1, (r, r))
    mean_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip  = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I   = mean_II - mean_I * mean_I
    a       = cov_Ip / (var_I + eps)
    b       = mean_p - a * mean_I
    mean_a  = cv2.boxFilter(a, -1, (r, r))
    mean_b  = cv2.boxFilter(b, -1, (r, r))
    return mean_a * I + mean_b


def dcp_dehaze(img_bgr):
    """Dehaze image using Dark Channel Prior + Guided Filter."""
    img = img_bgr.astype(np.float32) / 255.0
    img_rgb = img[:, :, ::-1]

    dc = dark_channel(img_rgb, patch=9)
    A = atm_light(img_rgb, dc)
    t = transmission(img_rgb, A, patch=9)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    t_ref = guided_filter(gray, t, r=20, eps=1e-3)
    t_ref = np.clip(t_ref, 0.1, 1.0)

    dehazed = (img_rgb - A) / t_ref[..., None] + A
    dehazed = np.clip(dehazed, 0.0, 1.0)
    return (dehazed[:, :, ::-1] * 255.0).round().astype(np.uint8)


# ---------------------------------------------------------------------------
# Model Loader
# ---------------------------------------------------------------------------

def load_eval_model(weights_path, device):
    model = build_dehazeformer_t().to(device)
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    ckpt = torch.load(weights_path, map_location=device)
    # Prefer EMA weights if present, then state_dict, then model, else raw dict
    if isinstance(ckpt, dict):
        state = ckpt.get("ema", ckpt.get("state_dict", ckpt.get("model", ckpt)))
    else:
        state = ckpt
    state = {k.replace("module.", ""): v for k, v in state.items()}

    res = model.load_state_dict(state, strict=False)
    if len(res.missing_keys) > 5:
        raise RuntimeError(f"Too many missing keys ({len(res.missing_keys)}) in {weights_path}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_psnr_ssim(pred_01, gt_01):
    """pred_01 and gt_01 are 1x3xHxW float tensors in [0, 1]."""
    pred_01 = pred_01.clamp(0.0, 1.0)
    gt_01   = gt_01.clamp(0.0, 1.0)
    mse = F.mse_loss(pred_01, gt_01, reduction='none').mean(dim=[1, 2, 3])
    p = 10.0 * torch.log10(1.0 / (mse + 1e-10))
    s = ssim(pred_01, gt_01, data_range=1.0, size_average=False)
    return float(p.mean().item()), float(s.mean().item())


def sequence_flicker(frames_uint8):
    """Mean absolute difference between consecutive frames (uint8 numpy HxWx3)."""
    if len(frames_uint8) < 2:
        return 0.0
    diffs = [
        np.mean(np.abs(frames_uint8[i].astype(np.float32) - frames_uint8[i-1].astype(np.float32)))
        for i in range(1, len(frames_uint8))
    ]
    return float(np.mean(diffs))


# ---------------------------------------------------------------------------
# Flicker evaluation on consecutive frames (stride 1)
# ---------------------------------------------------------------------------

def evaluate_consecutive_flicker(data_root, models_dict, device, max_res=720, max_seqs=3, max_frames=30):
    """
    Computes flicker on consecutive frames (stride 1) on up to 3 sequences, 30 frames each.
    Returns:
        method_flickers: dict {method_name: mean_flicker}
        gt_flicker: float
    """
    split_dir = None
    for candidate in ["Test", "test"]:
        p = Path(data_root) / candidate
        if p.exists():
            split_dir = p
            break
    if split_dir is None:
        return {m: 0.0 for m in models_dict}, 0.0

    hazy_root = split_dir / "hazy"
    gt_root   = split_dir / "gt"
    valid_exts = {".png", ".jpg", ".jpeg"}

    seq_dirs = sorted([d for d in hazy_root.iterdir() if d.is_dir()])[:max_seqs]

    seq_flickers = {m: [] for m in models_dict}
    seq_flickers["Hazy Input"] = []
    seq_flickers["Dark Channel Prior (DCP)"] = []
    gt_seq_flickers = []

    for sdir in seq_dirs:
        gt_sdir = gt_root / sdir.name
        if not gt_sdir.exists():
            continue
        frame_files = sorted([f for f in sdir.iterdir() if f.suffix.lower() in valid_exts])[:max_frames]
        if len(frame_files) < 2:
            continue

        hazy_frames = []
        gt_frames   = []
        model_frames = {m: [] for m in models_dict}
        dcp_frames  = []

        for f in frame_files:
            gt_f = gt_sdir / f.name
            if not gt_f.exists():
                continue

            h_img = Image.open(str(f)).convert("RGB")
            g_img = Image.open(str(gt_f)).convert("RGB")

            w, h = h_img.size
            longer = max(w, h)
            if max_res > 0 and longer > max_res:
                scale = max_res / longer
                nw = int(round(w * scale))
                nh = int(round(h * scale))
                h_img = h_img.resize((nw, nh), Image.Resampling.BILINEAR)
                g_img = g_img.resize((nw, nh), Image.Resampling.BILINEAR)

            # [-1, 1] tensor
            h_t = TF.to_tensor(h_img) * 2.0 - 1.0
            g_t = TF.to_tensor(g_img) * 2.0 - 1.0

            # uint8 for flicker
            gt_np = ((g_t * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
            hazy_np = ((h_t * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)

            gt_frames.append(gt_np)
            hazy_frames.append(hazy_np)

            # DCP
            dcp_bgr = dcp_dehaze(cv2.cvtColor(hazy_np, cv2.COLOR_RGB2BGR))
            dcp_rgb = cv2.cvtColor(dcp_bgr, cv2.COLOR_BGR2RGB)
            dcp_frames.append(dcp_rgb)

            # Models
            inp = h_t.unsqueeze(0).to(device)
            for m_name, model in models_dict.items():
                with torch.no_grad():
                    out = model(inp).clamp(-1.0, 1.0)
                out_01 = (out.squeeze(0) * 0.5 + 0.5).clamp(0.0, 1.0)
                out_np = (out_01.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
                model_frames[m_name].append(out_np)

        if len(gt_frames) >= 2:
            gt_seq_flickers.append(sequence_flicker(gt_frames))
            seq_flickers["Hazy Input"].append(sequence_flicker(hazy_frames))
            seq_flickers["Dark Channel Prior (DCP)"].append(sequence_flicker(dcp_frames))
            for m_name in models_dict:
                seq_flickers[m_name].append(sequence_flicker(model_frames[m_name]))

    res_flicker = {}
    for m in seq_flickers:
        vals = seq_flickers[m]
        res_flicker[m] = float(np.mean(vals)) if vals else 0.0

    mean_gt_flicker = float(np.mean(gt_seq_flickers)) if gt_seq_flickers else 0.0
    return res_flicker, mean_gt_flicker


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate DehazeFormer-T models and baselines on REVIDE.")
    p.add_argument("--data",       type=str, default="data/REVIDE",
                   help="Root of REVIDE dataset")
    p.add_argument("--weights",    nargs="*", default=[],
                   help="Named weights in format name=path [name=path ...]")
    p.add_argument("--device",     type=str, default="auto",
                   help="Device: 'auto', 'cuda', 'mps', or 'cpu'")
    p.add_argument("--max_res",    type=int, default=720,
                   help="Downscale longer edge to this (0=no limit)")
    p.add_argument("--stride",     type=int, default=10,
                   help="Frame stride for test set validation (default: 10)")
    p.add_argument("--max_frames", type=int, default=0,
                   help="Max validation frames to evaluate (0=all, for quick tests)")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    # Parse named weights
    weights_map = {}
    for item in args.weights:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            name = Path(item).stem
            path = item
        weights_map[name] = path

    print(f"Loading weights: {weights_map if weights_map else 'None (baselines only)'}")
    models = {}
    for name, path in weights_map.items():
        print(f"Loading model '{name}' from {path}...")
        models[name] = load_eval_model(path, device)

    # Validation dataset (reusing REVIDEValDataset from train.py)
    print(f"\nInitializing REVIDE validation dataset (data={args.data}, stride={args.stride}, max_res={args.max_res})...")
    val_dataset = REVIDEValDataset(root=args.data, split="test", stride=args.stride, max_res=args.max_res)
    total_val_samples = len(val_dataset)
    if args.max_frames > 0:
        total_val_samples = min(total_val_samples, args.max_frames)
    print(f"Evaluating PSNR & SSIM on {total_val_samples} validation frames...")

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Method lists
    methods = ["Hazy Input", "Dark Channel Prior (DCP)"] + list(models.keys())
    psnr_records = {m: [] for m in methods}
    ssim_records = {m: [] for m in methods}

    for idx, (hazy, gt) in enumerate(tqdm(val_loader, total=total_val_samples, desc="Evaluating")):
        if args.max_frames > 0 and idx >= args.max_frames:
            break

        hazy = hazy.to(device)
        gt   = gt.to(device)

        # [-1, 1] -> [0, 1]
        hazy_01 = hazy * 0.5 + 0.5
        gt_01   = gt * 0.5 + 0.5

        # 1. Hazy Input
        p, s = compute_psnr_ssim(hazy_01, gt_01)
        psnr_records["Hazy Input"].append(p)
        ssim_records["Hazy Input"].append(s)

        # 2. Dark Channel Prior (DCP) Baseline
        h_np = (hazy_01.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        dcp_bgr = dcp_dehaze(cv2.cvtColor(h_np, cv2.COLOR_RGB2BGR))
        dcp_rgb = cv2.cvtColor(dcp_bgr, cv2.COLOR_BGR2RGB)
        dcp_01 = torch.from_numpy(dcp_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        p, s = compute_psnr_ssim(dcp_01, gt_01)
        psnr_records["Dark Channel Prior (DCP)"].append(p)
        ssim_records["Dark Channel Prior (DCP)"].append(s)

        # 3. Model weights
        for name, model in models.items():
            with torch.no_grad():
                out = model(hazy).clamp(-1.0, 1.0)
            out_01 = out * 0.5 + 0.5
            p, s = compute_psnr_ssim(out_01, gt_01)
            psnr_records[name].append(p)
            ssim_records[name].append(s)

    # Evaluate Flicker on consecutive frames
    flicker_max_frames = 30 if args.max_frames <= 0 else min(30, args.max_frames)
    print(f"\nEvaluating video flicker on consecutive frames (stride 1, up to 3 sequences, {flicker_max_frames} frames each)...")
    flicker_records, gt_flicker = evaluate_consecutive_flicker(
        data_root=args.data,
        models_dict=models,
        device=device,
        max_res=args.max_res,
        max_seqs=3,
        max_frames=flicker_max_frames
    )

    # Aggregate results
    summary_rows = []
    for m in methods:
        mean_p = float(np.mean(psnr_records[m])) if psnr_records[m] else 0.0
        mean_s = float(np.mean(ssim_records[m])) if ssim_records[m] else 0.0
        flk    = flicker_records.get(m, 0.0)
        summary_rows.append({
            "Method": m,
            "PSNR (dB)": f"{mean_p:.2f}",
            "SSIM": f"{mean_s:.4f}",
            "Flicker": f"{flk:.2f}",
            "GT Flicker": f"{gt_flicker:.2f}"
        })

    # Prepare Markdown table & CSV output
    os.makedirs("results", exist_ok=True)
    md_path = "results/results.md"
    csv_path = "results/results.csv"

    # Write Markdown
    md_lines = [
        "# Evaluation Results on REVIDE\n",
        "| Method | PSNR (dB) | SSIM | Flicker | GT Flicker |",
        "|:---|:---:|:---:|:---:|:---:|"
    ]
    for row in summary_rows:
        md_lines.append(f"| {row['Method']} | {row['PSNR (dB)']} | {row['SSIM']} | {row['Flicker']} | {row['GT Flicker']} |")
    md_content = "\n".join(md_lines) + "\n"

    with open(md_path, "w") as f:
        f.write(md_content)

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Method", "PSNR (dB)", "SSIM", "Flicker", "GT Flicker"])
        writer.writeheader()
        writer.writerows(summary_rows)

    # Print summary to console
    print("\n" + "=" * 70)
    print("                      EVALUATION RESULTS")
    print("=" * 70)
    print(md_content)
    print("=" * 70)
    print(f"Saved results table to: {md_path}")
    print(f"Saved results CSV to:   {csv_path}\n")


if __name__ == "__main__":
    main()
