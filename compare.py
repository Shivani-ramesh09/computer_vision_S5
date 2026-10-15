"""
Compare Dehazing results side-by-side:
Hazy | Pretrained | Fine-tuned | DCP | Ground Truth

Usage:
    python compare.py --frame data/REVIDE/Test/hazy/C005/00000.JPG \
                      --weights pretrained=checkpoints/dehazeformer-t.pth fine-tuned=checkpoints/best.pth \
                      --output results/compare.png
"""

import os, sys, argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from PIL import Image

from model import build_dehazeformer_t


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
    dc = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    return cv2.erode(dc, kernel, borderType=cv2.BORDER_REPLICATE)


def atm_light(img, dc):
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
# Helper: Find test pairs from REVIDE
# ---------------------------------------------------------------------------

def get_test_pairs(data_root):
    split_dir = None
    for candidate in ["Test", "test"]:
        p = Path(data_root) / candidate
        if p.exists():
            split_dir = p
            break
    if split_dir is None:
        return []

    hazy_root = split_dir / "hazy"
    gt_root   = split_dir / "gt"
    valid_exts = {".png", ".jpg", ".jpeg"}

    pairs = []
    if hazy_root.exists() and gt_root.exists():
        for seq in sorted(hazy_root.iterdir()):
            if not seq.is_dir():
                continue
            for f in sorted(seq.iterdir()):
                if f.suffix.lower() in valid_exts:
                    gt_f = gt_root / seq.name / f.name
                    if gt_f.exists():
                        pairs.append((str(f), str(gt_f)))
    return pairs


# ---------------------------------------------------------------------------
# Label Banner
# ---------------------------------------------------------------------------

def label_panel(img_bgr, text):
    """Add a sleek top header banner with label text to an image panel."""
    h, w = img_bgr.shape[:2]
    banner_h = max(36, int(h * 0.065))
    banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    banner[:] = (30, 30, 30)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = banner_h / 48.0
    thickness = max(1, int(round(scale * 2)))
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    tx = max(8, (w - tw) // 2)
    ty = (banner_h + th) // 2 - 1
    cv2.putText(banner, text, (tx, ty), font, scale, (240, 240, 240), thickness, cv2.LINE_AA)
    return np.vstack([banner, img_bgr])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate side-by-side comparison image.")
    p.add_argument("--frame",   type=str, default=None,
                   help="Path to hazy image file OR integer index into REVIDE test set")
    p.add_argument("--index",   type=int, default=None,
                   help="Integer index into REVIDE test set")
    p.add_argument("--data",    type=str, default="data/REVIDE",
                   help="Path to REVIDE dataset root")
    p.add_argument("--weights", nargs="*", default=[],
                   help="Named weights in format name=path [name=path ...]")
    p.add_argument("--device",  type=str, default="auto",
                   help="Device: 'auto', 'cuda', 'mps', or 'cpu'")
    p.add_argument("--max_res", type=int, default=720,
                   help="Downscale longer edge to this (0=no limit)")
    p.add_argument("--output",  type=str, default="results/compare.png",
                   help="Output path for comparison image")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    # Discover test pairs
    test_pairs = get_test_pairs(args.data)

    hazy_path = None
    gt_path = None

    if args.frame is not None:
        if args.frame.isdigit():
            idx = int(args.frame)
            if idx < 0 or idx >= len(test_pairs):
                print(f"[ERROR] Index {idx} out of range (0..{len(test_pairs)-1})")
                sys.exit(1)
            hazy_path, gt_path = test_pairs[idx]
        elif os.path.isfile(args.frame):
            hazy_path = args.frame
            # Check for GT
            hp = Path(hazy_path)
            # Try replacing /hazy/ with /gt/
            parts = list(hp.parts)
            if "hazy" in parts:
                idx = parts.index("hazy")
                parts[idx] = "gt"
                candidate = Path(*parts)
                if candidate.exists():
                    gt_path = str(candidate)
            elif "Hazy" in parts:
                idx = parts.index("Hazy")
                parts[idx] = "gt" if "gt" in os.listdir(hp.parent.parent) else "GT"
                candidate = Path(*parts)
                if candidate.exists():
                    gt_path = str(candidate)

            if gt_path is None:
                # Search test pairs for match
                for hp_match, gp_match in test_pairs:
                    if Path(hp_match).name == hp.name and Path(hp_match).parent.name == hp.parent.name:
                        gt_path = gp_match
                        break
        else:
            print(f"[ERROR] Specified --frame not found: {args.frame}")
            sys.exit(1)
    elif args.index is not None:
        idx = args.index
        if idx < 0 or idx >= len(test_pairs):
            print(f"[ERROR] Index {idx} out of range (0..{len(test_pairs)-1})")
            sys.exit(1)
        hazy_path, gt_path = test_pairs[idx]
    else:
        if not test_pairs:
            print(f"[ERROR] No test image pairs found in {args.data}")
            sys.exit(1)
        hazy_path, gt_path = test_pairs[0]

    print(f"Input Hazy:  {hazy_path}")
    print(f"Ground Truth: {gt_path}")

    # Load images
    h_img = Image.open(hazy_path).convert("RGB")
    g_img = Image.open(gt_path).convert("RGB") if gt_path and os.path.isfile(gt_path) else None

    w, h = h_img.size
    longer = max(w, h)
    if args.max_res > 0 and longer > args.max_res:
        scale = args.max_res / longer
        nw = int(round(w * scale))
        nh = int(round(h * scale))
        h_img = h_img.resize((nw, nh), Image.Resampling.BILINEAR)
        if g_img is not None:
            g_img = g_img.resize((nw, nh), Image.Resampling.BILINEAR)

    # Convert to [-1, 1] tensor
    h_t = TF.to_tensor(h_img) * 2.0 - 1.0

    # BGR numpy arrays
    hazy_bgr = cv2.cvtColor(np.array(h_img), cv2.COLOR_RGB2BGR)
    gt_bgr   = cv2.cvtColor(np.array(g_img), cv2.COLOR_RGB2BGR) if g_img is not None else None

    # Parse named weights
    weights_map = {}
    for item in args.weights:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            name = Path(item).stem
            path = item
        weights_map[name] = path

    # Prepare panels list in order: Hazy | [models...] | DCP | Ground Truth
    panels = []
    panels.append((hazy_bgr, "Hazy Input"))

    # Models
    inp = h_t.unsqueeze(0).to(device)
    for name, path in weights_map.items():
        print(f"Running inference for '{name}' ({path})...")
        model = load_eval_model(path, device)
        with torch.no_grad():
            out = model(inp).clamp(-1.0, 1.0)
        out_01 = (out.squeeze(0) * 0.5 + 0.5).clamp(0.0, 1.0)
        out_np = (out_01.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
        model_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
        panels.append((model_bgr, name))

    # DCP Baseline
    print("Computing Dark Channel Prior (DCP) baseline...")
    dcp_bgr = dcp_dehaze(hazy_bgr)
    panels.append((dcp_bgr, "DCP"))

    # Ground Truth
    if gt_bgr is not None:
        panels.append((gt_bgr, "Ground Truth"))

    # Render side-by-side labeled image
    labeled_panels = [label_panel(p, lbl) for p, lbl in panels]
    combined = np.hstack(labeled_panels)

    out_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, combined)
    print(f"\n[DONE] Side-by-side comparison saved to: {out_path} ({combined.shape[1]}x{combined.shape[0]})")


if __name__ == "__main__":
    main()
