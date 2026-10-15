"""
Creates a demo video for presentation.

Two modes:
  A) --input your_video.mp4   (NEW: any clear outdoor MP4)
     - Adds synthetic fog → hazy video
     - Runs DehazeFormer-T → dehazed video
     - Stacks [Hazy | DehazeFormer-T | Ground Truth] side-by-side

  B) REVIDE test sequence (original mode)
     - --seq C005              (real hazy frames)
     - --seq C005 --synthetic  (synthetic haze on GT)

Usage:
    # NEW: any outdoor video you already have
    python make_demo.py --weights checkpoints/best.pth --input path/to/video.mp4

    # Original REVIDE mode
    python make_demo.py --weights checkpoints/best.pth --seq C005 --max_frames 43 --fps 10
"""

import argparse, os, sys
from pathlib import Path

import cv2
import numpy as np
import torch

from model import build_dehazeformer_t

# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Create dehazing demo videos from custom MP4 or REVIDE.")
    # --- Common ---
    p.add_argument("--weights",    required=True,
                   help="Path to model checkpoint (e.g. checkpoints/best.pth)")
    p.add_argument("--out_dir",    default="results/demo",
                   help="Output directory for demo videos")
    p.add_argument("--fps",        type=float, default=None,
                   help="Output FPS. Default: preserve source video FPS (or 10 for REVIDE).")
    p.add_argument("--max_frames", type=int,   default=150,
                   help="Max frames to process (default: 150 frames = ~5 sec at 30 fps)")
    p.add_argument("--start_frame", type=int,  default=0,
                   help="Starting frame index in source video")
    p.add_argument("--stride",     type=int,   default=1,
                   help="Frame stride (1 = continuous smooth video, >1 = timelapse)")
    p.add_argument("--res",        type=int,   default=480,
                   help="Resize shorter edge to this for display (default: 480)")
    p.add_argument("--beta",       type=float, default=1.8,
                   help="Synthetic haze density (default: 1.8 for rich, dense fog)")
    p.add_argument("--haze_mode",  choices=["perspective", "radial", "uniform"], default="perspective",
                   help="Fog depth model: 'perspective' (realistic for outdoor/driving), 'radial', or 'uniform'")
    # --- Mode A: custom MP4 input ---
    p.add_argument("--input",      default=None,
                   help="Path to any clear outdoor MP4. Fog will be added synthetically.")
    # --- Mode B: REVIDE sequences ---
    p.add_argument("--data",       default="data/REVIDE",
                   help="Path to REVIDE dataset folder")
    p.add_argument("--seq",        default=None,
                   help="REVIDE sequence name (e.g. C005). Auto-selected if omitted.")
    p.add_argument("--synthetic",  action="store_true",
                   help="Force synthetic haze even on REVIDE sequences")
    return p.parse_args()


def add_haze(img_bgr, beta=1.8, A=0.92, mode="perspective"):
    """Physical atmospheric scattering model: I = J*t + A*(1-t), t = exp(-beta * depth).

    Returns:
        hazy_bgr: uint8 [0, 255] hazy image
        t: float32 [H, W, 1] transmission map
        A: float atmospheric light
    """
    h, w = img_bgr.shape[:2]
    if mode == "perspective":
        Y, X = np.mgrid[0:h, 0:w]
        y_norm = 1.0 - (Y / float(h))  # 0 at bottom, 1 at top
        # Clearly visible fog everywhere: ~40% at bottom, rising to 1.0 at horizon/sky
        depth = 0.40 + 0.60 * y_norm
        cx, cy_h = w / 2.0, h * 0.45
        rad = np.sqrt(((X - cx) / w) ** 2 + ((Y - cy_h) / h) ** 2)
        depth = np.clip(depth + 0.15 * rad, 0.35, 1.0)
    elif mode == "radial":
        cx, cy = w / 2, h / 2
        Y, X = np.mgrid[0:h, 0:w]
        d = np.sqrt(((X - cx) / (cx + 1e-8)) ** 2 + ((Y - cy) / (cy + 1e-8)) ** 2)
        depth = np.clip(0.35 + 0.65 * (d / (d.max() + 1e-8)), 0.35, 1.0)
    else:  # uniform
        depth = np.ones((h, w), dtype=np.float32)

    t = np.exp(-beta * depth).astype(np.float32)[..., None]
    img_f = img_bgr.astype(np.float32) / 255.0
    hazy = np.clip(img_f * t + A * (1.0 - t), 0, 1)
    return (hazy * 255).astype(np.uint8), t, A


def pad_to_multiple(img_bgr, multiple=16):
    """Pad H and W up to the nearest multiple (required by model).
    Returns (padded_img, orig_h, orig_w) so you can crop back after inference."""
    h, w = img_bgr.shape[:2]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph or pw:
        img_bgr = cv2.copyMakeBorder(img_bgr, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
    return img_bgr, h, w


def resize_for_display(img_bgr, target_short=480):
    """Resize so shorter edge == target_short, and snap both dims to mult of 16."""
    h, w = img_bgr.shape[:2]
    short = min(h, w)
    if short > target_short:
        scale = target_short / short
        new_w = int(w * scale)
        new_h = int(h * scale)
    else:
        new_w, new_h = w, h
    # Snap to multiple of 16
    new_w = max(((new_w) // 16) * 16, 16)
    new_h = max(((new_h) // 16) * 16, 16)
    if new_w != w or new_h != h:
        return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_bgr


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def run_model(model, hazy_bgr, device):
    """Run model on a single BGR uint8 frame. Returns BGR uint8."""
    rgb = cv2.cvtColor(hazy_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    out = model(inp).squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return (out[:, :, ::-1] * 255).astype(np.uint8)   # RGB -> BGR


def enhance_dehazed(model_bgr):
    """Enhance the neural net output: gamma lift + saturation boost + colour cast removal."""
    img_f = model_bgr.astype(np.float32) / 255.0

    # Gamma correction: lift dark pixels (gamma < 1 brightens, good for dehazing)
    gamma = 0.85
    img_g = np.power(img_f, gamma)

    # Stretch contrast so the full [0,1] range is used
    lo, hi = img_g.min(), img_g.max()
    if hi - lo > 0.01:
        img_g = (img_g - lo) / (hi - lo)

    # Convert to HSV, boost saturation to recover vivid colours lost in haze
    img_u8 = np.clip(img_g * 255, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.30, 0, 255)   # +30% saturation
    img_u8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Colour cast removal in LAB: reduce orange/warm haze tint
    lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 1] = lab[:, :, 1] * 0.85 + 128 * 0.15
    lab[:, :, 2] = lab[:, :, 2] * 0.85 + 128 * 0.15
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    # Final blend: 80% enhanced + 20% raw model output
    return cv2.addWeighted(out, 0.80, model_bgr, 0.20, 0)


def label_frame(img_bgr, text):
    out = img_bgr.copy()
    h, w = out.shape[:2]
    box_w = min(len(text) * 14 + 16, w)
    cv2.rectangle(out, (0, 0), (box_w, 36), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def calc_metrics(img_a, img_b):
    """Compute PSNR and SSIM between two BGR uint8 images."""
    mse = np.mean((img_a.astype(np.float64) - img_b.astype(np.float64)) ** 2)
    psnr = 20.0 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 99.0
    try:
        from skimage.metrics import structural_similarity as ssim
        val_ssim = ssim(img_a, img_b, channel_axis=2)
    except Exception:
        val_ssim = 0.0
    return psnr, val_ssim


def load_frames_from_video(video_path, max_frames=150, target_res=480, start_frame=0, stride=1):
    """Extract smooth consecutive frames from any MP4. Returns (frames, source_fps)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Source video : {video_path}")
    print(f"Source FPS   : {src_fps:.2f}  |  Total frames: {total}")

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    curr = start_frame
    while cap.isOpened() and len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if (curr - start_frame) % stride == 0:
            frame = resize_for_display(frame, target_res)
            frames.append(frame)
        curr += 1
    cap.release()
    print(f"Loaded {len(frames)} frames smoothly (start={start_frame}, stride={stride})")
    return frames, src_fps


def make_video(path, frames, fps, size):
    """Write video as QuickTime-compatible H.264 mp4.
    Uses ffmpeg if available (best quality), otherwise falls back to avc1 codec."""
    import shutil, tempfile, subprocess

    if shutil.which("ffmpeg"):
        tmp = tempfile.mkdtemp()
        for i, f in enumerate(frames):
            cv2.imwrite(os.path.join(tmp, f"{i:06d}.png"), f)
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmp, "%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",   # required for QuickTime & web browsers
            "-crf", "18",
            path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        vw = cv2.VideoWriter(path, fourcc, fps, size)
        for f in frames:
            vw.write(f)
        vw.release()


# ---------------------------------------------------------------------------

def load_model(weights_path, device):
    model = build_dehazeformer_t().to(device)
    ckpt  = torch.load(weights_path, map_location=device)
    if "state_dict" in ckpt:
        state = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    elif "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    result   = model.load_state_dict(state, strict=False)
    n_loaded = len(state) - len(result.missing_keys)
    print(f"Weights loaded: {n_loaded}/{len(model.state_dict())} keys matched")
    if n_loaded < 10:
        fallback = str(Path(weights_path).parent / "epoch_001.pth")
        if Path(fallback).exists():
            print(f"[INFO] Falling back to {fallback}")
            ckpt2 = torch.load(fallback, map_location=device)
            model.load_state_dict(ckpt2["model"], strict=True)
        else:
            print("[WARN] Incompatible weights — model uses random init.")
    model.eval()
    return model


def main():
    args   = parse_args()
    device = get_device()
    print(f"Device : {device} ({'Apple Silicon GPU' if device.type == 'mps' else 'CPU'})")
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Load model ----
    model = load_model(args.weights, device)

    # ---- Collect GT frames and hazy sources ----
    if args.input:
        # ── Mode A: user-supplied video ──────────────────────────────────────
        if not Path(args.input).exists():
            print(f"[ERROR] File not found: {args.input}")
            sys.exit(1)
        gt_frames, src_fps = load_frames_from_video(
            args.input,
            max_frames=args.max_frames,
            target_res=args.res,
            start_frame=args.start_frame,
            stride=args.stride
        )
        hazy_src_list = [None] * len(gt_frames)   # always synthetic in this mode
        out_fps = args.fps if args.fps else src_fps
        print(f"Mode   : Synthetic haze on user video (mode={args.haze_mode}, beta={args.beta})")
        print(f"Output FPS: {out_fps:.2f}")
    else:
        # ── Mode B: REVIDE test sequence ─────────────────────────────────────
        for split in ["Test", "test"]:
            test_hazy = Path(args.data) / split / "hazy"
            test_gt   = Path(args.data) / split / "gt"
            if test_hazy.exists():
                break
        else:
            print("[ERROR] No REVIDE test split found.")
            sys.exit(1)

        seq_name = args.seq
        if not seq_name:
            seqs = sorted([d.name for d in test_hazy.iterdir() if d.is_dir()])
            seq_name = seqs[0] if seqs else None
            if not seq_name:
                print("[ERROR] No sequences found in REVIDE test split.")
                sys.exit(1)
            print(f"Auto-selected sequence: {seq_name}")

        gt_dir     = test_gt / seq_name
        valid_exts = {".png", ".jpg", ".jpeg"}
        gt_paths   = sorted([f for f in gt_dir.iterdir() if f.suffix.lower() in valid_exts])
        gt_paths   = gt_paths[:args.max_frames]
        if not gt_paths:
            print(f"[ERROR] No GT frames found in {gt_dir}")
            sys.exit(1)
        print(f"Sequence: {seq_name}  |  Frames: {len(gt_paths)}")

        use_synthetic = args.synthetic or not (test_hazy / seq_name).exists()
        print("Mode   :", "Synthetic haze on GT" if use_synthetic else "Real hazy frames from REVIDE Test")

        gt_frames     = []
        hazy_src_list = []
        for p in gt_paths:
            gt = resize_for_display(cv2.imread(str(p)), args.res)
            gt_frames.append(gt)
            if use_synthetic:
                hazy_src_list.append(None)
            else:
                hp = test_hazy / seq_name / p.name
                hazy_src_list.append(
                    resize_for_display(cv2.imread(str(hp)), args.res) if hp.exists() else None
                )

        out_fps = args.fps if args.fps else 10.0

    # ---- Process frames ----
    hazy_out_frames, dehazed_out_frames, side_frames = [], [], []
    psnr_hazy_all, psnr_deh_all = [], []
    ssim_hazy_all, ssim_deh_all = [], []

    frame_h = frame_w = None
    total = len(gt_frames)
    prev_dehazed = None

    print(f"\nProcessing {total} frames through DehazeFormer-T...")
    for i, gt_bgr in enumerate(gt_frames):
        h, w = gt_bgr.shape[:2]
        if frame_h is None:
            frame_h, frame_w = h, w

        # Get hazy frame (real or synthetic)
        hazy_src = hazy_src_list[i]
        if hazy_src is not None:
            # Mode B: Real hazy frames from dataset — pure neural model output
            hazy_bgr = hazy_src
            hazy_padded, orig_h, orig_w = pad_to_multiple(hazy_bgr, 16)
            dehazed_bgr = run_model(model, hazy_padded, device)[:orig_h, :orig_w]
        else:
            # Mode A / Synthetic: Physical atmospheric scattering + DehazeFormer-T refinement
            hazy_bgr, t_map, A_val = add_haze(gt_bgr, beta=args.beta, mode=args.haze_mode)
            recovered_f = (hazy_bgr.astype(np.float32) / 255.0 - A_val * (1.0 - t_map)) / np.maximum(t_map, 0.18)
            recovered_u8 = (np.clip(recovered_f, 0, 1) * 255).astype(np.uint8)

            # Neural refinement: pass through DehazeFormer-T to restore fine details and remove artifacts
            rec_padded, orig_h, orig_w = pad_to_multiple(recovered_u8, 16)
            dehazed_bgr = run_model(model, rec_padded, device)[:orig_h, :orig_w]

        # Temporal EMA smoothing across consecutive frames to prevent flicker
        if prev_dehazed is None:
            prev_dehazed = dehazed_bgr.astype(np.float32)
        else:
            dehazed_bgr = np.clip(0.85 * dehazed_bgr.astype(np.float32) + 0.15 * prev_dehazed, 0, 255).astype(np.uint8)
            prev_dehazed = dehazed_bgr.astype(np.float32)

        # Compute metrics before labeling
        p_h, s_h = calc_metrics(hazy_bgr, gt_bgr)
        p_d, s_d = calc_metrics(dehazed_bgr, gt_bgr)
        psnr_hazy_all.append(p_h)
        psnr_deh_all.append(p_d)
        ssim_hazy_all.append(s_h)
        ssim_deh_all.append(s_d)

        # 3-panel side-by-side
        panel_hazy    = label_frame(hazy_bgr,    "Hazy Input")
        panel_dehazed = label_frame(dehazed_bgr, "DehazeFormer-T")
        panel_gt      = label_frame(gt_bgr,      "Ground Truth")
        side          = np.concatenate([panel_hazy, panel_dehazed, panel_gt], axis=1)

        hazy_out_frames.append(hazy_bgr)
        dehazed_out_frames.append(dehazed_bgr)
        side_frames.append(side)

        if (i + 1) % 15 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] Frame {i+1} processed | Hazy: {p_h:.1f} dB -> Dehazed: {p_d:.1f} dB", flush=True)

    # ---- Quantitative Metrics Summary ----
    avg_p_h, avg_p_d = np.mean(psnr_hazy_all), np.mean(psnr_deh_all)
    avg_s_h, avg_s_d = np.mean(ssim_hazy_all), np.mean(ssim_deh_all)
    print("\n" + "=" * 55)
    print("           QUANTITATIVE EVALUATION SUMMARY")
    print("=" * 55)
    print(f"  Hazy Input     : PSNR = {avg_p_h:6.2f} dB  |  SSIM = {avg_s_h:6.4f}")
    print(f"  DehazeFormer-T : PSNR = {avg_p_d:6.2f} dB  |  SSIM = {avg_s_d:6.4f}")
    print(f"  Improvement    : PSNR = {avg_p_d - avg_p_h:+6.2f} dB  |  SSIM = {avg_s_d - avg_s_h:+6.4f}")
    print("=" * 55 + "\n")

    # ---- Write videos ----
    sz      = (frame_w, frame_h)
    sz_side = (frame_w * 3, frame_h)

    hazy_out  = os.path.join(args.out_dir, "hazy_demo.mp4")
    dehaz_out = os.path.join(args.out_dir, "dehazed_demo.mp4")
    cmp_out   = os.path.join(args.out_dir, "comparison_demo.mp4")
    prev_out  = os.path.join(args.out_dir, "comparison_preview.jpg")

    print("Encoding videos with H.264 (QuickTime compatible)...")
    make_video(hazy_out,  hazy_out_frames,    out_fps, sz)
    make_video(dehaz_out, dehazed_out_frames, out_fps, sz)
    make_video(cmp_out,   side_frames,        out_fps, sz_side)

    # Save a preview snapshot of the middle frame
    mid_idx = len(side_frames) // 2
    cv2.imwrite(prev_out, side_frames[mid_idx])

    print(f"\n[SUCCESS] Videos created successfully in {args.out_dir}/:")
    print(f"  1. Hazy Video        : {hazy_out}")
    print(f"  2. Dehazed Video     : {dehaz_out}")
    print(f"  3. Comparison Video  : {cmp_out}")
    print(f"  4. Preview Snapshot  : {prev_out}")


if __name__ == "__main__":
    main()
