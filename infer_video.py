"""
Infer DehazeFormer-T on a hazy video or single frame.
Per-frame dehazing + temporal EMA smoothing (alpha=0.8) to reduce video flicker.

Usage:
    # Zero-shot test on a single frame
    python infer_video.py --frame path/to/image.jpg --output results/dehazed_frame.png --weights checkpoints/dehazeformer-t.pth

    # Process a full video
    python infer_video.py --input hazy.mp4 --output dehazed.mp4 --weights checkpoints/best.pth
"""

import argparse, os, sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model import build_dehazeformer_t

EMA_ALPHA = 0.8          # weight of current frame in temporal EMA blend


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Infer DehazeFormer-T on a video or single frame.")
    p.add_argument("--input",    default=None,
                   help="Input hazy video path or directory of frames")
    p.add_argument("--output",   default="results/dehazed.mp4",
                   help="Output path for dehazed video or image")
    p.add_argument("--weights",  required=True,
                   help="Model checkpoint (.pth)")
    p.add_argument("--frame",    default=None,
                   help="Path to a single hazy image to dehaze (zero-shot quick test)")
    p.add_argument("--max_res",  type=int, default=720,
                   help="Downscale longer edge to this before processing (0=no limit)")
    p.add_argument("--fp16",     action="store_true",
                   help="Use half precision (CUDA GPU only)")
    p.add_argument("--no_smooth", action="store_true",
                   help="Disable temporal EMA smoothing across frames")
    return p.parse_args()


def load_model(weights_path, device, fp16=False):
    model = build_dehazeformer_t().to(device)

    if not os.path.isfile(weights_path):
        print(f"[ERROR] Weights file not found: {weights_path}")
        sys.exit(1)

    ckpt = torch.load(weights_path, map_location=device)
    # Prefer EMA weights if present, then state_dict, then model
    state = ckpt.get("ema", ckpt.get("state_dict", ckpt.get("model", ckpt)))
    state = {k.replace("module.", ""): v for k, v in state.items()}

    res = model.load_state_dict(state, strict=False)
    print(f"[Weights] Loaded {len(state) - len(res.missing_keys)}/{len(model.state_dict())} keys matched from {weights_path}")
    if len(res.missing_keys) > 0:
        print(f"  Missing keys: {res.missing_keys[:10]}")
    if len(res.unexpected_keys) > 0:
        print(f"  Unexpected keys: {res.unexpected_keys[:10]}")
    if len(res.missing_keys) > 5:
        raise RuntimeError(f"Too many missing keys ({len(res.missing_keys)}) when loading weights from {weights_path}")

    model.eval()
    if fp16 and device.type == "cuda":
        model = model.half()

    return model


def resize_if_needed(frame_bgr, max_res):
    """Downscale frame so longer edge <= max_res (preserving aspect ratio)."""
    if max_res <= 0:
        return frame_bgr, 1.0
    h, w = frame_bgr.shape[:2]
    longer = max(h, w)
    if longer <= max_res:
        return frame_bgr, 1.0
    scale = max_res / longer
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def frame_to_tensor(frame_bgr, device, fp16):
    """BGR uint8 HxWx3  ->  float tensor 1x3xHxW scaled to [-1, 1]."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t   = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    # Scale [0, 1] -> [-1, 1]
    t   = t * 2.0 - 1.0
    t   = t.to(device)
    if fp16 and device.type == "cuda":
        t = t.half()
    return t


def tensor_to_frame(t):
    """1x3xHxW float tensor in [-1, 1] -> BGR uint8 HxWx3."""
    out = t.squeeze(0).permute(1, 2, 0).float().clamp(-1.0, 1.0)
    # Scale [-1, 1] -> [0, 1]
    out = (out * 0.5 + 0.5).clamp(0.0, 1.0).cpu().numpy()
    arr = (out * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dehaze_single_frame(model, frame_path, output_path, device, max_res, fp16):
    """Dehaze a single image file."""
    if not os.path.isfile(frame_path):
        print(f"[ERROR] Frame not found: {frame_path}")
        sys.exit(1)

    frame_bgr = cv2.imread(frame_path)
    frame_rs, _ = resize_if_needed(frame_bgr, max_res)
    inp = frame_to_tensor(frame_rs, device, fp16)

    with torch.no_grad():
        out = model(inp)

    out_bgr = tensor_to_frame(out)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, out_bgr)
    print(f"\n[DONE] Dehazed single frame saved to: {output_path}")


def ensure_quicktime_compatible(video_path):
    """Ensure video uses H.264 / yuv420p so QuickTime Player on macOS can play it."""
    import shutil, subprocess
    ffmpeg_bin = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    if not os.path.isfile(video_path):
        return
    if not os.path.isfile(ffmpeg_bin):
        return
    tmp_path = str(video_path) + ".h264.mp4"
    cmd = [
        ffmpeg_bin, "-y", "-v", "error",
        "-i", str(video_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        tmp_path
    ]
    try:
        ret = subprocess.run(cmd)
        if ret.returncode == 0 and os.path.isfile(tmp_path):
            os.replace(tmp_path, video_path)
    except Exception:
        pass


def main():
    args = parse_args()
    device = get_device()
    use_fp16 = args.fp16 and device.type == "cuda"
    print(f"Device: {device} ({'Apple Silicon GPU' if device.type == 'mps' else 'CUDA GPU' if device.type == 'cuda' else 'CPU'})  FP16: {use_fp16}")

    model = load_model(args.weights, device, use_fp16)

    # --- Mode 1: Single Frame Dehazing ---
    if args.frame:
        out_path = args.output
        if out_path.endswith(".mp4"):
            out_path = "results/dehazed_frame.png"
        dehaze_single_frame(model, args.frame, out_path, device, args.max_res, use_fp16)
        return

    # --- Mode 2: Video or Directory of Frames ---
    if not args.input:
        print("[ERROR] Please provide either --input <video.mp4> or --frame <image.jpg>")
        sys.exit(1)

    is_dir = os.path.isdir(args.input)
    if is_dir:
        valid_exts = {".png", ".jpg", ".jpeg"}
        frame_paths = sorted([os.path.join(args.input, f) for f in os.listdir(args.input)
                              if os.path.splitext(f)[1].lower() in valid_exts])
        if not frame_paths:
            print(f"[ERROR] No image frames found in directory: {args.input}")
            sys.exit(1)
        fps = 25.0
        n_frames = len(frame_paths)
        first_frame = cv2.imread(frame_paths[0])
        print(f"Input folder: {args.input}  |  Frames: {n_frames}  |  FPS: {fps}")
    else:
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {args.input}")
            sys.exit(1)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, first_frame = cap.read()
        if not ret:
            print("[ERROR] Could not read first frame.")
            sys.exit(1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print(f"Input video: {args.input}  |  FPS: {fps:.2f}  |  Frames: {n_frames}")

    first_frame_rs, _ = resize_if_needed(first_frame, args.max_res)
    oh, ow = first_frame_rs.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    writer = cv2.VideoWriter(args.output, fourcc, fps, (ow, oh))

    prev_out = None

    with torch.no_grad():
        for frame_idx in range(n_frames):
            if is_dir:
                frame_bgr = cv2.imread(frame_paths[frame_idx])
            else:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

            frame_rs, _ = resize_if_needed(frame_bgr, args.max_res)
            inp = frame_to_tensor(frame_rs, device, use_fp16)

            with torch.autocast(device_type=device.type,
                                dtype=torch.float16 if use_fp16 else torch.float32,
                                enabled=use_fp16):
                out = model(inp)

            out = out.float()

            # Temporal EMA smoothing across consecutive frames (if enabled)
            if not args.no_smooth:
                if prev_out is None:
                    prev_out = out.clone()
                else:
                    out      = EMA_ALPHA * out + (1.0 - EMA_ALPHA) * prev_out
                    prev_out = out.clone()

            out_bgr = tensor_to_frame(out)
            writer.write(out_bgr)

            if (frame_idx + 1) % 15 == 0 or frame_idx == n_frames - 1:
                pct = (frame_idx + 1) / max(n_frames, 1) * 100
                print(f"  Processed {frame_idx + 1}/{n_frames} ({pct:.1f}%)", flush=True)

    if not is_dir:
        cap.release()
    writer.release()
    ensure_quicktime_compatible(args.output)
    print(f"\n[DONE] Dehazed video saved to: {args.output}")


if __name__ == "__main__":
    main()
