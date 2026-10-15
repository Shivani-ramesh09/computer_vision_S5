"""
Standalone Fog Generator:
Applies synthetic atmospheric scattering haze to any MP4 video and saves it
as an independent hazy video file.

Usage:
    python add_fog.py --input path/to/video.mp4 --output results/hazy_input.mp4 --beta 1.5
"""

import argparse, os, sys, shutil, tempfile, subprocess
from pathlib import Path
import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic hazy video from a clean MP4.")
    p.add_argument("--input",      required=True,
                   help="Path to source clean MP4 video")
    p.add_argument("--output",     default="results/hazy_input.mp4",
                   help="Path for saved hazy video")
    p.add_argument("--max_frames", type=int, default=150,
                   help="Max frames to process (default: 150)")
    p.add_argument("--start_frame", type=int, default=0,
                   help="Starting frame index")
    p.add_argument("--res",        type=int, default=480,
                   help="Resize shorter edge (default: 480)")
    p.add_argument("--beta",       type=float, default=1.5,
                   help="Fog thickness density (default: 1.5)")
    p.add_argument("--haze_mode",  choices=["perspective", "radial", "uniform"], default="perspective",
                   help="Depth model: 'perspective' (outdoor/driving), 'radial', or 'uniform'")
    return p.parse_args()


def resize_frame(img_bgr, target_short=480):
    h, w = img_bgr.shape[:2]
    short = min(h, w)
    if short > target_short:
        scale = target_short / short
        new_w = int(w * scale)
        new_h = int(h * scale)
    else:
        new_w, new_h = w, h
    new_w = max(((new_w) // 16) * 16, 16)
    new_h = max(((new_h) // 16) * 16, 16)
    if new_w != w or new_h != h:
        return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_bgr


def add_fog(img_bgr, beta=1.5, A=0.92, mode="perspective"):
    """Physical scattering model: I = J*t + A*(1-t), t = exp(-beta * depth)."""
    h, w = img_bgr.shape[:2]
    if mode == "perspective":
        Y, X = np.mgrid[0:h, 0:w]
        y_norm = 1.0 - (Y / float(h))
        depth = 0.35 + 0.65 * y_norm
        cx, cy_h = w / 2.0, h * 0.45
        rad = np.sqrt(((X - cx) / w) ** 2 + ((Y - cy_h) / h) ** 2)
        depth = np.clip(depth + 0.15 * rad, 0.30, 1.0)
    elif mode == "radial":
        cx, cy = w / 2, h / 2
        Y, X = np.mgrid[0:h, 0:w]
        d = np.sqrt(((X - cx) / (cx + 1e-8)) ** 2 + ((Y - cy) / (cy + 1e-8)) ** 2)
        depth = np.clip(0.35 + 0.65 * (d / (d.max() + 1e-8)), 0.30, 1.0)
    else:
        depth = np.ones((h, w), dtype=np.float32)

    t = np.exp(-beta * depth).astype(np.float32)[..., None]
    img_f = img_bgr.astype(np.float32) / 255.0
    hazy = np.clip(img_f * t + A * (1.0 - t), 0, 1)
    return (hazy * 255).astype(np.uint8)


def make_video(path, frames, fps, size):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if shutil.which("ffmpeg"):
        tmp = tempfile.mkdtemp()
        for i, f in enumerate(frames):
            cv2.imwrite(os.path.join(tmp, f"{i:06d}.png"), f)
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmp, "%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
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


def main():
    args = parse_args()
    if not os.path.exists(args.input):
        print(f"[ERROR] Input video not found: {args.input}")
        sys.exit(1)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.input}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Input video : {args.input}")
    print(f"Source FPS  : {fps:.2f} | Total frames: {total}")

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    frames = []
    print(f"Applying synthetic fog (beta={args.beta}, mode={args.haze_mode})...")
    while cap.isOpened() and len(frames) < args.max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rs = resize_frame(frame, args.res)
        hazy_frame = add_fog(frame_rs, beta=args.beta, mode=args.haze_mode)
        frames.append(hazy_frame)

    cap.release()
    print(f"Generated {len(frames)} hazy frames.")

    h, w = frames[0].shape[:2]
    print(f"Encoding output video to: {args.output}")
    make_video(args.output, frames, fps, (w, h))
    print(f"\n[DONE] Saved isolated hazy video to: {args.output}")


if __name__ == "__main__":
    main()
