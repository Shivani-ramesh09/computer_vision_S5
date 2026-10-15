"""
Blind Video Dehazing Test:
Takes an isolated hazy MP4 video file and runs 100% blind DehazeFormer-T inference
(no transmission maps, no optical hints, no synthetic shortcuts).

Usage:
    python test_dehaze.py \
      --input results/hazy_input.mp4 \
      --weights checkpoints/best.pth \
      --gt path/to/clean_video.mp4 \
      --out_dir results/blind_test
"""

import argparse, os, sys, shutil, tempfile, subprocess
from pathlib import Path
import cv2
import numpy as np
import torch

from model import build_dehazeformer_t


def parse_args():
    p = argparse.ArgumentParser(description="Blind DehazeFormer-T inference on a hazy video.")
    p.add_argument("--input",    required=True,
                   help="Path to hazy MP4 video file")
    p.add_argument("--weights",  default="checkpoints/best.pth",
                   help="Path to model checkpoint")
    p.add_argument("--gt",       default=None,
                   help="Optional path to clean ground truth MP4 for comparison and PSNR/SSIM evaluation")
    p.add_argument("--out_dir",  default="results/blind_test",
                   help="Directory for output videos")
    p.add_argument("--max_frames", type=int, default=150,
                   help="Max frames to process (default: 150)")
    p.add_argument("--ema_alpha", type=float, default=0.85,
                   help="Temporal EMA smoothing factor across frames (0.85 = smooth, 1.0 = raw)")
    return p.parse_args()


def pad_to_multiple(img_bgr, multiple=16):
    h, w = img_bgr.shape[:2]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph or pw:
        img_bgr = cv2.copyMakeBorder(img_bgr, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
    return img_bgr, h, w


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(weights_path, device):
    model = build_dehazeformer_t().to(device)
    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def run_model(model, img_bgr, device):
    """Run model on single frame. 100% pure neural network forward pass."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    out = model(inp).squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return (out[:, :, ::-1] * 255).astype(np.uint8)


def label_frame(img_bgr, text):
    out = img_bgr.copy()
    h, w = out.shape[:2]
    box_w = min(len(text) * 14 + 16, w)
    cv2.rectangle(out, (0, 0), (box_w, 36), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def calc_metrics(img_a, img_b):
    mse = np.mean((img_a.astype(np.float64) - img_b.astype(np.float64)) ** 2)
    psnr = 20.0 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 99.0
    try:
        from skimage.metrics import structural_similarity as ssim
        val_ssim = ssim(img_a, img_b, channel_axis=2)
    except Exception:
        val_ssim = 0.0
    return psnr, val_ssim


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
    device = get_device()
    print(f"Device : {device} ({'Apple Silicon GPU' if device.type == 'mps' else 'CPU'})")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading weights from: {args.weights}")
    model = load_model(args.weights, device)

    # Open hazy video
    cap_hazy = cv2.VideoCapture(args.input)
    if not cap_hazy.isOpened():
        print(f"[ERROR] Cannot open hazy video: {args.input}")
        sys.exit(1)
    fps = cap_hazy.get(cv2.CAP_PROP_FPS) or 30.0
    total_hazy = int(cap_hazy.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Input hazy video : {args.input} ({total_hazy} frames, {fps:.2f} FPS)")

    # Open GT video if provided
    cap_gt = None
    if args.gt:
        if not os.path.exists(args.gt):
            print(f"[WARN] GT video not found at: {args.gt}. Skipping GT comparison.")
        else:
            cap_gt = cv2.VideoCapture(args.gt)
            print(f"Ground truth video: {args.gt}")

    dehazed_frames = []
    side_frames = []
    psnr_hazy_list, psnr_deh_list = [], []
    ssim_hazy_list, ssim_deh_list = [], []

    prev_dehazed = None
    frame_idx = 0
    max_f = min(args.max_frames, total_hazy)

    print(f"\nRunning 100% blind DehazeFormer-T on {max_f} frames...")
    while cap_hazy.isOpened() and frame_idx < max_f:
        ok_h, hazy_bgr = cap_hazy.read()
        if not ok_h:
            break

        h, w = hazy_bgr.shape[:2]

        # 1. Blind inference through DehazeFormer-T
        hazy_padded, orig_h, orig_w = pad_to_multiple(hazy_bgr, 16)
        raw_out = run_model(model, hazy_padded, device)[:orig_h, :orig_w]

        # 2. Temporal smoothing to reduce video frame flicker
        if prev_dehazed is None:
            dehazed_bgr = raw_out
            prev_dehazed = raw_out.astype(np.float32)
        else:
            alpha = args.ema_alpha
            dehazed_f = alpha * raw_out.astype(np.float32) + (1.0 - alpha) * prev_dehazed
            dehazed_bgr = np.clip(dehazed_f, 0, 255).astype(np.uint8)
            prev_dehazed = dehazed_f

        dehazed_frames.append(dehazed_bgr)

        # 3. If GT video is available, read matching frame and evaluate
        if cap_gt is not None:
            ok_g, gt_bgr = cap_gt.read()
            if ok_g:
                # Resize GT to match hazy resolution
                gt_bgr = cv2.resize(gt_bgr, (w, h))

                p_h, s_h = calc_metrics(hazy_bgr, gt_bgr)
                p_d, s_d = calc_metrics(dehazed_bgr, gt_bgr)
                psnr_hazy_list.append(p_h)
                psnr_deh_list.append(p_d)
                ssim_hazy_list.append(s_h)
                ssim_deh_list.append(s_d)

                panel_h = label_frame(hazy_bgr, "Isolated Hazy Input")
                panel_d = label_frame(dehazed_bgr, "DehazeFormer-T (Blind)")
                panel_g = label_frame(gt_bgr, "Ground Truth")
                side = np.concatenate([panel_h, panel_d, panel_g], axis=1)
                side_frames.append(side)

        frame_idx += 1
        if frame_idx % 15 == 0 or frame_idx == max_f:
            if psnr_hazy_list:
                print(f"  [{frame_idx}/{max_f}] Hazy: {p_h:.1f} dB -> Dehazed: {p_d:.1f} dB", flush=True)
            else:
                print(f"  [{frame_idx}/{max_f}] Processed frame {frame_idx}", flush=True)

    cap_hazy.release()
    if cap_gt is not None:
        cap_gt.release()

    # Quantitative Summary
    if psnr_hazy_list:
        avg_ph, avg_pd = np.mean(psnr_hazy_list), np.mean(psnr_deh_list)
        avg_sh, avg_sd = np.mean(ssim_hazy_list), np.mean(ssim_deh_list)
        print("\n" + "=" * 55)
        print("        BLIND MODEL EVALUATION SUMMARY")
        print("=" * 55)
        print(f"  Isolated Hazy Input : PSNR = {avg_ph:6.2f} dB  |  SSIM = {avg_sh:6.4f}")
        print(f"  DehazeFormer-T      : PSNR = {avg_pd:6.2f} dB  |  SSIM = {avg_sd:6.4f}")
        print(f"  Real Gain           : PSNR = {avg_pd - avg_ph:+6.2f} dB  |  SSIM = {avg_sd - avg_sh:+6.4f}")
        print("=" * 55 + "\n")

    # Save output videos
    out_dehazed_path = os.path.join(args.out_dir, "dehazed_isolated.mp4")
    out_cmp_path = os.path.join(args.out_dir, "comparison_isolated.mp4")
    out_prev_path = os.path.join(args.out_dir, "comparison_preview.jpg")

    print(f"Writing dehazed video to: {out_dehazed_path}")
    make_video(out_dehazed_path, dehazed_frames, fps, (w, h))

    if side_frames:
        print(f"Writing comparison video to: {out_cmp_path}")
        make_video(out_cmp_path, side_frames, fps, (w * 3, h))
        cv2.imwrite(out_prev_path, side_frames[len(side_frames) // 2])

    print(f"\n[SUCCESS] Completed blind testing! Files saved in {args.out_dir}/:")
    print(f"  Dehazed video : {out_dehazed_path}")
    if side_frames:
        print(f"  Comparison    : {out_cmp_path}")
        print(f"  Preview image : {out_prev_path}")


if __name__ == "__main__":
    main()
