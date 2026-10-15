"""
Model statistics: params, file size, MACs @ 256x256 (thop), FPS @ 480p.
Saves to results/model_stats.txt.

Usage:
    python model_stats.py
    python model_stats.py --weights checkpoints/dehazeformer-t.pth
"""

import argparse, os, time
import torch

from model import build_dehazeformer_t


def fmt(n):
    return f"{n:,}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default=None,
                   help="Optional: path to .pth file to report disk size")
    p.add_argument("--fps_runs", type=int, default=100,
                   help="Number of forward passes for FPS measurement")
    return p.parse_args()


def main():
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device} ({'Apple Silicon GPU' if device.type == 'mps' else 'CUDA GPU' if device.type == 'cuda' else 'CPU'})")

    model = build_dehazeformer_t().to(device)
    model.eval()

    # --- Param count ---
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params < 1_000_000, f"Param count {total_params:,} >= 1M!"

    # --- File size from weights on disk ---
    file_size_mb = "N/A (weights not provided)"
    if args.weights and os.path.isfile(args.weights):
        size_bytes   = os.path.getsize(args.weights)
        file_size_mb = f"{size_bytes / 1e6:.2f} MB"

    # --- MACs via thop ---
    dummy_256 = torch.randn(1, 3, 256, 256).to(device)
    try:
        from thop import profile, clever_format
        macs, _ = profile(model, inputs=(dummy_256,), verbose=False)
        macs_str, _ = clever_format([macs, total_params], "%.3f")
    except ImportError:
        macs_str = "thop not installed (pip install thop)"
    except Exception as e:
        macs_str = f"Error: {e}"

    # --- FPS at 480p (854x480) ---
    dummy_480p = torch.randn(1, 3, 480, 854).to(device)
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_480p)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.fps_runs):
            _ = model(dummy_480p)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()

    elapsed = time.perf_counter() - t0
    fps = args.fps_runs / elapsed

    # --- Report ---
    lines = [
        "=" * 50,
        "DehazeFormer-T  —  Model Statistics",
        "=" * 50,
        f"Total parameters    : {fmt(total_params)}  ({total_params/1e6:.4f} M)",
        f"Trainable params    : {fmt(trainable_params)}",
        f"<1M assertion       : {'PASS' if total_params < 1_000_000 else 'FAIL'}",
        f"Checkpoint size     : {file_size_mb}",
        f"MACs @ 256x256      : {macs_str}",
        f"FPS @ 480p (1 img)  : {fps:.1f}  (device={device})",
        "=" * 50,
    ]
    report = "\n".join(lines)
    print(report)

    os.makedirs("results", exist_ok=True)
    with open("results/model_stats.txt", "w") as f:
        f.write(report + "\n")
    print("\nSaved to results/model_stats.txt")


if __name__ == "__main__":
    main()
