# Video Dehazing with DehazeFormer-T

Lightweight per-frame video dehazing using **DehazeFormer-T** (~0.7 M params) with temporal EMA smoothing.
Architecture: [IDKiro/DehazeFormer](https://github.com/IDKiro/DehazeFormer) (MIT License).

---

## 1  Environment Setup

```bash
# Python 3.9+
conda create -n dehaze python=3.9 -y
conda activate dehaze

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # adjust CUDA ver
pip install opencv-python-headless pillow numpy scikit-image einops thop
```

---

## 2  Pretrained Weights (RESIDE-OTS)

Official RESIDE-OTS pretrained weights for DehazeFormer-T are **not auto-downloaded**.

1. Visit [https://github.com/IDKiro/DehazeFormer/releases](https://github.com/IDKiro/DehazeFormer/releases)
2. Download **`dehazeformer-t.pth`** (RESIDE-OTS variant)
3. Place it at:

```
checkpoints/
└── dehazeformer-t.pth
```

You can then pass `--weights checkpoints/dehazeformer-t.pth` to any script.

---

## 3  Dataset Layout

### REVIDE (for training & eval)

```
data/
└── REVIDE/
    ├── train/
    │   ├── hazy/
    │   │   ├── seq_001/
    │   │   │   ├── 000001.png
    │   │   │   └── ...
    │   │   └── seq_002/ ...
    │   └── gt/
    │       ├── seq_001/
    │       │   ├── 000001.png
    │       │   └── ...
    │       └── seq_002/ ...
    └── test/
        ├── hazy/ ...
        └── gt/ ...
```

Download REVIDE from: [https://github.com/BookerDeWitt/REVIDE_Dataset](https://github.com/BookerDeWitt/REVIDE_Dataset)

---

## 4  Commands (copy-paste ready)

### Sanity check — model params & forward pass

```bash
python model.py
```

### Model statistics (params, MACs, FPS)

```bash
python model_stats.py
# With checkpoint on disk:
python model_stats.py --weights checkpoints/best.pth
```

### Smoke test — 2-iteration training on synthetic data

```bash
python train.py --smoke
```

### Train from scratch on REVIDE

```bash
python train.py \
  --data   data/REVIDE \
  --epochs 50 \
  --batch  4 \
  --lr     2e-4 \
  --crop   256 \
  --workers 4 \
  --ckpt_dir checkpoints
```

### Resume from checkpoint

```bash
python train.py \
  --data   data/REVIDE \
  --epochs 50 \
  --batch  4 \
  --lr     2e-4 \
  --ckpt_dir checkpoints \
  --resume checkpoints/epoch_010.pth
```

### Infer on a video

```bash
# Basic
python infer_video.py \
  --input   hazy_video.mp4 \
  --output  dehazed_output.mp4 \
  --weights checkpoints/best.pth

# Downscale to 720p + FP16 (GPU)
python infer_video.py \
  --input   hazy_video.mp4 \
  --output  dehazed_output.mp4 \
  --weights checkpoints/best.pth \
  --max_res 720 \
  --fp16
```

### Evaluate PSNR / SSIM / Flicker vs DCP baseline

```bash
python eval.py \
  --data    data/REVIDE \
  --weights checkpoints/best.pth \
  --split   test

# Quick eval (max 30 frames per sequence)
python eval.py \
  --data      data/REVIDE \
  --weights   checkpoints/best.pth \
  --split     test \
  --max_frames 30
```

---

## 5  File Overview

| File | Purpose |
|------|---------|
| `model.py` | DehazeFormer-T architecture (<1 M params) |
| `train.py` | Fine-tune on REVIDE (L1, AdamW, AMP, checkpoints) |
| `infer_video.py` | Per-frame inference + EMA temporal smoothing |
| `eval.py` | PSNR/SSIM/Flicker vs DCP baseline |
| `model_stats.py` | Params, MACs, FPS; saves `results/model_stats.txt` |

---

## 6  Notes

- **Temporal smoothing**: EMA with α=0.8 (`current_frame × 0.8 + prev_frame × 0.2`) applied after each dehazed frame to reduce flicker without optical flow.
- **Param guarantee**: `build_dehazeformer_t()` asserts `params < 1,000,000` at construction time.
- **AMP**: Mixed precision is enabled automatically on CUDA; falls back to FP32 on CPU.
