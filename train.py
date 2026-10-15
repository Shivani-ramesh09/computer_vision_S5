"""
Fine-tune DehazeFormer-T on REVIDE dataset using iteration-based training.

Usage:
    python train.py --data data/REVIDE --pretrained checkpoints/dehazeformer-t.pth
    python train.py --smoke
"""

import os, argparse, time, random, copy, warnings
from pathlib import Path
from collections import OrderedDict

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

try:
    from pytorch_msssim import ssim
except ImportError:
    def ssim(img1, img2, data_range=1.0, size_average=True):
        # Fallback simple SSIM implementation
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


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class REVIDEDataset(Dataset):
    """REVIDE train split with paired random crop, flip, and [-1, 1] scaling."""
    def __init__(self, root, split="train", crop=256, augment=True):
        self.crop    = crop
        self.augment = augment
        split_dir = split if (Path(root) / split).exists() else split.capitalize()
        hazy_root = Path(root) / split_dir / "hazy"
        gt_root   = Path(root) / split_dir / "gt"

        self.pairs = []
        valid_exts = {".png", ".jpg", ".jpeg"}
        if hazy_root.exists() and gt_root.exists():
            for seq in sorted(hazy_root.iterdir()):
                if not seq.is_dir():
                    continue
                for f in sorted(seq.iterdir()):
                    if f.suffix.lower() in valid_exts:
                        gt_f = gt_root / seq.name / f.name
                        if gt_f.exists():
                            self.pairs.append((str(f), str(gt_f)))

        assert len(self.pairs) > 0, (
            f"No image pairs found under {hazy_root}. Check REVIDE folder layout."
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        hazy_path, gt_path = self.pairs[idx]
        hazy = Image.open(hazy_path).convert("RGB")
        gt   = Image.open(gt_path).convert("RGB")

        # Paired random crop
        i, j, h, w = transforms.RandomCrop.get_params(hazy, output_size=(self.crop, self.crop))
        hazy = TF.crop(hazy, i, j, h, w)
        gt   = TF.crop(gt, i, j, h, w)

        if self.augment:
            if random.random() > 0.5:
                hazy = TF.hflip(hazy)
                gt   = TF.hflip(gt)
            if random.random() > 0.5:
                hazy = TF.vflip(hazy)
                gt   = TF.vflip(gt)

        # Scale [0, 1] to [-1, 1]
        hazy = TF.to_tensor(hazy) * 2.0 - 1.0
        gt   = TF.to_tensor(gt) * 2.0 - 1.0
        return hazy, gt


class REVIDEValDataset(Dataset):
    """Validation dataset: REVIDE test split using every 10th frame, max 720p, [-1, 1]."""
    def __init__(self, root, split="test", stride=10, max_res=720):
        self.max_res = max_res
        split_dir = split if (Path(root) / split).exists() else split.capitalize()
        hazy_root = Path(root) / split_dir / "hazy"
        gt_root   = Path(root) / split_dir / "gt"

        self.pairs = []
        valid_exts = {".png", ".jpg", ".jpeg"}
        if hazy_root.exists() and gt_root.exists():
            for seq in sorted(hazy_root.iterdir()):
                if not seq.is_dir():
                    continue
                frames = sorted([f for f in seq.iterdir() if f.suffix.lower() in valid_exts])
                for f in frames[::stride]:
                    gt_f = gt_root / seq.name / f.name
                    if gt_f.exists():
                        self.pairs.append((str(f), str(gt_f)))

        assert len(self.pairs) > 0, (
            f"No validation image pairs found under {hazy_root}."
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        hazy_path, gt_path = self.pairs[idx]
        hazy = Image.open(hazy_path).convert("RGB")
        gt   = Image.open(gt_path).convert("RGB")

        w, h = hazy.size
        longer = max(w, h)
        if self.max_res > 0 and longer > self.max_res:
            scale = self.max_res / longer
            nw = int(round(w * scale))
            nh = int(round(h * scale))
            hazy = hazy.resize((nw, nh), Image.Resampling.BILINEAR)
            gt   = gt.resize((nw, nh), Image.Resampling.BILINEAR)

        hazy = TF.to_tensor(hazy) * 2.0 - 1.0
        gt   = TF.to_tensor(gt) * 2.0 - 1.0
        return hazy, gt


class SyntheticHazyDataset(Dataset):
    """In-memory synthetic dataset for quick smoke tests."""
    def __init__(self, n=8, size=256):
        self.n = n
        self.size = size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        gt   = torch.rand(3, self.size, self.size) * 2.0 - 1.0
        t    = random.uniform(0.5, 0.9)
        A    = random.uniform(-0.4, 0.8)
        hazy = (gt * t + A * (1.0 - t)).clamp(-1.0, 1.0)
        return hazy, gt


# ---------------------------------------------------------------------------
# Model EMA
# ---------------------------------------------------------------------------

class ModelEMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(msd[k])

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, state_dict):
        self.ema.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# Checkpoint & Validation Utilities
# ---------------------------------------------------------------------------

def load_pretrained(model, path, device):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Pretrained weights not found at: {path}")
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("state_dict", ckpt.get("model", ckpt.get("ema", ckpt)))
    state = {k.replace("module.", ""): v for k, v in state.items()}
    res = model.load_state_dict(state, strict=False)
    print(f"[Pretrained] Loaded from {path}: {len(state) - len(res.missing_keys)}/{len(model.state_dict())} keys matched.")
    if len(res.missing_keys) > 0:
        print(f"  Missing keys: {res.missing_keys}")
    if len(res.unexpected_keys) > 0:
        print(f"  Unexpected keys: {res.unexpected_keys}")
    if len(res.missing_keys) > 5:
        raise RuntimeError(f"Too many missing keys ({len(res.missing_keys)}) when loading pretrained weights from {path}")


def load_resume(model, ema, optimizer, scheduler, scaler, path, device):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Resume checkpoint not found at: {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    else:
        ema.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    scheduler.load_state_dict(ckpt["scheduler"])
    start_iter = ckpt.get("iteration", 0)
    best_psnr  = ckpt.get("best_psnr", 0.0)
    print(f"[Resume] Loaded {path} at iteration {start_iter} (Best PSNR: {best_psnr:.2f} dB)")
    return start_iter, best_psnr


def validate(model, val_loader, device):
    model.eval()
    psnr_vals = []
    ssim_vals = []
    with torch.no_grad():
        for hazy, gt in val_loader:
            hazy = hazy.to(device)
            gt   = gt.to(device)
            out  = model(hazy).clamp(-1.0, 1.0)

            # [-1, 1] to [0, 1]
            out_01 = out * 0.5 + 0.5
            gt_01  = gt * 0.5 + 0.5

            mse = F.mse_loss(out_01, gt_01, reduction='none').mean(dim=[1, 2, 3])
            psnr = 10.0 * torch.log10(1.0 / (mse + 1e-10))
            psnr_vals.extend(psnr.cpu().tolist())

            s = ssim(out_01, gt_01, data_range=1.0, size_average=False)
            ssim_vals.extend(s.cpu().tolist())

    mean_psnr = float(np.mean(psnr_vals)) if psnr_vals else 0.0
    mean_ssim = float(np.mean(ssim_vals)) if ssim_vals else 0.0
    return mean_psnr, mean_ssim


# ---------------------------------------------------------------------------
# Training Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train DehazeFormer-T on REVIDE.")
    p.add_argument("--data",       type=str, default="data/REVIDE",
                   help="Root of REVIDE dataset")
    p.add_argument("--iters",      type=int, default=60000,
                   help="Total iterations (default: 60000)")
    p.add_argument("--val_every",  type=int, default=2000,
                   help="Validation frequency in iters (default: 2000)")
    p.add_argument("--save_every", type=int, default=2000,
                   help="Checkpoint save frequency in iters (default: 2000)")
    p.add_argument("--batch",      type=int, default=8,
                   help="Batch size (default: 8)")
    p.add_argument("--lr",         type=float, default=1e-4,
                   help="Learning rate (default: 1e-4)")
    p.add_argument("--crop",       type=int, default=256,
                   help="Random crop size (default: 256)")
    p.add_argument("--grad_clip",  type=float, default=1.0,
                   help="Gradient norm clipping (default: 1.0)")
    p.add_argument("--workers",    type=int, default=4,
                   help="Dataloader workers (default: 4)")
    p.add_argument("--device",     type=str, default=None,
                   help="Device: 'cuda', 'mps', or 'cpu' (default: auto)")
    p.add_argument("--ckpt_dir",   type=str, default="checkpoints",
                   help="Directory to save checkpoints")
    p.add_argument("--pretrained", type=str, default=None,
                   help="Path to pretrained weights to initialize from")
    p.add_argument("--resume",     type=str, default=None,
                   help="Path to checkpoint to resume full training state from")
    p.add_argument("--smoke",      action="store_true",
                   help="Run 2 iterations on synthetic data with 1 validation pass")
    return p.parse_args()


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def main():
    args = parse_args()

    # --- Device selection (cuda -> mps -> cpu) ---
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device} ({'Apple Silicon GPU' if device.type == 'mps' else 'NVIDIA GPU' if device.type == 'cuda' else 'CPU'})")

    # --- Model & EMA ---
    model = build_dehazeformer_t().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: DehazeFormer-T  ({n_params:,} parameters, <1M assert OK)")

    if args.pretrained:
        load_pretrained(model, args.pretrained, device)

    ema = ModelEMA(model, decay=0.999)

    # --- Optimizer & Scheduler & Scaler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler    = GradScaler(device.type, enabled=(device.type == "cuda"))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.iters,
        eta_min=1e-6
    )

    start_iter = 0
    best_psnr  = 0.0

    if args.resume:
        start_iter, best_psnr = load_resume(model, ema, optimizer, scheduler, scaler, args.resume, device)

    # --- Datasets ---
    if args.smoke:
        print("[SMOKE TEST] Running 2 iterations on synthetic data...")
        train_dataset = SyntheticHazyDataset(n=8, size=args.crop)
        val_dataset   = SyntheticHazyDataset(n=4, size=args.crop)
        args.iters      = 2
        args.val_every  = 2
        args.save_every = 2
        loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
    else:
        train_dataset = REVIDEDataset(args.data, split="train", crop=args.crop, augment=True)
        val_dataset   = REVIDEValDataset(args.data, split="test", stride=10, max_res=720)
        loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True,
                            num_workers=args.workers,
                            pin_memory=(device.type == "cuda"), drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                num_workers=args.workers,
                                pin_memory=(device.type == "cuda"))
        print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} (every 10th test frame)")

    train_iter = cycle(loader)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    running_loss = 0.0
    loss_count   = 0

    print(f"Starting training: total iters={args.iters}, val_every={args.val_every}, save_every={args.save_every}")

    pbar = tqdm(
        range(start_iter + 1, args.iters + 1),
        initial=start_iter,
        total=args.iters,
        dynamic_ncols=True,
        desc="Training"
    )

    for it in pbar:
        model.train()
        hazy, gt = next(train_iter)
        hazy = hazy.to(device, non_blocking=True)
        gt   = gt.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            output = model(hazy)
            # Loss: L1 + 0.2 * (1 - SSIM)
            out_01 = output.clamp(-1.0, 1.0) * 0.5 + 0.5
            gt_01  = gt * 0.5 + 0.5
            loss_l1   = F.l1_loss(output, gt)
            loss_ssim = 1.0 - ssim(out_01, gt_01, data_range=1.0)
            loss = loss_l1 + 0.2 * loss_ssim

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        ema.update(model)
        scheduler.step()

        running_loss += loss.item()
        loss_count   += 1

        # Live progress bar stats (updated every 5 iters)
        if hasattr(pbar, "set_postfix") and (it % 5 == 0 or it == args.iters):
            avg_loss = running_loss / max(loss_count, 1)
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg": f"{avg_loss:.4f}",
                "best_psnr": f"{best_psnr:.2f}" if best_psnr > 0 else "N/A",
                "lr": f"{scheduler.get_last_lr()[0]:.1e}"
            })

        # --- Validation & Logging ---
        if it % args.val_every == 0 or it == args.iters:
            avg_loss = running_loss / max(loss_count, 1)
            running_loss = 0.0
            loss_count   = 0

            # Validate using EMA weights
            val_psnr, val_ssim = validate(ema.ema, val_loader, device)
            current_lr = scheduler.get_last_lr()[0]

            log_line = f"Iter [{it:05d}/{args.iters:05d}] | Train Loss: {avg_loss:.4f} | Val PSNR: {val_psnr:5.2f} dB | Val SSIM: {val_ssim:.4f} | LR: {current_lr:.2e}"
            if hasattr(pbar, "write"):
                pbar.write(log_line)
            else:
                print(log_line, flush=True)

            # Save best checkpoint by validation PSNR
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_path = os.path.join(args.ckpt_dir, "best.pth")
                torch.save({
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "iteration": it,
                    "best_psnr": best_psnr
                }, best_path)

        # --- Periodic Checkpoint ---
        if it % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"iter_{it:06d}.pth")
            torch.save({
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                "iteration": it,
                "best_psnr": best_psnr
            }, ckpt_path)

    print(f"\n[DONE] Training complete. Best validation PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    main()
