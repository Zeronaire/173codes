"""
training script for NAFNet image restoration (denoising / deblurring, etc.)

Folder structure (paired):
  train_lq/
      0001.png
      0002.png
      ...
  train_gt/
      0001.png
      0002.png
      ...
(Same for validation if provided.)

Example:
  python train_single_file.py \
      --train_lq_dir ./datasets/SIDD/train/input_patches \
      --train_gt_dir ./datasets/SIDD/train/gt_patches \
      --val_lq_dir ./datasets/SIDD/val/input_patches \
      --val_gt_dir ./datasets/SIDD/val/gt_patches \
      --save_dir ./experiments/nafnet_sidd_run1 \
      --width 32 --enc_blk_nums 2 2 4 8 --middle_blk_num 12 --dec_blk_nums 2 2 2 2

If you want to resume:
  python train_single_file.py ... --resume latest

This script assumes the repository (megvii-research/NAFNet) is in your PYTHONPATH so that:
  from basicsr.models.archs.NAFNet_arch import NAFNet
works correctly.

If parameter names in the upstream repo differ, adjust the NAFNet(...) call accordingly.
"""

import os
import math
import argparse
import time
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Import the model from the repo
try:
    from nafnet_src.NAFNet_arch import NAFNet
except ImportError as e:
    raise ImportError("Could not import NAFNet. Ensure you run this inside the NAFNet repo or add it to PYTHONPATH.") from e


# ----------------------------
# Dataset
# ----------------------------
class PairedImageFolder(Dataset):
    def __init__(self, lq_dir, gt_dir, transform=None):
        self.lq_dir = Path(lq_dir)
        self.gt_dir = Path(gt_dir)
        self.transform = transform
        self.lq_files = sorted([p for p in self.lq_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif")])
        if len(self.lq_files) == 0:
            raise RuntimeError(f"No images found in {lq_dir}")
        # Match filenames in gt
        self.pairs = []
        for lq_path in self.lq_files:
            gt_path = self.gt_dir / lq_path.name
            if not gt_path.exists():
                raise RuntimeError(f"Ground-truth file {gt_path} not found for {lq_path.name}")
            self.pairs.append((lq_path, gt_path))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lq_path, gt_path = self.pairs[idx]
        lq = Image.open(lq_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGB")

        if self.transform:
            seed = torch.seed()
            torch.manual_seed(seed)
            lq = self.transform(lq)
            torch.manual_seed(seed)
            gt = self.transform(gt)
        else:
            to_tensor = transforms.ToTensor()
            lq = to_tensor(lq)
            gt = to_tensor(gt)

        return {"lq": lq, "gt": gt, "lq_path": str(lq_path), "gt_path": str(gt_path)}


# ----------------------------
# Utilities
# ----------------------------
def psnr(pred, target, eps=1e-10):
    mse = F.mse_loss(pred, target)
    return 10 * torch.log10(1.0 / (mse + eps))


def save_checkpoint(state, path, is_best=False, max_keep=5):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    if is_best:
        best_path = path.parent / "model_best.pth"
        torch.save(state, best_path)

    # Rotate old checkpoints
    ckpts = sorted(path.parent.glob("checkpoint_*.pth"), key=os.path.getmtime)
    if len(ckpts) > max_keep:
        for old in ckpts[:-max_keep]:
            old.unlink(missing_ok=True)


def get_cosine_lr(iter_idx, total_iters, base_lr, eta_min=1e-7):
    # True cosine annealing as used in configs
    return eta_min + 0.5 * (base_lr - eta_min) * (1 + math.cos(math.pi * iter_idx / total_iters))


def str2list_int(values: List[str]):
    return [int(v) for v in values]


# ----------------------------
# Training Loop
# ----------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Transforms (basic; add flips/rots if wanted)
    tfm_train = transforms.Compose([
        transforms.RandomHorizontalFlip() if args.augment else transforms.Lambda(lambda x: x),
        transforms.RandomVerticalFlip() if args.augment else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
    ])
    tfm_val = transforms.ToTensor()

    train_dataset = PairedImageFolder(args.train_lq_dir, args.train_gt_dir, transform=tfm_train)
    val_dataset = None
    if args.val_lq_dir and args.val_gt_dir:
        val_dataset = PairedImageFolder(args.val_lq_dir, args.val_gt_dir, transform=tfm_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=max(1, args.num_workers // 2),
            pin_memory=True,
            drop_last=False,
        )

    # Model
    # Adjust constructor param names if upstream differs.
    model = NAFNet(
        img_channel=args.in_channels,
        out_channel=args.out_channels,
        width=args.width,
        enc_blk_nums=args.enc_blk_nums,
        middle_blk_num=args.middle_blk_num,
        dec_blk_nums=args.dec_blk_nums,
    ).to(device)

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    # Optimizer
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.9),
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_iter = 0
    best_psnr = -1.0
    save_dir = Path(args.save_dir)
    ckpt_dir = save_dir / "checkpoints"
    logs_path = save_dir / "train_log.txt"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Resume
    if args.resume:
        if args.resume == "latest":
            ckpts = sorted(ckpt_dir.glob("checkpoint_*.pth"), key=os.path.getmtime)
            if ckpts:
                resume_path = ckpts[-1]
            else:
                raise FileNotFoundError("No checkpoints to resume from.")
        else:
            resume_path = Path(args.resume)
        state = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(state["model"])
        optim.load_state_dict(state["optim"])
        scaler.load_state_dict(state["scaler"])
        start_iter = state["iter"] + 1
        best_psnr = state.get("best_psnr", best_psnr)
        print(f"Resumed from {resume_path}, starting at iter {start_iter}")

    # Loss (L1 or MSE)
    if args.loss == "l1":
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()

    total_iters = args.total_iters
    print_every = args.print_freq
    val_every = args.val_freq if val_loader else None
    ckpt_every = args.ckpt_freq

    with open(logs_path, "a") as f:
        f.write(f"Start training (start_iter={start_iter}, total_iters={total_iters})\n")

    model.train()

    t_start = time.time()
    data_iter = iter(train_loader)
    for it in range(start_iter, total_iters):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        lq = batch["lq"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)

        # Update LR (cosine)
        lr = get_cosine_lr(it, total_iters - 1, args.lr, eta_min=args.eta_min)
        for pg in optim.param_groups:
            pg["lr"] = lr

        optim.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            pred = model(lq)
            loss = criterion(pred, gt)

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optim)
        scaler.update()

        if (it + 1) % print_every == 0 or it == 0:
            elapsed = time.time() - t_start
            msg = f"[Iter {it+1}/{total_iters}] loss={loss.item():.4f} lr={lr:.3e} elapsed={elapsed/60:.1f}m"
            print(msg)
            with open(logs_path, "a") as f:
                f.write(msg + "\n")

        # Validation
        if val_every and ((it + 1) % val_every == 0 or (it + 1) == total_iters):
            model.eval()
            val_loss_acc = 0.0
            val_psnr_acc = 0.0
            count = 0
            with torch.no_grad():
                for vb in val_loader:
                    lqv = vb["lq"].to(device)
                    gtv = vb["gt"].to(device)
                    with torch.cuda.amp.autocast(enabled=args.amp):
                        predv = model(lqv)
                        vloss = criterion(predv, gtv)
                    val_loss_acc += vloss.item() * lqv.size(0)
                    val_psnr_acc += psnr(torch.clamp(predv, 0, 1), gtv).item() * lqv.size(0)
                    count += lqv.size(0)
            val_loss = val_loss_acc / count
            val_psnr_mean = val_psnr_acc / count
            is_best = val_psnr_mean > best_psnr
            if is_best:
                best_psnr = val_psnr_mean
            msg = f"[Val @ iter {it+1}] val_loss={val_loss:.4f} val_psnr={val_psnr_mean:.3f} best_psnr={best_psnr:.3f}"
            print(msg)
            with open(logs_path, "a") as f:
                f.write(msg + "\n")
            # Save after validation
            state = {
                "iter": it,
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "scaler": scaler.state_dict(),
                "best_psnr": best_psnr,
                "args": vars(args),
            }
            save_checkpoint(state, ckpt_dir / f"checkpoint_{it+1:08d}.pth", is_best=is_best)
            model.train()

        # Periodic checkpoint (even without validation)
        elif ckpt_every and ((it + 1) % ckpt_every == 0):
            state = {
                "iter": it,
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "scaler": scaler.state_dict(),
                "best_psnr": best_psnr,
                "args": vars(args),
            }
            save_checkpoint(state, ckpt_dir / f"checkpoint_{it+1:08d}.pth", is_best=False)

    print("Training complete.")


# ----------------------------
# Main / Argument parsing
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Single-file NAFNet training")
    # Data
    parser.add_argument("--train_lq_dir", required=True)
    parser.add_argument("--train_gt_dir", required=True)
    parser.add_argument("--val_lq_dir", default="")
    parser.add_argument("--val_gt_dir", default="")
    # Model params (from configs)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--enc_blk_nums", type=int, nargs="+", default=[1,1,1,28],
                        help="Encoder block counts per stage.")
    parser.add_argument("--middle_blk_num", type=int, default=1)
    parser.add_argument("--dec_blk_nums", type=int, nargs="+", default=[1,1,1,1],
                        help="Decoder block counts per stage.")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--out_channels", type=int, default=3)
    # Optimization
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eta_min", type=float, default=1e-7)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--total_iters", type=int, default=400000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--loss", choices=["l1", "mse"], default="l1")
    # Logging / saving
    parser.add_argument("--print_freq", type=int, default=200)
    parser.add_argument("--val_freq", type=int, default=20000)
    parser.add_argument("--ckpt_freq", type=int, default=5000)
    parser.add_argument("--save_dir", type=str, default="./experiments/nafnet_run")
    parser.add_argument("--resume", type=str, default="", help="'latest' or path to checkpoint")
    # Misc
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision")
    parser.add_argument("--augment", action="store_true", help="Simple flips for training")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile if available")

    args = parser.parse_args()

    # Basic validation
    if len(args.enc_blk_nums) != len(args.dec_blk_nums):
        print("Warning: enc_blk_nums and dec_blk_nums lengths differ; ensure this matches architecture expectations.")

    train(args)


if __name__ == "__main__":
    main()