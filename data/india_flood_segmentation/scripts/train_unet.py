#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class IndiaFloodDataset(Dataset):
    def __init__(self, root: Path, split: str):
        self.image_dir = root / "raw" / split / "images"
        self.label_dir = root / "raw" / split / "labels"
        self.ids = sorted(p.stem for p in self.image_dir.glob("*.tif"))
        if not self.ids:
            raise FileNotFoundError(f"No image chips found in {self.image_dir}")
        missing = [cid for cid in self.ids if not (self.label_dir / f"{cid}.tif").exists()]
        if missing:
            raise FileNotFoundError(f"Missing labels for {split}: {missing[:5]}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        cid = self.ids[index]
        image = tifffile.imread(self.image_dir / f"{cid}.tif").astype("float32")
        label = tifffile.imread(self.label_dir / f"{cid}.tif").astype("int16")
        if image.shape != (2, 512, 512):
            raise ValueError(f"Unexpected image shape for {cid}: {image.shape}")
        if label.shape != (512, 512):
            raise ValueError(f"Unexpected label shape for {cid}: {label.shape}")
        # Fixed dB normalization keeps inference reproducible across scenes.
        image = np.clip(image, -40.0, 5.0)
        image = (image + 40.0) / 45.0
        valid = label >= 0
        target = np.clip(label, 0, 1).astype("float32")
        return torch.from_numpy(image), torch.from_numpy(target[None]), torch.from_numpy(valid[None]), cid


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(2, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def masked_bce_dice(logits, target, valid):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)
    probs = torch.sigmoid(logits) * valid
    target = target * valid
    intersection = (probs * target).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denom + 1.0)).mean()
    return bce + dice_loss


def masked_metrics(logits, target, valid):
    pred = (torch.sigmoid(logits) >= 0.5).float()
    pred = pred * valid
    target = target * valid
    intersection = (pred * target).sum().item()
    union = ((pred + target) > 0).float().sum().item()
    denom = pred.sum().item() + target.sum().item()
    return (intersection / max(union, 1.0), (2 * intersection) / max(denom, 1.0))


def run_epoch(model, loader, optimizer, device, train: bool):
    model.train(train)
    losses, intersections, unions, denoms = [], 0.0, 0.0, 0.0
    for image, target, valid, _ in loader:
        image, target, valid = image.to(device), target.to(device), valid.to(device)
        with torch.set_grad_enabled(train):
            logits = model(image)
            loss = masked_bce_dice(logits, target, valid)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        pred = (torch.sigmoid(logits) >= 0.5).float() * valid
        target = target * valid
        intersections += float((pred * target).sum().detach().cpu())
        unions += float(((pred + target) > 0).float().sum().detach().cpu())
        denoms += float((pred + target).sum().detach().cpu())
    iou = intersections / max(unions, 1.0)
    dice = (2 * intersections) / max(denoms, 1.0)
    return float(np.mean(losses)), iou, dice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/india_unet"))
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = IndiaFloodDataset(args.data_root, "train")
    val_ds = IndiaFloodDataset(args.data_root, "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_iou = -1.0
    print(f"device={device} train_chips={len(train_ds)} val_chips={len(val_ds)}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_iou, train_dice = run_epoch(model, train_loader, optimizer, device, True)
        with torch.no_grad():
            val_loss, val_iou, val_dice = run_epoch(model, val_loader, optimizer, device, False)
        row = {"epoch": epoch, "train_loss": train_loss, "train_iou": train_iou, "train_dice": train_dice, "val_loss": val_loss, "val_iou": val_iou, "val_dice": val_dice}
        history.append(row)
        print(json.dumps(row))
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": row}, args.output_dir / "best_unet.pt")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"best_val_iou={best_iou:.4f}")


if __name__ == "__main__":
    main()
