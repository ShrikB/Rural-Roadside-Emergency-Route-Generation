from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from Model import unet_model
from Data_Process import train_ds, val_ds


@dataclass
class TrainConfig:
    epochs: int = 15
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "unet_model4.pt"
    early_stopping_patience: int = 9


def batch_iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    """IoU for binary segmentation.

    logits:  (N,1,H,W)
    targets: (N,1,H,W) in {0,1}
    """
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    targets = (targets > 0.5).float()

    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = (preds + targets - preds * targets).sum(dim=(1, 2, 3))
    iou = (inter + eps) / (union + eps)
    return float(iou.mean().item())


def train_one_epoch(model: nn.Module, loader, criterion, optimizer, device: str):
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    n = 0

    for imgs, masks in tqdm(loader, desc="train", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        total_loss += float(loss.item()) * bs
        total_iou += batch_iou_from_logits(logits.detach(), masks.detach()) * bs
        n += bs

    return total_loss / max(n, 1), total_iou / max(n, 1)


@torch.no_grad()
def eval_one_epoch(model: nn.Module, loader, criterion, device: str):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    n = 0

    for imgs, masks in tqdm(loader, desc="val", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, masks)

        bs = imgs.size(0)
        total_loss += float(loss.item()) * bs
        total_iou += batch_iou_from_logits(logits, masks) * bs
        n += bs

    return total_loss / max(n, 1), total_iou / max(n, 1)


def main():
    torch.backends.cudnn.enabled = False
    cfg = TrainConfig()

    model = unet_model(imageL=512, imageW=512, channels=3).to(cfg.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    bad_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_iou = train_one_epoch(model, train_ds, criterion, optimizer, cfg.device)
        va_loss, va_iou = eval_one_epoch(model, val_ds, criterion, cfg.device)
        scheduler.step(va_loss)

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train loss {tr_loss:.4f} iou {tr_iou:.4f} | "
            f"val loss {va_loss:.4f} iou {va_iou:.4f}"
        )

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": va_loss,
                    "config": cfg.__dict__,
                },
                cfg.save_path,
            )
            print(f"Saved best checkpoint -> {cfg.save_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stopping_patience:
                print("Early stopping triggered")
                break


if __name__ == "__main__":
    # Make the saved file land next to this script when run from elsewhere
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
