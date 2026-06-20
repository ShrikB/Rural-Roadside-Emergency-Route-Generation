"""Data loading (PyTorch).

Exports:
- train_ds, val_ds, test_ds: torch.utils.data.DataLoader

Notes:
- Assumes DeepGlobe-style naming: *_sat.jpg and *_mask.png
- Produces tensors:
  - image: float32 (C,H,W) in [0,1]
  - mask:  float32 (1,H,W) in {0,1}
"""

from __future__ import annotations

import os
from glob import glob
from typing import List, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader


base_dir = "deepglobe/"
train_dir = os.path.join(base_dir, "train/")
test_dir = os.path.join(base_dir, "testset/")

img_size: Tuple[int, int] = (512, 512)
batch_size: int = 4
num_workers: int = 2


def _find_pairs(dir_path: str) -> Tuple[List[str], List[str]]:
    imgs = sorted(glob(os.path.join(dir_path, "*_sat.jpg")))
    masks = sorted(glob(os.path.join(dir_path, "*_mask.png")))
    return imgs, masks


class SegmentationDataset(Dataset):
    def __init__(self, img_paths: List[str], mask_paths: List[str], size: Tuple[int, int]):
        if len(img_paths) != len(mask_paths):
            raise ValueError(f"Mismatched images ({len(img_paths)}) and masks ({len(mask_paths)})")
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.size = size

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.img_paths[idx]).convert("RGB").resize(self.size, resample=Image.BILINEAR)
        mask = Image.open(self.mask_paths[idx]).convert("L").resize(self.size, resample=Image.NEAREST)

        img_np = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3)
        mask_np = (np.asarray(mask, dtype=np.uint8) > 128).astype(np.float32)  # (H,W)

        img_t = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()  # (3,H,W)
        mask_t = torch.from_numpy(mask_np)[None, ...].contiguous()  # (1,H,W)

        return img_t, mask_t


img_paths, mask_paths = _find_pairs(train_dir)
test_imgs, test_masks = _find_pairs(test_dir)

train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    img_paths,
    mask_paths,
    test_size=0.20,
    random_state=49,
    shuffle=True,
)

train_dataset = SegmentationDataset(train_imgs, train_masks, img_size)
val_dataset = SegmentationDataset(val_imgs, val_masks, img_size)
test_dataset = SegmentationDataset(test_imgs, test_masks, img_size)

train_ds = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)

val_ds = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

test_ds = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)


if __name__ == "__main__":
    # quick sanity check
    imgs, msks = next(iter(train_ds))
    print("Batch img shape:", tuple(imgs.shape))
    print("Batch mask shape:", tuple(msks.shape))
