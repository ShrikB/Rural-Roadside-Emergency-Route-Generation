"""Inference engine: loads the trained U-Net and returns binary masks.

Public API:
    predict_mask(pil_image) -> np.ndarray  # (512, 512) uint8, 1=road, 0=obstacle
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Model import unet_model

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unet_model4.pt")
_IMG_SIZE = (512, 512)

_model: torch.nn.Module | None = None
_device: str | None = None


def _load_model() -> tuple[torch.nn.Module, str]:
    global _model, _device
    if _model is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.backends.cudnn.enabled = False
        _model = unet_model(512, 512, 3).to(_device)
        ckpt = torch.load(_MODEL_PATH, map_location=_device, weights_only=False)
        _model.load_state_dict(ckpt["model_state_dict"])
        _model.eval()
    return _model, _device  # type: ignore[return-value]


def predict_mask(pil_image: Image.Image) -> np.ndarray:
    """Run U-Net inference on a PIL image.

    Returns a (512, 512) uint8 array:
        1 = traversable (road/path)
        0 = obstacle (background)
    """
    model, device = _load_model()
    img = pil_image.convert("RGB").resize(_IMG_SIZE, Image.BILINEAR)
    img_np = np.asarray(img, dtype=np.float32) / 255.0
    img_t = (
        torch.from_numpy(img_np)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
    with torch.no_grad():
        logits = model(img_t)
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    return (probs > 0.5).astype(np.uint8)
