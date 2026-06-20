"""Model definition (PyTorch).

Provides a U-Net for binary segmentation.

Contract:
- Input:  (N, 3, H, W) float32 in [0, 1]
- Output: (N, 1, H, W) logits (NOT sigmoid)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
  def __init__(self, in_ch: int, out_ch: int):
    super().__init__()
    self.net = nn.Sequential(
      nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_ch),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_ch),
      nn.ReLU(inplace=True),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)


class UNet(nn.Module):
  def __init__(
    self,
    in_channels: int = 3,
    out_channels: int = 1,
    features: tuple[int, ...] = (64, 128, 256, 512),
  ):
    super().__init__()

    self.downs = nn.ModuleList()
    self.pools = nn.ModuleList()

    ch = in_channels
    for f in features:
      self.downs.append(DoubleConv(ch, f))
      self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
      ch = f

    self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

    self.upconvs = nn.ModuleList()
    self.ups = nn.ModuleList()

    up_ch = features[-1] * 2
    for f in reversed(features):
      self.upconvs.append(nn.ConvTranspose2d(up_ch, f, kernel_size=2, stride=2))
      self.ups.append(DoubleConv(f * 2, f))
      up_ch = f

    self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    skips: list[torch.Tensor] = []

    for down, pool in zip(self.downs, self.pools):
      x = down(x)
      skips.append(x)
      x = pool(x)

    x = self.bottleneck(x)

    for upconv, up, skip in zip(self.upconvs, self.ups, reversed(skips)):
      x = upconv(x)

      # if shapes mismatch by 1px (odd dimensions), resize
      if x.shape[-2:] != skip.shape[-2:]:
        x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

      x = torch.cat([skip, x], dim=1)
      x = up(x)

    return self.final_conv(x)


def unet_model(imageL: int = 512, imageW: int = 512, channels: int = 3) -> UNet:
  """Back-compat factory matching the older Keras-style call sites."""
  _ = imageL, imageW
  return UNet(in_channels=channels, out_channels=1)