from __future__ import annotations

import os

import matplotlib.pyplot as plt
import torch

from Data_Process import test_ds
from Model import unet_model


def batch_iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    targets = (targets > 0.5).float()
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = (preds + targets - preds * targets).sum(dim=(1, 2, 3))
    iou = (inter + eps) / (union + eps)
    return float(iou.mean().item())


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = unet_model(imageL=512, imageW=512, channels=3).to(device)
    ckpt_path = "unet_model4.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total_iou = 0.0
    n = 0
    for imgs, masks in test_ds:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(imgs)
        bs = imgs.size(0)
        total_iou += batch_iou_from_logits(logits, masks) * bs
        n += bs

    print({"mean_iou": total_iou / max(n, 1), "num_samples": n})

    # visualize a batch
    imgs, masks = next(iter(test_ds))
    imgs = imgs.to(device)
    masks = masks.to(device)
    logits = model(imgs)
    preds = (torch.sigmoid(logits) > 0.5).float()

    B = imgs.shape[0]
    
    # Create an output directory for the saved images
    os.makedirs("output_images", exist_ok=True)
    
    for i in range(min(B, 4)):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(imgs[i].detach().cpu().permute(1, 2, 0))
        axes[0].set_title("Input RGB")
        axes[0].axis("off")

        axes[1].imshow(masks[i, 0].detach().cpu(), cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(preds[i, 0].detach().cpu(), cmap="gray")
        axes[2].set_title("Prediction")
        axes[2].axis("off")
        
        # Save the figure instead of showing it
        save_path = os.path.join("output_images", f"prediction_result_{i}.png")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
        
        # Close the figure to free up memory
        plt.close(fig)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    torch.backends.cudnn.enabled = False

    main()