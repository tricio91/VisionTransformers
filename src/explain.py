"""Attention rollout for the ViT classifier.

Draws a heatmap over the input image showing which patches the model actually
leaned on for its decision. Useful for letting a QA operator confirm the model
is looking at the defect itself and not at some edge or background artefact.

Run:
    python -m src.explain --checkpoint checkpoints/best_model.pt \
        --image sample.bmp --output results/attention.png
"""

from __future__ import annotations

import argparse

import numpy as np
import timm
import torch
from matplotlib import cm
from PIL import Image

from .data import build_transforms


def load_model(ckpt_path: str, device: str):
    """Rebuild the trained ViT from a checkpoint, ready for inference."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model = timm.create_model(
        ckpt["model_name"], pretrained=False, num_classes=len(ckpt["classes"])
    )
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model, ckpt["classes"], ckpt["img_size"]


class AttentionRollout:
    """Collects the attention maps from every ViT block and folds them together.

    Follows Abnar & Zuidema (2020): average heads, add the residual connection,
    normalise, then multiply the per-layer matrices to trace how information
    flows from the input patches up to the class token.
    """

    def __init__(self, model, head_fusion: str = "mean", discard_ratio: float = 0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.attentions: list[torch.Tensor] = []
        self.hooks = []
        # timm's Attention runs a fused kernel by default, which hides the
        # softmax weights. Turn it off so we can grab them via attn_drop.
        for m in model.modules():
            if hasattr(m, "attn_drop"):
                if hasattr(m, "fused_attn"):
                    m.fused_attn = False
                self.hooks.append(m.attn_drop.register_forward_hook(self._grab))

    def _grab(self, module, inputs, output):
        # attn_drop's input is the softmaxed attention: [B, heads, N, N]
        self.attentions.append(inputs[0].detach().cpu())

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, prefix_tokens: int = 1) -> np.ndarray:
        self.attentions.clear()
        self.model(x)

        n_tokens = self.attentions[0].size(-1)
        result = torch.eye(n_tokens)
        for attn in self.attentions:
            if self.head_fusion == "max":
                fused = attn.max(dim=1)[0][0]
            elif self.head_fusion == "min":
                fused = attn.min(dim=1)[0][0]
            else:
                fused = attn.mean(dim=1)[0]

            # zero out the weakest links to keep the map readable, but never
            # drop the class token's own column
            flat = fused.view(-1)
            n_drop = int(flat.numel() * self.discard_ratio)
            if n_drop:
                weakest = flat.argsort()[:n_drop]
                weakest = weakest[weakest != 0]
                flat[weakest] = 0

            fused = fused + torch.eye(n_tokens)
            fused = fused / fused.sum(dim=-1, keepdim=True)
            result = fused @ result

        # class-token row, restricted to the patch tokens
        mask = result[0, prefix_tokens:]
        grid = int(mask.numel() ** 0.5)
        mask = mask.reshape(grid, grid).numpy()
        return mask / mask.max()


def compute_overlay(image_path: str, mask: np.ndarray, img_size: int,
                    alpha: float = 0.5) -> np.ndarray:
    """Blend the heatmap onto the image and return it as an RGB uint8 array."""
    img = Image.open(image_path).convert("RGB").resize((img_size, img_size))
    base = np.asarray(img, dtype=np.float32) / 255.0

    heat = Image.fromarray((mask * 255).astype(np.uint8)).resize(
        (img_size, img_size), Image.BILINEAR
    )
    heat = cm.jet(np.asarray(heat) / 255.0)[..., :3]

    blended = (1 - alpha) * base + alpha * heat
    return (blended * 255).astype(np.uint8)


def save_overlay(image_path: str, mask: np.ndarray, img_size: int,
                 out_path: str, alpha: float = 0.5) -> None:
    """Save a single heatmap overlay to disk."""
    Image.fromarray(compute_overlay(image_path, mask, img_size, alpha)).save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Attention rollout heatmap")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="results/attention.png")
    parser.add_argument("--discard-ratio", type=float, default=0.9)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes, img_size = load_model(args.checkpoint, device)

    _, eval_tf = build_transforms(img_size)
    x = eval_tf(Image.open(args.image)).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = classes[model(x).argmax(1).item()]

    prefix = getattr(model, "num_prefix_tokens", 1)
    rollout = AttentionRollout(model, discard_ratio=args.discard_ratio)
    mask = rollout(x, prefix_tokens=prefix)
    rollout.remove()

    save_overlay(args.image, mask, img_size, args.output)
    print(f"Predicted: {pred}  ->  saved heatmap to {args.output}")


if __name__ == "__main__":
    main()
