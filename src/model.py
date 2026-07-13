"""Model factory and utilities (build ViT, freeze/unfreeze, EMA)."""

from __future__ import annotations

import copy

import timm
import torch
import torch.nn as nn


def build_model(cfg: dict) -> nn.Module:
    """Create a ViT-style classifier from timm with the right head."""
    mcfg = cfg["model"]
    model = timm.create_model(
        mcfg["name"],
        pretrained=mcfg["pretrained"],
        num_classes=mcfg["num_classes"],
        drop_rate=mcfg.get("dropout", 0.0),  # dropout before the classifier head
    )
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze/unfreeze everything except the classification head.

    Works across timm architectures by treating the classifier submodule
    (returned by `get_classifier()`) as the head and everything else as backbone.
    """
    head = model.get_classifier()
    head_params = {id(p) for p in head.parameters()}
    for p in model.parameters():
        if id(p) not in head_params:
            p.requires_grad = trainable
    # Head is always trainable.
    for p in head.parameters():
        p.requires_grad = True


class ModelEMA:
    """Exponential Moving Average of model weights.

    Keeping a smoothed copy of the weights usually generalizes slightly better
    than the raw trained weights; at save time we keep whichever wins on val.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for ema_p, p in zip(self.ema.parameters(), model.parameters()):
            ema_p.mul_(d).add_(p.detach(), alpha=1.0 - d)
        for ema_b, b in zip(self.ema.buffers(), model.buffers()):
            ema_b.copy_(b)
