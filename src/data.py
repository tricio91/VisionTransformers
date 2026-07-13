"""Dataset and dataloaders for the NEU Surface Defect Database.

The NEU images are 200x200 grayscale .bmp files organized one folder per
class (torchvision ImageFolder layout). Since ViT backbones expect 224x224
RGB input, we resize and replicate the single channel into three.

Preprocessing here is kept identical to what `export_onnx.py` bakes into the
ONNX metadata, so training and production see exactly the same pixels.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets.folder import IMG_EXTENSIONS


def _has_direct_image(path: str) -> bool:
    """True if `path` holds at least one image file as a direct child."""
    with os.scandir(path) as it:
        return any(e.is_file() and e.name.lower().endswith(IMG_EXTENSIONS) for e in it)


class DefectImageFolder(datasets.ImageFolder):
    """ImageFolder that only treats sub-folders holding images *directly* as classes.

    Plain ImageFolder turns every sub-directory of the root into a class and
    walks it recursively. That pulls in stray entries left in the data root —
    a `.zip`, or an extracted dump like `archive/NEU-DET/...` — as bogus extra
    classes. Here we keep only the folders that actually contain image files,
    so the class list matches the real dataset regardless of what else sits
    next to it.
    """

    def find_classes(self, directory: str):
        classes = sorted(
            e.name for e in os.scandir(directory) if e.is_dir() and _has_direct_image(e.path)
        )
        if not classes:
            raise FileNotFoundError(f"No class folders with images found in {directory!r}.")
        return classes, {name: i for i, name in enumerate(classes)}

# ImageNet statistics — the pretrained backbones were trained with these.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    classes: list[str]
    class_weights: torch.Tensor | None


def build_transforms(img_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """Return (train_tf, eval_tf). Eval must match ONNX preprocessing exactly."""
    train_tf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return train_tf, eval_tf


def _compute_class_weights(targets: list[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights, normalized to mean 1.0."""
    counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0  # avoid div-by-zero
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()  # normalize to mean 1.0
    return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(cfg: dict) -> DataBundle:
    """Build stratified-ish train/val/test loaders from an ImageFolder root."""
    dcfg, mcfg, tcfg = cfg["data"], cfg["model"], cfg["train"]
    img_size = mcfg["img_size"]
    seed = dcfg["seed"]

    train_tf, eval_tf = build_transforms(img_size)

    # One base dataset to read labels/paths, plus two views with different tf.
    base = DefectImageFolder(dcfg["data_dir"])
    classes = base.classes
    targets = [s[1] for s in base.samples]
    n = len(base)

    # Reproducible split.
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_test = int(n * dcfg["test_split"])
    n_val = int(n * dcfg["val_split"])
    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]

    train_ds_full = DefectImageFolder(dcfg["data_dir"], transform=train_tf)
    eval_ds_full = DefectImageFolder(dcfg["data_dir"], transform=eval_tf)

    train_ds = Subset(train_ds_full, train_idx)
    val_ds = Subset(eval_ds_full, val_idx)
    test_ds = Subset(eval_ds_full, test_idx)

    class_weights = None
    if tcfg.get("class_weights"):
        train_targets = [targets[i] for i in train_idx]
        class_weights = _compute_class_weights(train_targets, len(classes))

    common = dict(
        batch_size=tcfg["batch_size"],
        num_workers=dcfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    return DataBundle(
        train_loader=DataLoader(train_ds, shuffle=True, drop_last=True, **common),
        val_loader=DataLoader(val_ds, shuffle=False, **common),
        test_loader=DataLoader(test_ds, shuffle=False, **common),
        classes=classes,
        class_weights=class_weights,
    )
