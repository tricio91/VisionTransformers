"""Two-stage fine-tuning of a ViT on the NEU surface-defect dataset.

Phase 1 (epochs < unfreeze_epoch): only the classification head trains,
backbone frozen. Phase 2: everything unfreezes, backbone gets a much lower LR.

Run:
    python -m src.train --config configs/config.yaml

Outputs (all under results/):
    metrics.json                    per-epoch history + final test scores
    confusion_matrix.png            confusion matrix on the test set
    training_curves.png             loss and accuracy per epoch (train vs val)
    validation_gradcam_grid.png     5x5 grid of val images with attention maps
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .data import build_dataloaders, build_transforms
from .explain import AttentionRollout, compute_overlay
from .model import ModelEMA, build_model, set_backbone_trainable


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model, tcfg) -> torch.optim.Optimizer:
    """Separate param groups so head and backbone can use different LRs."""
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    head_params, backbone_params = [], []
    for p in model.parameters():
        (head_params if id(p) in head_ids else backbone_params).append(p)
    return torch.optim.AdamW(
        [
            {"params": head_params, "lr": tcfg["lr_head"]},
            {"params": backbone_params, "lr": tcfg["lr_backbone"]},
        ],
        weight_decay=tcfg["weight_decay"],
    )


def lr_factor(epoch: int, warmup: int, total: int) -> float:
    """Linear warmup then cosine annealing, returned as a multiplier."""
    if epoch < warmup:
        return (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    """Run the model over a loader, returning true/pred labels and mean loss."""
    model.eval()
    y_true, y_pred = [], []
    loss_sum, n = 0.0, 0
    for x, y in loader:
        x, yb = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += criterion(logits, yb).item() * x.size(0)
        n += x.size(0)
        y_pred.extend(logits.argmax(1).cpu().tolist())
        y_true.extend(y.tolist())
    return y_true, y_pred, loss_sum / n


def save_confusion_matrix(y_true, y_pred, classes, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_training_curves(history, path) -> None:
    """Plot loss and accuracy per epoch, train vs validation, side by side."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    ax_loss.plot(epochs, history["train_loss"], marker="o", label="train")
    ax_loss.plot(epochs, history["val_loss"], marker="o", label="validation")
    ax_loss.set_title("Loss per epoch")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, history["train_acc"], marker="o", label="train")
    ax_acc.plot(epochs, history["val_acc"], marker="o", label="validation")
    ax_acc.set_title("Accuracy per epoch")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_gradcam_grid(model, val_subset, classes, img_size, device, path, n=25) -> None:
    """Render a 5x5 grid of validation images with attention overlays.

    Each tile shows the heatmap of where the model looked, plus the predicted
    class and the ground truth. Titles are green when the prediction is right
    and red when it's wrong, so mistakes jump out at a glance.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, eval_tf = build_transforms(img_size)
    base_ds = val_subset.dataset
    indices = list(val_subset.indices)[:n]
    prefix = getattr(model, "num_prefix_tokens", 1)

    rollout = AttentionRollout(model)
    fig, axes = plt.subplots(5, 5, figsize=(15, 16))
    for ax, idx in zip(axes.flat, indices):
        img_path, gt = base_ds.samples[idx]
        x = eval_tf(Image.open(img_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x).argmax(1).item()
        mask = rollout(x, prefix_tokens=prefix)
        ax.imshow(compute_overlay(img_path, mask, img_size))
        ax.axis("off")
        correct = pred == gt
        ax.set_title(
            f"pred: {classes[pred]}\ngt: {classes[gt]}",
            color="green" if correct else "red",
            fontsize=9,
        )
    for ax in list(axes.flat)[len(indices):]:
        ax.axis("off")
    rollout.remove()

    fig.suptitle("Validation predictions with attention rollout", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ViT on NEU defects")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg, ocfg = cfg["train"], cfg["output"]
    img_size = cfg["model"]["img_size"]
    set_seed(cfg["data"]["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data = build_dataloaders(cfg)
    print(f"Classes: {data.classes}")

    model = build_model(cfg).to(device)
    set_backbone_trainable(model, False)  # start frozen

    weight = data.class_weights.to(device) if data.class_weights is not None else None
    criterion = nn.CrossEntropyLoss(
        weight=weight, label_smoothing=tcfg["label_smoothing"]
    )
    optimizer = build_optimizer(model, tcfg)
    scaler = torch.cuda.amp.GradScaler(enabled=tcfg["mixed_precision"] and device == "cuda")
    ema = ModelEMA(model, tcfg["ema_decay"]) if tcfg["use_ema"] else None

    os.makedirs(ocfg["ckpt_dir"], exist_ok=True)
    os.makedirs(ocfg["results_dir"], exist_ok=True)
    ckpt_path = os.path.join(ocfg["ckpt_dir"], ocfg["ckpt_name"])

    best_val = 0.0
    epochs_no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(tcfg["epochs"]):
        if epoch == tcfg["unfreeze_epoch"]:
            set_backbone_trainable(model, True)
            print(f"[epoch {epoch}] backbone unfrozen - phase 2")

        factor = lr_factor(epoch, tcfg["warmup_epochs"], tcfg["epochs"])
        for g, base_lr in zip(optimizer.param_groups, [tcfg["lr_head"], tcfg["lr_backbone"]]):
            g["lr"] = base_lr * factor

        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for x, y in data.train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = model(x)
                loss = criterion(outputs, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema:
                ema.update(model)
            running_loss += loss.item() * x.size(0)
            correct += (outputs.argmax(1) == y).sum().item()
            total += x.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        # Validation on the raw model drives the curves.
        yt, yp, val_loss = evaluate(model, data.val_loader, device, criterion)
        val_acc = accuracy_score(yt, yp)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Checkpoint selection: keep whichever of raw / EMA scores higher.
        sel_acc, used_ema = val_acc, False
        if ema:
            yte, ype, _ = evaluate(ema.ema, data.val_loader, device, criterion)
            ema_acc = accuracy_score(yte, ype)
            if ema_acc >= sel_acc:
                sel_acc, used_ema = ema_acc, True

        print(
            f"epoch {epoch:02d} | loss {train_loss:.4f} | acc {train_acc:.4f} "
            f"| val_loss {val_loss:.4f} | val_acc {sel_acc:.4f}"
            f"{' (EMA)' if used_ema else ''}"
        )

        if sel_acc > best_val:
            best_val = sel_acc
            epochs_no_improve = 0
            state = (ema.ema if used_ema else model).state_dict()
            torch.save(
                {
                    "model": state,
                    "used_ema": used_ema,
                    "best_val_acc": best_val,
                    "epoch": epoch,
                    "classes": data.classes,
                    "img_size": img_size,
                    "model_name": cfg["model"]["name"],
                },
                ckpt_path,
            )
            print(f"  saved checkpoint (val_acc={best_val:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= tcfg["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    # ---- Reload best checkpoint for the final artefacts ----
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    yt, yp, _ = evaluate(model, data.test_loader, device, criterion)
    test_acc = accuracy_score(yt, yp)
    test_f1 = f1_score(yt, yp, average="macro")
    report = classification_report(yt, yp, target_names=data.classes, digits=4)
    print("\n=== TEST REPORT ===")
    print(f"accuracy: {test_acc:.4f} | macro-F1: {test_f1:.4f}")
    print(report)

    results = ocfg["results_dir"]
    save_confusion_matrix(yt, yp, data.classes,
                          os.path.join(results, "confusion_matrix.png"))
    save_training_curves(history, os.path.join(results, "training_curves.png"))
    save_gradcam_grid(model, data.val_loader.dataset, data.classes, img_size,
                      device, os.path.join(results, "validation_gradcam_grid.png"))

    with open(os.path.join(results, "metrics.json"), "w") as f:
        json.dump(
            {
                "best_val_acc": best_val,
                "test_acc": test_acc,
                "test_macro_f1": test_f1,
                "model_name": cfg["model"]["name"],
                "history": history,
            },
            f,
            indent=2,
        )
    print(f"\nResults written to {results}/")


if __name__ == "__main__":
    main()
