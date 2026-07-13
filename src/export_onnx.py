"""Export a trained checkpoint to ONNX for production serving.

The exported graph ends in a softmax, so consumers get calibrated
probabilities directly — no PyTorch needed at inference time. Class names,
image size and normalization constants are stored as ONNX metadata so the
serving code is fully self-describing.

Run:
    python -m src.export_onnx \
        --checkpoint checkpoints/best_model.pt \
        --output neu_defect_vit.onnx \
        [--opset 17] [--dynamic-batch] [--no-verify]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import timm
import torch
import torch.nn as nn

from .data import MEAN, STD


class InferenceModel(nn.Module):
    """Wraps the backbone and appends softmax so ONNX outputs probabilities."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.backbone(x), dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export checkpoint to ONNX")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--output", default="neu_defect_vit.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Allow variable batch size at inference (default: batch=1).",
    )
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    classes = ckpt["classes"]
    img_size = ckpt["img_size"]
    model_name = ckpt["model_name"]
    print(f"Loaded {model_name} | {len(classes)} classes | img_size={img_size}")

    backbone = timm.create_model(
        model_name, pretrained=False, num_classes=len(classes)
    )
    backbone.load_state_dict(ckpt["model"])
    model = InferenceModel(backbone).eval()

    dummy = torch.randn(1, 3, img_size, img_size)
    dynamic_axes = (
        {"input": {0: "batch"}, "probabilities": {0: "batch"}}
        if args.dynamic_batch
        else None
    )

    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["input"],
        output_names=["probabilities"],
        opset_version=args.opset,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    print(f"Exported to {args.output}")

    # ---- Embed self-describing metadata ----
    import onnx

    onnx_model = onnx.load(args.output)
    meta = {
        "classes": json.dumps(classes),
        "img_size": str(img_size),
        "mean": json.dumps(MEAN),
        "std": json.dumps(STD),
        "model_name": model_name,
        "layout": "NCHW",
        "channels": "RGB",
    }
    for k, v in meta.items():
        entry = onnx_model.metadata_props.add()
        entry.key, entry.value = k, v
    onnx.save(onnx_model, args.output)
    onnx.checker.check_model(onnx_model)
    print("Metadata embedded and model structure validated.")

    # ---- Verify ONNX == PyTorch within tolerance ----
    if not args.no_verify:
        import onnxruntime as ort

        with torch.no_grad():
            torch_out = model(dummy).numpy()
        sess = ort.InferenceSession(
            args.output, providers=["CPUExecutionProvider"]
        )
        onnx_out = sess.run(None, {"input": dummy.numpy()})[0]
        max_diff = float(np.abs(torch_out - onnx_out).max())
        print(f"Max |PyTorch - ONNX| = {max_diff:.2e}")
        assert max_diff < 1e-4, "ONNX output diverges from PyTorch!"
        print("Verification passed — ONNX matches PyTorch.")


if __name__ == "__main__":
    main()
