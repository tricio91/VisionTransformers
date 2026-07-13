"""Production inference with the exported ONNX model.

Depends only on onnxruntime + numpy + pillow (see requirements-inference.txt).
No PyTorch, no timm. Preprocessing is read from the ONNX metadata, so this
script stays correct even if you retrain with a different backbone or image size.

Run:
    python -m src.predict --onnx neu_defect_vit.onnx --image sample.bmp
    python -m src.predict --onnx neu_defect_vit.onnx --image sample.bmp --topk 3 --json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import onnxruntime as ort
from PIL import Image


class DefectClassifier:
    """Self-describing ONNX classifier: reads its own preprocessing metadata."""

    def __init__(self, onnx_path: str, providers: list[str] | None = None):
        self.session = ort.InferenceSession(
            onnx_path,
            providers=providers or ["CPUExecutionProvider"],
        )
        meta = dict(self.session.get_modelmeta().custom_metadata_map)
        self.classes = json.loads(meta["classes"])
        self.img_size = int(meta["img_size"])
        self.mean = np.array(json.loads(meta["mean"]), dtype=np.float32)
        self.std = np.array(json.loads(meta["std"]), dtype=np.float32)
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std          # HWC normalize
        arr = np.transpose(arr, (2, 0, 1))          # -> CHW
        return arr[np.newaxis, :].astype(np.float32)  # -> NCHW

    def predict(self, image_path: str, topk: int = 1) -> list[dict]:
        x = self.preprocess(image_path)
        probs = self.session.run(None, {self.input_name: x})[0][0]
        order = np.argsort(probs)[::-1][:topk]
        return [
            {"class": self.classes[i], "confidence": float(probs[i])}
            for i in order
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX defect inference")
    parser.add_argument("--onnx", default="neu_defect_vit.onnx")
    parser.add_argument("--image", required=True)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    args = parser.parse_args()

    clf = DefectClassifier(args.onnx)
    results = clf.predict(args.image, topk=args.topk)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            print(f"{r['class']:<18} {r['confidence'] * 100:6.2f}%")


if __name__ == "__main__":
    main()
