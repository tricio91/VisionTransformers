"""Shrink the exported ONNX model to INT8 with static (calibrated) quantisation.

Both weights *and* activations are quantised to 8-bit using ranges measured on a
set of real images (calibration), so you point it at a folder of representative
images (one sub-folder per class). The file comes out ~4x smaller and runs
faster on CPU. Class metadata is copied across, so predict.py works on the
quantised model unchanged.

Run:
    python -m src.quantize_onnx --input neu_defect_vit.onnx \
        --output neu_defect_vit.int8.onnx --calib-dir data/NEU-CLS --num-calib 100
"""

from __future__ import annotations

import argparse
import os
import random
import tempfile

import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from .predict import DefectClassifier

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def copy_metadata(src_path: str, dst_path: str) -> None:
    """Carry the class names / preprocessing metadata over to the INT8 file.

    The quantiser can drop custom metadata, so we copy it across if it's
    missing. This keeps predict.py working on the quantised model unchanged.
    """
    dst = onnx.load(dst_path)
    if dst.metadata_props:
        return
    src = onnx.load(src_path)
    for prop in src.metadata_props:
        entry = dst.metadata_props.add()
        entry.key, entry.value = prop.key, prop.value
    onnx.save(dst, dst_path)


def _collect_calibration_images(calib_dir: str, num: int, seed: int = 42) -> list[str]:
    """Gather up to `num` image paths, spread across the class sub-folders."""
    paths: list[str] = []
    for entry in sorted(os.scandir(calib_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        imgs = [
            os.path.join(entry.path, f)
            for f in sorted(os.listdir(entry.path))
            if f.lower().endswith(IMG_EXTS)
        ]
        paths.extend(imgs)  # only folders with direct images contribute
    if not paths:
        raise FileNotFoundError(f"No calibration images found under {calib_dir!r}.")
    random.Random(seed).shuffle(paths)
    return paths[:num]


class _ImageCalibrationReader(CalibrationDataReader):
    """Feeds preprocessed images to the static quantiser, one at a time.

    Preprocessing is reused verbatim from DefectClassifier so the calibration
    inputs are identical to what the model sees in production.
    """

    def __init__(self, clf: DefectClassifier, image_paths: list[str]):
        self.clf = clf
        self.input_name = clf.input_name
        self._paths = image_paths
        self._iter = iter(image_paths)

    def get_next(self):
        path = next(self._iter, None)
        if path is None:
            return None
        return {self.input_name: self.clf.preprocess(path)}

    def rewind(self):
        self._iter = iter(self._paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="INT8 static (calibrated) quantisation")
    parser.add_argument("--input", default="neu_defect_vit.onnx")
    parser.add_argument("--output", default="neu_defect_vit.int8.onnx")
    parser.add_argument("--calib-dir", default="data/NEU-CLS",
                        help="Folder of images for calibration (one sub-folder per class).")
    parser.add_argument("--num-calib", type=int, default=100,
                        help="How many images to calibrate on.")
    args = parser.parse_args()

    clf = DefectClassifier(args.input)
    images = _collect_calibration_images(args.calib_dir, args.num_calib)
    reader = _ImageCalibrationReader(clf, images)
    print(f"Calibrating on {len(images)} images from {args.calib_dir}")

    # Static quantisation needs a shape-inferred / cleaned graph first.
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        prepped = tmp.name
    try:
        quant_pre_process(args.input, prepped)
        quantize_static(
            prepped,
            args.output,
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
        )
    finally:
        if os.path.exists(prepped):
            os.remove(prepped)

    copy_metadata(args.input, args.output)

    before = os.path.getsize(args.input) / 1e6
    after = os.path.getsize(args.output) / 1e6
    print(f"{args.input}: {before:.1f} MB")
    print(f"{args.output}: {after:.1f} MB  ({before / after:.1f}x smaller)")
    print("Metadata preserved — run it with predict.py as usual.")


if __name__ == "__main__":
    main()
