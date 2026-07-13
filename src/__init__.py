"""NEU surface-defect classification package."""

__version__ = "1.0.0"

# Canonical class order. The exported ONNX model relies on this exact order:
# output logit i corresponds to CLASSES[i].
CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# Human-readable descriptions used in reports / demos.
CLASS_DESCRIPTIONS = {
    "crazing": "Network of fine surface cracks",
    "inclusion": "Foreign material embedded in the surface",
    "patches": "Localized discolored regions",
    "pitted_surface": "Small cavities / pitting",
    "rolled-in_scale": "Scale pressed into the surface during rolling",
    "scratches": "Linear mechanical scratches",
}
