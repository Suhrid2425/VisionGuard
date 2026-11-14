
"""
VisionGuard - Production Inference Interface

Uses the TensorFlow/Keras PatchCore implementation.
Maintains compatibility with the Django inspection application.
"""

from pathlib import Path

import numpy as np

# Support both Django package imports and direct script execution.
try:
    from .tf_inference import predict_image as tensorflow_predict_image
except ImportError:
    from tf_inference import predict_image as tensorflow_predict_image


SUPPORTED_CATEGORIES = {"bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"}


def predict_image(image_path, category="bottle"):
    """
    Run TensorFlow PatchCore inference on an inspection image.

    Args:
        image_path: Path to the uploaded inspection image.
        category: Product category. Currently supports 'bottle'.

    Returns:
        dict:
            anomaly_score: float
            predicted_label: bool
            anomaly_map: numpy.ndarray (float32)
            predicted_mask: numpy.ndarray (uint8)
    """

    # 1. Validate category.
    if category not in SUPPORTED_CATEGORIES:
        raise ValueError(
            f"AI inspection for '{category}' is not available yet. "
            "Currently supported category: bottle."
        )

    # 2. Validate image path.
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Inspection image not found: {image_path}"
        )

    # 3. Run TensorFlow inference.
    result = tensorflow_predict_image(
        image_path=image_path,
        category=category,
    )

    # 4. Validate the inference result.
    required_keys = {
        "anomaly_score",
        "predicted_label",
        "anomaly_map",
        "predicted_mask",
    }

    missing_keys = required_keys - result.keys()

    if missing_keys:
        raise RuntimeError(
            f"TensorFlow inference returned an incomplete result. "
            f"Missing keys: {sorted(missing_keys)}"
        )

    # 5. Normalize output types for Django compatibility.
    anomaly_score = float(result["anomaly_score"])
    predicted_label = bool(result["predicted_label"])

    anomaly_map = np.asarray(
        result["anomaly_map"],
        dtype=np.float32,
    )

    predicted_mask = np.asarray(
        result["predicted_mask"],
        dtype=np.uint8,
    )

    if anomaly_map.ndim != 2 or predicted_mask.ndim != 2:
        raise RuntimeError(
            "Expected anomaly_map and predicted_mask "
            "to be two-dimensional arrays."
        )

    if anomaly_map.shape != predicted_mask.shape:
        raise RuntimeError(
            "Anomaly map and predicted mask dimensions do not match."
        )

    # 6. Return the same interface used by the previous engine.
    return {
        "anomaly_score": anomaly_score,
        "predicted_label": predicted_label,
        "anomaly_map": anomaly_map,
        "predicted_mask": predicted_mask,
    }

