
from pathlib import Path
import json

import numpy as np
import tensorflow as tf

from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter

try:
    from .tf_feature_extractor import (
        build_feature_extractor,
        extract_features,
    )
except ImportError:
    from tf_feature_extractor import (
        build_feature_extractor,
        extract_features,
    )


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "ml_engine" / "models"

CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

IMAGE_SIZE = 256
FEATURE_HEIGHT = 32
FEATURE_WIDTH = 32
GAUSSIAN_SIGMA = 4.0


# ---------------------------------------------------------
# MODEL CACHES
# ---------------------------------------------------------

_feature_extractor = None
_memory_banks = {}
_thresholds_cache = {}


def validate_category(category):
    if category not in CATEGORIES:
        raise ValueError(
            f"Unsupported category: {category}. "
            f"Supported categories: {CATEGORIES}"
        )


def get_feature_extractor():
    global _feature_extractor

    if _feature_extractor is None:
        print("Loading TensorFlow feature extractor...")
        _feature_extractor = build_feature_extractor()

    return _feature_extractor


# ---------------------------------------------------------
# CATEGORY-SPECIFIC MEMORY BANK
# ---------------------------------------------------------

def get_memory_bank(category="bottle"):

    validate_category(category)

    if category not in _memory_banks:

        memory_bank_path = (
    MODEL_DIR
    / "mvtec_ad"
    / category
    / "memory_bank.npz"
)

        if not memory_bank_path.exists():
            raise FileNotFoundError(
                f"Memory bank not found for {category}: "
                f"{memory_bank_path}"
            )

        print(f"Loading {category} memory bank...")

        with np.load(
            memory_bank_path,
            allow_pickle=False,
        ) as data:

            memory_bank = data["memory_bank"].astype(
                np.float32
            )

        if memory_bank.ndim != 2:
            raise ValueError(
                "Memory bank must be a 2D array."
            )

        _memory_banks[category] = memory_bank

        print(
            f"{category} memory bank loaded:",
            memory_bank.shape,
        )

    return _memory_banks[category]


# ---------------------------------------------------------
# NEAREST-NEIGHBOUR DISTANCE
# ---------------------------------------------------------

def calculate_patch_distances(features, memory_bank):

    if features.shape[1] != memory_bank.shape[1]:
        raise ValueError(
            "Feature dimension mismatch: "
            f"{features.shape[1]} vs {memory_bank.shape[1]}"
        )

    distances = cdist(
        features,
        memory_bank,
        metric="euclidean",
    )

    nearest_distances = np.min(
        distances,
        axis=1,
    )

    return nearest_distances.astype(np.float32)


# ---------------------------------------------------------
# ANOMALY MAP GENERATION
# ---------------------------------------------------------

def generate_anomaly_map(patch_distances):

    expected_patches = FEATURE_HEIGHT * FEATURE_WIDTH

    if len(patch_distances) != expected_patches:
        raise ValueError(
            f"Expected {expected_patches} patch distances, "
            f"received {len(patch_distances)}."
        )

    anomaly_map = patch_distances.reshape(
        FEATURE_HEIGHT,
        FEATURE_WIDTH,
    )

    anomaly_map = tf.image.resize(
        anomaly_map[..., np.newaxis],
        size=(IMAGE_SIZE, IMAGE_SIZE),
        method="bilinear",
    ).numpy()[..., 0]

    anomaly_map = gaussian_filter(
        anomaly_map,
        sigma=GAUSSIAN_SIGMA,
    )

    return anomaly_map.astype(np.float32)


# ---------------------------------------------------------
# IMAGE-LEVEL ANOMALY SCORE
# ---------------------------------------------------------

def calculate_image_score(patch_distances):

    number_of_patches = max(
        1,
        int(np.ceil(len(patch_distances) * 0.01)),
    )

    highest_distances = np.partition(
        patch_distances,
        -number_of_patches,
    )[-number_of_patches:]

    return float(np.mean(highest_distances))


# ---------------------------------------------------------
# ANALYZE IMAGE
# ---------------------------------------------------------

def analyze_image(image_path, category="bottle"):

    validate_category(category)

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    feature_extractor = get_feature_extractor()

    memory_bank = get_memory_bank(category)

    features = extract_features(
        feature_extractor,
        image_path,
    )

    patch_features = features.reshape(
        -1,
        features.shape[-1],
    ).astype(np.float32)

    patch_distances = calculate_patch_distances(
        patch_features,
        memory_bank,
    )

    anomaly_map = generate_anomaly_map(
        patch_distances
    )

    anomaly_score = calculate_image_score(
        patch_distances
    )

    return {
        "anomaly_score": anomaly_score,
        "anomaly_map": anomaly_map,
        "patch_distances": patch_distances,
    }


# ---------------------------------------------------------
# CATEGORY-SPECIFIC THRESHOLDS
# ---------------------------------------------------------

def load_thresholds(category="bottle"):

    validate_category(category)

    if category not in _thresholds_cache:

        threshold_path = (
    MODEL_DIR
    / "mvtec_ad"
    / category
    / "thresholds.json"
)

        if not threshold_path.exists():
            raise FileNotFoundError(
                f"Calibrated thresholds not found for "
                f"{category}: {threshold_path}. "
                "Run threshold calibration first."
            )

        with open(
            threshold_path,
            "r",
            encoding="utf-8",
        ) as file:
            thresholds = json.load(file)

        required_keys = [
            "image_threshold",
            "pixel_threshold",
        ]

        for key in required_keys:
            if key not in thresholds:
                raise ValueError(
                    f"Missing calibration value: {key}"
                )

        _thresholds_cache[category] = thresholds

    return _thresholds_cache[category]


# ---------------------------------------------------------
# PRODUCTION PREDICTION INTERFACE
# ---------------------------------------------------------

def predict_image(image_path, category="bottle"):

    validate_category(category)

    thresholds = load_thresholds(category)

    result = analyze_image(
        image_path,
        category=category,
    )

    anomaly_score = result["anomaly_score"]
    anomaly_map = result["anomaly_map"]

    image_threshold = float(
        thresholds["image_threshold"]
    )

    pixel_threshold = float(
        thresholds["pixel_threshold"]
    )

    predicted_label = bool(
        anomaly_score > image_threshold
    )

    predicted_mask = (
        anomaly_map > pixel_threshold
    ).astype(np.uint8)

    return {
        "anomaly_score": float(anomaly_score),
        "predicted_label": predicted_label,
        "anomaly_map": anomaly_map,
        "predicted_mask": predicted_mask,
    }


# ---------------------------------------------------------
# BASIC TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    sample_image = (
        PROJECT_ROOT
        / "datasets"
        / "MVTecAD"
        / "bottle"
        / "test"
        / "broken_large"
        / "000.png"
    )

    print("=" * 60)
    print("VISIONGUARD - TENSORFLOW PATCHCORE INFERENCE")
    print("=" * 60)

    result = analyze_image(sample_image)

    print("\nInference successful.")

    print("Category: bottle")
    print("Image anomaly score:", result["anomaly_score"])
    print("Anomaly map shape:", result["anomaly_map"].shape)
    print("Patch distance shape:", result["patch_distances"].shape)

    print("\nClassification is not enabled yet.")
