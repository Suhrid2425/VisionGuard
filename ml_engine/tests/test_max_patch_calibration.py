
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_engine.tf_inference import analyze_image


DATASET_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "MVTecAD"
    / "bottle"
)

VALIDATION_DIR = (
    DATASET_DIR
    / "validation"
    / "good"
)

TEST_DIR = DATASET_DIR / "test"

SAFETY_MULTIPLIER = 1.05

TEST_SAMPLES = [
    ("good", "000.png"),
    ("broken_large", "000.png"),
    ("broken_small", "000.png"),
    ("contamination", "000.png"),
    ("contamination", "003.png"),
    ("contamination", "019.png"),
]


def get_max_patch_score(image_path):

    result = analyze_image(image_path)

    distances = np.asarray(
        result["patch_distances"],
        dtype=np.float32
    ).reshape(-1)

    if not np.all(np.isfinite(distances)):
        raise ValueError(
            f"Non-finite patch distances: {image_path}"
        )

    return float(np.max(distances))


def main():

    validation_images = sorted(
        VALIDATION_DIR.glob("*.png")
    )

    if not validation_images:
        raise FileNotFoundError(
            f"No validation images found: {VALIDATION_DIR}"
        )

    print("=" * 65)
    print("MAXIMUM PATCH SCORE CALIBRATION")
    print("=" * 65)

    print(
        f"Normal validation images: {len(validation_images)}"
    )

    # --------------------------------------------------
    # CALIBRATE USING NORMAL VALIDATION IMAGES ONLY
    # --------------------------------------------------

    validation_scores = []

    for image_path in validation_images:

        score = get_max_patch_score(image_path)

        validation_scores.append(score)

    validation_scores = np.asarray(
        validation_scores,
        dtype=np.float32
    )

    max_normal_score = float(
        np.max(validation_scores)
    )

    max_patch_threshold = (
        max_normal_score * SAFETY_MULTIPLIER
    )

    print("\nNORMAL VALIDATION STATISTICS")

    print(
        f"Minimum maximum-patch score: "
        f"{np.min(validation_scores):.6f}"
    )

    print(
        f"Mean maximum-patch score: "
        f"{np.mean(validation_scores):.6f}"
    )

    print(
        f"Maximum normal score: "
        f"{max_normal_score:.6f}"
    )

    print(
        f"Safety multiplier: "
        f"{SAFETY_MULTIPLIER}"
    )

    print(
        f"\nCALIBRATED MAX-PATCH THRESHOLD: "
        f"{max_patch_threshold:.6f}"
    )

    # --------------------------------------------------
    # DIAGNOSTIC COMPARISON ON TEST SAMPLES
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("TEST SAMPLE COMPARISON")
    print("=" * 65)

    print(
        "NOTE: Test samples are used only for evaluation, "
        "not threshold calibration."
    )

    for category, filename in TEST_SAMPLES:

        image_path = (
            TEST_DIR
            / category
            / filename
        )

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        score = get_max_patch_score(image_path)

        prediction = (
            score > max_patch_threshold
        )

        print("\n" + "-" * 55)
        print(f"Image: {category}/{filename}")
        print(f"Maximum patch score: {score:.6f}")
        print(f"Calibrated threshold: {max_patch_threshold:.6f}")

        print(
            "Maximum-patch prediction:",
            "DEFECTIVE" if prediction else "NORMAL"
        )

    print("\n" + "=" * 65)
    print("CALIBRATION EXPERIMENT COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
