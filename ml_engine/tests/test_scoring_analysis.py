
from pathlib import Path
import sys
import math

import numpy as np

# --------------------------------------------------
# PROJECT PATH
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_engine.tf_inference import analyze_image, load_thresholds


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATASET_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "MVTecAD"
    / "bottle"
)

SAMPLES = [
    ("good", "000.png"),
    ("broken_large", "000.png"),
    ("broken_small", "000.png"),
    ("contamination", "000.png"),
    ("contamination", "003.png"),
    ("contamination", "019.png"),
]

TOP_PERCENTAGES = [1, 2, 5]


# --------------------------------------------------
# SCORING FUNCTIONS
# --------------------------------------------------

def calculate_top_k_mean(distances, percentage):

    distances = np.asarray(
        distances,
        dtype=np.float32
    ).reshape(-1)

    k = max(
        1,
        math.ceil(len(distances) * percentage / 100)
    )

    top_distances = np.partition(
        distances,
        len(distances) - k
    )[-k:]

    return float(np.mean(top_distances)), k


def analyze_sample(category, filename, image_threshold, pixel_threshold):

    image_path = (
        DATASET_DIR
        / "test"
        / category
        / filename
    )

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    result = analyze_image(image_path)

    distances = np.asarray(
        result["patch_distances"],
        dtype=np.float32
    ).reshape(-1)

    anomaly_map = np.asarray(
        result["anomaly_map"],
        dtype=np.float32
    )

    current_score = float(result["anomaly_score"])

    # Alternative image scores
    top_scores = {}

    for percentage in TOP_PERCENTAGES:

        score, k = calculate_top_k_mean(
            distances,
            percentage
        )

        top_scores[percentage] = (score, k)

    maximum_patch_distance = float(np.max(distances))

    # Pixel-level anomaly information
    pixel_mask = anomaly_map > pixel_threshold

    anomaly_pixel_percentage = (
        np.count_nonzero(pixel_mask)
        / pixel_mask.size
    ) * 100

    maximum_anomaly_value = float(np.max(anomaly_map))

    # --------------------------------------------------
    # REPORT
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print(f"IMAGE: {category}/{filename}")
    print("=" * 70)

    print(f"Current image score: {current_score:.6f}")
    print(f"Image threshold:     {image_threshold:.6f}")

    print(
        "Current prediction: ",
        "DEFECTIVE" if current_score > image_threshold else "NORMAL"
    )

    print(
        f"Threshold margin:   {current_score - image_threshold:+.6f}"
    )

    print("\nAlternative top-patch scores:")

    for percentage, (score, k) in top_scores.items():

        diagnostic_prediction = (
            "DEFECTIVE"
            if score > image_threshold
            else "NORMAL"
        )

        print(
            f"Top {percentage}% "
            f"({k} patches): "
            f"{score:.6f} | "
            f"{diagnostic_prediction}"
        )

    print("\nPatch distance statistics:")

    print(f"Maximum: {maximum_patch_distance:.6f}")
    print(f"Mean:    {np.mean(distances):.6f}")
    print(f"Median:  {np.median(distances):.6f}")
    print(f"Std:     {np.std(distances):.6f}")

    print("\nAnomaly map statistics:")

    print(f"Maximum anomaly value: {maximum_anomaly_value:.6f}")
    print(
        f"Pixels above calibrated pixel threshold: "
        f"{anomaly_pixel_percentage:.2f}%"
    )

    print(
        "\nNOTE: Patch distances and anomaly-map pixel values "
        "are different representations. "
        "The calibrated pixel threshold must not be applied "
        "directly to raw patch distances."
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    thresholds = load_thresholds()

    image_threshold = float(
        thresholds["image_threshold"]
    )

    pixel_threshold = float(
        thresholds["pixel_threshold"]
    )

    print("=" * 70)
    print("PATCHCORE IMAGE SCORING DIAGNOSTIC")
    print("=" * 70)

    print(f"Image threshold: {image_threshold:.6f}")
    print(f"Pixel threshold: {pixel_threshold:.6f}")

    for category, filename in SAMPLES:

        analyze_sample(
            category,
            filename,
            image_threshold,
            pixel_threshold
        )

    print("\n" + "=" * 70)
    print("SCORING ANALYSIS COMPLETED")
    print("=" * 70)
