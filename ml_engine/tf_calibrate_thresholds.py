
from pathlib import Path
import json
import time

import numpy as np

from tf_inference import analyze_image, CATEGORIES


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "ml_engine" / "models"

CATEGORIES_TO_CALIBRATE =  ["bottle"]

IMAGE_THRESHOLD_MARGIN = 1.05
PIXEL_THRESHOLD_MARGIN = 1.05
PIXEL_PERCENTILE = 99.9


# ---------------------------------------------------------
# CALIBRATE ONE CATEGORY
# ---------------------------------------------------------

def calibrate_category(category):

    print("\n" + "=" * 65)
    print(f"VISIONGUARD - CALIBRATING: {category.upper()}")
    print("=" * 65)

    category_dir = MODEL_DIR / "mvtec_ad" / category

    manifest_path = category_dir / "calibration_manifest.json"

    output_file = category_dir / "thresholds.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Calibration manifest not found: {manifest_path}"
        )

    # Load exact held-out calibration image paths.
    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    if manifest.get("category") != category:
        raise ValueError(
            f"Manifest category mismatch for {category}"
        )

    if manifest.get("test_set_used") is not False:
        raise ValueError(
            f"Invalid manifest: test set must not be used ({category})"
        )

    image_paths = [
        Path(path)
        for path in manifest["calibration_images"]
    ]

    expected_count = manifest["calibration_image_count"]

    if len(image_paths) != expected_count:
        raise ValueError(
            f"{category}: Manifest image count mismatch."
        )

    missing_images = [
        str(path) for path in image_paths
        if not path.exists()
    ]

    if missing_images:
        raise FileNotFoundError(
            f"{category}: {len(missing_images)} calibration "
            f"images are missing. First missing: {missing_images[0]}"
        )

    if not image_paths:
        raise ValueError(
            f"No calibration images found for {category}"
        )

    print("Calibration images:", len(image_paths))
    print("Memory bank:", category_dir / "memory_bank.npz")

    image_scores = []
    pixel_maps = []

    start_time = time.time()

    # -----------------------------------------------------
    # Analyze normal calibration images
    # -----------------------------------------------------

    for index, image_path in enumerate(image_paths, start=1):

        result = analyze_image(
            image_path,
            category=category,
        )

        image_scores.append(
            result["anomaly_score"]
        )

        pixel_maps.append(
            result["anomaly_map"].reshape(-1)
        )

        print(
            f"\rProcessed: {index}/{len(image_paths)} "
            f"| Score: {result['anomaly_score']:.4f}",
            end="",
            flush=True,
        )

    print()

    image_scores = np.asarray(
        image_scores,
        dtype=np.float32,
    )

    all_normal_pixels = np.concatenate(pixel_maps)

    # -----------------------------------------------------
    # Calculate image threshold
    # -----------------------------------------------------

    maximum_normal_score = float(
        np.max(image_scores)
    )

    image_threshold = (
        maximum_normal_score * IMAGE_THRESHOLD_MARGIN
    )

    # -----------------------------------------------------
    # Calculate pixel threshold
    # -----------------------------------------------------

    normal_pixel_percentile = float(
        np.percentile(
            all_normal_pixels,
            PIXEL_PERCENTILE,
        )
    )

    pixel_threshold = (
        normal_pixel_percentile * PIXEL_THRESHOLD_MARGIN
    )

    # -----------------------------------------------------
    # Save calibration results
    # -----------------------------------------------------

    metadata = {

        "model": "TensorFlow PatchCore",

        "category": category,

        "calibration_dataset": (
            f"MVTec AD {category}/train/good "
            "held-out calibration split"
        ),

        "calibration_image_count": len(image_paths),

        "random_seed": manifest["random_seed"],

        "test_set_used": False,

        "image_threshold": float(image_threshold),

        "pixel_threshold": float(pixel_threshold),

        "image_threshold_method": (
            "Maximum normal calibration image score "
            "multiplied by 1.05"
        ),

        "pixel_threshold_method": (
            "99.9th percentile of normal calibration "
            "anomaly-map pixels multiplied by 1.05"
        ),

        "image_threshold_margin": IMAGE_THRESHOLD_MARGIN,

        "pixel_threshold_margin": PIXEL_THRESHOLD_MARGIN,

        "pixel_percentile": PIXEL_PERCENTILE,

        "normal_image_score_statistics": {
            "minimum": float(np.min(image_scores)),
            "maximum": maximum_normal_score,
            "mean": float(np.mean(image_scores)),
            "median": float(np.median(image_scores)),
            "std": float(np.std(image_scores)),
            "p95": float(np.percentile(image_scores, 95)),
            "p99": float(np.percentile(image_scores, 99)),
        },

        "normal_pixel_statistics": {
            "minimum": float(np.min(all_normal_pixels)),
            "maximum": float(np.max(all_normal_pixels)),
            "mean": float(np.mean(all_normal_pixels)),
            "p99": float(np.percentile(all_normal_pixels, 99)),
            "p999": normal_pixel_percentile,
        },

        "validation_image_scores": [
            float(score) for score in image_scores
        ],

        "limitations": (
            "Calibration uses only held-out normal images. "
            "It estimates normal-score thresholds but cannot "
            "optimize anomaly sensitivity or establish recall. "
            "Calibration reliability depends on sample size."
        ),
    }

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    elapsed = time.time() - start_time

    # -----------------------------------------------------
    # Print report
    # -----------------------------------------------------

    print("\nCALIBRATION COMPLETED:", category.upper())

    print(f"Normal images: {len(image_paths)}")

    print(f"Image threshold: {image_threshold:.6f}")

    print(f"Pixel threshold: {pixel_threshold:.6f}")

    print(f"Normal image maximum: {maximum_normal_score:.6f}")

    print(
        f"Normal pixel {PIXEL_PERCENTILE}th percentile: "
        f"{normal_pixel_percentile:.6f}"
    )

    print("Saved to:", output_file)

    print(f"Elapsed: {elapsed / 60:.2f} minutes")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 65)
    print("VISIONGUARD - MULTI-CATEGORY PATCHCORE CALIBRATION")
    print("=" * 65)

    print("\nCategories to calibrate:", len(CATEGORIES_TO_CALIBRATE))

    print("Bottle calibration will remain unchanged.")

    overall_start = time.time()

    successful = []
    failed = []

    for category in CATEGORIES_TO_CALIBRATE:

        try:
            calibrate_category(category)
            successful.append(category)

        except Exception as error:

            failed.append(category)

            print(f"\nERROR calibrating {category}: {error}")

    print("\n" + "=" * 65)
    print("CALIBRATION SUMMARY")
    print("=" * 65)

    print("\nSuccessful:", len(successful))

    for category in successful:
        print("  SUCCESS:", category)

    print("\nFailed:", len(failed))

    for category in failed:
        print("  FAILED:", category)

    print(
        f"\nTotal elapsed: "
        f"{(time.time() - overall_start) / 60:.2f} minutes"
    )

    if failed:
        raise RuntimeError(
            "Some categories failed calibration. "
            "Check the errors above."
        )

    print("\nAll 14 categories calibrated successfully.")


if __name__ == "__main__":
    main()
