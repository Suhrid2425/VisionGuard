from pathlib import Path
import json
import time

import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

try:
    # When executed from the project root as a script
    from ml_engine.tf_inference import (
        analyze_image,
        load_thresholds,
    )

except ModuleNotFoundError:
    # When Python executes this file directly from ml_engine
    from tf_inference import (
        analyze_image,
        load_thresholds,
    )


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "datasets" / "MVTecAD" / "bottle"

TEST_DIR = DATASET_DIR / "test"
GROUND_TRUTH_DIR = DATASET_DIR / "ground_truth"

OUTPUT_DIR = PROJECT_ROOT / "ml_engine" / "evaluation"
OUTPUT_FILE = OUTPUT_DIR / "tensorflow_patchcore_results.json"

IMAGE_SIZE = (256, 256)

EXPECTED_TEST_IMAGES = 83


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def load_ground_truth_mask(image_path, defect_type):

    if defect_type == "good":
        return np.zeros(IMAGE_SIZE, dtype=np.uint8)

    mask_path = (
        GROUND_TRUTH_DIR
        / defect_type
        / f"{image_path.stem}_mask.png"
    )

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Ground-truth mask not found: {mask_path}"
        )

    with Image.open(mask_path) as mask_image:
        mask_image = mask_image.convert("L")
        mask_image = mask_image.resize(
            IMAGE_SIZE,
            Image.Resampling.NEAREST,
        )

        mask = np.asarray(mask_image)

    return (mask > 0).astype(np.uint8)


def calculate_metrics(y_true, y_pred, y_scores):

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "f1_score": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "auroc": float(
            roc_auc_score(y_true, y_scores)
        ),

        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).tolist(),
    }


# ---------------------------------------------------------
# MAIN EVALUATION
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("VISIONGUARD - TENSORFLOW PATCHCORE TEST EVALUATION")
    print("=" * 70)

    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Test directory not found: {TEST_DIR}"
        )

    if not GROUND_TRUTH_DIR.exists():
        raise FileNotFoundError(
            f"Ground-truth directory not found: {GROUND_TRUTH_DIR}"
        )

    thresholds = load_thresholds()

    image_threshold = float(
        thresholds["image_threshold"]
    )

    pixel_threshold = float(
        thresholds["pixel_threshold"]
    )

    print("\nUsing fixed thresholds:")
    print("Image threshold:", image_threshold)
    print("Pixel threshold:", pixel_threshold)

    # -----------------------------------------------------
    # Collect test images
    # -----------------------------------------------------

    test_images = []

    defect_types = sorted(
        path for path in TEST_DIR.iterdir()
        if path.is_dir()
    )

    for defect_dir in defect_types:

        defect_type = defect_dir.name

        for image_path in sorted(defect_dir.iterdir()):

            if image_path.suffix.lower() in [
                ".png",
                ".jpg",
                ".jpeg",
            ]:
                test_images.append(
                    (image_path, defect_type)
                )

    if len(test_images) != EXPECTED_TEST_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_TEST_IMAGES} test images, "
            f"found {len(test_images)}."
        )

    print("\nTest images:", len(test_images))

    # -----------------------------------------------------
    # Store evaluation values
    # -----------------------------------------------------

    image_true = []
    image_pred = []
    image_scores = []

    pixel_true = []
    pixel_pred = []
    pixel_scores = []

    category_results = {}

    start_time = time.time()

    # -----------------------------------------------------
    # Process test images
    # -----------------------------------------------------

    for index, (image_path, defect_type) in enumerate(
        test_images,
        start=1,
    ):

        print(
            f"\n[{index}/{len(test_images)}] "
            f"Processing: {defect_type}/{image_path.name}"
        )

        result = analyze_image(image_path)

        anomaly_score = float(
            result["anomaly_score"]
        )

        anomaly_map = result["anomaly_map"]

        predicted_label = int(
            anomaly_score > image_threshold
        )

        predicted_mask = (
            anomaly_map > pixel_threshold
        ).astype(np.uint8)

        ground_truth_mask = load_ground_truth_mask(
            image_path,
            defect_type,
        )

        actual_label = int(defect_type != "good")

        image_true.append(actual_label)
        image_pred.append(predicted_label)
        image_scores.append(anomaly_score)

        pixel_true.append(
            ground_truth_mask.reshape(-1)
        )

        pixel_pred.append(
            predicted_mask.reshape(-1)
        )

        pixel_scores.append(
            anomaly_map.reshape(-1)
        )

        # Per-category image classification statistics
        if defect_type not in category_results:
            category_results[defect_type] = {
                "total": 0,
                "correct": 0,
                "predicted_anomalies": 0,
            }

        category_results[defect_type]["total"] += 1

        if predicted_label == actual_label:
            category_results[defect_type]["correct"] += 1

        if predicted_label == 1:
            category_results[defect_type]["predicted_anomalies"] += 1

        print(
            f"Score: {anomaly_score:.4f} | "
            f"Actual: {actual_label} | "
            f"Predicted: {predicted_label}"
        )

    # -----------------------------------------------------
    # Convert arrays
    # -----------------------------------------------------

    image_true = np.asarray(image_true, dtype=np.uint8)
    image_pred = np.asarray(image_pred, dtype=np.uint8)
    image_scores = np.asarray(image_scores, dtype=np.float32)

    pixel_true = np.concatenate(pixel_true)
    pixel_pred = np.concatenate(pixel_pred)
    pixel_scores = np.concatenate(pixel_scores)

    # -----------------------------------------------------
    # Calculate image metrics
    # -----------------------------------------------------

    image_metrics = calculate_metrics(
        image_true,
        image_pred,
        image_scores,
    )

    # -----------------------------------------------------
    # Calculate pixel metrics
    # -----------------------------------------------------

    pixel_metrics = calculate_metrics(
        pixel_true,
        pixel_pred,
        pixel_scores,
    )

    # -----------------------------------------------------
    # Category accuracy
    # -----------------------------------------------------

    for category, values in category_results.items():

        values["accuracy"] = (
            values["correct"] / values["total"]
        )

    # -----------------------------------------------------
    # Prepare report
    # -----------------------------------------------------

    report = {
        "model": "TensorFlow PatchCore",
        "dataset": "MVTec AD - bottle",
        "test_images": len(test_images),

        "thresholds": {
            "image_threshold": image_threshold,
            "pixel_threshold": pixel_threshold,
        },

        "image_level_metrics": image_metrics,

        "pixel_level_metrics": pixel_metrics,

        "category_results": category_results,

        "evaluation_notes": [
            "Thresholds were calibrated using normal validation images.",
            "Test images were not used for threshold calibration.",
            "Pixel ground-truth masks were resized using nearest-neighbor interpolation.",
            "Pixel metrics were calculated at 256x256 resolution.",
            "This evaluation measures the TensorFlow implementation, not an exact reproduction of Anomalib PatchCore."
        ],
    }

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # Print final report
    # -----------------------------------------------------

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("IMAGE-LEVEL EVALUATION")
    print("=" * 70)

    for metric, value in image_metrics.items():
        print(f"{metric}: {value}")

    print("\n" + "=" * 70)
    print("PIXEL-LEVEL EVALUATION")
    print("=" * 70)

    for metric, value in pixel_metrics.items():
        print(f"{metric}: {value}")

    print("\n" + "=" * 70)
    print("CATEGORY RESULTS")
    print("=" * 70)

    for category, values in category_results.items():
        print(f"\n{category}:")
        print("Total:", values["total"])
        print("Correct:", values["correct"])
        print("Accuracy:", round(values["accuracy"] * 100, 2), "%")
        print(
            "Predicted anomalies:",
            values["predicted_anomalies"],
        )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)

    print("\nResults saved to:")
    print(OUTPUT_FILE)

    print(f"\nElapsed time: {elapsed / 60:.2f} minutes")


if __name__ == "__main__":
    main()
