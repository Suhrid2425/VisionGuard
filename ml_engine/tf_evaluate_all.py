
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

from ml_engine.tf_inference import (
    analyze_image,
    load_thresholds,
    CATEGORIES,
)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "datasets" / "MVTecAD"

OUTPUT_DIR = PROJECT_ROOT / "ml_engine" / "evaluation"

OUTPUT_FILE = (
    OUTPUT_DIR / "tensorflow_patchcore_all_categories.json"
)

IMAGE_SIZE = (256, 256)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def load_ground_truth_mask(
    image_path,
    defect_type,
    ground_truth_dir,
):

    if defect_type == "good":
        return np.zeros(IMAGE_SIZE, dtype=np.uint8)

    mask_path = (
        ground_truth_dir
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

    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "AUROC requires both normal and anomalous samples."
        )

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


def calculate_pixel_metrics(
    pixel_true,
    pixel_pred,
    pixel_scores,
):

    return calculate_metrics(
        pixel_true,
        pixel_pred,
        pixel_scores,
    )


def calculate_macro_average(category_results, metric_name):

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "auroc",
    ]

    result = {}

    for metric in metric_names:

        values = [
            category_results[category][metric_name][metric]
            for category in category_results
        ]

        result[metric] = float(np.mean(values))

    return result


# ---------------------------------------------------------
# EVALUATE ONE CATEGORY
# ---------------------------------------------------------

def evaluate_category(category):

    print("\n" + "=" * 70)
    print(f"EVALUATING CATEGORY: {category.upper()}")
    print("=" * 70)

    category_dir = DATASET_ROOT / category

    test_dir = category_dir / "test"

    ground_truth_dir = category_dir / "ground_truth"

    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test directory not found: {test_dir}"
        )

    if not ground_truth_dir.exists():
        raise FileNotFoundError(
            f"Ground-truth directory not found: {ground_truth_dir}"
        )

    # Load this category's own calibrated thresholds.

    thresholds = load_thresholds(category)

    image_threshold = float(
        thresholds["image_threshold"]
    )

    pixel_threshold = float(
        thresholds["pixel_threshold"]
    )

    print("Image threshold:", image_threshold)
    print("Pixel threshold:", pixel_threshold)

    # Collect test images.

    test_images = []

    defect_dirs = sorted(
        path for path in test_dir.iterdir()
        if path.is_dir()
    )

    for defect_dir in defect_dirs:

        defect_type = defect_dir.name

        for image_path in sorted(defect_dir.iterdir()):

            if image_path.suffix.lower() in IMAGE_EXTENSIONS:

                test_images.append(
                    (image_path, defect_type)
                )

    if not test_images:
        raise ValueError(
            f"No test images found for {category}"
        )

    print("Test images:", len(test_images))

    # Image-level values.

    image_true = []
    image_pred = []
    image_scores = []

    # Pixel-level values.
    # Kept category-specific to limit memory usage.

    pixel_true_parts = []
    pixel_pred_parts = []
    pixel_score_parts = []

    defect_results = {}

    start_time = time.time()

    # Process images.

    for index, (image_path, defect_type) in enumerate(
        test_images,
        start=1,
    ):

        print(
            f"\r[{index}/{len(test_images)}] "
            f"{defect_type}/{image_path.name}",
            end="",
            flush=True,
        )

        result = analyze_image(
            image_path,
            category=category,
        )

        anomaly_score = float(
            result["anomaly_score"]
        )

        anomaly_map = np.asarray(
            result["anomaly_map"],
            dtype=np.float32,
        )

        if anomaly_map.shape != IMAGE_SIZE:
            raise ValueError(
                f"Unexpected anomaly map shape: "
                f"{anomaly_map.shape} for {image_path}"
            )

        predicted_label = int(
            anomaly_score > image_threshold
        )

        predicted_mask = (
            anomaly_map > pixel_threshold
        ).astype(np.uint8)

        ground_truth_mask = load_ground_truth_mask(
            image_path,
            defect_type,
            ground_truth_dir,
        )

        actual_label = int(
            defect_type != "good"
        )

        image_true.append(actual_label)
        image_pred.append(predicted_label)
        image_scores.append(anomaly_score)

        pixel_true_parts.append(
            ground_truth_mask.reshape(-1)
        )

        pixel_pred_parts.append(
            predicted_mask.reshape(-1)
        )

        pixel_score_parts.append(
            anomaly_map.reshape(-1)
        )

        # Defect-type statistics.

        if defect_type not in defect_results:

            defect_results[defect_type] = {
                "total": 0,
                "correct": 0,
                "predicted_anomalies": 0,
            }

        defect_results[defect_type]["total"] += 1

        if predicted_label == actual_label:

            defect_results[defect_type]["correct"] += 1

        if predicted_label == 1:

            defect_results[defect_type][
                "predicted_anomalies"
            ] += 1

    print()

    # Convert image-level arrays.

    image_true = np.asarray(
        image_true,
        dtype=np.uint8,
    )

    image_pred = np.asarray(
        image_pred,
        dtype=np.uint8,
    )

    image_scores = np.asarray(
        image_scores,
        dtype=np.float32,
    )

    # Calculate image metrics.

    image_metrics = calculate_metrics(
        image_true,
        image_pred,
        image_scores,
    )

    # Combine pixels for this category only.

    pixel_true = np.concatenate(pixel_true_parts)

    pixel_pred = np.concatenate(pixel_pred_parts)

    pixel_scores = np.concatenate(pixel_score_parts)

    # Calculate pixel metrics.

    pixel_metrics = calculate_pixel_metrics(
        pixel_true,
        pixel_pred,
        pixel_scores,
    )

    # Defect-type accuracy.

    for defect_type, values in defect_results.items():

        values["accuracy"] = float(
            values["correct"] / values["total"]
        )

    elapsed = time.time() - start_time

    report = {
        "category": category,

        "test_images": len(test_images),

        "thresholds": {
            "image_threshold": image_threshold,
            "pixel_threshold": pixel_threshold,
        },

        "image_level_metrics": image_metrics,

        "pixel_level_metrics": pixel_metrics,

        "defect_type_results": defect_results,

        "elapsed_seconds": round(elapsed, 2),
    }

    print("\nIMAGE-LEVEL METRICS")

    for metric, value in image_metrics.items():
        print(f"{metric}: {value}")

    print("\nPIXEL-LEVEL METRICS")

    for metric, value in pixel_metrics.items():
        print(f"{metric}: {value}")

    print(
        f"\n{category.upper()} completed in "
        f"{elapsed / 60:.2f} minutes"
    )

    return report


# ---------------------------------------------------------
# EVALUATE ALL CATEGORIES
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("VISIONGUARD - ALL CATEGORY PATCHCORE EVALUATION")
    print("=" * 70)

    print("Total categories:", len(CATEGORIES))

    start_time = time.time()

    category_reports = {}

    all_image_true = []
    all_image_pred = []
    all_image_scores = []

    # Aggregate pixel confusion counts without
    # retaining every category's pixel arrays.

    total_pixel_cm = np.zeros(
        (2, 2),
        dtype=np.int64,
    )

    category_pixel_aurocs = []

    # Evaluate each category.

    for category in CATEGORIES:

        report = evaluate_category(category)

        category_reports[category] = report

        # Store image-level data for combined metrics.

        category_dir = DATASET_ROOT / category

        test_dir = category_dir / "test"

        thresholds = report["thresholds"]

        image_threshold = thresholds["image_threshold"]

        for defect_dir in sorted(
            path for path in test_dir.iterdir()
            if path.is_dir()
        ):

            defect_type = defect_dir.name

            for image_path in sorted(defect_dir.iterdir()):

                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                # Reuse category results would require storing
                # image-level arrays. Instead, collect them during
                # a separate lightweight inference pass below.

                result = analyze_image(
                    image_path,
                    category=category,
                )

                score = float(result["anomaly_score"])

                actual = int(defect_type != "good")

                predicted = int(score > image_threshold)

                all_image_true.append(actual)
                all_image_pred.append(predicted)
                all_image_scores.append(score)

        # Aggregate pixel confusion matrices.

        category_cm = np.asarray(
            report["pixel_level_metrics"]["confusion_matrix"],
            dtype=np.int64,
        )

        total_pixel_cm += category_cm

        category_pixel_aurocs.append(
            report["pixel_level_metrics"]["auroc"]
        )

    # Combined image-level metrics.

    all_image_true = np.asarray(
        all_image_true,
        dtype=np.uint8,
    )

    all_image_pred = np.asarray(
        all_image_pred,
        dtype=np.uint8,
    )

    all_image_scores = np.asarray(
        all_image_scores,
        dtype=np.float32,
    )

    overall_image_metrics = calculate_metrics(
        all_image_true,
        all_image_pred,
        all_image_scores,
    )

    # Combined pixel metrics from aggregate confusion counts.

    tn, fp = total_pixel_cm[0]
    fn, tp = total_pixel_cm[1]

    total_pixels = int(total_pixel_cm.sum())

    overall_pixel_metrics = {
        "accuracy": float(
            (tn + tp) / total_pixels
        ),

        "precision": float(
            tp / (tp + fp) if (tp + fp) else 0.0
        ),

        "recall": float(
            tp / (tp + fn) if (tp + fn) else 0.0
        ),

        "f1_score": float(
            (2 * tp) / (2 * tp + fp + fn)
            if (2 * tp + fp + fn)
            else 0.0
        ),

        "confusion_matrix": total_pixel_cm.tolist(),

        "auroc_macro_average": float(
            np.mean(category_pixel_aurocs)
        ),
    }

    # Macro-average across categories.

    macro_image_metrics = calculate_macro_average(
        category_reports,
        "image_level_metrics",
    )

    macro_pixel_metrics = calculate_macro_average(
        category_reports,
        "pixel_level_metrics",
    )

    elapsed = time.time() - start_time

    # Final report.

    final_report = {

        "model": "TensorFlow PatchCore",

        "dataset": "MVTec AD",

        "categories_evaluated": len(category_reports),

        "category_names": list(category_reports.keys()),

        "total_test_images": int(len(all_image_true)),

        "overall_image_level_metrics": overall_image_metrics,

        "macro_average_image_metrics": macro_image_metrics,

        "overall_pixel_level_metrics": overall_pixel_metrics,

        "macro_average_pixel_metrics": macro_pixel_metrics,

        "category_results": category_reports,

        "evaluation_notes": [
            "Each category uses its own memory bank and calibrated thresholds.",
            "Test images are not used for threshold calibration.",
            "Image-level overall metrics are calculated across all test images.",
            "Macro-average metrics give equal weight to each category.",
            "Pixel confusion counts are aggregated across all categories.",
            "Pixel AUROC is reported as a macro-average of category AUROCs.",
            "Ground-truth masks are resized using nearest-neighbor interpolation.",
            "Pixel metrics are calculated at 256x256 resolution.",
            "This evaluates the custom TensorFlow PatchCore implementation, not an exact reproduction of Anomalib PatchCore."
        ],

        "elapsed_seconds": round(elapsed, 2),

        "elapsed_minutes": round(elapsed / 60, 2),
    }

    # Save report.

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
            final_report,
            file,
            indent=4,
        )

    # Print summary.

    print("\n" + "=" * 70)
    print("OVERALL IMAGE-LEVEL METRICS")
    print("=" * 70)

    for metric, value in overall_image_metrics.items():
        print(f"{metric}: {value}")

    print("\n" + "=" * 70)
    print("MACRO-AVERAGE IMAGE METRICS")
    print("=" * 70)

    for metric, value in macro_image_metrics.items():
        print(f"{metric}: {value}")

    print("\n" + "=" * 70)
    print("OVERALL PIXEL-LEVEL METRICS")
    print("=" * 70)

    for metric, value in overall_pixel_metrics.items():
        print(f"{metric}: {value}")

    print("\n" + "=" * 70)
    print("MACRO-AVERAGE PIXEL METRICS")
    print("=" * 70)

    for metric, value in macro_pixel_metrics.items():
        print(f"{metric}: {value}")

    print("\n" + "=" * 70)
    print("CATEGORY SUMMARY")
    print("=" * 70)

    for category, report in category_reports.items():

        image = report["image_level_metrics"]

        pixel = report["pixel_level_metrics"]

        print(f"\n{category.upper()}")

        print(
            "Image accuracy:",
            round(image["accuracy"] * 100, 2),
            "%",
        )

        print(
            "Image AUROC:",
            round(image["auroc"], 4),
        )

        print(
            "Pixel AUROC:",
            round(pixel["auroc"], 4),
        )

        print(
            "Pixel F1:",
            round(pixel["f1_score"], 4),
        )

    print("\n" + "=" * 70)
    print("ALL-CATEGORY EVALUATION COMPLETED")
    print("=" * 70)

    print("\nCategories:", len(category_reports))

    print("Test images:", len(all_image_true))

    print("Results saved to:")

    print(OUTPUT_FILE)

    print(
        f"\nTotal elapsed time: "
        f"{elapsed / 60:.2f} minutes"
    )


if __name__ == "__main__":
    main()
