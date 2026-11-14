
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_engine.tf_inference import predict_image


TEST_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "MVTecAD"
    / "bottle"
    / "test"
)

GROUND_TRUTH_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "MVTecAD"
    / "bottle"
    / "ground_truth"
)

CATEGORIES = [
    "broken_large",
    "broken_small",
    "contamination",
]

IMAGE_SIZE = (256, 256)


def load_ground_truth(mask_path):
    """Load a ground-truth mask and resize it using nearest neighbor."""

    mask = Image.open(mask_path).convert("L")

    mask = mask.resize(
        IMAGE_SIZE,
        Image.Resampling.NEAREST,
    )

    mask = np.array(mask)

    return (mask > 0).astype(np.uint8)


def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(
        y_true == 1,
        y_pred == 1,
    ).sum()

    union = np.logical_or(
        y_true == 1,
        y_pred == 1,
    ).sum()

    if union == 0:
        return 1.0

    return intersection / union


def calculate_dice(y_true, y_pred):
    intersection = np.logical_and(
        y_true == 1,
        y_pred == 1,
    ).sum()

    total = y_true.sum() + y_pred.sum()

    if total == 0:
        return 1.0

    return (2.0 * intersection) / total


def run_evaluation():
    print("\nStarting pixel-level localization evaluation...\n")

    all_true_pixels = []
    all_pred_pixels = []
    all_anomaly_scores = []

    category_results = []

    total_images = 0

    for category in CATEGORIES:
        image_dir = TEST_DIR / category
        mask_dir = GROUND_TRUTH_DIR / category

        if not image_dir.exists():
            raise FileNotFoundError(image_dir)

        if not mask_dir.exists():
            raise FileNotFoundError(mask_dir)

        images = sorted(image_dir.glob("*.png"))

        if not images:
            raise FileNotFoundError(
                f"No test images found in {image_dir}"
            )

        category_true = []
        category_pred = []
        category_scores = []

        print(
            f"\nEvaluating {category}: "
            f"{len(images)} images"
        )

        for image_path in images:
            mask_path = (
                mask_dir / f"{image_path.stem}_mask.png"
            )

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Ground-truth mask missing: {mask_path}"
                )

            # Load actual defect mask
            ground_truth = load_ground_truth(mask_path)

            # Run model prediction
            result = predict_image(str(image_path))

            predicted_mask = result["predicted_mask"]
            anomaly_map = result["anomaly_map"]

            if predicted_mask.shape != IMAGE_SIZE:
                raise ValueError(
                    f"Unexpected predicted mask shape: "
                    f"{predicted_mask.shape}"
                )

            if anomaly_map.shape != IMAGE_SIZE:
                raise ValueError(
                    f"Unexpected anomaly map shape: "
                    f"{anomaly_map.shape}"
                )

            if not np.isfinite(anomaly_map).all():
                raise ValueError(
                    f"Non-finite anomaly map: {image_path}"
                )

            # Flatten pixels for global metrics
            true_pixels = ground_truth.ravel()
            pred_pixels = predicted_mask.ravel()
            score_pixels = anomaly_map.ravel()

            category_true.append(true_pixels)
            category_pred.append(pred_pixels)
            category_scores.append(score_pixels)

            all_true_pixels.append(true_pixels)
            all_pred_pixels.append(pred_pixels)
            all_anomaly_scores.append(score_pixels)

            total_images += 1

        category_true = np.concatenate(category_true)
        category_pred = np.concatenate(category_pred)

        category_iou = calculate_iou(
            category_true,
            category_pred,
        )

        category_dice = calculate_dice(
            category_true,
            category_pred,
        )

        category_f1 = f1_score(
            category_true,
            category_pred,
            zero_division=0,
        )

        category_results.append({
            "category": category,
            "images": len(images),
            "iou": category_iou,
            "dice": category_dice,
            "f1": category_f1,
        })

        print(f"Images evaluated: {len(images)}")
        print(f"Pixel F1: {category_f1:.4f}")
        print(f"IoU: {category_iou:.4f}")
        print(f"Dice: {category_dice:.4f}")

    # Combine all pixels
    y_true = np.concatenate(all_true_pixels)
    y_pred = np.concatenate(all_pred_pixels)
    y_scores = np.concatenate(all_anomaly_scores)

    # Calculate global metrics
    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    iou = calculate_iou(y_true, y_pred)
    dice = calculate_dice(y_true, y_pred)

    auc = roc_auc_score(y_true, y_scores)

    # Final report
    print("\n" + "=" * 60)
    print("PIXEL-LEVEL LOCALIZATION REPORT")
    print("=" * 60)

    print(f"Total defective images: {total_images}")
    print(f"Total pixels evaluated: {len(y_true):,}")

    print("\nGLOBAL METRICS")
    print("-" * 40)

    print(f"Pixel Accuracy:  {accuracy * 100:.2f}%")
    print(f"Pixel Precision: {precision * 100:.2f}%")
    print(f"Pixel Recall:    {recall * 100:.2f}%")
    print(f"Pixel F1-score:  {f1 * 100:.2f}%")
    print(f"IoU:             {iou:.4f}")
    print(f"Dice:            {dice:.4f}")
    print(f"Pixel ROC-AUC:   {auc:.4f}")

    print("\nCATEGORY-WISE RESULTS")
    print("-" * 40)

    for item in category_results:
        print(
            f"{item['category']:16s} | "
            f"F1: {item['f1']:.4f} | "
            f"IoU: {item['iou']:.4f} | "
            f"Dice: {item['dice']:.4f}"
        )

    print("\n" + "=" * 60)
    print("PIXEL-LEVEL EVALUATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
