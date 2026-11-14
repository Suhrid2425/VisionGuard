
import sys
from pathlib import Path

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_engine.tf_inference import predict_image, load_thresholds


TEST_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "MVTecAD"
    / "bottle"
    / "test"
)

CATEGORIES = [
    "good",
    "broken_large",
    "broken_small",
    "contamination",
]


def run_evaluation():
    print("\nStarting full MVTec AD Bottle evaluation...\n")

    thresholds = load_thresholds()

    image_threshold = float(thresholds["image_threshold"])

    y_true = []
    y_pred = []
    y_scores = []

    category_results = {}
    false_positives = []
    false_negatives = []

    total_images = 0

    for category in CATEGORIES:
        category_dir = TEST_DIR / category

        if not category_dir.exists():
            raise FileNotFoundError(
                f"Dataset category not found: {category_dir}"
            )

        images = sorted(category_dir.glob("*.png"))

        if not images:
            raise FileNotFoundError(
                f"No PNG images found in {category_dir}"
            )

        expected_label = 0 if category == "good" else 1

        category_true = []
        category_pred = []

        print(
            f"\nEvaluating {category}: "
            f"{len(images)} images"
        )

        for image_path in images:
            result = predict_image(str(image_path))

            score = float(result["anomaly_score"])
            predicted_label = int(result["predicted_label"])

            if not np.isfinite(score):
                raise ValueError(
                    f"Invalid anomaly score: {image_path}"
                )

            if predicted_label not in (0, 1):
                raise ValueError(
                    f"Invalid predicted label: {image_path}"
                )

            y_true.append(expected_label)
            y_pred.append(predicted_label)
            y_scores.append(score)

            category_true.append(expected_label)
            category_pred.append(predicted_label)

            total_images += 1

            if predicted_label == 1 and expected_label == 0:
                false_positives.append({
                    "category": category,
                    "image": image_path.name,
                    "score": score,
                })

            elif predicted_label == 0 and expected_label == 1:
                false_negatives.append({
                    "category": category,
                    "image": image_path.name,
                    "score": score,
                })

        category_accuracy = accuracy_score(
            category_true,
            category_pred
        )

        category_results[category] = {
            "total": len(images),
            "correct": sum(
                actual == predicted
                for actual, predicted
                in zip(category_true, category_pred)
            ),
            "accuracy": category_accuracy,
        }

        print(
            f"Category accuracy: "
            f"{category_accuracy * 100:.2f}%"
        )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    # Overall classification metrics
    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(y_true, y_scores)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = matrix.ravel()

    # Print final report
    print("\n" + "=" * 65)
    print("FULL DATASET EVALUATION REPORT")
    print("=" * 65)

    print(f"Total images: {total_images}")
    print(f"Image threshold: {image_threshold:.6f}")

    print("\nOVERALL METRICS")
    print("-" * 40)

    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-score:  {f1 * 100:.2f}%")
    print(f"ROC-AUC:   {auc:.4f}")

    print("\nCONFUSION MATRIX")
    print("-" * 40)
    print("Rows = Actual | Columns = Predicted")
    print("             Normal  Defective")
    print(f"Normal       {tn:6d}  {fp:9d}")
    print(f"Defective    {fn:6d}  {tp:9d}")

    print("\nPER-CATEGORY PERFORMANCE")
    print("-" * 40)

    for category, data in category_results.items():
        print(
            f"{category:16s} | "
            f"{data['correct']}/{data['total']} correct | "
            f"Accuracy: {data['accuracy'] * 100:.2f}%"
        )

    print("\nFALSE POSITIVES")
    print("-" * 40)

    if false_positives:
        for item in false_positives:
            print(
                f"{item['category']}/{item['image']} | "
                f"Score: {item['score']:.6f}"
            )
    else:
        print("None")

    print("\nFALSE NEGATIVES")
    print("-" * 40)

    if false_negatives:
        for item in false_negatives:
            print(
                f"{item['category']}/{item['image']} | "
                f"Score: {item['score']:.6f}"
            )
    else:
        print("None")

    print("\n" + "=" * 65)
    print("EVALUATION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    run_evaluation()
