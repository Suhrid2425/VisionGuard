
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

try:
    from .tf_feature_extractor import (
        build_feature_extractor,
        extract_features,
    )
    from .tf_inference import (
        get_memory_bank,
        calculate_patch_distances,
        load_thresholds,
    )
except ImportError:
    from tf_feature_extractor import (
        build_feature_extractor,
        extract_features,
    )
    from tf_inference import (
        get_memory_bank,
        calculate_patch_distances,
        load_thresholds,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent

GRID_TEST_DIR = (
    PROJECT_ROOT / "datasets" / "MVTecAD" / "grid" / "test"
)

DEFECT_TYPES = [
    "bent",
    "broken",
    "glue",
    "metal_contamination",
    "thread",
]

TOP_PERCENT = 0.01


def calculate_scores(distances):
    """
    Compare alternative image-level aggregation methods.
    All scores are calculated from the same patch distances.
    """

    return {
        "top_1_percent_mean": float(
            np.mean(
                np.sort(distances)[-max(1, int(np.ceil(len(distances) * TOP_PERCENT))):]
            )
        ),

        "top_5_percent_mean": float(
            np.mean(
                np.sort(distances)[-max(1, int(np.ceil(len(distances) * 0.05))):]
            )
        ),

        "top_10_percent_mean": float(
            np.mean(
                np.sort(distances)[-max(1, int(np.ceil(len(distances) * 0.10))):]
            )
        ),

        "maximum": float(np.max(distances)),

        "overall_mean": float(np.mean(distances)),

        "percentile_99": float(np.percentile(distances, 99)),

        "percentile_95": float(np.percentile(distances, 95)),
    }


def main():

    print("=" * 70)
    print("GRID - IMAGE SCORING METHOD COMPARISON")
    print("=" * 70)

    print("\nBuilding feature extractor...")
    extractor = build_feature_extractor()

    print("\nLoading Grid memory bank...")
    memory_bank = get_memory_bank("grid")

    thresholds = load_thresholds("grid")
    current_threshold = float(thresholds["image_threshold"])

    print("\nCurrent image threshold:", round(current_threshold, 4))

    labels = []
    scores_by_method = {}

    total_images = 0

    folders = ["good"] + DEFECT_TYPES

    for defect_type in folders:

        image_dir = GRID_TEST_DIR / defect_type

        if not image_dir.exists():
            print("Skipping missing folder:", defect_type)
            continue

        image_paths = sorted(image_dir.glob("*.png"))

        print(f"\nProcessing {defect_type}: {len(image_paths)} images")

        for image_path in image_paths:

            features = extract_features(
                extractor,
                image_path,
            )

            patch_features = features.reshape(
                -1,
                features.shape[-1],
            ).astype(np.float32)

            distances = calculate_patch_distances(
                patch_features,
                memory_bank,
            )

            image_scores = calculate_scores(distances)

            labels.append(int(defect_type != "good"))

            for method, score in image_scores.items():

                scores_by_method.setdefault(method, []).append(score)

            total_images += 1

    labels = np.asarray(labels, dtype=np.int32)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("Total images:", total_images)
    print("Normal images:", int(np.sum(labels == 0)))
    print("Defective images:", int(np.sum(labels == 1)))

    print("\nAUROC BY SCORING METHOD")
    print("-" * 70)

    for method, scores in scores_by_method.items():

        scores = np.asarray(scores, dtype=np.float64)

        auc = roc_auc_score(labels, scores)

        predictions = scores > current_threshold

        true_positive = int(
            np.sum((predictions == 1) & (labels == 1))
        )

        false_positive = int(
            np.sum((predictions == 1) & (labels == 0))
        )

        false_negative = int(
            np.sum((predictions == 0) & (labels == 1))
        )

        recall = true_positive / max(1, int(np.sum(labels == 1)))

        print(f"\nMethod: {method}")
        print("AUROC:", round(float(auc), 4))
        print("Recall at current threshold:", round(recall, 4))
        print("TP:", true_positive)
        print("FP:", false_positive)
        print("FN:", false_negative)

        print("Normal score range:",
              round(float(np.min(scores[labels == 0])), 4),
              "to",
              round(float(np.max(scores[labels == 0])), 4))

        print("Defect score range:",
              round(float(np.min(scores[labels == 1])), 4),
              "to",
              round(float(np.max(scores[labels == 1])), 4))

    print("\nDiagnostic completed.")
    print("No model files or thresholds were modified.")


if __name__ == "__main__":
    main()
