
from pathlib import Path
import json
import numpy as np

from ml_engine.tf_feature_extractor import (
    build_feature_extractor,
    extract_features,
)

from ml_engine.tf_inference import (
    get_memory_bank,
    calculate_patch_distances,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_PATH = (
    PROJECT_ROOT
    / "ml_engine"
    / "models"
    / "mvtec_ad"
    / "grid"
    / "calibration_manifest.json"
)

TOP_PERCENT = 0.01


def calculate_scores(distances):

    sorted_distances = np.sort(distances)

    return {
        "top_1_percent_mean": float(
            np.mean(sorted_distances[-max(1, int(np.ceil(len(distances) * 0.01))):])
        ),
        "top_5_percent_mean": float(
            np.mean(sorted_distances[-max(1, int(np.ceil(len(distances) * 0.05))):])
        ),
        "top_10_percent_mean": float(
            np.mean(sorted_distances[-max(1, int(np.ceil(len(distances) * 0.10))):])
        ),
        "maximum": float(np.max(distances)),
        "overall_mean": float(np.mean(distances)),
        "percentile_99": float(np.percentile(distances, 99)),
        "percentile_95": float(np.percentile(distances, 95)),
    }


def main():

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("test_set_used") is not False:
        raise ValueError("Calibration manifest does not confirm test_set_used=False.")

    image_paths = manifest.get("calibration_images", [])

    if not image_paths:
        raise ValueError("No calibration images found.")

    print("=" * 70)
    print("GRID NORMAL CALIBRATION SCORING DISTRIBUTION")
    print("=" * 70)

    print("Calibration images:", len(image_paths))

    extractor = build_feature_extractor()
    memory_bank = get_memory_bank("grid")

    all_scores = {}

    for index, image_path in enumerate(image_paths, start=1):

        features = extract_features(extractor, Path(image_path))

        patch_features = features.reshape(
            -1,
            features.shape[-1],
        ).astype(np.float32)

        distances = calculate_patch_distances(
            patch_features,
            memory_bank,
        )

        scores = calculate_scores(distances)

        for method, score in scores.items():
            all_scores.setdefault(method, []).append(score)

        if index % 10 == 0:
            print(f"Processed {index}/{len(image_paths)} images")

    print("\n" + "=" * 70)
    print("NORMAL CALIBRATION SCORE SUMMARY")
    print("=" * 70)

    for method, values in all_scores.items():

        values = np.asarray(values)

        print(f"\nMethod: {method}")
        print("Mean:", round(float(np.mean(values)), 4))
        print("Median:", round(float(np.median(values)), 4))
        print("Maximum:", round(float(np.max(values)), 4))
        print("95th percentile:", round(float(np.percentile(values, 95)), 4))
        print("99th percentile:", round(float(np.percentile(values, 99)), 4))

    print("\nCompleted. No model or threshold files modified.")


if __name__ == "__main__":
    main()
