
import sys
from pathlib import Path

import numpy as np

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

CATEGORIES = [
    "good",
    "broken_large",
    "broken_small",
    "contamination",
]


def run_tests():
    print("\nStarting defective-image validation...\n")

    results = []

    for category in CATEGORIES:
        category_dir = TEST_DIR / category

        assert category_dir.exists(), (
            f"Dataset category not found: {category_dir}"
        )

        images = sorted(category_dir.glob("*.png"))

        assert images, (
            f"No PNG images found in {category_dir}"
        )

        # Use the first sorted image for reproducibility.
        image_path = images[0]

        # Ground truth comes from the dataset folder.
        expected_anomaly = category != "good"

        print(f"\nCategory: {category}")
        print(f"Image: {image_path.name}")

        prediction = predict_image(str(image_path))

        predicted_label = prediction["predicted_label"]
        anomaly_score = prediction["anomaly_score"]
        anomaly_map = prediction["anomaly_map"]
        predicted_mask = prediction["predicted_mask"]

        # Validate prediction output.
        assert isinstance(predicted_label, bool)
        assert np.isfinite(anomaly_score)
        assert anomaly_score >= 0

        assert anomaly_map.shape == (256, 256)
        assert np.isfinite(anomaly_map).all()

        assert predicted_mask.shape == (256, 256)
        assert predicted_mask.dtype == np.uint8
        assert np.isin(predicted_mask, [0, 1]).all()

        matched = predicted_label == expected_anomaly

        results.append({
            "category": category,
            "image": image_path.name,
            "expected": expected_anomaly,
            "predicted": predicted_label,
            "score": anomaly_score,
            "matched": matched,
        })

        print(f"Expected anomaly: {expected_anomaly}")
        print(f"Predicted anomaly: {predicted_label}")
        print(f"Anomaly score: {anomaly_score:.6f}")
        print(
            f"Defective pixel percentage: "
            f"{predicted_mask.mean() * 100:.2f}%"
        )
        print(f"Classification: {'MATCH' if matched else 'MISMATCH'}")

    # Final summary
    print("\n" + "=" * 55)
    print("DEFECTIVE-IMAGE VALIDATION SUMMARY")
    print("=" * 55)

    matches = sum(item["matched"] for item in results)

    for item in results:
        status = "MATCH" if item["matched"] else "MISMATCH"

        print(
            f"{item['category']:16s} | "
            f"Expected: {str(item['expected']):5s} | "
            f"Predicted: {str(item['predicted']):5s} | "
            f"Score: {item['score']:.4f} | "
            f"{status}"
        )

    print(f"\nCorrect classifications: {matches}/{len(results)}")

    if matches == len(results):
        print("All four sample classifications match ground truth.")
    else:
        print(
            "Some sample classifications differ from ground truth. "
            "Review the scores; do not change thresholds based on "
            "these test samples."
        )


if __name__ == "__main__":
    run_tests()
