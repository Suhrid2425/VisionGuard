
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_engine.tf_inference import analyze_image


VALIDATION_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "MVTecAD"
    / "bottle"
    / "validation"
    / "good"
)

TOP_N = 10


def main():

    images = sorted(VALIDATION_DIR.glob("*.png"))

    if not images:
        raise FileNotFoundError(
            f"No validation images found: {VALIDATION_DIR}"
        )

    results = []

    print("Analyzing normal validation images...")

    for image_path in images:

        result = analyze_image(image_path)

        distances = np.asarray(
            result["patch_distances"],
            dtype=np.float32
        ).reshape(-1)

        anomaly_map = np.asarray(
            result["anomaly_map"],
            dtype=np.float32
        )

        max_patch = float(np.max(distances))

        top_1_percent_count = max(
            1,
            int(np.ceil(len(distances) * 0.01))
        )

        top_distances = np.partition(
            distances,
            len(distances) - top_1_percent_count
        )[-top_1_percent_count:]

        top_1_percent_mean = float(
            np.mean(top_distances)
        )

        results.append({
            "filename": image_path.name,
            "max_patch": max_patch,
            "top_1_percent": top_1_percent_mean,
            "max_anomaly_map": float(np.max(anomaly_map)),
        })

    results.sort(
        key=lambda item: item["max_patch"],
        reverse=True
    )

    print("\n" + "=" * 85)
    print("TOP NORMAL VALIDATION OUTLIERS")
    print("=" * 85)

    print(
        f"{'Rank':<6}"
        f"{'Filename':<18}"
        f"{'Max Patch':<18}"
        f"{'Top 1% Mean':<18}"
        f"{'Max Anomaly Map':<18}"
    )

    print("-" * 85)

    for rank, item in enumerate(results[:TOP_N], start=1):

        print(
            f"{rank:<6}"
            f"{item['filename']:<18}"
            f"{item['max_patch']:<18.6f}"
            f"{item['top_1_percent']:<18.6f}"
            f"{item['max_anomaly_map']:<18.6f}"
        )

    print("\nHighest normal maximum-patch score:")
    print(f"{results[0]['filename']}: {results[0]['max_patch']:.6f}")

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
