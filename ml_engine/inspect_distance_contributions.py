
from pathlib import Path
import json
import numpy as np
from scipy.spatial.distance import cdist

from .tf_feature_extractor import (
    build_feature_extractor,
    extract_features,
)
from .tf_inference import get_memory_bank

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "ml_engine" / "models" / "mvtec_ad"

CATEGORIES = [
    "bottle",
    "grid",
    "pill",
    "screw",
    "transistor",
]

SAMPLES_PER_CATEGORY = 3


def main():

    extractor = build_feature_extractor()

    for category in CATEGORIES:

        print("\n" + "=" * 60)
        print(category.upper())
        print("=" * 60)

        manifest_path = (
            MODEL_DIR / category / "calibration_manifest.json"
        )

        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        image_paths = [
            Path(p) for p in manifest["calibration_images"]
        ][:SAMPLES_PER_CATEGORY]

        memory_bank = get_memory_bank(category)

        shallow_bank = memory_bank[:, :512]
        deep_bank = memory_bank[:, 512:]

        shallow_ratios = []
        deep_ratios = []

        for image_path in image_paths:

            features = extract_features(
                extractor,
                image_path,
            )[0]

            patches = features.reshape(-1, 1536)

            # Find nearest memory-bank neighbour
            combined_distances = cdist(
                patches,
                memory_bank,
                metric="euclidean",
            )

            nearest_indices = np.argmin(
                combined_distances,
                axis=1,
            )

            # Compare both groups against the SAME neighbour
            shallow_query = patches[:, :512]
            deep_query = patches[:, 512:]

            shallow_diff = (
                shallow_query -
                shallow_bank[nearest_indices]
            )

            deep_diff = (
                deep_query -
                deep_bank[nearest_indices]
            )

            shallow_squared = np.sum(
                shallow_diff ** 2,
                axis=1,
            )

            deep_squared = np.sum(
                deep_diff ** 2,
                axis=1,
            )

            total_squared = (
                shallow_squared + deep_squared
            )

            valid = total_squared > 0

            shallow_ratio = (
                shallow_squared[valid] /
                total_squared[valid]
            )

            deep_ratio = (
                deep_squared[valid] /
                total_squared[valid]
            )

            shallow_ratios.extend(shallow_ratio)
            deep_ratios.extend(deep_ratio)

        print(
            "Shallow distance contribution:",
            f"{np.mean(shallow_ratios) * 100:.2f}%"
        )

        print(
            "Deep distance contribution:",
            f"{np.mean(deep_ratios) * 100:.2f}%"
        )

        print("Diagnostic completed.")


if __name__ == "__main__":
    main()
