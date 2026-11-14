from pathlib import Path
import numpy as np

from .tf_feature_extractor import (
    build_feature_extractor,
    extract_features,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "MVTecAD"

CATEGORIES = [
    "bottle",
    "grid",
    "pill",
    "screw",
    "transistor",
]

SAMPLES_PER_CATEGORY = 5


def main():

    print("Building feature extractor...")
    extractor = build_feature_extractor()

    shallow_values = []
    deep_values = []

    for category in CATEGORIES:

        image_dir = (
            DATASET_DIR
            / category
            / "train"
            / "good"
        )

        image_paths = sorted(image_dir.glob("*.png"))

        image_paths = image_paths[:SAMPLES_PER_CATEGORY]

        print(f"\nCategory: {category}")

        for image_path in image_paths:

            features = extract_features(
                extractor,
                image_path,
            )[0]

            shallow = features[:, :, :512]
            deep = features[:, :, 512:]

            shallow_values.append(shallow.reshape(-1, 512))
            deep_values.append(deep.reshape(-1, 1024))

    shallow_values = np.concatenate(shallow_values, axis=0)
    deep_values = np.concatenate(deep_values, axis=0)

    print("\n" + "=" * 60)
    print("FEATURE SCALE DIAGNOSTIC")
    print("=" * 60)

    for name, values in [
        ("SHALLOW FEATURES", shallow_values),
        ("DEEP FEATURES", deep_values),
    ]:

        print(f"\n{name}")
        print("Shape:", values.shape)
        print("Mean:", float(np.mean(values)))
        print("Standard deviation:", float(np.std(values)))
        print("Minimum:", float(np.min(values)))
        print("Maximum:", float(np.max(values)))

        patch_norms = np.linalg.norm(values, axis=1)

        print("Mean patch L2 norm:", float(np.mean(patch_norms)))
        print("Median patch L2 norm:", float(np.median(patch_norms)))

    print("\nDiagnostic completed.")


if __name__ == "__main__":
    main()