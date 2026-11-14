
from pathlib import Path

import numpy as np
from PIL import Image

from scipy.ndimage import zoom

try:
    from .tf_feature_extractor import (
        build_feature_extractor,
        extract_features,
    )
    from .tf_inference import (
        get_memory_bank,
        calculate_patch_distances,
    )
except ImportError:
    from tf_feature_extractor import (
        build_feature_extractor,
        extract_features,
    )
    from tf_inference import (
        get_memory_bank,
        calculate_patch_distances,
    )


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GRID_TEST_DIR = (
    PROJECT_ROOT / "datasets" / "MVTecAD" / "grid" / "test"
)

GRID_MASK_DIR = (
    PROJECT_ROOT / "datasets" / "MVTecAD" / "grid" / "ground_truth"
)

DEFECT_TYPES = [
    "bent",
    "broken",
    "glue",
    "metal_contamination",
    "thread",
]

FEATURE_SIZE = 32


# ---------------------------------------------------------
# MASK TO FEATURE-MAP RESOLUTION
# ---------------------------------------------------------

def load_patch_mask(mask_path):
    """
    Convert the ground-truth mask into a 32x32 patch mask.

    A patch is considered defective when at least 1%
    of its corresponding image area is defective.
    """

    with Image.open(mask_path) as image:
        mask = image.convert("L")

        mask_array = np.asarray(
            mask,
            dtype=np.float32,
        ) / 255.0

    # Area-based resizing preserves small defect regions
    # better than nearest-neighbour resizing.
    from PIL import Image as PILImage

    resized = PILImage.fromarray(
        (mask_array * 255).astype(np.uint8)
    ).resize(
        (FEATURE_SIZE, FEATURE_SIZE),
        PILImage.Resampling.BOX,
    )

    patch_mask = (
        np.asarray(resized, dtype=np.float32) / 255.0
    ) >= 0.01

    return patch_mask.reshape(-1)


# ---------------------------------------------------------
# MAIN DIAGNOSTIC
# ---------------------------------------------------------

def main():

    print("=" * 65)
    print("VISIONGUARD - GRID FEATURE RESPONSE DIAGNOSTIC")
    print("=" * 65)

    print("\nBuilding feature extractor...")
    extractor = build_feature_extractor()

    print("\nLoading Grid memory bank...")
    memory_bank = get_memory_bank("grid")

    all_defect_distances = []
    all_normal_distances = []

    for defect_type in DEFECT_TYPES:

        image_dir = GRID_TEST_DIR / defect_type
        mask_dir = GRID_MASK_DIR / defect_type

        if not image_dir.exists() or not mask_dir.exists():
            print(f"\nSkipping missing folder: {defect_type}")
            continue

        defect_distances = []
        normal_distances = []

        image_paths = sorted(image_dir.glob("*.png"))

        print(f"\nDefect type: {defect_type}")
        print("-" * 65)

        for image_path in image_paths:

            mask_path = mask_dir / f"{image_path.stem}_mask.png"

            if not mask_path.exists():
                print(f"Missing mask: {mask_path.name}")
                continue

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

            patch_mask = load_patch_mask(mask_path)

            if len(patch_mask) != len(distances):
                raise ValueError(
                    f"Mask/feature size mismatch: {image_path.name}"
                )

            inside = distances[patch_mask]
            outside = distances[~patch_mask]

            if len(inside) == 0 or len(outside) == 0:
                print(f"Insufficient mask regions: {image_path.name}")
                continue

            defect_distances.extend(inside.tolist())
            normal_distances.extend(outside.tolist())

            all_defect_distances.extend(inside.tolist())
            all_normal_distances.extend(outside.tolist())

        if defect_distances and normal_distances:

            print("Images analysed:", len(image_paths))
            print(
                "Mean defect-region distance:",
                round(float(np.mean(defect_distances)), 4),
            )
            print(
                "Mean normal-region distance:",
                round(float(np.mean(normal_distances)), 4),
            )
            print(
                "Median defect-region distance:",
                round(float(np.median(defect_distances)), 4),
            )
            print(
                "Median normal-region distance:",
                round(float(np.median(normal_distances)), 4),
            )

    # -----------------------------------------------------
    # OVERALL RESULTS
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("OVERALL GRID PATCH RESPONSE")
    print("=" * 65)

    if all_defect_distances and all_normal_distances:

        defect_mean = float(np.mean(all_defect_distances))
        normal_mean = float(np.mean(all_normal_distances))

        print(
            "Total defective patches:",
            len(all_defect_distances),
        )

        print(
            "Total normal patches:",
            len(all_normal_distances),
        )

        print(
            "Mean defective-patch distance:",
            round(defect_mean, 4),
        )

        print(
            "Mean normal-patch distance:",
            round(normal_mean, 4),
        )

        print(
            "Defect / normal distance ratio:",
            round(defect_mean / max(normal_mean, 1e-8), 4),
        )

        print("\nDiagnostic completed.")

    else:
        print("No valid patch comparisons were collected.")


if __name__ == "__main__":
    main()
