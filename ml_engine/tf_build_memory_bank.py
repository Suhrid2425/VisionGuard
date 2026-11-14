
from pathlib import Path
import argparse
import json
import time

import numpy as np

from tf_feature_extractor import (
    build_feature_extractor,
    extract_features,
)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "MVTecAD"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml_engine"
    / "models"
    / "mvtec_ad"
)

CATEGORIES = [
    "bottle",
    "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile",
    "toothbrush", "transistor", "wood", "zipper",
]

CANDIDATE_POOL_SIZE = 10_000
MEMORY_BANK_SIZE = 1_000
PROJECTION_DIMENSION = 64

RANDOM_SEED = 42
TRAIN_RATIO = 0.80


# ---------------------------------------------------------
# DATA SPLIT
# ---------------------------------------------------------

def split_normal_images(category):
    train_dir = DATASET_DIR / category / "train" / "good"

    if not train_dir.exists():
        raise FileNotFoundError(train_dir)

    image_paths = sorted(train_dir.glob("*.png"))

    if len(image_paths) < 2:
        raise RuntimeError(
            f"Not enough training images: {category}"
        )

    rng = np.random.default_rng(RANDOM_SEED)

    indices = rng.permutation(len(image_paths))

    train_count = int(len(image_paths) * TRAIN_RATIO)

    train_indices = indices[:train_count]
    calibration_indices = indices[train_count:]

    train_images = [
        str(image_paths[i]) for i in train_indices
    ]

    calibration_images = [
        str(image_paths[i]) for i in calibration_indices
    ]

    return train_images, calibration_images


def save_calibration_manifest(category, calibration_images):
    category_dir = OUTPUT_DIR / category
    category_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "category": category,
        "calibration_images": calibration_images,
        "calibration_image_count": len(calibration_images),
        "random_seed": RANDOM_SEED,
        "train_ratio": TRAIN_RATIO,
        "calibration_ratio": 1.0 - TRAIN_RATIO,
        "test_set_used": False,
    }

    manifest_path = category_dir / "calibration_manifest.json"

    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=4)

    print("Calibration manifest:", manifest_path)


# ---------------------------------------------------------
# CANDIDATE PATCH SAMPLING
# ---------------------------------------------------------

def collect_candidate_patches(feature_extractor, image_paths):

    rng = np.random.default_rng(RANDOM_SEED)

    candidate_features = None
    candidate_priorities = None

    total_patches_seen = 0

    for index, image_path in enumerate(image_paths, start=1):

        features = extract_features(
            feature_extractor,
            image_path,
        )

        patches = features.reshape(
            -1,
            features.shape[-1],
        ).astype(np.float32)

        priorities = rng.random(
            len(patches),
            dtype=np.float32,
        )

        if candidate_features is None:
            candidate_features = patches
            candidate_priorities = priorities

        else:
            combined_features = np.concatenate(
                [candidate_features, patches],
                axis=0,
            )

            combined_priorities = np.concatenate(
                [candidate_priorities, priorities],
                axis=0,
            )

            keep_count = min(
                CANDIDATE_POOL_SIZE,
                len(combined_priorities),
            )

            selected = np.argpartition(
                combined_priorities,
                -keep_count,
            )[-keep_count:]

            candidate_features = combined_features[selected]
            candidate_priorities = combined_priorities[selected]

        total_patches_seen += len(patches)

        print(
            f"\rProcessed: {index}/{len(image_paths)}"
            f" | Patches: {total_patches_seen:,}"
            f" | Candidates: {len(candidate_features):,}",
            end="",
            flush=True,
        )

    print()

    return candidate_features, total_patches_seen


# ---------------------------------------------------------
# RANDOM PROJECTION
# ---------------------------------------------------------

def project_features(features):

    rng = np.random.default_rng(RANDOM_SEED)

    projection = rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(PROJECTION_DIMENSION),
        size=(
            features.shape[1],
            PROJECTION_DIMENSION,
        ),
    ).astype(np.float32)

    return (features @ projection).astype(np.float32)


# ---------------------------------------------------------
# CORESET SELECTION
# ---------------------------------------------------------

def select_coreset(features, target_size):

    projected = project_features(features)

    number_of_candidates = len(projected)

    if target_size > number_of_candidates:
        raise ValueError(
            f"Memory bank size {target_size} exceeds "
            f"candidate count {number_of_candidates}."
        )

    print(
        f"Selecting {target_size} patches "
        f"from {number_of_candidates} candidates..."
    )

    centroid = projected.mean(axis=0)

    distances = np.sum(
        (projected - centroid) ** 2,
        axis=1,
    )

    selected_indices = np.empty(
        target_size,
        dtype=np.int64,
    )

    selected_mask = np.zeros(
        number_of_candidates,
        dtype=bool,
    )

    for step in range(target_size):

        selected_index = int(np.argmax(distances))

        selected_indices[step] = selected_index
        selected_mask[selected_index] = True

        selected_point = projected[selected_index]

        new_distances = np.sum(
            (projected - selected_point) ** 2,
            axis=1,
        )

        distances = np.minimum(distances, new_distances)
        distances[selected_mask] = -1.0

        if (step + 1) % 100 == 0 or step == 0:
            print(
                f"\rSelected: {step + 1}/{target_size}",
                end="",
                flush=True,
            )

    print()

    return features[selected_indices]


# ---------------------------------------------------------
# SAVE MEMORY BANK
# ---------------------------------------------------------

def save_memory_bank(
    memory_bank,
    category,
    image_count,
    patch_count,
):

    category_dir = OUTPUT_DIR / category
    category_dir.mkdir(parents=True, exist_ok=True)

    output_file = category_dir / "memory_bank.npz"

    metadata = {
        "model": "TensorFlow PatchCore",
        "category": category,
        "backbone": "ResNet50",
        "feature_layers": [
            "conv3_block4_out",
            "conv4_block6_out",
        ],
        "image_size": 256,
        "feature_dimensions": int(memory_bank.shape[1]),
        "memory_bank_size": int(memory_bank.shape[0]),
        "training_images": int(image_count),
        "total_training_patches": int(patch_count),
        "candidate_pool_size": CANDIDATE_POOL_SIZE,
        "projection_dimension": PROJECTION_DIMENSION,
        "coreset_method": "Greedy farthest-first on random projection",
        "random_seed": RANDOM_SEED,
        "train_ratio": TRAIN_RATIO,
    }

    np.savez_compressed(
        output_file,
        memory_bank=memory_bank.astype(np.float32),
        metadata=json.dumps(metadata),
    )

    print("\nMemory bank saved:", output_file)
    print("Shape:", memory_bank.shape)


# ---------------------------------------------------------
# CATEGORY TRAINING
# ---------------------------------------------------------

def train_category(category):

    print("\n" + "=" * 65)
    print(f"VISIONGUARD - TRAINING: {category.upper()}")
    print("=" * 65)

    train_images, calibration_images = split_normal_images(category)

    print("Memory-bank training images:", len(train_images))
    print("Calibration images:", len(calibration_images))

    # Save exact calibration image list.
    save_calibration_manifest(category, calibration_images)

    print("\nBuilding feature extractor...")

    feature_extractor = build_feature_extractor()

    start_time = time.time()

    print("\nExtracting training patches...")

    candidate_features, total_patches = collect_candidate_patches(
        feature_extractor,
        train_images,
    )

    print("\nCandidate feature shape:", candidate_features.shape)

    memory_bank = select_coreset(
        candidate_features,
        MEMORY_BANK_SIZE,
    )

    save_memory_bank(
        memory_bank=memory_bank,
        category=category,
        image_count=len(train_images),
        patch_count=total_patches,
    )

    print("\nCategory completed:", category)
    print("Training images:", len(train_images))
    print("Calibration images:", len(calibration_images))
    print("Memory bank shape:", memory_bank.shape)
    print(f"Elapsed: {(time.time() - start_time) / 60:.2f} minutes")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="VisionGuard leakage-free PatchCore builder"
    )

    parser.add_argument(
        "--category",
        required=True,
        choices=CATEGORIES,
    )

    args = parser.parse_args()

    train_category(args.category)


if __name__ == "__main__":
    main()
