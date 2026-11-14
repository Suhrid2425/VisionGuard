
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "MVTecAD"
    / "bottle"
)

GROUND_TRUTH_DIR = DATASET_DIR / "ground_truth"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml_engine"
    / "test_outputs"
    / "pixel_localization"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# IMPORT MODEL
# --------------------------------------------------

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_engine.tf_inference import predict_image


# --------------------------------------------------
# SELECT REPRESENTATIVE IMAGES
# --------------------------------------------------

SAMPLES = [
    ("broken_large", "000.png"),
    ("broken_small", "000.png"),
    ("contamination", "000.png"),
    ("contamination", "003.png"),
    ("contamination", "019.png"),
]


# --------------------------------------------------
# IMAGE UTILITIES
# --------------------------------------------------

def load_original_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((256, 256), Image.Resampling.BILINEAR)

    return np.asarray(image)


def load_ground_truth_mask(category, filename):
    mask_filename = filename.replace(".png", "_mask.png")

    mask_path = (
        GROUND_TRUTH_DIR
        / category
        / mask_filename
    )

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Ground-truth mask not found: {mask_path}"
        )

    mask = Image.open(mask_path).convert("L")

    mask = mask.resize(
        (256, 256),
        Image.Resampling.NEAREST
    )

    mask = np.asarray(mask)

    return (mask > 0).astype(np.uint8)


def create_overlay(image, mask, color):
    overlay = image.astype(np.float32).copy()

    mask_pixels = mask == 1

    color_array = np.array(color, dtype=np.float32)

    overlay[mask_pixels] = (
        0.55 * overlay[mask_pixels]
        + 0.45 * color_array
    )

    return np.clip(overlay, 0, 255).astype(np.uint8)


# --------------------------------------------------
# VISUALIZATION
# --------------------------------------------------

def visualize_sample(category, filename):

    print("\n" + "=" * 60)
    print(f"Analyzing: {category}/{filename}")
    print("=" * 60)

    image_path = (
        DATASET_DIR
        / "test"
        / category
        / filename
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Test image not found: {image_path}"
        )

    # Original image
    original = load_original_image(image_path)

    # Ground truth
    ground_truth = load_ground_truth_mask(
        category,
        filename
    )

    # Model prediction
    result = predict_image(image_path)

    predicted_mask = result["predicted_mask"]
    anomaly_map = result["anomaly_map"]

    score = float(result["anomaly_score"])
    predicted_label = result["predicted_label"]

    # Validate shapes
    if predicted_mask.shape != (256, 256):
        raise ValueError(
            f"Unexpected predicted mask shape: {predicted_mask.shape}"
        )

    if anomaly_map.shape != (256, 256):
        raise ValueError(
            f"Unexpected anomaly map shape: {anomaly_map.shape}"
        )

    predicted_mask = predicted_mask.astype(np.uint8)

    # Overlays
    ground_truth_overlay = create_overlay(
        original,
        ground_truth,
        color=(0, 255, 0)
    )

    prediction_overlay = create_overlay(
        original,
        predicted_mask,
        color=(255, 0, 0)
    )

    # Metrics for this individual image
    intersection = np.logical_and(
        ground_truth == 1,
        predicted_mask == 1
    ).sum()

    union = np.logical_or(
        ground_truth == 1,
        predicted_mask == 1
    ).sum()

    gt_pixels = int(ground_truth.sum())
    predicted_pixels = int(predicted_mask.sum())

    iou = intersection / union if union else 1.0

    gt_percentage = gt_pixels / ground_truth.size * 100
    predicted_percentage = predicted_pixels / predicted_mask.size * 100

    # Display six panels
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, 10)
    )

    fig.suptitle(
        f"{category}/{filename} | "
        f"Score: {score:.4f} | "
        f"Predicted defective: {predicted_label}",
        fontsize=14
    )

    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original Image")

    axes[0, 1].imshow(
        ground_truth,
        cmap="gray",
        vmin=0,
        vmax=1
    )
    axes[0, 1].set_title("Ground Truth Mask")

    axes[0, 2].imshow(
        predicted_mask,
        cmap="gray",
        vmin=0,
        vmax=1
    )
    axes[0, 2].set_title("Predicted Mask")

    heatmap = axes[1, 0].imshow(
        anomaly_map,
        cmap="inferno"
    )
    axes[1, 0].set_title("Anomaly Heatmap")
    fig.colorbar(heatmap, ax=axes[1, 0], fraction=0.046)

    axes[1, 1].imshow(ground_truth_overlay)
    axes[1, 1].set_title("Ground Truth Overlay (Green)")

    axes[1, 2].imshow(prediction_overlay)
    axes[1, 2].set_title("Prediction Overlay (Red)")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()

    output_filename = (
        f"{category}_{filename.replace('.png', '')}_comparison.png"
    )

    output_path = OUTPUT_DIR / output_filename

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    # Print individual image statistics
    print(f"Anomaly score: {score:.6f}")
    print(f"Predicted defective: {predicted_label}")
    print(f"Ground-truth defect pixels: {gt_percentage:.2f}%")
    print(f"Predicted defect pixels: {predicted_percentage:.2f}%")
    print(f"Image IoU: {iou:.4f}")
    print(f"Saved visualization: {output_path}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    for category, filename in SAMPLES:
        visualize_sample(category, filename)

    print("\n" + "=" * 60)
    print("VISUAL ERROR ANALYSIS COMPLETED")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)
