import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "datasets" / "MVTecAD" / "bottle"

TEST_DIR = DATASET_DIR / "test"
GROUND_TRUTH_DIR = DATASET_DIR / "ground_truth"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml_engine"
    / "test_outputs"
    / "false_negative_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_engine.tf_inference import predict_image


# --------------------------------------------------
# FALSE-NEGATIVE SAMPLES
# --------------------------------------------------

SAMPLES = [
    "contamination/003.png",
    "contamination/019.png",
]

IMAGE_SIZE = (256, 256)


# --------------------------------------------------
# OVERLAY FUNCTION
# --------------------------------------------------

def create_overlay(image, mask, color, alpha=0.45):
    """
    Overlay a binary mask on an RGB image.

    color:
        RGB tuple, for example (255, 0, 0)
    """

    image_array = np.asarray(image).astype(np.float32)
    mask = mask.astype(bool)

    overlay = image_array.copy()

    color_array = np.array(color, dtype=np.float32)

    overlay[mask] = (
        (1 - alpha) * overlay[mask]
        + alpha * color_array
    )

    return np.clip(overlay, 0, 255).astype(np.uint8)


# --------------------------------------------------
# ANALYZE ONE IMAGE
# --------------------------------------------------

def analyze_sample(relative_path):

    image_path = TEST_DIR / relative_path

    filename = Path(relative_path).name
    category = Path(relative_path).parent.name

    ground_truth_path = (
        GROUND_TRUTH_DIR
        / category
        / f"{Path(filename).stem}_mask.png"
    )

    print("\n" + "=" * 65)
    print(f"IMAGE: {relative_path}")
    print("=" * 65)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Test image not found: {image_path}"
        )

    if not ground_truth_path.exists():
        raise FileNotFoundError(
            f"Ground-truth mask not found: {ground_truth_path}"
        )

    # Original image
    original = Image.open(image_path).convert("RGB")
    original = original.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    # Ground-truth mask
    ground_truth_image = Image.open(
        ground_truth_path
    ).convert("L")

    ground_truth_image = ground_truth_image.resize(
        IMAGE_SIZE,
        Image.Resampling.NEAREST
    )

    ground_truth = (
        np.asarray(ground_truth_image) > 0
    ).astype(np.uint8)

    # Model prediction
    result = predict_image(image_path)

    anomaly_score = float(result["anomaly_score"])
    predicted_label = bool(result["predicted_label"])

    anomaly_map = np.asarray(
        result["anomaly_map"],
        dtype=np.float32
    )

    predicted_mask = np.asarray(
        result["predicted_mask"],
        dtype=np.uint8
    )

    # Validate shapes
    if anomaly_map.shape != IMAGE_SIZE[::-1]:
        raise ValueError(
            f"Unexpected anomaly map shape: {anomaly_map.shape}"
        )

    if predicted_mask.shape != IMAGE_SIZE[::-1]:
        raise ValueError(
            f"Unexpected predicted mask shape: {predicted_mask.shape}"
        )

    if ground_truth.shape != predicted_mask.shape:
        raise ValueError(
            "Ground-truth and predicted mask shapes do not match."
        )

    # Pixel-level metrics for this image
    gt_flat = ground_truth.flatten().astype(bool)
    pred_flat = predicted_mask.flatten().astype(bool)

    tp = int(np.sum(gt_flat & pred_flat))
    fp = int(np.sum(~gt_flat & pred_flat))
    fn = int(np.sum(gt_flat & ~pred_flat))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    iou = (
        tp / (tp + fp + fn)
        if (tp + fp + fn)
        else 0.0
    )

    gt_percentage = float(np.mean(gt_flat) * 100)
    predicted_percentage = float(np.mean(pred_flat) * 100)

    print(f"Image-level score:       {anomaly_score:.6f}")
    print(f"Image-level prediction:  {'DEFECTIVE' if predicted_label else 'NORMAL'}")
    print(f"Ground-truth pixels:     {gt_percentage:.2f}%")
    print(f"Predicted pixels:        {predicted_percentage:.2f}%")
    print(f"Pixel precision:         {precision:.4f}")
    print(f"Pixel recall:            {recall:.4f}")
    print(f"Pixel IoU:               {iou:.4f}")

    # Create overlays
    original_array = np.asarray(original)

    gt_overlay = create_overlay(
        original_array,
        ground_truth,
        color=(0, 255, 0)
    )

    prediction_overlay = create_overlay(
        original_array,
        predicted_mask,
        color=(255, 0, 0)
    )

    # --------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 10)
    )

    fig.suptitle(
        f"{relative_path} | "
        f"Image score: {anomaly_score:.4f} | "
        f"Prediction: {'DEFECTIVE' if predicted_label else 'NORMAL'}",
        fontsize=14
    )

    axes[0, 0].imshow(original_array)
    axes[0, 0].set_title("Original Image")

    axes[0, 1].imshow(
        ground_truth,
        cmap="gray",
        vmin=0,
        vmax=1
    )
    axes[0, 1].set_title("Ground-Truth Defect Mask")

    axes[0, 2].imshow(
        predicted_mask,
        cmap="gray",
        vmin=0,
        vmax=1
    )
    axes[0, 2].set_title("Predicted Anomaly Mask")

    heatmap = axes[1, 0].imshow(
        anomaly_map,
        cmap="inferno"
    )
    axes[1, 0].set_title("Anomaly Heatmap")
    fig.colorbar(heatmap, ax=axes[1, 0], fraction=0.046)

    axes[1, 1].imshow(gt_overlay)
    axes[1, 1].set_title("Ground Truth Overlay (Green)")

    axes[1, 2].imshow(prediction_overlay)
    axes[1, 2].set_title("Prediction Overlay (Red)")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    output_path = OUTPUT_DIR / (
        f"{category}_{Path(filename).stem}_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Visualization saved: {output_path}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 65)
    print("FALSE-NEGATIVE VISUAL ANALYSIS")
    print("=" * 65)

    for sample in SAMPLES:
        analyze_sample(sample)

    print("\n" + "=" * 65)
    print("VISUAL ANALYSIS COMPLETED")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()