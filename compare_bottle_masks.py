
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from ml_engine.tf_inference import analyze_image, load_thresholds

# Input directories
image_dir = Path("datasets/MVTecAD/bottle/test/contamination")
mask_dir = Path("datasets/MVTecAD/bottle/ground_truth/contamination")

# Output directory
output_dir = Path("ml_engine/evaluation/bottle_mask_comparison")
output_dir.mkdir(parents=True, exist_ok=True)

# Previously calibrated pixel threshold
thresholds = load_thresholds("bottle")
pixel_threshold = thresholds["pixel_threshold"]

missed_images = [
    "003.png",
    "004.png",
    "012.png",
    "019.png",
    "020.png",
]

print("=" * 65)
print("GROUND TRUTH VS PATCHCORE PREDICTION")
print("=" * 65)
print("Pixel threshold:", pixel_threshold)

for filename in missed_images:

    image_path = image_dir / filename
    mask_path = mask_dir / f"{Path(filename).stem}_mask.png"

    # Run model inference
    result = analyze_image(image_path, "bottle")

    # Load original image
    image = np.asarray(
        Image.open(image_path).convert("RGB").resize((256, 256))
    )

    # Load ground-truth mask
    ground_truth = Image.open(mask_path).convert("L")
    ground_truth = ground_truth.resize((256, 256), Image.Resampling.NEAREST)
    ground_truth = np.asarray(ground_truth) > 0

    # Get anomaly map
    anomaly_map = np.asarray(result["anomaly_map"]).squeeze()

    # Resize anomaly map if necessary
    if anomaly_map.shape != (256, 256):
        anomaly_map = np.asarray(
            Image.fromarray(anomaly_map.astype(np.float32)).resize(
                (256, 256), Image.Resampling.BILINEAR
            )
        )

    # Convert anomaly map into binary prediction
    prediction = anomaly_map > pixel_threshold

    # Calculate pixel metrics
    tp = np.logical_and(prediction, ground_truth).sum()
    fp = np.logical_and(prediction, ~ground_truth).sum()
    fn = np.logical_and(~prediction, ground_truth).sum()

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0

    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original image")

    axes[1].imshow(ground_truth, cmap="gray")
    axes[1].set_title("Ground-truth mask")

    axes[2].imshow(anomaly_map, cmap="jet")
    axes[2].set_title("Predicted anomaly heatmap")

    axes[3].imshow(image)
    axes[3].imshow(prediction, cmap="Reds", alpha=0.5)
    axes[3].set_title("Predicted binary mask")

    fig.suptitle(
        f"{filename} | Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | IoU: {iou:.3f}"
    )

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()

    save_path = output_dir / f"{Path(filename).stem}_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(
        f"{filename} | Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | IoU: {iou:.4f}"
    )

print("\nAll comparison images generated.")
print("Saved in:", output_dir)
