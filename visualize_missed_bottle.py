
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ml_engine.tf_inference import analyze_image, load_thresholds


image_dir = Path("datasets/MVTecAD/bottle/test/contamination")

output_dir = Path("ml_engine/evaluation/bottle_diagnostics")
output_dir.mkdir(parents=True, exist_ok=True)

missed_images = [
    "003.png",
    "004.png",
    "012.png",
    "019.png",
    "020.png",
]

thresholds = load_thresholds("bottle")
image_threshold = thresholds["image_threshold"]
pixel_threshold = thresholds["pixel_threshold"]

for filename in missed_images:

    image_path = image_dir / filename

    result = analyze_image(image_path, "bottle")

    image = np.asarray(
        Image.open(image_path).convert("RGB").resize((256, 256))
    )

    anomaly_map = result["anomaly_map"]
    anomaly_score = result["anomaly_score"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original image")

    axes[1].imshow(anomaly_map, cmap="jet")
    axes[1].set_title("Anomaly heatmap")

    axes[2].imshow(image)
    axes[2].imshow(anomaly_map, cmap="jet", alpha=0.45)
    axes[2].set_title("Overlay")

    fig.suptitle(
        f"{filename} | Score: {anomaly_score:.3f} | "
        f"Threshold: {image_threshold:.3f}"
    )

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()

    save_path = output_dir / f"{Path(filename).stem}_diagnostic.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", save_path)

print("\nAll five diagnostic visualizations generated.")
print("Pixel threshold:", pixel_threshold)
