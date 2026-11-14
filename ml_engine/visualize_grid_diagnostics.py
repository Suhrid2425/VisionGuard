
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from ml_engine.tf_inference import analyze_image, load_thresholds

BASE_DIR = Path("datasets/MVTecAD/grid/test")
OUTPUT_DIR = Path("ml_engine/evaluation/grid_diagnostics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

thresholds = load_thresholds("grid")
pixel_threshold = thresholds["pixel_threshold"]

samples = [
    ("bent", "000.png"),
    ("broken", "000.png"),
    ("glue", "000.png"),
    ("metal_contamination", "000.png"),
    ("thread", "000.png"),
]

for defect_type, filename in samples:
    image_path = BASE_DIR / defect_type / filename
    mask_path = (
        Path("datasets/MVTecAD/grid/ground_truth")
        / defect_type / filename.replace(".png", "_mask.png")
    )

    if not image_path.exists():
        print(f"Missing image: {image_path}")
        continue

    result = analyze_image(str(image_path), category="grid")

    original = Image.open(image_path).convert("RGB")
    anomaly_map = np.asarray(result["anomaly_map"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original Image")

    if mask_path.exists():
        mask = Image.open(mask_path).convert("L")
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Ground Truth Mask")
    else:
        axes[1].axis("off")
        axes[1].set_title("Ground Truth Not Available")

    axes[2].imshow(original)
    axes[2].imshow(anomaly_map, cmap="jet", alpha=0.5)
    axes[2].set_title(
        f"Anomaly Map | Score: {result['anomaly_score']:.3f}"
    )

    for ax in axes:
        ax.axis("off")

    plt.suptitle(f"Grid - {defect_type}")
    plt.tight_layout()

    output_path = OUTPUT_DIR / f"{defect_type}_diagnostic.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")

print("\nGrid visual diagnostics completed.")
