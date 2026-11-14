
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_engine.tf_inference import analyze_image, load_thresholds


VALIDATION_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "MVTecAD"
    / "bottle"
    / "validation"
    / "good"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml_engine"
    / "test_outputs"
    / "normal_outliers"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    "073.png",
    "064.png",
    "037.png",
]


def visualize_image(filename, pixel_threshold):

    image_path = VALIDATION_DIR / filename

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    original = Image.open(image_path).convert("RGB")

    original = original.resize(
        (256, 256),
        Image.Resampling.BILINEAR
    )

    original = np.asarray(original)

    result = analyze_image(image_path)

    distances = np.asarray(
        result["patch_distances"],
        dtype=np.float32
    ).reshape(-1)

    anomaly_map = np.asarray(
        result["anomaly_map"],
        dtype=np.float32
    )

    predicted_mask = (
        anomaly_map > pixel_threshold
    ).astype(np.uint8)

    # Locate maximum-distance patch
    max_index = int(np.argmax(distances))

    patch_row = max_index // 32
    patch_col = max_index % 32

    patch_x = patch_col * 8
    patch_y = patch_row * 8

    max_distance = float(distances[max_index])

    print("\n" + "=" * 60)
    print(f"IMAGE: {filename}")
    print("=" * 60)

    print(f"Maximum patch distance: {max_distance:.6f}")
    print(f"Maximum patch index: {max_index}")
    print(f"Patch row: {patch_row}")
    print(f"Patch column: {patch_col}")
    print(f"Approximate pixel location: ({patch_x}, {patch_y})")
    print(
        f"Pixels above threshold: "
        f"{np.mean(predicted_mask) * 100:.2f}%"
    )

    # --------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 11)
    )

    fig.suptitle(
        f"Normal Validation Outlier: {filename}\n"
        f"Maximum Patch Distance: {max_distance:.4f}",
        fontsize=14
    )

    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original Image")

    heatmap = axes[0, 1].imshow(
        anomaly_map,
        cmap="inferno"
    )

    axes[0, 1].set_title("Anomaly Heatmap")

    fig.colorbar(
        heatmap,
        ax=axes[0, 1],
        fraction=0.046
    )

    axes[1, 0].imshow(
        predicted_mask,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[1, 0].set_title(
        "Predicted Pixel Anomaly Mask"
    )

    # Display the 32x32 patch distance grid
    patch_grid = distances.reshape(32, 32)

    patch_heatmap = axes[1, 1].imshow(
        patch_grid,
        cmap="inferno"
    )

    axes[1, 1].set_title(
        "Patch Distance Grid (32 x 32)"
    )

    # Highlight the highest-distance patch
    rectangle = Rectangle(
        (patch_col - 0.5, patch_row - 0.5),
        1,
        1,
        fill=False,
        edgecolor="cyan",
        linewidth=2
    )

    axes[1, 1].add_patch(rectangle)

    fig.colorbar(
        patch_heatmap,
        ax=axes[1, 1],
        fraction=0.046
    )

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{Path(filename).stem}_normal_analysis.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


if __name__ == "__main__":

    thresholds = load_thresholds()

    pixel_threshold = float(
        thresholds["pixel_threshold"]
    )

    for filename in SAMPLES:
        visualize_image(
            filename,
            pixel_threshold
        )

    print("\nNormal outlier visualization completed.")
