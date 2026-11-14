
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

from ml_engine.tf_feature_extractor import (
    build_feature_extractor,
    extract_features,
)

from ml_engine.tf_inference import (
    get_memory_bank,
    calculate_patch_distances,
    generate_anomaly_map,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_PATH = (
    PROJECT_ROOT
    / "ml_engine"
    / "models"
    / "mvtec_ad"
    / "grid"
    / "calibration_manifest.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml_engine"
    / "evaluation"
    / "grid_normal_diagnostics"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("test_set_used") is not False:
        raise ValueError("Calibration manifest does not confirm test_set_used=False.")

    image_paths = manifest["calibration_images"]

    extractor = build_feature_extractor()
    memory_bank = get_memory_bank("grid")

    results = []

    for image_path in image_paths:

        image_path = Path(image_path)

        features = extract_features(extractor, image_path)

        patch_features = features.reshape(
            -1,
            features.shape[-1],
        ).astype(np.float32)

        distances = calculate_patch_distances(
            patch_features,
            memory_bank,
        )

        p99 = float(np.percentile(distances, 99))

        results.append((p99, image_path, distances))

    results.sort(key=lambda item: item[0], reverse=True)

    print("\nGenerating visualizations for five highest P99 normal images...")

    for score, image_path, distances in results[:5]:

        with open(image_path, "rb") as f:
            from PIL import Image
            original = Image.open(f).convert("RGB")

        anomaly_map = generate_anomaly_map(distances)

        original = original.resize((256, 256))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original)
        axes[0].set_title("Normal Image")

        axes[1].imshow(anomaly_map, cmap="jet")
        axes[1].set_title("Anomaly Heatmap")

        axes[2].imshow(original)
        axes[2].imshow(
            anomaly_map,
            cmap="jet",
            alpha=0.45,
        )
        axes[2].set_title("Overlay")

        for ax in axes:
            ax.axis("off")

        fig.suptitle(
            f"Grid Normal: {image_path.name} | P99: {score:.4f}"
        )

        fig.tight_layout()

        output_path = OUTPUT_DIR / f"{image_path.stem}_diagnostic.png"

        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        print("Saved:", output_path)

    print("\nCompleted.")


if __name__ == "__main__":
    main()
