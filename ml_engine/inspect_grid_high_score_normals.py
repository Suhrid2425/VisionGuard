
from pathlib import Path
import json
import numpy as np

from ml_engine.tf_feature_extractor import (
    build_feature_extractor,
    extract_features,
)

from ml_engine.tf_inference import (
    get_memory_bank,
    calculate_patch_distances,
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

def main():

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("test_set_used") is not False:
        raise ValueError("Calibration manifest does not confirm test_set_used=False.")

    image_paths = manifest["calibration_images"]

    extractor = build_feature_extractor()
    memory_bank = get_memory_bank("grid")

    results = []

    for index, image_path in enumerate(image_paths, start=1):

        features = extract_features(extractor, Path(image_path))

        patch_features = features.reshape(
            -1,
            features.shape[-1],
        ).astype(np.float32)

        distances = calculate_patch_distances(
            patch_features,
            memory_bank,
        )

        score = float(np.percentile(distances, 99))
        maximum = float(np.max(distances))
        top_1 = float(
            np.mean(
                np.sort(distances)[-max(1, int(np.ceil(len(distances) * 0.01))):]
            )
        )

        results.append({
            "image": Path(image_path).name,
            "path": str(image_path),
            "percentile_99": score,
            "maximum": maximum,
            "top_1_percent_mean": top_1,
        })

        if index % 10 == 0:
            print(f"Processed {index}/{len(image_paths)}")

    results.sort(
        key=lambda item: item["percentile_99"],
        reverse=True,
    )

    print("\nTOP 10 NORMAL CALIBRATION IMAGES")
    print("-" * 75)

    for item in results[:10]:

        print(
            f"{item['image']} | "
            f"P99: {item['percentile_99']:.4f} | "
            f"Maximum: {item['maximum']:.4f} | "
            f"Top 1%: {item['top_1_percent_mean']:.4f}"
        )

    print("\nBOTTOM 5 NORMAL CALIBRATION IMAGES")
    print("-" * 75)

    for item in results[-5:]:

        print(
            f"{item['image']} | "
            f"P99: {item['percentile_99']:.4f} | "
            f"Maximum: {item['maximum']:.4f}"
        )

    print("\nDiagnostic completed. No files modified.")


if __name__ == "__main__":
    main()
