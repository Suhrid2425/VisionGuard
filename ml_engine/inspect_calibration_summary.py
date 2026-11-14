import json
from pathlib import Path
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / "models" / "mvtec_ad"

CATEGORIES = ["grid", "pill", "screw", "transistor"]

for category in CATEGORIES:
    print("\n" + "=" * 65)
    print(category.upper())
    print("=" * 65)

    threshold_path = MODEL_DIR / category / "thresholds.json"
    manifest_path = MODEL_DIR / category / "calibration_manifest.json"

    with open(threshold_path, encoding="utf-8") as f:
        thresholds = json.load(f)

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    print("Calibration images:", manifest["calibration_image_count"])
    print("Test set used:", manifest["test_set_used"])

    print("\nThresholds:")
    print("Image:", thresholds.get("image_threshold"))
    print("Pixel:", thresholds.get("pixel_threshold"))

    print("\nCalibration metadata:")

    for key, value in thresholds.items():
        if isinstance(value, (int, float, str, bool)):
            print(f"{key}: {value}")

        elif isinstance(value, list) and value:
            if all(isinstance(x, (int, float)) for x in value):
                arr = np.asarray(value, dtype=float)
                print(
                    f"{key}: count={len(arr)}, "
                    f"min={arr.min():.4f}, "
                    f"max={arr.max():.4f}, "
                    f"mean={arr.mean():.4f}"
                )