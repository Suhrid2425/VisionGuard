
import json
from pathlib import Path

manifest_path = Path(
    "ml_engine/models/mvtec_ad/grid/calibration_manifest.json"
)

with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

print("=" * 60)
print("GRID VALIDATION DATA CHECK")
print("=" * 60)

print("Manifest keys:", list(manifest.keys()))
print("Category:", manifest.get("category"))
print("Test set used:", manifest.get("test_set_used"))

print("\nCalibration image count:")
print(manifest.get("calibration_image_count"))

print("\nCalibration images:")
for path in manifest.get("calibration_images", []):
    print(path)

print("\nAnomaly/defect validation data:")
for key, value in manifest.items():
    if any(word in key.lower() for word in [
        "defect", "anomal", "validation", "test", "label"
    ]):
        print(f"{key}: {value}")
