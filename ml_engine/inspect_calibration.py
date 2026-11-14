import json
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "models" / "mvtec_ad"

CATEGORIES = [
    "grid",
    "pill",
    "screw",
    "transistor",
]

for category in CATEGORIES:
    print("\n" + "=" * 60)
    print("CATEGORY:", category.upper())
    print("=" * 60)

    threshold_file = MODEL_DIR / category / "thresholds.json"
    manifest_file = MODEL_DIR / category / "calibration_manifest.json"

    for file_path in [threshold_file, manifest_file]:
        print("\nFILE:", file_path.name)

        if not file_path.exists():
            print("File not found!")
            continue

        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        print(json.dumps(data, indent=2))