import json
from pathlib import Path

results_path = (
    Path(__file__).resolve().parent
    / "evaluation"
    / "tensorflow_patchcore_all_categories.json"
)

with open(results_path, "r", encoding="utf-8") as file:
    data = json.load(file)

print(
    f"{'CATEGORY':<15}"
    f"{'IMG THRESHOLD':>16}"
    f"{'IMG RECALL':>13}"
    f"{'PIX THRESHOLD':>16}"
    f"{'PIX RECALL':>13}"
)

print("-" * 73)

for category, result in data["category_results"].items():
    image_threshold = result["thresholds"]["image_threshold"]
    image_recall = result["image_level_metrics"]["recall"] * 100

    pixel_threshold = result["thresholds"]["pixel_threshold"]
    pixel_recall = result["pixel_level_metrics"]["recall"] * 100

    print(
        f"{category:<15}"
        f"{image_threshold:>16.4f}"
        f"{image_recall:>12.2f}%"
        f"{pixel_threshold:>16.4f}"
        f"{pixel_recall:>12.2f}%"
    )