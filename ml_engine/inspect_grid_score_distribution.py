import json
from pathlib import Path
import numpy as np

RESULT_PATH = Path(
    "ml_engine/evaluation/tensorflow_patchcore_all_categories.json"
)

with open(RESULT_PATH, "r", encoding="utf-8") as f:
    results = json.load(f)

grid = results["category_results"]["grid"]

print("=" * 65)
print("GRID IMAGE SCORE DISTRIBUTION")
print("=" * 65)

print("\nGrid result keys:")
print(list(grid.keys()))

print("\nImage-level metrics:")
print(grid.get("image_level_metrics"))

print("\nDefect-wise results:")
for key, value in grid.items():
    if key not in ["image_level_metrics", "pixel_level_metrics"]:
        print(f"\n{key}: {value}")