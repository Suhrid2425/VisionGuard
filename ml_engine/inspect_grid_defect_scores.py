
from pathlib import Path
import numpy as np

from ml_engine.tf_inference import analyze_image, load_thresholds

TEST_DIR = Path("datasets/MVTecAD/grid/test")

thresholds = load_thresholds("grid")
image_threshold = thresholds["image_threshold"]

print("\nGRID DEFECT SCORE ANALYSIS")
print("=" * 60)
print(f"Image threshold: {image_threshold:.4f}")

for defect_dir in sorted(TEST_DIR.iterdir()):
    if not defect_dir.is_dir() or defect_dir.name == "good":
        continue

    scores = []

    for image_path in sorted(defect_dir.glob("*.png")):
        result = analyze_image(str(image_path), category="grid")
        scores.append(float(result["anomaly_score"]))

    if not scores:
        continue

    scores = np.array(scores)

    print(f"\nDefect: {defect_dir.name}")
    print(f"Images: {len(scores)}")
    print(f"Minimum: {scores.min():.4f}")
    print(f"Mean: {scores.mean():.4f}")
    print(f"Maximum: {scores.max():.4f}")
    print(f"Detected at existing threshold: "
          f"{np.sum(scores > image_threshold)}/{len(scores)}")
