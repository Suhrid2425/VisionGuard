import json
import numpy as np

from ml_engine.tf_inference import analyze_image, load_thresholds

MANIFEST_PATH = "ml_engine/models/mvtec_ad/grid/calibration_manifest.json"

with open(MANIFEST_PATH, "r") as f:
    manifest = json.load(f)

image_paths = manifest["calibration_images"]

thresholds = load_thresholds("grid")
image_threshold = thresholds["image_threshold"]

scores = []

print("\nGRID NORMAL CALIBRATION SCORE ANALYSIS")
print("-" * 55)

for image_path in image_paths:
    result = analyze_image(image_path, category="grid")
    score = float(result["anomaly_score"])
    scores.append(score)

    print(f"{image_path} | Score: {score:.4f}")

scores = np.array(scores)

print("\nSUMMARY")
print("-" * 55)
print(f"Number of images: {len(scores)}")
print(f"Minimum score:   {scores.min():.4f}")
print(f"Mean score:      {scores.mean():.4f}")
print(f"Median score:    {np.median(scores):.4f}")
print(f"Maximum score:   {scores.max():.4f}")
print(f"Threshold:       {image_threshold:.4f}")
print(f"Images above threshold: {np.sum(scores > image_threshold)}")