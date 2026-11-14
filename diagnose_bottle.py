
from pathlib import Path

from ml_engine.tf_inference import analyze_image, load_thresholds


thresholds = load_thresholds("bottle")
image_threshold = thresholds["image_threshold"]

test_dir = Path("datasets/MVTecAD/bottle/test/contamination")

print("=" * 60)
print("BOTTLE CONTAMINATION DIAGNOSTIC")
print("=" * 60)

print("Image threshold:", image_threshold)
print()

for image_path in sorted(test_dir.glob("*.png")):

    result = analyze_image(image_path, "bottle")
    score = result["anomaly_score"]

    if score > image_threshold:
        prediction = "ANOMALY"
    else:
        prediction = "MISSED"

    print(
        image_path.name,
        "| Score:", round(score, 4),
        "| Prediction:", prediction
    )
