from pathlib import Path

from ml_engine.inference import predict_image


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_IMAGE = (
    PROJECT_ROOT
    / "datasets"
    / "MVTecAD"
    / "bottle"
    / "test"
    / "broken_large"
    / "000.png"
)

print("Testing VisionGuard inference service...")

result = predict_image(
    image_path=TEST_IMAGE,
    category="bottle",
)

print("\nInference results")
print("-" * 40)
print("Anomaly score:", result["anomaly_score"])
print("Predicted label:", result["predicted_label"])
print("Anomaly map:", result["anomaly_map"].shape)
print("Predicted mask:", result["predicted_mask"].shape)
print("Defect pixels:", result["predicted_mask"].sum())

print("\nInference service test completed.")