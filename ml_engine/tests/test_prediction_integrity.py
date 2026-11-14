
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_engine.tf_inference import predict_image, load_thresholds


SAMPLE_IMAGE = (
    PROJECT_ROOT
    / "datasets"
    / "MVTecAD"
    / "bottle"
    / "test"
    / "good"
    / "000.png"
)


def run_tests():
    print("\nStarting prediction integrity tests...\n")

    # Test 1: Sample image exists
    assert SAMPLE_IMAGE.exists(), (
        f"Sample image not found: {SAMPLE_IMAGE}"
    )
    print("PASS: Sample image exists")

    # Test 2: Thresholds load correctly
    thresholds = load_thresholds()

    assert "image_threshold" in thresholds
    assert "pixel_threshold" in thresholds

    image_threshold = float(thresholds["image_threshold"])
    pixel_threshold = float(thresholds["pixel_threshold"])

    assert np.isfinite(image_threshold)
    assert np.isfinite(pixel_threshold)
    assert image_threshold > 0
    assert pixel_threshold > 0

    print("PASS: Thresholds loaded and validated")

    # Test 3: Run actual prediction
    result = predict_image(str(SAMPLE_IMAGE))

    required_keys = {
        "anomaly_score",
        "predicted_label",
        "anomaly_map",
        "predicted_mask",
    }

    assert required_keys.issubset(result.keys())

    print("PASS: Prediction returned all required fields")

    # Test 4: Validate anomaly score
    score = result["anomaly_score"]

    assert np.isscalar(score)
    assert np.isfinite(score)
    assert score >= 0

    print("PASS: Anomaly score is valid")

    # Test 5: Validate classification
    expected_label = bool(score > image_threshold)

    assert result["predicted_label"] == expected_label
    assert isinstance(result["predicted_label"], bool)

    print("PASS: Classification matches image threshold")

    # Test 6: Validate anomaly map
    anomaly_map = result["anomaly_map"]

    assert anomaly_map.shape == (256, 256)
    assert anomaly_map.dtype == np.float32
    assert np.isfinite(anomaly_map).all()
    assert (anomaly_map >= 0).all()

    print("PASS: Anomaly map shape, dtype and values are valid")

    # Test 7: Validate predicted mask
    mask = result["predicted_mask"]

    assert mask.shape == (256, 256)
    assert mask.dtype == np.uint8
    assert np.isin(mask, [0, 1]).all()

    print("PASS: Predicted mask shape, dtype and values are valid")

    # Test 8: Verify pixel threshold consistency
    expected_mask = (
        anomaly_map > pixel_threshold
    ).astype(np.uint8)

    assert np.array_equal(mask, expected_mask)

    print("PASS: Predicted mask matches pixel threshold")

    # Test 9: Invalid category must be rejected
    try:
        predict_image(
            str(SAMPLE_IMAGE),
            category="invalid_category"
        )
    except ValueError:
        print("PASS: Invalid category is rejected")
    else:
        raise AssertionError(
            "Invalid category was not rejected"
        )

    print("\nAll 9 prediction integrity tests passed.")

    print("\nPrediction summary:")
    print(f"Anomaly score: {score}")
    print(f"Image threshold: {image_threshold}")
    print(f"Pixel threshold: {pixel_threshold}")
    print(f"Predicted label: {result['predicted_label']}")
    print(f"Mask shape: {mask.shape}")
    print(
        f"Defective pixel percentage: "
        f"{mask.mean() * 100:.2f}%"
    )


if __name__ == "__main__":
    run_tests()
