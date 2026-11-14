
from pathlib import Path
import sys

import numpy as np


# ---------------------------------------------------------
# PROJECT IMPORT PATH
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from ml_engine.tf_inference import analyze_image


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

SAMPLE_IMAGE = (
    PROJECT_ROOT
    / "datasets"
    / "MVTecAD"
    / "bottle"
    / "test"
    / "good"
    / "000.png"
)

EXPECTED_PATCH_COUNT = 1024
EXPECTED_MAP_SHAPE = (256, 256)

_analysis_cache = None


# ---------------------------------------------------------
# HELPER
# ---------------------------------------------------------

def get_analysis():

    global _analysis_cache

    if _analysis_cache is None:

        assert SAMPLE_IMAGE.exists(), (
            f"Sample image not found: {SAMPLE_IMAGE}"
        )

        _analysis_cache = analyze_image(SAMPLE_IMAGE)

    return _analysis_cache


# ---------------------------------------------------------
# TEST 1: INFERENCE AND MEMORY BANK LOADING
# ---------------------------------------------------------

def test_inference_and_memory_bank_loading():

    result = get_analysis()

    assert "patch_distances" in result
    assert "anomaly_map" in result
    assert "anomaly_score" in result

    patch_distances = result["patch_distances"]

    assert patch_distances.shape == (
        EXPECTED_PATCH_COUNT,
    )

    assert np.isfinite(patch_distances).all()

    assert np.all(patch_distances >= 0)

    print("Inference completed successfully.")
    print("Patch distances:", patch_distances.shape)


# ---------------------------------------------------------
# TEST 2: PATCH DISTANCES
# ---------------------------------------------------------

def test_patch_distances():

    result = get_analysis()

    patch_distances = result["patch_distances"]

    assert patch_distances.shape == (1024,)

    assert patch_distances.dtype == np.float32

    assert np.isfinite(patch_distances).all()

    assert np.all(patch_distances >= 0)

    print("Patch distance validation passed.")


# ---------------------------------------------------------
# TEST 3: ANOMALY MAP
# ---------------------------------------------------------

def test_anomaly_map():

    result = get_analysis()

    anomaly_map = result["anomaly_map"]

    assert anomaly_map.shape == (256, 256)

    assert anomaly_map.dtype == np.float32

    assert np.isfinite(anomaly_map).all()

    assert np.all(anomaly_map >= 0)

    print("Anomaly map:", anomaly_map.shape)


# ---------------------------------------------------------
# TEST 4: IMAGE SCORE
# ---------------------------------------------------------

def test_image_score():

    result = get_analysis()

    anomaly_score = result["anomaly_score"]

    assert np.isscalar(anomaly_score)

    assert np.isfinite(anomaly_score)

    assert anomaly_score >= 0

    print("Anomaly score:", float(anomaly_score))


# ---------------------------------------------------------
# TEST 5: REPEATABILITY
# ---------------------------------------------------------

def test_inference_repeatability():

    first_result = get_analysis()

    second_result = analyze_image(SAMPLE_IMAGE)

    np.testing.assert_allclose(
        first_result["patch_distances"],
        second_result["patch_distances"],
        rtol=1e-5,
        atol=1e-5,
    )

    np.testing.assert_allclose(
        first_result["anomaly_map"],
        second_result["anomaly_map"],
        rtol=1e-5,
        atol=1e-5,
    )

    np.testing.assert_allclose(
        first_result["anomaly_score"],
        second_result["anomaly_score"],
        rtol=1e-5,
        atol=1e-5,
    )

    print("Repeated inference results are consistent.")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    tests = [
        test_inference_and_memory_bank_loading,
        test_patch_distances,
        test_anomaly_map,
        test_image_score,
        test_inference_repeatability,
    ]

    print("=" * 65)
    print("VISIONGUARD - INFERENCE INTEGRITY TEST")
    print("=" * 65)

    for test in tests:

        print(f"\nRunning: {test.__name__}")

        test()

        print("PASSED")

    print("\n" + "=" * 65)
    print("ALL INFERENCE INTEGRITY TESTS PASSED")
    print("=" * 65)
