
from pathlib import Path
import json

import numpy as np


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MEMORY_BANK_PATH = (
    PROJECT_ROOT
    / "ml_engine"
    / "models"
    / "bottle_memory_bank.npz"
)

EXPECTED_MEMORY_BANK_SIZE = 1_000
EXPECTED_FEATURE_DIMENSIONS = 1_536


# ---------------------------------------------------------
# MEMORY BANK INTEGRITY TESTS
# ---------------------------------------------------------

def test_memory_bank_file_exists():
    assert MEMORY_BANK_PATH.exists(), (
        f"Memory bank not found: {MEMORY_BANK_PATH}"
    )


def load_memory_bank():
    assert MEMORY_BANK_PATH.exists(), (
        f"Memory bank not found: {MEMORY_BANK_PATH}"
    )

    with np.load(
        MEMORY_BANK_PATH,
        allow_pickle=False,
    ) as data:

        assert "memory_bank" in data.files, (
            "Missing memory_bank array."
        )

        assert "metadata" in data.files, (
            "Missing metadata."
        )

        memory_bank = data["memory_bank"]
        metadata = data["metadata"]

    metadata = json.loads(str(metadata.item()))

    return memory_bank, metadata


def test_memory_bank_shape():
    memory_bank, _ = load_memory_bank()

    assert memory_bank.shape == (
        EXPECTED_MEMORY_BANK_SIZE,
        EXPECTED_FEATURE_DIMENSIONS,
    ), (
        f"Unexpected memory bank shape: {memory_bank.shape}"
    )


def test_memory_bank_dtype():
    memory_bank, _ = load_memory_bank()

    assert memory_bank.dtype == np.float32, (
        f"Expected float32, got {memory_bank.dtype}"
    )


def test_memory_bank_finite_values():
    memory_bank, _ = load_memory_bank()

    assert np.isfinite(memory_bank).all(), (
        "Memory bank contains NaN or infinity."
    )


def test_memory_bank_has_no_duplicate_vectors():
    memory_bank, _ = load_memory_bank()

    unique_vectors = np.unique(
        memory_bank,
        axis=0,
    )

    assert len(unique_vectors) == len(memory_bank), (
        f"Duplicate vectors detected: "
        f"{len(memory_bank) - len(unique_vectors)}"
    )


def test_memory_bank_metadata():
    memory_bank, metadata = load_memory_bank()

    assert metadata["category"] == "bottle"

    assert metadata["backbone"] == "ResNet50"

    assert metadata["feature_dimensions"] == (
        memory_bank.shape[1]
    )

    assert metadata["memory_bank_size"] == (
        memory_bank.shape[0]
    )

    assert metadata["image_size"] == 256

    assert metadata["random_seed"] == 42


def test_memory_bank_value_statistics():
    memory_bank, _ = load_memory_bank()

    assert np.isfinite(memory_bank.min())
    assert np.isfinite(memory_bank.max())
    assert np.isfinite(memory_bank.mean())

    print("\nMemory Bank Statistics")
    print("----------------------")
    print("Shape:", memory_bank.shape)
    print("Dtype:", memory_bank.dtype)
    print("Minimum:", float(memory_bank.min()))
    print("Maximum:", float(memory_bank.max()))
    print("Mean:", float(memory_bank.mean()))
    print("Standard deviation:", float(memory_bank.std()))


if __name__ == "__main__":

    tests = [
        test_memory_bank_file_exists,
        test_memory_bank_shape,
        test_memory_bank_dtype,
        test_memory_bank_finite_values,
        test_memory_bank_has_no_duplicate_vectors,
        test_memory_bank_metadata,
        test_memory_bank_value_statistics,
    ]

    print("=" * 60)
    print("VISIONGUARD - MEMORY BANK INTEGRITY TEST")
    print("=" * 60)

    for test in tests:
        print(f"\nRunning: {test.__name__}")

        test()

        print("PASSED")

    print("\nAll memory bank integrity tests passed.")
