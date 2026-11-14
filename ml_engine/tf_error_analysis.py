from pathlib import Path
import json
import csv

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    # Package import
    from ml_engine.tf_inference import (
        analyze_image,
        load_thresholds,
    )

except ModuleNotFoundError:
    # Direct script execution from ml_engine
    from tf_inference import (
        analyze_image,
        load_thresholds,
    )


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "datasets" / "MVTecAD" / "bottle"
TEST_DIR = DATASET_DIR / "test"
GROUND_TRUTH_DIR = DATASET_DIR / "ground_truth"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml_engine"
    / "evaluation"
    / "error_analysis"
)

IMAGE_SIZE = (256, 256)


# ---------------------------------------------------------
# IMAGE UTILITIES
# ---------------------------------------------------------

def load_original_image(image_path):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.BILINEAR,
        )
        return image


def load_ground_truth(image_path, defect_type):

    if defect_type == "good":
        return np.zeros(
            (IMAGE_SIZE[1], IMAGE_SIZE[0]),
            dtype=np.uint8,
        )

    mask_path = (
        GROUND_TRUTH_DIR
        / defect_type
        / f"{image_path.stem}_mask.png"
    )

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Ground-truth mask missing: {mask_path}"
        )

    with Image.open(mask_path) as image:
        image = image.convert("L")
        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.NEAREST,
        )
        mask = np.asarray(image)

    return (mask > 0).astype(np.uint8)


def create_mask_visualization(mask):

    mask_rgb = np.zeros(
        (mask.shape[0], mask.shape[1], 3),
        dtype=np.uint8,
    )

    mask_rgb[mask > 0] = [255, 40, 40]

    return Image.fromarray(mask_rgb)


def create_heatmap(anomaly_map):

    minimum = float(np.min(anomaly_map))
    maximum = float(np.max(anomaly_map))

    normalized = (
        (anomaly_map - minimum)
        / (maximum - minimum + 1e-8)
    )

    # Jet-style color mapping
    red = np.clip(
        1.5 - np.abs(4 * normalized - 3),
        0,
        1,
    )

    green = np.clip(
        1.5 - np.abs(4 * normalized - 2),
        0,
        1,
    )

    blue = np.clip(
        1.5 - np.abs(4 * normalized - 1),
        0,
        1,
    )

    heatmap = np.stack(
        [red, green, blue],
        axis=-1,
    )

    heatmap = (heatmap * 255).astype(np.uint8)

    return Image.fromarray(heatmap)


def create_overlay(original, heatmap):

    return Image.blend(
        original.convert("RGB"),
        heatmap.convert("RGB"),
        alpha=0.45,
    )


def create_comparison(
    image_path,
    defect_type,
    original,
    ground_truth,
    predicted_mask,
    anomaly_map,
    score,
    threshold,
    predicted_label,
    actual_label,
):

    ground_truth_image = create_mask_visualization(
        ground_truth
    )

    predicted_mask_image = create_mask_visualization(
        predicted_mask
    )

    heatmap = create_heatmap(anomaly_map)

    overlay = create_overlay(
        original,
        heatmap,
    )

    panels = [
        ("Original", original),
        ("Ground Truth", ground_truth_image),
        ("Predicted Mask", predicted_mask_image),
        ("Anomaly Heatmap", heatmap),
        ("Heatmap Overlay", overlay),
    ]

    panel_width = IMAGE_SIZE[0]
    panel_height = IMAGE_SIZE[1]
    header_height = 75

    canvas = Image.new(
        "RGB",
        (
            panel_width * len(panels),
            panel_height + header_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    for index, (label, panel) in enumerate(panels):

        x = index * panel_width

        draw.text(
            (x + 8, 8),
            label,
            fill="black",
            font=font,
        )

        canvas.paste(
            panel.convert("RGB"),
            (x, header_height),
        )

    status = (
        "CORRECT"
        if predicted_label == actual_label
        else "MISCLASSIFIED"
    )

    summary = (
        f"{defect_type}/{image_path.name} | "
        f"Score: {score:.4f} | "
        f"Threshold: {threshold:.4f} | "
        f"Actual: {actual_label} | "
        f"Predicted: {predicted_label} | "
        f"{status}"
    )

    draw.text(
        (8, 42),
        summary,
        fill="black",
        font=font,
    )

    return canvas


# ---------------------------------------------------------
# MAIN ERROR ANALYSIS
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("VISIONGUARD - VISUAL ERROR ANALYSIS")
    print("=" * 70)

    thresholds = load_thresholds()

    image_threshold = float(
        thresholds["image_threshold"]
    )

    pixel_threshold = float(
        thresholds["pixel_threshold"]
    )

    print("\nFixed image threshold:", image_threshold)
    print("Fixed pixel threshold:", pixel_threshold)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Collect images
    test_images = []

    for defect_dir in sorted(TEST_DIR.iterdir()):

        if not defect_dir.is_dir():
            continue

        defect_type = defect_dir.name

        for image_path in sorted(defect_dir.iterdir()):

            if image_path.suffix.lower() in [
                ".png",
                ".jpg",
                ".jpeg",
            ]:
                test_images.append(
                    (image_path, defect_type)
                )

    print("\nTotal test images:", len(test_images))

    records = []

    # -----------------------------------------------------
    # Analyze every image
    # -----------------------------------------------------

    for index, (image_path, defect_type) in enumerate(
        test_images,
        start=1,
    ):

        print(
            f"\rAnalyzing {index}/{len(test_images)}: "
            f"{defect_type}/{image_path.name}",
            end="",
            flush=True,
        )

        result = analyze_image(image_path)

        score = float(result["anomaly_score"])
        anomaly_map = result["anomaly_map"]

        predicted_label = int(
            score > image_threshold
        )

        predicted_mask = (
            anomaly_map > pixel_threshold
        ).astype(np.uint8)

        ground_truth = load_ground_truth(
            image_path,
            defect_type,
        )

        actual_label = int(
            defect_type != "good"
        )

        record = {
            "image_path": image_path,
            "defect_type": defect_type,
            "score": score,
            "actual_label": actual_label,
            "predicted_label": predicted_label,
            "correct": predicted_label == actual_label,
            "anomaly_map": anomaly_map,
            "predicted_mask": predicted_mask,
            "ground_truth": ground_truth,
        }

        records.append(record)

    print("\n\nInference completed.")

    # -----------------------------------------------------
    # Identify errors and representative examples
    # -----------------------------------------------------

    false_negatives = [
        record for record in records
        if record["actual_label"] == 1
        and record["predicted_label"] == 0
    ]

    false_positives = [
        record for record in records
        if record["actual_label"] == 0
        and record["predicted_label"] == 1
    ]

    selected = {}

    # Always include all classification errors
    for record in false_negatives + false_positives:

        key = (
            f"{record['defect_type']}_"
            f"{record['image_path'].stem}"
        )

        selected[key] = record

    # Include one correctly detected example per defect category
    for category in sorted(
        set(record["defect_type"] for record in records)
    ):

        candidates = [
            record for record in records
            if record["defect_type"] == category
            and record["correct"]
        ]

        if candidates:

            record = candidates[0]

            key = (
                f"{record['defect_type']}_"
                f"{record['image_path'].stem}"
            )

            selected[key] = record

    # -----------------------------------------------------
    # Save comparison images
    # -----------------------------------------------------

    for key, record in selected.items():

        original = load_original_image(
            record["image_path"]
        )

        comparison = create_comparison(
            image_path=record["image_path"],
            defect_type=record["defect_type"],
            original=original,
            ground_truth=record["ground_truth"],
            predicted_mask=record["predicted_mask"],
            anomaly_map=record["anomaly_map"],
            score=record["score"],
            threshold=image_threshold,
            predicted_label=record["predicted_label"],
            actual_label=record["actual_label"],
        )

        output_path = OUTPUT_DIR / f"{key}.png"

        comparison.save(output_path)

        print("Saved:", output_path.name)

    # -----------------------------------------------------
    # Save error report
    # -----------------------------------------------------

    error_report = {
        "image_threshold": image_threshold,
        "pixel_threshold": pixel_threshold,
        "total_test_images": len(records),

        "false_negatives": [
            {
                "image": record["image_path"].name,
                "category": record["defect_type"],
                "score": record["score"],
            }
            for record in false_negatives
        ],

        "false_positives": [
            {
                "image": record["image_path"].name,
                "category": record["defect_type"],
                "score": record["score"],
            }
            for record in false_positives
        ],

        "visualizations": [
            f"{key}.png" for key in selected
        ],
    }

    report_path = OUTPUT_DIR / "error_report.json"

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            error_report,
            file,
            indent=4,
        )

    # Save a simple CSV for convenient inspection
    csv_path = OUTPUT_DIR / "all_predictions.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "image",
            "category",
            "anomaly_score",
            "actual_label",
            "predicted_label",
            "correct",
        ])

        for record in records:

            writer.writerow([
                record["image_path"].name,
                record["defect_type"],
                record["score"],
                record["actual_label"],
                record["predicted_label"],
                record["correct"],
            ])

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 70)

    print("\nFalse negatives:", len(false_negatives))

    for record in false_negatives:
        print(
            f"  {record['defect_type']}/"
            f"{record['image_path'].name} "
            f"| Score: {record['score']:.4f}"
        )

    print("\nFalse positives:", len(false_positives))

    for record in false_positives:
        print(
            f"  {record['defect_type']}/"
            f"{record['image_path'].name} "
            f"| Score: {record['score']:.4f}"
        )

    print("\nVisualizations saved:", len(selected))
    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nReport:", report_path)
    print("CSV:", csv_path)

    print("\nVisual error analysis completed.")


if __name__ == "__main__":
    main()
