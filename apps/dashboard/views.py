
from pathlib import Path

from django.shortcuts import render


def home(request):
    # Temporary dashboard statistics
    total_inspections = 0
    defects_detected = 0
    quality_rate = 0.0
    classified_count = 0

    # Locate trained PatchCore model artifacts
    project_root = Path(__file__).resolve().parents[2]
    model_directory = project_root / "ml_engine" / "models" / "mvtec_ad"

    categories = [
        "bottle", "cable", "capsule", "carpet", "grid",
        "hazelnut", "leather", "metal_nut", "pill", "screw",
        "tile", "toothbrush", "transistor", "wood", "zipper",
    ]

    ready_categories = []

    for category in categories:
        category_directory = model_directory / category

        memory_bank = category_directory / "memory_bank.npz"
        thresholds = category_directory / "thresholds.json"
        manifest = category_directory / "calibration_manifest.json"

        if (
            memory_bank.is_file()
            and thresholds.is_file()
            and manifest.is_file()
            and memory_bank.stat().st_size > 0
        ):
            ready_categories.append(category)

    active_models = len(ready_categories)
    total_models = len(categories)
    all_models_ready = active_models == total_models

    activity = [
        {
            "day": day,
            "date": "",
            "count": 0,
            "bar_width": 0,
        }
        for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    ]

    recent_inspections = []

    context = {
        "total_inspections": total_inspections,
        "defects_detected": defects_detected,
        "quality_rate": quality_rate,
        "active_models": active_models,
        "total_models": total_models,
        "recent_inspections": recent_inspections,
        "activity": activity,
        "all_models_ready": all_models_ready,
        "classified_count": classified_count,
    }

    return render(request, "dashboard/home.html", context)
