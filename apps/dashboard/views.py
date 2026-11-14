
from pathlib import Path

from django.shortcuts import render


def home(request):
    # ---------------------------------------------------------
    # 1. Temporary dashboard statistics
    # Database queries are disabled for Vercel deployment.
    # ---------------------------------------------------------
    total_inspections = 0
    defects_detected = 0
    quality_rate = 0.0
    classified_count = 0

    # ---------------------------------------------------------
    # 2. Check whether trained PatchCore models exist
    # ---------------------------------------------------------
    project_root = Path(__file__).resolve().parents[2]
    model_directory = project_root / "ml_engine" / "models"

    categories = [
        "bottle", "cable", "capsule", "carpet", "grid",
        "hazelnut", "leather", "metal_nut", "pill", "screw",
        "tile", "toothbrush", "transistor", "wood", "zipper",
    ]

    ready_categories = []

    for category in categories:
        if category == "bottle":
            memory_bank = model_directory / "bottle_memory_bank.npz"
            thresholds = model_directory / "bottle_thresholds.json"
        else:
            category_directory = model_directory / "mvtec_ad" / category
            memory_bank = category_directory / "memory_bank.npz"
            thresholds = category_directory / "thresholds.json"

        if memory_bank.is_file() and thresholds.is_file():
            ready_categories.append(category)

    active_models = len(ready_categories)
    total_models = len(categories)
    all_models_ready = active_models == total_models

    # ---------------------------------------------------------
    # 3. Temporary seven-day activity
    # ---------------------------------------------------------
    activity = [
        {
            "day": day,
            "date": "",
            "count": 0,
            "bar_width": 0,
        }
        for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    ]

    # ---------------------------------------------------------
    # 4. Recent inspections unavailable without a database
    # ---------------------------------------------------------
    recent_inspections = []

    # ---------------------------------------------------------
    # 5. Dashboard context
    # ---------------------------------------------------------
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
