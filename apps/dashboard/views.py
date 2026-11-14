
from datetime import timedelta
from pathlib import Path

from django.db.models import Count, Q
from django.shortcuts import render
from django.utils import timezone

from apps.inspections.models import Inspection


def home(request):
    # ---------------------------------------------------------
    # 1. Database-backed inspection statistics
    # ---------------------------------------------------------
    total_inspections = Inspection.objects.count()

    completed_inspections = Inspection.objects.filter(
        status="completed",
        predicted_label__isnull=False,
    )

    classified_count = completed_inspections.count()

    defects_detected = completed_inspections.filter(
        predicted_label=True
    ).count()

    normal_inspections = completed_inspections.filter(
        predicted_label=False
    ).count()

    quality_rate = (
        round((normal_inspections / classified_count) * 100, 1)
        if classified_count
        else 0.0
    )

    # ---------------------------------------------------------
    # 2. Check whether the trained PatchCore model exists
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
    # 3. Inspection activity for the last seven days
    # ---------------------------------------------------------
    today = timezone.localdate()
    start_date = today - timedelta(days=6)

    activity_records = (
        Inspection.objects.filter(created_at__date__gte=start_date)
        .values("created_at__date")
        .annotate(total=Count("id"))
        .order_by("created_at__date")
    )

    activity_by_date = {
        record["created_at__date"]: record["total"]
        for record in activity_records
    }

    activity = []

    for day_offset in range(7):
        current_day = start_date + timedelta(days=day_offset)
        activity.append({
            "day": current_day.strftime("%a"),
            "date": current_day.strftime("%d %b"),
            "count": activity_by_date.get(current_day, 0),
        })

    maximum_activity = max(
        (item["count"] for item in activity),
        default=0,
    )

    for item in activity:
        item["bar_width"] = (
            round((item["count"] / maximum_activity) * 100)
            if maximum_activity
            else 0
        )

    # ---------------------------------------------------------
    # 4. Recent inspection records
    # ---------------------------------------------------------
    recent_inspections = Inspection.objects.all()[:5]

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



