
import logging
import os
from datetime import timedelta
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps

from django.core.files.base import ContentFile
from django.db.models import Count, Q
from django.db.models.functions import TruncDate
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .forms import InspectionForm
from .models import Inspection
from ml_engine.inference import predict_image


logger = logging.getLogger(__name__)


# ============================================================
# 1. CREATE INSPECTION
# ============================================================


def create_inspection(request):

    if request.method == 'POST':

        form = InspectionForm(request.POST, request.FILES)

        if form.is_valid():

            inspection = form.save(commit=False)
            inspection.status = 'processing'
            inspection.save()

            try:

                # 1. Run TensorFlow/Keras PatchCore inference.
                result = predict_image(
                    image_path=inspection.image.path,
                    category=inspection.category,
                )

                # 2. Validate anomaly score.
                anomaly_score = float(result['anomaly_score'])

                if not np.isfinite(anomaly_score) or anomaly_score < 0:
                    raise ValueError(
                        'Anomaly score must be finite and non-negative.'
                    )

                # 3. Validate predicted label.
                predicted_label = result['predicted_label']

                if not isinstance(predicted_label, (bool, np.bool_)):
                    raise ValueError(
                        'Predicted label must be a boolean value.'
                    )

                # 4. Read model outputs.
                anomaly_map = np.asarray(
                    result['anomaly_map'],
                    dtype=np.float32,
                )

                raw_mask = np.asarray(
                    result['predicted_mask'],
                )

                # 5. Validate anomaly map.
                if anomaly_map.ndim != 2 or anomaly_map.size == 0:
                    raise ValueError(
                        'Anomaly map must be a non-empty 2D array.'
                    )

                if not np.all(np.isfinite(anomaly_map)):
                    raise ValueError(
                        'Anomaly map contains invalid values.'
                    )

                # 6. Validate predicted mask before conversion.
                if raw_mask.ndim != 2 or raw_mask.size == 0:
                    raise ValueError(
                        'Predicted mask must be a non-empty 2D array.'
                    )

                if not np.all(np.isin(raw_mask, [0, 1])):
                    raise ValueError(
                        'Predicted mask must contain only 0 and 1.'
                    )

                if anomaly_map.shape != raw_mask.shape:
                    raise ValueError(
                        'Anomaly map and predicted mask must have matching shapes.'
                    )

                predicted_mask = raw_mask.astype(np.uint8)

                # Store validated prediction values.
                inspection.anomaly_score = anomaly_score
                inspection.predicted_label = bool(predicted_label)

                # 7. Load the original uploaded image.
                with Image.open(inspection.image.path) as source:
                    original_image = source.convert('RGB')

                original_size = original_image.size

                # 8. Normalize anomaly distances for visualization.
                # This does not modify the actual model score or prediction.

                lower_value = float(np.min(anomaly_map))

                upper_value = float(
                    np.percentile(anomaly_map, 99.5)
                )

                value_range = upper_value - lower_value

                if value_range > 1e-8:

                    normalized_map = (
                        anomaly_map - lower_value
                    ) / value_range

                else:

                    normalized_map = np.zeros_like(anomaly_map)

                heatmap_array = (
                    np.clip(normalized_map, 0.0, 1.0) * 255
                ).astype(np.uint8)

                heatmap_gray = Image.fromarray(
                    heatmap_array
                ).convert('L')

                heatmap_gray = heatmap_gray.resize(
                    original_size,
                    Image.Resampling.BILINEAR,
                )

                # 9. Apply VisionGuard heatmap colors.
                heatmap = ImageOps.colorize(
                    heatmap_gray,
                    black='#08111f',
                    mid='#00c8ff',
                    white='#ff3b30',
                ).convert('RGB')

                # 10. Save anomaly heatmap.
                buffer = BytesIO()

                heatmap.save(
                    buffer,
                    format='PNG',
                )

                inspection.anomaly_map.save(
                    f'inspection_{inspection.pk}_heatmap.png',
                    ContentFile(buffer.getvalue()),
                    save=False,
                )

                # 11. Generate heatmap overlay.
                overlay = Image.blend(
                    original_image,
                    heatmap,
                    alpha=0.45,
                )

                buffer = BytesIO()

                overlay.save(
                    buffer,
                    format='PNG',
                )

                inspection.anomaly_overlay.save(
                    f'inspection_{inspection.pk}_overlay.png',
                    ContentFile(buffer.getvalue()),
                    save=False,
                )

                # 12. Generate predicted defect mask.
                mask_array = predicted_mask * 255

                mask_image = Image.fromarray(
                    mask_array
                ).convert('L')

                mask_image = mask_image.resize(
                    original_size,
                    Image.Resampling.NEAREST,
                )

                buffer = BytesIO()

                mask_image.save(
                    buffer,
                    format='PNG',
                )

                inspection.predicted_mask.save(
                    f'inspection_{inspection.pk}_mask.png',
                    ContentFile(buffer.getvalue()),
                    save=False,
                )

                # 13. Calculate predicted anomalous pixel coverage.
                # This is mask coverage, not physical defect area.

                inspection.defect_percentage = round(
                    float(np.mean(predicted_mask > 0) * 100),
                    2,
                )

                # 14. Mark inspection as completed.
                inspection.status = 'completed'
                inspection.error_message = ''

                inspection.save()

            except Exception:

                logger.exception(
                    'VisionGuard inspection failed: %s',
                    inspection.pk,
                )

                # 15. Delete partially generated result images.
                result_fields = [
                    'anomaly_map',
                    'predicted_mask',
                    'anomaly_overlay',
                ]

                for field_name in result_fields:

                    result_file = getattr(
                        inspection,
                        field_name,
                    )

                    if result_file and result_file.name:

                        try:
                            result_file.storage.delete(
                                result_file.name
                            )

                        except Exception:

                            logger.exception(
                                'Failed to clean up %s for inspection %s',
                                field_name,
                                inspection.pk,
                            )

                        setattr(
                            inspection,
                            field_name,
                            None,
                        )

                # 16. Clear incomplete AI results.
                inspection.anomaly_score = None
                inspection.predicted_label = None
                inspection.defect_percentage = None

                # Preserve the original uploaded image.
                inspection.status = 'failed'

                inspection.error_message = (
                    'AI inspection failed. Check server logs.'
                )

                inspection.save()

            return redirect(
                'inspections:detail',
                pk=inspection.pk,
            )

    else:

        form = InspectionForm()

    return render(
        request,
        'inspections/create.html',
        {'form': form},
    )



# ============================================================
# 2. INSPECTION DETAIL
# ============================================================

def inspection_detail(request, pk):

    inspection = get_object_or_404(
        Inspection,
        pk=pk,
    )

    return render(
        request,
        'inspections/detail.html',
        {'inspection': inspection},
    )


# ============================================================
# 3. DOWNLOAD INSPECTION PDF REPORT
# ============================================================

def download_inspection_report(request, pk):

    inspection = get_object_or_404(
        Inspection,
        pk=pk,
    )

    if inspection.status != 'completed':

        return HttpResponse(
            'PDF reports are available only for completed inspections.',
            status=400,
            content_type='text/plain',
        )

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=21 * mm,
        bottomMargin=20 * mm,
        title=f'VisionGuard Inspection Report #{inspection.id}',
        author='VisionGuard',
    )

    # --------------------------------------------------------
    # HEADER AND FOOTER
    # --------------------------------------------------------

    def add_page_header_footer(canvas, doc):

        canvas.saveState()

        page_width, page_height = A4

        canvas.setFont('Helvetica-Bold', 10)
        canvas.setFillColor(
            colors.HexColor('#111827')
        )

        canvas.drawString(
            18 * mm,
            page_height - 11 * mm,
            'VISIONGUARD',
        )

        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(
            colors.HexColor('#64748b')
        )

        canvas.drawRightString(
            page_width - 18 * mm,
            page_height - 11 * mm,
            f'Inspection #{inspection.id}',
        )

        canvas.setStrokeColor(
            colors.HexColor('#e2e8f0')
        )

        canvas.line(
            18 * mm,
            page_height - 14 * mm,
            page_width - 18 * mm,
            page_height - 14 * mm,
        )

        canvas.line(
            18 * mm,
            15 * mm,
            page_width - 18 * mm,
            15 * mm,
        )

        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(
            colors.HexColor('#64748b')
        )

        canvas.drawString(
            18 * mm,
            10 * mm,
            'VisionGuard | Industrial AI Inspection',
        )

        canvas.drawRightString(
            page_width - 18 * mm,
            10 * mm,
            f'Page {doc.page}',
        )

        canvas.restoreState()

    # --------------------------------------------------------
    # REPORT STYLES
    # --------------------------------------------------------

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=23,
        leading=27,
        textColor=colors.HexColor('#111827'),
        alignment=TA_CENTER,
        spaceAfter=5,
    )

    subtitle_style = ParagraphStyle(
        'ReportSubtitle',
        parent=styles['Normal'],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#64748b'),
        alignment=TA_CENTER,
        spaceAfter=12,
    )

    section_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#111827'),
        spaceBefore=9,
        spaceAfter=6,
        keepWithNext=True,
    )

    image_heading_style = ParagraphStyle(
        'ImageHeading',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#111827'),
        spaceAfter=4,
    )

    description_style = ParagraphStyle(
        'ImageDescription',
        parent=styles['BodyText'],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#64748b'),
        spaceAfter=5,
    )

    normal_style = ParagraphStyle(
        'ReportBody',
        parent=styles['BodyText'],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#334155'),
    )

    story = []

    # --------------------------------------------------------
    # PAGE 1: REPORT SUMMARY
    # --------------------------------------------------------

    story.append(
        Paragraph(
            'VISIONGUARD',
            title_style,
        )
    )

    story.append(
        Paragraph(
            'Industrial AI Visual Inspection Report',
            subtitle_style,
        )
    )

    story.append(
        Paragraph(
            f'<b>Inspection Report:</b> #{inspection.id}',
            normal_style,
        )
    )

    story.append(
        Spacer(1, 10)
    )

    # Prediction result.
    if inspection.predicted_label is True:

        prediction = 'DEFECTIVE'
        prediction_color = colors.HexColor('#dc2626')

    elif inspection.predicted_label is False:

        prediction = 'NORMAL'
        prediction_color = colors.HexColor('#15803d')

    else:

        prediction = 'UNKNOWN'
        prediction_color = colors.HexColor('#64748b')

    prediction_table = Table(
        [
            ['AI INSPECTION RESULT', prediction]
        ],
        colWidths=[
            85 * mm,
            85 * mm,
        ],
    )

    prediction_table.setStyle(
        TableStyle([
            (
                'BACKGROUND',
                (0, 0),
                (0, 0),
                colors.HexColor('#f1f5f9'),
            ),
            (
                'TEXTCOLOR',
                (0, 0),
                (0, 0),
                colors.HexColor('#334155'),
            ),
            (
                'TEXTCOLOR',
                (1, 0),
                (1, 0),
                prediction_color,
            ),
            (
                'FONTNAME',
                (0, 0),
                (-1, -1),
                'Helvetica-Bold',
            ),
            (
                'FONTSIZE',
                (0, 0),
                (-1, -1),
                11,
            ),
            (
                'ALIGN',
                (0, 0),
                (-1, -1),
                'CENTER',
            ),
            (
                'VALIGN',
                (0, 0),
                (-1, -1),
                'MIDDLE',
            ),
            (
                'BOX',
                (0, 0),
                (-1, -1),
                0.7,
                colors.HexColor('#e2e8f0'),
            ),
            (
                'TOPPADDING',
                (0, 0),
                (-1, -1),
                12,
            ),
            (
                'BOTTOMPADDING',
                (0, 0),
                (-1, -1),
                12,
            ),
        ])
    )

    story.append(prediction_table)

    # Inspection metrics.
    story.append(
        Paragraph(
            'Inspection Metrics',
            section_style,
        )
    )

    anomaly_score = (
        f'{inspection.anomaly_score:.4f}'
        if inspection.anomaly_score is not None
        else 'Not available'
    )

    defect_percentage = (
        f'{inspection.defect_percentage:.2f}%'
        if inspection.defect_percentage is not None
        else 'Not available'
    )

    metrics_data = [
        ['Metric', 'Value'],
        ['Anomaly Score', anomaly_score],
        ['Predicted Result', prediction],
        ['Predicted Mask Coverage', defect_percentage],
        ['Product Category', str(inspection.category).title()],
        ['Inspection Status', str(inspection.status).title()],
        [
            'Created At',
            inspection.created_at.strftime(
                '%d %b %Y, %I:%M %p'
            ),
        ],
    ]

    metrics_table = Table(
        metrics_data,
        colWidths=[
            75 * mm,
            95 * mm,
        ],
        repeatRows=1,
    )

    metrics_table.setStyle(
        TableStyle([
            (
                'BACKGROUND',
                (0, 0),
                (-1, 0),
                colors.HexColor('#1e293b'),
            ),
            (
                'TEXTCOLOR',
                (0, 0),
                (-1, 0),
                colors.white,
            ),
            (
                'FONTNAME',
                (0, 0),
                (-1, 0),
                'Helvetica-Bold',
            ),
            (
                'FONTNAME',
                (0, 1),
                (-1, -1),
                'Helvetica',
            ),
            (
                'FONTSIZE',
                (0, 0),
                (-1, -1),
                8.5,
            ),
            (
                'GRID',
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor('#e2e8f0'),
            ),
            (
                'ROWBACKGROUNDS',
                (0, 1),
                (-1, -1),
                [
                    colors.white,
                    colors.HexColor('#f8fafc'),
                ],
            ),
            (
                'VALIGN',
                (0, 0),
                (-1, -1),
                'MIDDLE',
            ),
            (
                'TOPPADDING',
                (0, 0),
                (-1, -1),
                6,
            ),
            (
                'BOTTOMPADDING',
                (0, 0),
                (-1, -1),
                6,
            ),
        ])
    )

    story.append(metrics_table)

    # Original image.
    story.append(
        Paragraph(
            'Original Product Image',
            section_style,
        )
    )

    story.append(
        Paragraph(
            'Original image submitted for AI inspection.',
            description_style,
        )
    )

    if inspection.image and os.path.exists(inspection.image.path):

        original_image = RLImage(
            inspection.image.path,
            width=115 * mm,
            height=78 * mm,
            kind='proportional',
        )

        original_image.hAlign = 'CENTER'

        story.append(original_image)

    else:

        story.append(
            Paragraph(
                'Original image unavailable.',
                normal_style,
            )
        )

    # --------------------------------------------------------
    # PAGE 2: AI VISUAL ANALYSIS
    # --------------------------------------------------------

    story.append(PageBreak())

    story.append(
        Paragraph(
            'AI Visual Analysis',
            title_style,
        )
    )

    story.append(
        Paragraph(
            'PatchCore anomaly localization and visualization',
            subtitle_style,
        )
    )

    # Helper to create report images.
    def create_report_image(field, width, height):

        if not field:
            return None

        try:

            image_path = field.path

            if not os.path.exists(image_path):
                return None

            image = RLImage(
                image_path,
                width=width,
                height=height,
                kind='proportional',
            )

            image.hAlign = 'CENTER'

            return image

        except (OSError, ValueError):

            return None

    # Heatmap and mask.
    heatmap_image = create_report_image(
        inspection.anomaly_map,
        76 * mm,
        65 * mm,
    )

    mask_image = create_report_image(
        inspection.predicted_mask,
        76 * mm,
        65 * mm,
    )

    heatmap_cell = [
        Paragraph(
            'Anomaly Heatmap',
            image_heading_style,
        ),
        Paragraph(
            'Warmer regions indicate stronger anomaly responses.',
            description_style,
        ),
    ]

    if heatmap_image:

        heatmap_cell.append(heatmap_image)

    else:

        heatmap_cell.append(
            Paragraph(
                'Heatmap unavailable.',
                normal_style,
            )
        )

    mask_cell = [
        Paragraph(
            'Predicted Defect Mask',
            image_heading_style,
        ),
        Paragraph(
            'White pixels represent predicted anomalous regions.',
            description_style,
        ),
    ]

    if mask_image:

        mask_cell.append(mask_image)

    else:

        mask_cell.append(
            Paragraph(
                'Predicted mask unavailable.',
                normal_style,
            )
        )

    visualization_table = Table(
        [
            [heatmap_cell, mask_cell]
        ],
        colWidths=[
            85 * mm,
            85 * mm,
        ],
    )

    visualization_table.setStyle(
        TableStyle([
            (
                'VALIGN',
                (0, 0),
                (-1, -1),
                'TOP',
            ),
            (
                'LEFTPADDING',
                (0, 0),
                (-1, -1),
                4,
            ),
            (
                'RIGHTPADDING',
                (0, 0),
                (-1, -1),
                4,
            ),
            (
                'TOPPADDING',
                (0, 0),
                (-1, -1),
                5,
            ),
            (
                'BOTTOMPADDING',
                (0, 0),
                (-1, -1),
                8,
            ),
        ])
    )

    story.append(visualization_table)

    # Anomaly overlay.
    story.append(
        Paragraph(
            'Anomaly Overlay',
            section_style,
        )
    )

    story.append(
        Paragraph(
            'Heatmap blended with the original product image '
            'for easier visual interpretation.',
            description_style,
        )
    )

    overlay_image = create_report_image(
        inspection.anomaly_overlay,
        120 * mm,
        78 * mm,
    )

    if overlay_image:

        story.append(
            KeepTogether([
                overlay_image,
                Spacer(1, 8),
            ])
        )

    else:

        story.append(
            Paragraph(
                'Anomaly overlay unavailable.',
                normal_style,
            )
        )

    # Technical notes.
    story.append(
        Paragraph(
            'Technical Notes',
            section_style,
        )
    )

    disclaimer = (
        'VisionGuard uses PatchCore anomaly detection to identify '
        'unusual visual patterns. The anomaly score is a model score, '
        'not a probability of defect. Predicted mask coverage represents '
        'the percentage of pixels classified as anomalous; it does not '
        'measure the actual physical defect area. Results should be '
        'reviewed by a qualified inspector before making production decisions.'
    )

    story.append(
        Paragraph(
            disclaimer,
            normal_style,
        )
    )

    story.append(
        Spacer(1, 10)
    )

    story.append(
        Paragraph(
            'Generated by VisionGuard | AI-powered Industrial Inspection',
            subtitle_style,
        )
    )

    # --------------------------------------------------------
    # BUILD PDF
    # --------------------------------------------------------

    document.build(
        story,
        onFirstPage=add_page_header_footer,
        onLaterPages=add_page_header_footer,
    )

    buffer.seek(0)

    response = HttpResponse(
        buffer.getvalue(),
        content_type='application/pdf',
    )

    response['Content-Disposition'] = (
        f'attachment; filename="VisionGuard_Inspection_{inspection.id}.pdf"'
    )

    return response


# ============================================================
# 4. INSPECTION HISTORY
# ============================================================

def inspection_history(request):

    # Start with all inspections, newest first.
    inspections = Inspection.objects.all().order_by('-created_at')

    # Read filter parameters.
    status_filter = request.GET.get('status', '').strip()
    category_filter = request.GET.get('category', '').strip()
    search_query = request.GET.get('q', '').strip()

    # Filter by status.
    valid_statuses = [
        'pending',
        'processing',
        'completed',
        'failed',
    ]

    if status_filter in valid_statuses:

        inspections = inspections.filter(
            status=status_filter
        )

    # Filter by product category.
    valid_categories = [choice[0] for choice in Inspection.CATEGORY_CHOICES]

    if category_filter in valid_categories:

        inspections = inspections.filter(
            category=category_filter
        )

    # Search by inspection ID.
    if search_query:

        if search_query.isdigit():

            inspections = inspections.filter(
                pk=int(search_query)
            )

        else:

            inspections = inspections.none()

    # Dashboard statistics.
    context = {

        'inspections': inspections,

        'total_count': Inspection.objects.count(),

        'completed_count': Inspection.objects.filter(
            status='completed'
        ).count(),

        'failed_count': Inspection.objects.filter(
            status='failed'
        ).count(),

        'pending_count': Inspection.objects.filter(
            status__in=['pending', 'processing']
        ).count(),

        # Preserve selected filters in the template.
        'status_filter': status_filter,
        'category_filter': category_filter,
        'category_choices': Inspection.CATEGORY_CHOICES,
        'search_query': search_query,
    }

    return render(
        request,
        'inspections/history.html',
        context,
    )


# ============================================================
# 5. INSPECTION ANALYTICS
# ============================================================

def inspection_analytics(request):

    # Only successfully completed inspections are used.
    completed = Inspection.objects.filter(
        status='completed'
    )

    total_completed = completed.count()

    defective_count = completed.filter(
        predicted_label=True
    ).count()

    normal_count = completed.filter(
        predicted_label=False
    ).count()

    unknown_count = (
        total_completed - defective_count - normal_count
    )

    defect_rate = (
        round(
            (defective_count / total_completed) * 100,
            2,
        )
        if total_completed
        else 0
    )

    # Last 7 calendar days, including today.
    today = timezone.localdate()

    start_date = today - timedelta(days=6)

    daily_results = (
        completed
        .filter(
            created_at__date__gte=start_date,
            created_at__date__lte=today,
        )
        .annotate(
            day=TruncDate('created_at')
        )
        .values('day')
        .annotate(
            total=Count('id'),
            defective=Count(
                'id',
                filter=Q(predicted_label=True),
            ),
            normal=Count(
                'id',
                filter=Q(predicted_label=False),
            ),
        )
        .order_by('day')
    )

    # Fill missing dates with zero.
    daily_lookup = {
        item['day']: item
        for item in daily_results
    }

    trend_labels = []
    trend_total = []
    trend_defective = []
    trend_normal = []

    for day_offset in range(7):

        current_day = start_date + timedelta(
            days=day_offset
        )

        result = daily_lookup.get(
            current_day,
            {},
        )

        trend_labels.append(
            current_day.strftime('%d %b')
        )

        trend_total.append(
            result.get('total', 0)
        )

        trend_defective.append(
            result.get('defective', 0)
        )

        trend_normal.append(
            result.get('normal', 0)
        )

    # Category-wise inspection statistics.
    category_stats = list(
        completed
        .values('category')
        .annotate(
            total=Count('id'),
            defective=Count(
                'id',
                filter=Q(predicted_label=True),
            ),
            normal=Count(
                'id',
                filter=Q(predicted_label=False),
            ),
        )
        .order_by('category')
    )

    category_labels = [
        item['category'].title()
        for item in category_stats
    ]

    category_totals = [
        item['total']
        for item in category_stats
    ]

    # Recently detected defects.
    recent_defects = (
        completed
        .filter(predicted_label=True)
        .order_by('-created_at')[:5]
    )

    # Chart data.
    chart_data = {

        'distribution': {

            'labels': [
                'Normal',
                'Defective',
                'Unknown',
            ],

            'values': [
                normal_count,
                defective_count,
                unknown_count,
            ],
        },

        'trend': {

            'labels': trend_labels,
            'total': trend_total,
            'defective': trend_defective,
            'normal': trend_normal,
        },

        'categories': {

            'labels': category_labels,
            'values': category_totals,
        },
    }

    context = {

        'total_completed': total_completed,
        'defective_count': defective_count,
        'normal_count': normal_count,
        'unknown_count': unknown_count,
        'defect_rate': defect_rate,
        'recent_defects': recent_defects,
        'chart_data': chart_data,
    }

    return render(
        request,
        'inspections/analytics.html',
        context,
    )

