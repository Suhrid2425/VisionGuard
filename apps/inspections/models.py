
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from django.db.models import Q


class Inspection(models.Model):

    CATEGORY_CHOICES = [
        ('bottle', 'Bottle'),
        ('cable', 'Cable'),
        ('capsule', 'Capsule'),
        ('carpet', 'Carpet'),
        ('grid', 'Grid'),
        ('hazelnut', 'Hazelnut'),
        ('leather', 'Leather'),
        ('metal_nut', 'Metal Nut'),
        ('pill', 'Pill'),
        ('screw', 'Screw'),
        ('tile', 'Tile'),
        ('toothbrush', 'Toothbrush'),
        ('transistor', 'Transistor'),
        ('wood', 'Wood'),
        ('zipper', 'Zipper'),
    ]

    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    category = models.CharField(
        max_length=30,
        choices=CATEGORY_CHOICES
    )

    image = models.ImageField(
        upload_to='inspections/%Y/%m/%d/'
    )

    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending'
    )

    anomaly_score = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0)]
    )

    predicted_label = models.BooleanField(
        null=True,
        blank=True
    )

    anomaly_map = models.ImageField(
        upload_to='inspection_results/heatmaps/',
        null=True,
        blank=True
    )

    predicted_mask = models.ImageField(
        upload_to='inspection_results/masks/',
        null=True,
        blank=True
    )

    anomaly_overlay = models.ImageField(
        upload_to='inspection_results/overlays/',
        null=True,
        blank=True
    )

    defect_percentage = models.FloatField(
        null=True,
        blank=True,
        validators=[
            MinValueValidator(0),
            MaxValueValidator(100)
        ]
    )

    error_message = models.TextField(
        blank=True,
        default=''
    )

    created_at = models.DateTimeField(auto_now_add=True)

    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

        indexes = [
            models.Index(
                fields=['status', '-created_at'],
                name='insp_status_created_idx'
            ),
            models.Index(
                fields=['category', '-created_at'],
                name='insp_category_created_idx'
            ),
            models.Index(
                fields=['-created_at'],
                name='insp_created_idx'
            ),
        ]

        constraints = [
            models.CheckConstraint(
                condition=Q(anomaly_score__isnull=True) |
                          Q(anomaly_score__gte=0),
                name='insp_score_nonnegative'
            ),
            models.CheckConstraint(
                condition=Q(defect_percentage__isnull=True) |
                          (
                              Q(defect_percentage__gte=0) &
                              Q(defect_percentage__lte=100)
                          ),
                name='insp_defect_pct_valid'
            ),
        ]

    def __str__(self):
        return (
            f"Inspection #{self.id} - "
            f"{self.get_category_display()}"
        )

    @property
    def is_anomalous(self):
        return self.predicted_label is True
