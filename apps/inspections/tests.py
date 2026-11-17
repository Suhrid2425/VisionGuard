
import os
import tempfile
from datetime import timedelta
from io import BytesIO
from unittest.mock import patch

import numpy as np
from PIL import Image

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from apps.inspections.models import Inspection
from apps.inspections.forms import InspectionForm


class InspectionTests(TestCase):

    # ---------------- TEST SETUP ----------------

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

        self.settings_override = override_settings(
            MEDIA_ROOT=self.temp_dir.name
        )
        self.settings_override.enable()

        self.addCleanup(self.settings_override.disable)
        self.addCleanup(self.temp_dir.cleanup)

    # ---------------- TEST HELPERS ----------------

    def create_test_image(self, image_format="PNG"):
        buffer = BytesIO()
        image = Image.new("RGB", (100, 100), color="white")
        image.save(buffer, format=image_format)
        buffer.seek(0)

        extension = "png" if image_format == "PNG" else "jpg"

        return SimpleUploadedFile(
            f"test.{extension}",
            buffer.getvalue(),
            content_type=f"image/{extension}",
        )

    def create_inspection(
        self,
        status="completed",
        category="bottle",
        predicted_label=False,
    ):
        return Inspection.objects.create(
            category=category,
            image="inspections/test.jpg",
            status=status,
            anomaly_score=25.0,
            predicted_label=predicted_label,
        )

    def create_mock_prediction(self):
        anomaly_map = np.linspace(
            0.0, 1.0, 100, dtype=np.float32
        ).reshape(10, 10)

        predicted_mask = np.zeros((10, 10), dtype=np.uint8)
        predicted_mask[:5, :5] = 1

        return {
            "anomaly_score": 42.5,
            "predicted_label": True,
            "anomaly_map": anomaly_map,
            "predicted_mask": predicted_mask,
        }

    def assert_prediction_rejected(self, result):
        """
        Submit a prediction containing invalid model output
        and verify that the inspection fails safely.
        """

        with patch(
            "apps.inspections.views.predict_image",
            return_value=result,
        ) as mock_predict:

            response = self.client.post(
                reverse("inspections:create"),
                {
                    "category": "bottle",
                    "image": self.create_test_image(),
                },
            )

            mock_predict.assert_called_once()

        inspection = Inspection.objects.latest("pk")

        self.assertEqual(response.status_code, 302)
        self.assertEqual(inspection.status, "failed")

        self.assertEqual(
            inspection.error_message,
            "AI inspection failed. Check server logs.",
        )

        # Original upload must be preserved.
        self.assertTrue(inspection.image)
        self.assertTrue(os.path.exists(inspection.image.path))

        # Invalid AI results must not remain in the database.
        self.assertIsNone(inspection.anomaly_score)
        self.assertIsNone(inspection.predicted_label)
        self.assertIsNone(inspection.defect_percentage)

        # No generated result images should remain.
        self.assertFalse(inspection.anomaly_map)
        self.assertFalse(inspection.predicted_mask)
        self.assertFalse(inspection.anomaly_overlay)

        return inspection

    # ---------------- PAGE TESTS ----------------

    def test_inspection_history_loads(self):
        response = self.client.get(reverse("inspections:history"))

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "inspections/history.html")

    def test_inspection_detail_loads(self):
        inspection = self.create_inspection()

        response = self.client.get(
            reverse("inspections:detail", args=[inspection.pk])
        )

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "inspections/detail.html")

    def test_inspection_analytics_loads(self):
        response = self.client.get(reverse("inspections:analytics"))

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "inspections/analytics.html")

    def test_invalid_inspection_id_returns_404(self):
        response = self.client.get(
            reverse("inspections:detail", args=[99999])
        )

        self.assertEqual(response.status_code, 404)

    # ---------------- FORM VALIDATION TESTS ----------------

    def test_valid_png_upload(self):
        form = InspectionForm(
            data={"category": "bottle"},
            files={"image": self.create_test_image("PNG")},
        )

        self.assertTrue(form.is_valid(), form.errors)

    def test_valid_jpeg_upload(self):
        form = InspectionForm(
            data={"category": "bottle"},
            files={"image": self.create_test_image("JPEG")},
        )

        self.assertTrue(form.is_valid(), form.errors)

    def test_image_is_required(self):
        form = InspectionForm(data={"category": "bottle"})

        self.assertFalse(form.is_valid())
        self.assertIn("image", form.errors)

    def test_invalid_category_is_rejected(self):
        form = InspectionForm(
            data={"category": "invalid_category"},
            files={"image": self.create_test_image()},
        )

        self.assertFalse(form.is_valid())
        self.assertIn("category", form.errors)

    def test_gif_image_is_rejected(self):
        form = InspectionForm(
            data={"category": "bottle"},
            files={"image": self.create_test_image("GIF")},
        )

        self.assertFalse(form.is_valid())
        self.assertIn("image", form.errors)

    # ---------------- HISTORY FILTER TESTS ----------------

    def test_history_filters_by_status(self):
        completed = self.create_inspection(status="completed")
        self.create_inspection(status="failed")

        response = self.client.get(
            reverse("inspections:history"),
            {"status": "completed"},
        )

        self.assertEqual(response.status_code, 200)
        results = list(response.context["inspections"])

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].pk, completed.pk)

    def test_history_searches_by_inspection_id(self):
        inspection = self.create_inspection()
        self.create_inspection()

        response = self.client.get(
            reverse("inspections:history"),
            {"q": str(inspection.pk)},
        )

        self.assertEqual(response.status_code, 200)
        results = list(response.context["inspections"])

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].pk, inspection.pk)

    def test_history_returns_no_results_for_invalid_search(self):
        self.create_inspection()

        response = self.client.get(
            reverse("inspections:history"),
            {"q": "invalid"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.context["inspections"]), 0)

    def test_history_filters_by_category(self):
        bottle = self.create_inspection(category="bottle")
        self.create_inspection(category="cable")

        response = self.client.get(
            reverse("inspections:history"),
            {"category": "bottle"},
        )

        self.assertEqual(response.status_code, 200)
        results = list(response.context["inspections"])

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].pk, bottle.pk)

    # ---------------- PDF REPORT TESTS ----------------

    def test_completed_inspection_generates_pdf(self):
        inspection = self.create_inspection(status="completed")

        response = self.client.get(
            reverse("inspections:download_report", args=[inspection.pk])
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/pdf")

        self.assertIn(
            f"VisionGuard_Inspection_{inspection.pk}.pdf",
            response["Content-Disposition"],
        )

        self.assertTrue(response.content.startswith(b"%PDF"))

    def test_incomplete_inspection_cannot_generate_pdf(self):
        inspection = self.create_inspection(status="processing")

        response = self.client.get(
            reverse("inspections:download_report", args=[inspection.pk])
        )

        self.assertEqual(response.status_code, 400)

        self.assertContains(
            response,
            "PDF reports are available only for completed inspections.",
            status_code=400,
        )

    def test_invalid_pdf_inspection_id_returns_404(self):
        response = self.client.get(
            reverse("inspections:download_report", args=[99999])
        )

        self.assertEqual(response.status_code, 404)

    # ---------------- ANALYTICS TESTS ----------------

    def test_analytics_counts_completed_inspections(self):
        self.create_inspection(status="completed", predicted_label=True)
        self.create_inspection(status="completed", predicted_label=False)
        self.create_inspection(status="failed", predicted_label=True)

        response = self.client.get(reverse("inspections:analytics"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["total_completed"], 2)

    def test_analytics_counts_defective_normal_and_unknown(self):
        self.create_inspection(predicted_label=True)
        self.create_inspection(predicted_label=False)
        self.create_inspection(predicted_label=None)

        response = self.client.get(reverse("inspections:analytics"))

        self.assertEqual(response.context["defective_count"], 1)
        self.assertEqual(response.context["normal_count"], 1)
        self.assertEqual(response.context["unknown_count"], 1)

    def test_analytics_calculates_defect_rate(self):
        self.create_inspection(predicted_label=True)
        self.create_inspection(predicted_label=True)
        self.create_inspection(predicted_label=False)
        self.create_inspection(predicted_label=False)

        response = self.client.get(reverse("inspections:analytics"))

        self.assertEqual(response.context["defect_rate"], 50.0)

    def test_analytics_zero_inspections_has_zero_defect_rate(self):
        response = self.client.get(reverse("inspections:analytics"))

        self.assertEqual(response.context["total_completed"], 0)
        self.assertEqual(response.context["defect_rate"], 0)

    def test_analytics_recent_defects_contains_only_completed_defects(self):
        defective = self.create_inspection(
            status="completed",
            predicted_label=True,
        )

        self.create_inspection(
            status="completed",
            predicted_label=False,
        )

        self.create_inspection(
            status="failed",
            predicted_label=True,
        )

        response = self.client.get(reverse("inspections:analytics"))

        recent_defects = list(response.context["recent_defects"])

        self.assertEqual(len(recent_defects), 1)
        self.assertEqual(recent_defects[0].pk, defective.pk)

    def test_analytics_seven_day_trend(self):
        today = timezone.now()
        yesterday = today - timedelta(days=1)

        today_inspection = self.create_inspection(
            status="completed",
            predicted_label=True,
        )

        yesterday_inspection = self.create_inspection(
            status="completed",
            predicted_label=False,
        )

        Inspection.objects.filter(pk=today_inspection.pk).update(
            created_at=today
        )

        Inspection.objects.filter(pk=yesterday_inspection.pk).update(
            created_at=yesterday
        )

        response = self.client.get(reverse("inspections:analytics"))

        trend = response.context["chart_data"]["trend"]

        self.assertEqual(len(trend["labels"]), 7)
        self.assertEqual(len(trend["total"]), 7)
        self.assertEqual(trend["total"][-1], 1)
        self.assertEqual(trend["total"][-2], 1)

    # ---------------- INSPECTION WORKFLOW TESTS ----------------

    def test_create_inspection_page_loads(self):
        response = self.client.get(reverse("inspections:create"))

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "inspections/create.html")

    @patch("apps.inspections.views.predict_image")
    def test_successful_inspection_saves_all_results(self, mock_predict):
        mock_predict.return_value = self.create_mock_prediction()

        response = self.client.post(
            reverse("inspections:create"),
            {
                "category": "bottle",
                "image": self.create_test_image(),
            },
        )

        inspection = Inspection.objects.get()

        self.assertEqual(response.status_code, 302)

        self.assertRedirects(
            response,
            reverse("inspections:detail", args=[inspection.pk]),
        )

        mock_predict.assert_called_once()

        self.assertEqual(inspection.status, "completed")
        self.assertEqual(inspection.anomaly_score, 42.5)
        self.assertTrue(inspection.predicted_label)
        self.assertEqual(inspection.defect_percentage, 25.0)

        self.assertTrue(inspection.image)
        self.assertTrue(inspection.anomaly_map)
        self.assertTrue(inspection.predicted_mask)
        self.assertTrue(inspection.anomaly_overlay)

        self.assertTrue(os.path.exists(inspection.image.path))
        self.assertTrue(os.path.exists(inspection.anomaly_map.path))
        self.assertTrue(os.path.exists(inspection.predicted_mask.path))
        self.assertTrue(os.path.exists(inspection.anomaly_overlay.path))

    @patch("apps.inspections.views.predict_image")
    def test_inference_failure_marks_inspection_failed(self, mock_predict):
        mock_predict.side_effect = RuntimeError("Mock AI failure")

        response = self.client.post(
            reverse("inspections:create"),
            {
                "category": "bottle",
                "image": self.create_test_image(),
            },
        )

        inspection = Inspection.objects.get()

        self.assertEqual(response.status_code, 302)
        self.assertEqual(inspection.status, "failed")

        self.assertEqual(
            inspection.error_message,
            "AI inspection failed. Check server logs.",
        )

        mock_predict.assert_called_once()

    @patch("apps.inspections.views.predict_image")
    def test_invalid_model_output_marks_inspection_failed(self, mock_predict):
        result = self.create_mock_prediction()

        result["predicted_mask"] = np.zeros(
            (8, 8), dtype=np.uint8
        )

        mock_predict.return_value = result

        response = self.client.post(
            reverse("inspections:create"),
            {
                "category": "bottle",
                "image": self.create_test_image(),
            },
        )

        inspection = Inspection.objects.get()

        self.assertEqual(response.status_code, 302)
        self.assertEqual(inspection.status, "failed")

        self.assertEqual(
            inspection.error_message,
            "AI inspection failed. Check server logs.",
        )

    @patch("apps.inspections.views.predict_image")
    def test_invalid_upload_does_not_start_ai_inference(self, mock_predict):
        response = self.client.post(
            reverse("inspections:create"),
            {
                "category": "bottle",
                "image": self.create_test_image("GIF"),
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(Inspection.objects.count(), 0)

        mock_predict.assert_not_called()

    # ---------------- FAILURE CLEANUP TEST ----------------

    @patch("apps.inspections.views.predict_image")
    def test_visualization_failure_preserves_original_and_cleans_partial_outputs(
        self, mock_predict
    ):
        mock_predict.return_value = self.create_mock_prediction()

        uploaded_image = self.create_test_image()

        original_save = Image.Image.save
        save_calls = {"count": 0}

        def fail_on_second_save(image, fp, *args, **kwargs):
            save_calls["count"] += 1

            if save_calls["count"] == 2:
                raise OSError("Simulated visualization failure")

            return original_save(image, fp, *args, **kwargs)

        with patch.object(
            Image.Image,
            "save",
            autospec=True,
            side_effect=fail_on_second_save,
        ):
            response = self.client.post(
                reverse("inspections:create"),
                {
                    "category": "bottle",
                    "image": uploaded_image,
                },
            )

        inspection = Inspection.objects.get()

        self.assertEqual(response.status_code, 302)
        self.assertEqual(inspection.status, "failed")

        # The original uploaded image must remain available.
        self.assertTrue(inspection.image)
        self.assertTrue(os.path.exists(inspection.image.path))

        # Partial AI result files must be removed.
        self.assertFalse(inspection.anomaly_map)
        self.assertFalse(inspection.predicted_mask)
        self.assertFalse(inspection.anomaly_overlay)

        # Partial AI results must not remain in the database.
        self.assertIsNone(inspection.anomaly_score)
        self.assertIsNone(inspection.predicted_label)
        self.assertIsNone(inspection.defect_percentage)

        self.assertEqual(
            inspection.error_message,
            "AI inspection failed. Check server logs.",
        )

        self.assertGreaterEqual(save_calls["count"], 2)

    # ---------------- MODEL OUTPUT EDGE-CASE TESTS ----------------

    def test_negative_nan_and_infinite_anomaly_scores_are_rejected(self):
        invalid_scores = [
            -1.0,
            float("nan"),
            float("inf"),
            float("-inf"),
        ]

        for score in invalid_scores:
            with self.subTest(anomaly_score=score):
                result = self.create_mock_prediction()
                result["anomaly_score"] = score

                self.assert_prediction_rejected(result)

    def test_non_boolean_prediction_labels_are_rejected(self):
        invalid_labels = [
            1,
            0,
            "True",
            "False",
            None,
        ]

        for label in invalid_labels:
            with self.subTest(predicted_label=label):
                result = self.create_mock_prediction()
                result["predicted_label"] = label

                self.assert_prediction_rejected(result)

    def test_non_finite_and_invalid_dimension_anomaly_maps_are_rejected(self):
        invalid_maps = [
            np.full((10, 10), np.nan, dtype=np.float32),
            np.full((10, 10), np.inf, dtype=np.float32),
            np.empty((0, 0), dtype=np.float32),
            np.zeros((100,), dtype=np.float32),
        ]

        for anomaly_map in invalid_maps:
            with self.subTest(shape=anomaly_map.shape):
                result = self.create_mock_prediction()
                result["anomaly_map"] = anomaly_map

                self.assert_prediction_rejected(result)

    def test_invalid_values_and_dimensions_in_predicted_mask_are_rejected(self):
        invalid_masks = [
            np.full((10, 10), 2, dtype=np.uint8),
            np.full((10, 10), -1, dtype=np.int16),
            np.full((10, 10), np.nan, dtype=np.float32),
            np.empty((0, 0), dtype=np.uint8),
            np.zeros((100,), dtype=np.uint8),
        ]

        for predicted_mask in invalid_masks:
            with self.subTest(shape=predicted_mask.shape):
                result = self.create_mock_prediction()
                result["predicted_mask"] = predicted_mask

                self.assert_prediction_rejected(result)



class InspectionUploadSecurityTests(TestCase):

    # ---------------- TEST HELPERS ----------------

    def make_upload(self, content=None, name="test.png"):
        from io import BytesIO
        from django.core.files.uploadedfile import SimpleUploadedFile
        from PIL import Image

        if content is None:
            buffer = BytesIO()

            Image.new(
                "RGB",
                (100, 100),
                color="white"
            ).save(buffer, format="PNG")

            content = buffer.getvalue()

        return SimpleUploadedFile(
            name,
            content,
            content_type="image/png"
        )

    def make_real_image_upload(
        self,
        image_format="PNG",
        name=None,
        content_type=None
    ):
        from io import BytesIO
        from django.core.files.uploadedfile import SimpleUploadedFile
        from PIL import Image

        buffer = BytesIO()

        Image.new(
            "RGB",
            (100, 100),
            color="white"
        ).save(buffer, format=image_format)

        extension = "png" if image_format == "PNG" else "jpg"

        return SimpleUploadedFile(
            name or f"test.{extension}",
            buffer.getvalue(),
            content_type or f"image/{extension}"
        )

    def make_form(self, image):
        form = InspectionForm()
        form.cleaned_data = {"image": image}
        return form

    def assert_image_rejected(self, image, expected_message):
        from django.core.exceptions import ValidationError

        form = self.make_form(image)

        with self.assertRaisesRegex(
            ValidationError,
            expected_message
        ):
            form.clean_image()

    def make_mock_image(
        self,
        image_format="PNG",
        frames=1,
        size=(100, 100)
    ):
        from unittest.mock import MagicMock

        mocked_image = MagicMock()
        mocked_image.__enter__.return_value = mocked_image
        mocked_image.format = image_format
        mocked_image.n_frames = frames
        mocked_image.size = size

        return mocked_image

    # ---------------- IMAGE SECURITY TESTS ----------------

    def test_oversized_file_is_rejected(self):
        image = self.make_upload(
            b"x" * (InspectionForm.MAX_FILE_SIZE + 1)
        )

        self.assert_image_rejected(
            image,
            "Image size must not exceed 10 MB."
        )

    def test_corrupted_image_is_rejected(self):
        image = self.make_upload(
            b"This is not a valid image."
        )

        self.assert_image_rejected(
            image,
            "Invalid or corrupted image."
        )

    def test_unsupported_image_format_is_rejected(self):
        from unittest.mock import patch

        image = self.make_upload()
        mocked_image = self.make_mock_image(
            image_format="TIFF"
        )

        with patch(
            "apps.inspections.forms.Image.open",
            return_value=mocked_image
        ):
            self.assert_image_rejected(
                image,
                "Only JPG and PNG images are supported."
            )

    def test_multi_frame_image_is_rejected(self):
        from unittest.mock import patch

        image = self.make_upload()
        mocked_image = self.make_mock_image(frames=2)

        with patch(
            "apps.inspections.forms.Image.open",
            return_value=mocked_image
        ):
            self.assert_image_rejected(
                image,
                "Animated images are not supported."
            )

    def test_excessive_image_dimensions_are_rejected(self):
        from unittest.mock import patch

        image = self.make_upload()

        mocked_image = self.make_mock_image(
            size=(10001, 100)
        )

        with patch(
            "apps.inspections.forms.Image.open",
            return_value=mocked_image
        ):
            self.assert_image_rejected(
                image,
                "Image dimensions must not exceed"
            )

    def test_excessive_pixel_count_is_rejected(self):
        from unittest.mock import patch

        image = self.make_upload()

        mocked_image = self.make_mock_image(
            size=(5001, 5000)
        )

        with patch(
            "apps.inspections.forms.Image.open",
            return_value=mocked_image
        ):
            self.assert_image_rejected(
                image,
                "Image resolution is too large."
            )

    def test_decompression_bomb_error_is_rejected(self):
        from unittest.mock import patch
        from PIL import Image

        image = self.make_upload()

        with patch(
            "apps.inspections.forms.Image.open",
            side_effect=Image.DecompressionBombError
        ):
            self.assert_image_rejected(
                image,
                "Image resolution is too large."
            )

    # ---------------- COMPLETE FORM VALIDATION TESTS ----------------

    def test_valid_png_passes_complete_form_validation(self):
        image = self.make_real_image_upload("PNG")

        form = InspectionForm(
            data={"category": "bottle"},
            files={"image": image}
        )

        self.assertTrue(form.is_valid(), form.errors)
        self.assertIn("image", form.cleaned_data)

    def test_valid_jpeg_passes_complete_form_validation(self):
        image = self.make_real_image_upload("JPEG")

        form = InspectionForm(
            data={"category": "bottle"},
            files={"image": image}
        )

        self.assertTrue(form.is_valid(), form.errors)
        self.assertIn("image", form.cleaned_data)

    def test_corrupted_image_fails_complete_form_validation(self):
        image = self.make_upload(
            b"This is not a valid image."
        )

        form = InspectionForm(
            data={"category": "bottle"},
            files={"image": image}
        )

        self.assertFalse(form.is_valid())
        self.assertIn("image", form.errors)

    def test_empty_image_file_is_rejected(self):
        from django.core.files.uploadedfile import SimpleUploadedFile

        image = SimpleUploadedFile(
            "empty.png",
            b"",
            content_type="image/png"
        )

        form = InspectionForm(
            data={"category": "bottle"},
            files={"image": image}
        )

        self.assertFalse(form.is_valid())
        self.assertIn("image", form.errors)

    def test_mismatched_mime_type_does_not_override_actual_format(self):
        image = self.make_real_image_upload(
            image_format="PNG",
            name="test.png",
            content_type="image/jpeg"
        )

        form = InspectionForm(
            data={"category": "bottle"},
            files={"image": image}
        )

        # Actual image data is a valid PNG.
        # Client-provided MIME type is not trusted.
        self.assertTrue(form.is_valid(), form.errors)
