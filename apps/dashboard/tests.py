from django.test import TestCase
from django.urls import reverse
from apps.inspections.models import Inspection


class DashboardTests(TestCase):

    def test_dashboard_homepage_loads(self):
        response = self.client.get(reverse("dashboard:home"))

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "dashboard/home.html")

    def test_dashboard_statistics(self):
        Inspection.objects.create(
            category="bottle",
            image="inspections/test.jpg",
            status="completed",
            predicted_label=False,
            anomaly_score=20.0,
        )

        Inspection.objects.create(
            category="bottle",
            image="inspections/test2.jpg",
            status="completed",
            predicted_label=True,
            anomaly_score=40.0,
        )

        response = self.client.get(reverse("dashboard:home"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["total_inspections"], 2)
        self.assertEqual(response.context["defects_detected"], 1)
        self.assertEqual(response.context["quality_rate"], 50.0)