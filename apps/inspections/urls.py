from django.urls import path
from . import views
from .views import download_inspection_report
app_name = 'inspections'

urlpatterns = [
    path('new/', views.create_inspection, name='create'),
    path('history/', views.inspection_history, name='history'),
    path('<int:pk>/', views.inspection_detail, name='detail'),
    path('analytics/', views.inspection_analytics, name='analytics'),
    path(
    '<int:pk>/report/',
    download_inspection_report,
    name='download_report',
),
]