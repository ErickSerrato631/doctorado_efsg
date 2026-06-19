"""
urls.py

URLs de la app steady_states_app.
"""

from django.urls import path
from . import views

app_name = 'steady_states_app'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('sistema/almacenamiento/', views.storage_health, name='storage_health'),
    path('visualizaciones/', views.scenario_figures_gallery, name='scenario_figures_gallery'),
    path(
        'visualizaciones/imagen/<str:kind>/<str:scenario_name>/',
        views.serve_scenario_figure,
        name='scenario_figure',
    ),
    path(
        'visualizaciones/png/<str:scenario_name>/<str:basename>/',
        views.serve_scenario_png_basename,
        name='scenario_figure_named',
    ),
    path(
        'visualizaciones/campos/<str:scenario_name>/<str:basename>/',
        views.serve_simulation_field_png,
        name='scenario_simulation_field_png',
    ),
    path(
        'escenario/<str:scenario_name>/ajax/',
        views.scenario_catalog_ajax,
        name='scenario_catalog_ajax',
    ),
    path(
        'escenario/<str:scenario_name>/ajax/tab/<slug:tab>/',
        views.scenario_catalog_tab_ajax,
        name='scenario_catalog_tab_ajax',
    ),
    path('escenario/<str:scenario_name>/', views.scenario_catalog_detail, name='scenario_catalog_detail'),
    path(
        'analysis/complete-3d/modal/',
        views.analysis_complete_3d_modal,
        name='analysis_complete_3d_modal',
    ),
    path('analysis/<str:analysis_type>/', views.analysis_config, name='analysis_config'),
    path('analysis/run/<int:analysis_id>/', views.run_analysis, name='run_analysis'),
    path('analysis/progress/<int:analysis_id>/', views.get_analysis_progress, name='get_analysis_progress'),
    path('analysis/cancel/<int:analysis_id>/', views.cancel_analysis, name='cancel_analysis'),
    path('analysis/reset-stuck/', views.reset_stuck_analyses, name='reset_stuck_analyses'),
    path('analysis/list/', views.analysis_list, name='analysis_list'),
    path('plot/<int:analysis_id>/', views.plot_results, name='plot_results'),
    path('api/experiment-status/', views.experiment_status_api, name='experiment_status_api'),
]

