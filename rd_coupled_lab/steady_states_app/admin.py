"""
admin.py

Configuración del admin de Django para los modelos.
"""

from django.contrib import admin
from .models import AnalysisRun, SteadyState, Scenario


@admin.register(AnalysisRun)
class AnalysisRunAdmin(admin.ModelAdmin):
    list_display = ['name', 'analysis_type', 'status', 'created_at', 'completed_at']
    list_filter = ['status', 'analysis_type', 'created_at']
    search_fields = ['name', 'error_message']
    readonly_fields = ['created_at', 'started_at', 'completed_at', 'duration']
    
    fieldsets = (
        ('Información Básica', {
            'fields': ('name', 'analysis_type', 'status')
        }),
        ('Configuración', {
            'fields': ('config_params',)
        }),
        ('Resultados', {
            'fields': ('results_summary', 'csv_file_path')
        }),
        ('Metadatos', {
            'fields': ('created_at', 'started_at', 'completed_at', 'duration', 'error_message', 'notes')
        }),
    )


@admin.register(SteadyState)
class SteadyStateAdmin(admin.ModelAdmin):
    list_display = ['c_star', 's_star', 'i_star', 'max_real', 'unstable', 'analysis_run']
    list_filter = ['unstable', 'analysis_run', 'mu']
    search_fields = ['analysis_run__name']
    readonly_fields = ['created_at']


@admin.register(Scenario)
class ScenarioAdmin(admin.ModelAdmin):
    list_display = ['name', 'allee_type', 'mu', 'use_adaptive_control', 'is_active', 'created_at']
    list_filter = ['allee_type', 'use_adaptive_control', 'is_active', 'created_at']
    search_fields = ['name']
    readonly_fields = ['created_at', 'updated_at']
