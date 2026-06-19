"""
models.py

Modelos Django para almacenar análisis de estados estacionarios y resultados.
"""

from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
import json
from pathlib import Path


class AnalysisRun(models.Model):
    """
    Modelo para almacenar información sobre ejecuciones de análisis.
    """
    ANALYSIS_TYPES = [
        ('complete_3d', 'EE en (c, s, i)'),
    ]
    
    STATUS_CHOICES = [
        ('pending', 'Pendiente'),
        ('running', 'Ejecutando'),
        ('completed', 'Completado'),
        ('failed', 'Fallido'),
    ]
    
    # Información básica
    name = models.CharField(max_length=200, help_text="Nombre descriptivo del análisis")
    analysis_type = models.CharField(max_length=20, choices=ANALYSIS_TYPES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    
    # Parámetros de configuración (almacenados como JSON)
    config_params = models.JSONField(default=dict, help_text="Parámetros de configuración del análisis")
    
    # Resultados
    results_summary = models.JSONField(default=dict, null=True, blank=True, 
                                      help_text="Resumen de resultados (estadísticas)")
    csv_file_path = models.CharField(max_length=500, null=True, blank=True,
                                    help_text="Ruta al archivo CSV con resultados")
    json_file_path = models.CharField(max_length=500, null=True, blank=True,
                                     help_text="Ruta al archivo JSON con resultados")
    
    # Metadatos
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(null=True, blank=True)
    
    # Información adicional
    notes = models.TextField(blank=True, help_text="Notas adicionales sobre el análisis")
    
    # Campos de progreso para ejecución asíncrona
    progress_percent = models.FloatField(default=0.0, validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                         help_text="Porcentaje de progreso (0-100)")
    progress_message = models.TextField(blank=True, help_text="Mensaje de estado actual del análisis")
    total_combinations = models.IntegerField(null=True, blank=True, 
                                            help_text="Total de combinaciones de parámetros a procesar")
    processed_combinations = models.IntegerField(default=0, 
                                                 help_text="Número de combinaciones procesadas")
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Ejecución de Análisis"
        verbose_name_plural = "Ejecuciones de Análisis"
    
    def __str__(self):
        return f"{self.name} ({self.get_analysis_type_display()}) - {self.get_status_display()}"
    
    @property
    def duration(self):
        """Calcula la duración del análisis si está completado."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


class SteadyState(models.Model):
    """
    Modelo para almacenar estados estacionarios individuales encontrados.
    """
    analysis_run = models.ForeignKey(AnalysisRun, on_delete=models.CASCADE, 
                                     related_name='steady_states')
    
    # Parámetros del estado estacionario
    rc = models.FloatField(validators=[MinValueValidator(0)])
    rs = models.FloatField(validators=[MinValueValidator(0)])
    rd = models.FloatField(validators=[MinValueValidator(0)])
    alpha = models.FloatField(validators=[MinValueValidator(0)])
    delta = models.FloatField(validators=[MinValueValidator(0)])
    beta = models.FloatField(validators=[MinValueValidator(0)])
    eta = models.FloatField(validators=[MinValueValidator(0)])
    mu = models.FloatField(validators=[MinValueValidator(0)])
    
    # Valores del estado estacionario
    c_star = models.FloatField(help_text="Concentración de células cancerosas en equilibrio")
    s_star = models.FloatField(help_text="Concentración de células sanas en equilibrio")
    i_star = models.FloatField(null=True, blank=True, 
                              help_text="Concentración de células inmunes en equilibrio")
    
    # Autovalores del Jacobiano
    eig1_real = models.FloatField()
    eig1_imag = models.FloatField(default=0)
    eig2_real = models.FloatField()
    eig2_imag = models.FloatField(default=0)
    eig3_real = models.FloatField(null=True, blank=True)
    eig3_imag = models.FloatField(default=0)
    
    # Propiedades de estabilidad
    unstable = models.BooleanField(default=False, help_text="True si el equilibrio es inestable")
    max_real = models.FloatField(help_text="Máxima parte real de los autovalores")
    
    # Metadatos
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-max_real', 'c_star']
        indexes = [
            models.Index(fields=['analysis_run', 'unstable']),
            models.Index(fields=['c_star', 's_star']),
        ]
    
    def __str__(self):
        return f"c*={self.c_star:.3f}, s*={self.s_star:.3f}, unstable={self.unstable}"


class Scenario(models.Model):
    """
    Modelo para almacenar información sobre escenarios en scenarios.json.
    """
    name = models.CharField(max_length=200, unique=True)
    allee_type = models.CharField(max_length=10, choices=[('WEAK', 'Weak'), ('STRONG', 'Strong')])
    mu = models.FloatField()
    use_adaptive_control = models.BooleanField(default=False)
    
    # Parámetros del escenario (almacenados como JSON)
    params = models.JSONField(default=dict)
    
    # Información del estado estacionario asociado (si existe)
    steady_state = models.ForeignKey(SteadyState, on_delete=models.SET_NULL, 
                                     null=True, blank=True, related_name='scenarios')
    
    # Metadatos
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True, help_text="Si está activo en scenarios.json")
    
    class Meta:
        ordering = ['name']
    
    def __str__(self):
        return f"{self.name} ({self.allee_type}, μ={self.mu})"
