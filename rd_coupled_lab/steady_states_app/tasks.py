"""
tasks.py

Módulo para ejecución asíncrona de análisis en threads separados.
"""

import threading
from django.utils import timezone
from django.db import transaction
from .models import AnalysisRun
from pathlib import Path
import sys
import os

# Importar módulos necesarios (misma lógica que views.py)
from django.conf import settings

from .lab_paths import get_lab_results_root

ALLEE_ROOT = Path(settings.ALLEE_DIR)
if str(ALLEE_ROOT) not in sys.path:
    sys.path.insert(0, str(ALLEE_ROOT))
original_cwd = os.getcwd()
try:
    os.chdir(str(ALLEE_ROOT))
except Exception:
    pass

try:
    from steady_states import (
        scan_grid_3d, filter_physical_3d,
        generate_scenarios_from_control_3d
    )
    import numpy as np
    import pandas as pd
    from itertools import product

    MODULES_LOADED = True
except ImportError as e:
    MODULES_LOADED = False
    IMPORT_ERROR = str(e)
finally:
    try:
        os.chdir(original_cwd)
    except Exception:
        pass

RESULTS_DIR = get_lab_results_root()


def update_progress(analysis_id, processed, total, message):
    """
    Actualiza el progreso de un análisis en la base de datos.
    
    Args:
        analysis_id: ID del análisis
        processed: Número de combinaciones procesadas
        total: Total de combinaciones
        message: Mensaje de estado
    """
    try:
        with transaction.atomic():
            analysis = AnalysisRun.objects.select_for_update().get(id=analysis_id)
            
            # Verificar si fue cancelado
            if analysis.status != 'running':
                return False
            
            percent = (processed / total * 100) if total > 0 else 0.0
            analysis.progress_percent = percent
            analysis.progress_message = message
            analysis.total_combinations = total
            analysis.processed_combinations = processed
            analysis.save(update_fields=['progress_percent', 'progress_message', 
                                       'total_combinations', 'processed_combinations'])
            return True
    except AnalysisRun.DoesNotExist:
        return False
    except Exception as e:
        print(f"Error actualizando progreso: {e}")
        return False


def run_analysis_async(analysis_id):
    """
    Ejecuta un análisis en un thread separado.
    
    Args:
        analysis_id: ID del análisis a ejecutar
    """
    try:
        # Obtener análisis
        analysis = AnalysisRun.objects.get(id=analysis_id)
        
        # Cambiar directorio de trabajo
        original_cwd = os.getcwd()
        try:
            os.chdir(str(ALLEE_ROOT))
        except:
            pass
        
        try:
            # Importar funciones necesarias desde views
            import sys
            import importlib
            sys.path.insert(0, str(ALLEE_ROOT))
            
            # Importar módulos necesarios
            from steady_states_app import views
            run_complete_3d = views.run_complete_3d
            save_steady_states_to_db = views.save_steady_states_to_db
            
            # Crear callback de progreso
            def progress_callback(processed, total, message):
                return update_progress(analysis_id, processed, total, message)
            
            # Búsqueda de estados estacionarios (c, s, i)
            if analysis.analysis_type == 'complete_3d':
                result = run_complete_3d(analysis)
            else:
                raise ValueError(f"Tipo de análisis desconocido: {analysis.analysis_type}. Solo se admite 'complete_3d'.")
            
            # Guardar resultados
            with transaction.atomic():
                analysis = AnalysisRun.objects.select_for_update().get(id=analysis_id)
                if analysis.status == 'running':  # Solo actualizar si no fue cancelado
                    analysis.status = 'completed'
                    analysis.completed_at = timezone.now()
                    analysis.results_summary = result.get('summary', {})
                    analysis.csv_file_path = result.get('csv_path', '')
                    analysis.json_file_path = result.get('json_path', '')
                    analysis.progress_percent = 100.0
                    analysis.progress_message = "Análisis completado exitosamente"
                    analysis.save()
                    
                    # Guardar estados estacionarios en BD
                    if 'dataframe' in result:
                        save_steady_states_to_db(analysis, result['dataframe'])
        
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error en análisis {analysis_id}: {e}\n{error_trace}")
            
            # Actualizar estado a fallido
            with transaction.atomic():
                try:
                    analysis = AnalysisRun.objects.select_for_update().get(id=analysis_id)
                    analysis.status = 'failed'
                    analysis.completed_at = timezone.now()
                    analysis.error_message = str(e)
                    analysis.progress_message = f"Error: {str(e)}"
                    analysis.save()
                except:
                    pass
        
        finally:
            # Restaurar directorio de trabajo
            try:
                os.chdir(original_cwd)
            except:
                pass
                
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error crítico en run_analysis_async: {e}\n{error_trace}")
        
        # Intentar marcar como fallido
        try:
            with transaction.atomic():
                analysis = AnalysisRun.objects.select_for_update().get(id=analysis_id)
                analysis.status = 'failed'
                analysis.completed_at = timezone.now()
                analysis.error_message = f"Error crítico: {str(e)}"
                analysis.progress_message = f"Error crítico: {str(e)}"
                analysis.save()
        except:
            pass



