"""
views.py

Vistas Django para procesamiento y visualización de estados estacionarios.
"""

import sys
import os
from pathlib import Path
from urllib.parse import unquote
from typing import Any, Optional, Dict, cast
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse, FileResponse, Http404, HttpResponseNotAllowed
from django.urls import reverse
from django.views.decorators.http import require_http_methods
from django.contrib import messages
from django.utils import timezone
from django.core.cache import cache
from datetime import timedelta
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Usar backend sin GUI
import matplotlib.pyplot as plt
import base64
from io import BytesIO

from .models import AnalysisRun, SteadyState, Scenario
from .forms import Complete3DForm
from .tasks import run_analysis_async
from .experiment_status import (
    annotate_catalog_gates,
    collect_experiment_status,
    merge_scenario_row_into_experiment_cache,
    row_dict_for_scenario_name,
    EXPERIMENT_STATUS_CACHE_KEY,
    EXPERIMENT_STATUS_CACHE_TTL,
)
from .model_lab_context import build_model_equations_context
from .storage_diagnostics import collect_storage_diagnostics
from .scenarios_catalog import (
    get_all_scenario_names,
    load_normalized_catalog,
    pipeline_results_subdir_for_scenario,
)
import threading

# Raíz del código de simulación (Models/Allee): escenarios, steady_states, etc.
from django.conf import settings

ALLEE_ROOT = Path(settings.ALLEE_DIR)
sys.path.insert(0, str(ALLEE_ROOT))

from .lab_paths import get_lab_results_root
from .scenario_figures import (
    ALLOWED_KINDS,
    FigureKind,
    catalog_figure_select_options,
    get_figures_dir,
    is_png_allowed_for_scenario,
    list_scenario_figure_rows,
    list_simulation_field_pngs,
    load_scenario_names_from_json,
    resolve_figure_path,
    resolve_png_path_in_root,
    resolve_simulation_field_png_path,
    scenario_names_whitelist,
)

RESULTS_DIR = get_lab_results_root()
PATHS_AVAILABLE = True

# Caché de contexto SymPy/LaTeX por escenario (invalidación por mtime del archivo de catálogo)
CATALOG_EQ_CTX_CACHE_PREFIX = "catalog_eq_ctx_v1"
CATALOG_EQ_CTX_CACHE_TTL = 300

# Cambiar al directorio Allee para asegurar que las importaciones relativas funcionen
original_cwd = os.getcwd()
try:
    os.chdir(str(ALLEE_ROOT))
except:
    pass

# Importar módulos necesarios
try:
    from steady_states import (
        scan_grid_3d, filter_physical_3d,
        resolve_run_dir, generate_scenarios_from_control_3d
    )
    from model_parameters import load_from_scenarios_json
    MODULES_LOADED = True
    IMPORT_ERROR = None
except ImportError as e:
    import traceback
    error_msg = f"Error importando módulos: {e}\n{traceback.format_exc()}"
    print(error_msg)
    # Guardar error para mostrar en la vista
    MODULES_LOADED = False
    IMPORT_ERROR = error_msg
finally:
    # Restaurar el directorio de trabajo original
    try:
        os.chdir(original_cwd)
    except:
        pass


def _scenarios_json_mtime_ns(scenarios_file: Path) -> int:
    try:
        return int(scenarios_file.stat().st_mtime_ns)
    except OSError:
        return 0


def _catalog_read_scenario_json(scenario_name: str) -> tuple[str, Path, dict, dict]:
    """Devuelve (name, scenarios_file, scenarios_data, selected_scenario_detail) o Http404."""
    name = (scenario_name or "").strip()
    if not name:
        raise Http404("Nombre de escenario vacío")
    scenarios_file = Path(settings.SCENARIOS_JSON_PATH)
    if not scenarios_file.is_file():
        raise Http404("No se encontró el catálogo de escenarios (steady_states_full_run.json)")
    scenarios_data, cat_err = load_normalized_catalog(scenarios_file)
    if cat_err:
        raise Http404(f"No se pudo leer el catálogo de escenarios: {cat_err}")
    raw_list = scenarios_data.get("scenarios") or []
    selected_detail = None
    for s in raw_list:
        if isinstance(s, dict) and s.get("name") == name:
            selected_detail = s
            break
    if selected_detail is None:
        raise Http404("Escenario no definido en el catálogo")
    return name, scenarios_file, scenarios_data, selected_detail


def _catalog_get_scenario_row(name: str) -> dict:
    row = None
    payload = cache.get(EXPERIMENT_STATUS_CACHE_KEY)
    if payload:
        row = next((r for r in payload.get("rows", []) if r.get("name") == name), None)
    if row is None:
        row = row_dict_for_scenario_name(name)
        if row:
            merge_scenario_row_into_experiment_cache(row)
    if row is None:
        raise Http404("Estado de experimento no disponible para este escenario")
    return row


def _get_eq_ctx_cached(
    request,
    scenarios_file: Path,
    scenario_name: str,
    scenarios_data: dict,
) -> dict:
    mkey = _scenarios_json_mtime_ns(scenarios_file)
    key = f"{CATALOG_EQ_CTX_CACHE_PREFIX}:{scenario_name}:{mkey}"
    hit = cache.get(key)
    if hit is not None:
        return hit
    eq_ctx = build_model_equations_context(
        scenarios_file, scenario_name, scenarios_data=scenarios_data
    )
    for msg in eq_ctx.get("flash_errors") or []:
        messages.error(request, msg)
    if eq_ctx.get("model_load_error") is None and not (eq_ctx.get("flash_errors") or []):
        cache.set(key, eq_ctx, CATALOG_EQ_CTX_CACHE_TTL)
    return eq_ctx


def _build_scenario_catalog_modal_ultra_shell_context(scenario_name: str) -> dict:
    """
    Marco del modal sin escaneo de RESULTS_DIR: solo validación vía el catálogo JSON.
    Las pestañas cargan contenido vía GET .../ajax/tab/<slug>/.
    """
    name, scenarios_file, _scenarios_data, _selected = _catalog_read_scenario_json(scenario_name)
    return {
        "scenario_name": name,
        "scenarios_file": str(scenarios_file),
    }


def _figures_catalog_context_fields(
    scenario_name: str, normalized_catalog: Optional[Dict[str, Any]] = None
) -> dict:
    """Campos comunes para la pestaña Figuras (Paper/figures + PNG de simulación en RESULTS_DIR/.../images/)."""
    nm = (scenario_name or "").strip()
    cat = normalized_catalog if normalized_catalog is not None else {}
    nm_results = pipeline_results_subdir_for_scenario(cat, nm)
    figures_root = get_figures_dir()
    results_root = get_lab_results_root()
    figures_catalog_error = None
    selected_figure_row = None
    try:
        fig_rows = list_scenario_figure_rows([nm], figures_root)
        selected_figure_row = fig_rows[0] if fig_rows else None
    except Exception as e:
        figures_catalog_error = str(e)
    figures_png_options: list = []
    simulation_fields_count = 0
    try:
        simulation_fields_count = len(list_simulation_field_pngs(nm_results, results_root))
        figures_png_options = catalog_figure_select_options(
            nm,
            figures_root,
            results_root=results_root,
            normalized_catalog=cat,
        )
    except Exception as e:
        if figures_catalog_error is None:
            figures_catalog_error = str(e)
    return {
        "figures_dir": str(figures_root),
        "figures_dir_exists": figures_root.is_dir(),
        "figures_catalog_error": figures_catalog_error,
        "selected_figure_row": selected_figure_row,
        "figures_png_options": figures_png_options,
        "results_dir_display": str(results_root),
        "simulation_pipeline_folder": nm_results,
        "simulation_fields_count": simulation_fields_count,
    }


def _build_scenario_catalog_context(request, scenario_name: str) -> dict:
    """
    Contexto completo para la página de detalle del catálogo (todas las pestañas server-side).
    """
    name, scenarios_file, scenarios_data, _sel = _catalog_read_scenario_json(scenario_name)
    row = _catalog_get_scenario_row(name)
    eq_ctx = _get_eq_ctx_cached(request, scenarios_file, name, scenarios_data)
    stages_catalog = annotate_catalog_gates(row["stages"])

    fig_fields = _figures_catalog_context_fields(name, scenarios_data)

    return {
        "scenario_name": name,
        "scenario_row": row,
        "stages_catalog": stages_catalog,
        "scenarios_file": str(scenarios_file),
        "sympy_available": eq_ctx.get("sympy_available", False),
        "model_load_error": eq_ctx.get("model_load_error"),
        "latex_Rc": eq_ctx.get("latex_Rc"),
        "latex_Rs": eq_ctx.get("latex_Rs"),
        "latex_Ri": eq_ctx.get("latex_Ri"),
        "params_table": eq_ctx.get("params_table"),
        "params_source": eq_ctx.get("params_source"),
        "allee_type": eq_ctx.get("allee_type"),
        "use_adaptive_control": eq_ctx.get("use_adaptive_control"),
        **fig_fields,
    }


@require_http_methods(["GET"])
def scenario_catalog_tab_ajax(request, scenario_name: str, tab: str):
    """HTML interno de una pestaña del catálogo (carga diferida en el modal)."""
    tab_key = (tab or "").strip().lower()
    templates_map = {
        "ecuaciones": "steady_states_app/includes/scenario_catalog_fragment_equations.html",
        "parametros": "steady_states_app/includes/scenario_catalog_fragment_params.html",
        "figuras": "steady_states_app/includes/scenario_catalog_fragment_figures.html",
        "pipeline": "steady_states_app/includes/scenario_catalog_fragment_pipeline.html",
    }
    template_name = templates_map.get(tab_key)
    if not template_name:
        return HttpResponse("Pestaña no válida", status=404, content_type="text/plain; charset=utf-8")

    try:
        name, scenarios_file, scenarios_data, _sel = _catalog_read_scenario_json(scenario_name)
    except Http404:
        return HttpResponse(
            '<div class="alert alert-danger mb-0">Escenario no disponible.</div>',
            status=404,
            content_type="text/html; charset=utf-8",
        )

    catalog_uid = "dashboardModal"
    ctx: dict = {"catalog_uid": catalog_uid, "scenario_name": name, "scenarios_file": str(scenarios_file)}

    try:
        if tab_key in ("ecuaciones", "parametros"):
            eq_ctx = _get_eq_ctx_cached(request, scenarios_file, name, scenarios_data)
            ctx.update(eq_ctx)
        elif tab_key == "figuras":
            ctx.update(_figures_catalog_context_fields(name, scenarios_data))
        elif tab_key == "pipeline":
            row = _catalog_get_scenario_row(name)
            ctx["scenario_row"] = row
            ctx["stages_catalog"] = annotate_catalog_gates(row["stages"])
    except Http404:
        return HttpResponse(
            '<div class="alert alert-danger mb-0">No se pudo cargar esta pestaña.</div>',
            status=404,
            content_type="text/html; charset=utf-8",
        )

    return render(request, template_name, ctx, content_type="text/html; charset=utf-8")


def scenario_catalog_detail(request, scenario_name: str):
    """
    Catálogo por escenario: ecuaciones, diccionario de parámetros (<code>to_dict</code>) y pipeline en disco.
    """
    context = _build_scenario_catalog_context(request, scenario_name)
    context["catalog_uid"] = "page"
    return render(request, "steady_states_app/scenario_catalog_detail.html", context)


def scenario_catalog_ajax(request, scenario_name: str):
    """
    Fragmento HTML del catálogo para el modal: marco de pestañas (sin escaneo de pipeline);
    contenido vía scenario_catalog_tab_ajax.
    """
    try:
        context = _build_scenario_catalog_modal_ultra_shell_context(scenario_name)
    except Http404:
        return HttpResponse(
            '<div class="alert alert-danger mb-0">No se pudo cargar el catálogo de este escenario.</div>',
            status=404,
            content_type="text/html; charset=utf-8",
        )
    context["catalog_uid"] = "dashboardModal"
    return render(
        request,
        "steady_states_app/includes/scenario_catalog_tabbed_lazy.html",
        context,
        content_type="text/html; charset=utf-8",
    )


def scenario_figures_gallery(request):
    """
    Lista escenarios del catálogo JSON y muestra qué PNG existen en FIGURES_DIR
    (steady_*.png, estabilidad_lineal_*.png).
    """
    scenarios_file = Path(settings.SCENARIOS_JSON_PATH)
    figures_root = get_figures_dir()
    catalog_error = None
    scenario_names: list = []
    rows: list = []

    if scenarios_file.is_file():
        try:
            scenario_names = load_scenario_names_from_json(scenarios_file)
            rows = list_scenario_figure_rows(scenario_names, figures_root)
        except Exception as e:
            catalog_error = str(e)
    else:
        catalog_error = f'No se encontró {scenarios_file}'

    selected = (request.GET.get('scenario') or '').strip()
    if selected and selected not in scenario_names:
        selected = ''
    if not selected and scenario_names:
        selected = scenario_names[0]

    selected_row = next((r for r in rows if r['name'] == selected), None)

    context = {
        'scenarios_file': str(scenarios_file),
        'figures_dir': str(figures_root),
        'figures_dir_exists': figures_root.is_dir(),
        'catalog_error': catalog_error,
        'scenario_rows': rows,
        'scenario_names': scenario_names,
        'selected_scenario': selected,
        'selected_row': selected_row,
    }
    return render(request, 'steady_states_app/scenario_figures.html', context)


def serve_scenario_figure(request, kind: str, scenario_name: str):
    """Sirve un PNG solo si kind es válido y el nombre está en el catálogo."""
    if kind not in ALLOWED_KINDS:
        raise Http404('Tipo de figura no válido')
    scenarios_file = Path(settings.SCENARIOS_JSON_PATH)
    allowed = scenario_names_whitelist(scenarios_file)
    path = resolve_figure_path(
        cast(FigureKind, kind),
        scenario_name,
        allowed,
        get_figures_dir(),
    )
    if path is None or not path.is_file():
        raise Http404('Figura no encontrada')
    return FileResponse(path.open('rb'), content_type='image/png')


@require_http_methods(["GET"])
def serve_scenario_png_basename(request, scenario_name: str, basename: str):
    """
    Sirve un PNG por nombre de archivo bajo FIGURES_DIR, solo si está permitido para el escenario
    (lista alineada con catalog_figure_select_options: canónicos + nombre del escenario en el fichero).
    """
    bn = unquote(basename or "").strip()
    if not bn or "/" in bn or "\\" in bn or bn != Path(bn).name:
        raise Http404("Nombre de archivo no válido")
    if not bn.lower().endswith(".png"):
        raise Http404("Solo PNG")
    sn = (scenario_name or "").strip()
    figures_root = get_figures_dir().resolve()
    if not is_png_allowed_for_scenario(bn, sn, figures_root):
        raise Http404("Archivo no permitido para este escenario")
    path = resolve_png_path_in_root(figures_root, bn)
    if path is None or not path.is_file():
        raise Http404("Figura no encontrada")
    return FileResponse(path.open("rb"), content_type="image/png")


@require_http_methods(["GET"])
def serve_simulation_field_png(request, scenario_name: str, basename: str):
    """
    Sirve ``fields_block_*_step_*.png`` desde RESULTS_DIR/<escenario>/images/
    (salida de cancer_dynamics con SAVE_IMAGES=Y). Solo escenarios listados en el catálogo.
    """
    bn = unquote(basename or "").strip()
    if not bn or "/" in bn or "\\" in bn or bn != Path(bn).name:
        raise Http404("Nombre de archivo no válido")
    if not bn.lower().endswith(".png"):
        raise Http404("Solo PNG")
    scenarios_file = Path(settings.SCENARIOS_JSON_PATH)
    allowed = scenario_names_whitelist(scenarios_file)
    sn = (scenario_name or "").strip()
    if sn not in allowed:
        raise Http404("Escenario no permitido")
    catalog_data, catalog_err = load_normalized_catalog(scenarios_file)
    disk_sn = sn if catalog_err else pipeline_results_subdir_for_scenario(catalog_data, sn)
    path = resolve_simulation_field_png_path(disk_sn, bn, get_lab_results_root())
    if path is None or not path.is_file():
        raise Http404("Figura de simulación no encontrada")
    return FileResponse(path.open("rb"), content_type="image/png")


def storage_health(request):
    """
    Diagnóstico manual de rutas de resultados (RESULTS_DIR / Drive / WSL) y catálogo JSON.
    """
    context = {"storage": collect_storage_diagnostics()}
    return render(request, "steady_states_app/storage_health.html", context)


def dashboard(request):
    """Panel principal: KPIs de EE, escenarios en JSON y estado de experimentos en disco."""
    # Estadísticas (búsquedas EE en Django)
    total_analyses = AnalysisRun.objects.count()
    completed_analyses = AnalysisRun.objects.filter(status='completed').count()
    failed_analyses = AnalysisRun.objects.filter(status='failed').count()
    
    # Escenarios disponibles
    scenarios_file = Path(settings.SCENARIOS_JSON_PATH)
    scenarios_count = 0
    if scenarios_file.exists() and MODULES_LOADED:
        try:
            scenarios_list = get_all_scenario_names(scenarios_file)
            scenarios_count = len(scenarios_list)
        except Exception as e:
            print(f"Error al cargar escenarios: {e}")
            pass
    
    exp_payload = cache.get(EXPERIMENT_STATUS_CACHE_KEY)
    if exp_payload is None:
        exp_payload = collect_experiment_status()
        cache.set(
            EXPERIMENT_STATUS_CACHE_KEY,
            exp_payload,
            EXPERIMENT_STATUS_CACHE_TTL,
        )

    context = {
        'total_analyses': total_analyses,
        'completed_analyses': completed_analyses,
        'failed_analyses': failed_analyses,
        'scenarios_count': scenarios_count,
        'experiment_status': exp_payload,
        'experiment_status_cache_ttl': EXPERIMENT_STATUS_CACHE_TTL,
    }

    return render(request, 'steady_states_app/dashboard.html', context)


def experiment_status_api(request):
    """JSON del estado de experimentos en disco (misma caché que el dashboard). ?refresh=1 invalida caché."""
    if request.GET.get("refresh"):
        cache.delete(EXPERIMENT_STATUS_CACHE_KEY)
    payload = cache.get(EXPERIMENT_STATUS_CACHE_KEY)
    if payload is None:
        payload = collect_experiment_status()
        cache.set(EXPERIMENT_STATUS_CACHE_KEY, payload, EXPERIMENT_STATUS_CACHE_TTL)
    return JsonResponse(payload)


def _analysis_request_is_xhr(request) -> bool:
    return request.headers.get("X-Requested-With") == "XMLHttpRequest"


def analysis_complete_3d_modal(request):
    """
    Fragmento HTML del formulario de EE (c,s,i) para el modal del Dashboard (GET / AJAX).
    """
    if request.method != "GET":
        return HttpResponseNotAllowed(["GET"])
    form = Complete3DForm()
    form.helper.form_tag = False
    return render(
        request,
        "steady_states_app/includes/analysis_complete_3d_modal_body.html",
        {"form": form},
        content_type="text/html; charset=utf-8",
    )


def analysis_config(request, analysis_type='complete-3d'):
    """
    Vista para configurar la búsqueda de estados estacionarios en (c, s, i).

    GET complete-3d redirige al Dashboard (el formulario se abre en modal).
    POST: si es petición AJAX (modal), responde JSON al crear análisis o HTML 422 si hay errores.
    """
    old_routes = ['basic', 'systematic-2d', 'control-3d']
    if analysis_type in old_routes:
        messages.info(
            request,
            f'El análisis "{analysis_type}" ha sido consolidado en la búsqueda de EE en (c, s, i). '
            'Abre el formulario desde el Dashboard.',
        )
        return redirect(f"{reverse('steady_states_app:dashboard')}?abrir_generar_ee=1")

    if analysis_type != 'complete-3d':
        return redirect('steady_states_app:dashboard')

    if request.method == 'POST':
        form = Complete3DForm(request.POST)
        if form.is_valid():
            analysis_run = AnalysisRun.objects.create(
                name=f"EE (c,s,i) — {timezone.now().strftime('%Y-%m-%d %H:%M')}",
                analysis_type='complete_3d',
                status='pending',
                config_params=form.cleaned_data,
            )
            if _analysis_request_is_xhr(request):
                return JsonResponse(
                    {
                        "ok": True,
                        "analysis_id": analysis_run.id,
                        "run_url": reverse(
                            'steady_states_app:run_analysis',
                            args=[analysis_run.id],
                        ),
                    }
                )
            return redirect('steady_states_app:run_analysis', analysis_id=analysis_run.id)
        if _analysis_request_is_xhr(request):
            form.helper.form_tag = False
            return render(
                request,
                "steady_states_app/includes/analysis_complete_3d_modal_body.html",
                {"form": form},
                status=422,
                content_type="text/html; charset=utf-8",
            )
        context = {'form': form, 'analysis_type': 'complete-3d'}
        return render(request, 'steady_states_app/analysis_complete_3d.html', context)

    return redirect(f"{reverse('steady_states_app:dashboard')}?abrir_generar_ee=1")


def run_analysis(request, analysis_id):
    """Ejecuta un análisis y muestra resultados."""
    analysis = get_object_or_404(AnalysisRun, id=analysis_id)
    
    # Verificar si el análisis lleva mucho tiempo ejecutándose (más de 6 horas)
    if analysis.status == 'running' and analysis.started_at:
        time_elapsed = timezone.now() - analysis.started_at
        if time_elapsed > timedelta(hours=6):
            # Resetear análisis que lleva más de 6 horas
            analysis.status = 'failed'
            analysis.completed_at = timezone.now()
            analysis.error_message = f"Análisis cancelado automáticamente: llevaba ejecutándose más de {time_elapsed}"
            analysis.save()
            messages.warning(request, f'El análisis fue cancelado automáticamente por exceder el tiempo máximo (6 horas).')
    
    if request.method == 'POST' or analysis.status == 'pending':
        # Marcar como ejecutando
        analysis.status = 'running'
        analysis.started_at = timezone.now()
        analysis.progress_percent = 0.0
        analysis.progress_message = "Iniciando análisis..."
        analysis.save()
        
        # Ejecutar análisis en thread separado (asíncrono)
        thread = threading.Thread(
            target=run_analysis_async,
            args=(analysis.id,),
            daemon=True
        )
        thread.start()
        
        messages.info(request, 'Análisis iniciado. El progreso se actualizará automáticamente.')
    
    # Obtener estados estacionarios asociados
    steady_states = analysis.steady_states.all()[:100]  # Limitar a 100 para visualización
    
    # Intentar cargar información de parámetros desde JSON si existe
    analysis_info = {}
    if analysis.status == 'completed':
        try:
            django_results_dir = RESULTS_DIR / 'django_analyses'
            json_filename = f'steady_states_{analysis.analysis_type}_{analysis.id}.json'
            json_path = django_results_dir / json_filename
            
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    # Convertir initial_guesses de listas a formato más amigable para templates
                    initial_guesses_raw = json_data.get('initial_guesses', [])
                    initial_guesses_formatted = []
                    for i, guess in enumerate(initial_guesses_raw, 1):
                        if isinstance(guess, (list, tuple)) and len(guess) >= 2:
                            initial_guesses_formatted.append({
                                'point': i,
                                'c0': guess[0],
                                's0': guess[1]
                            })
                    
                    analysis_info = {
                        'base_parameters': json_data.get('base_parameters', {}),
                        'initial_guesses': initial_guesses_formatted,
                        'method_info': json_data.get('method', 'Newton-Raphson'),
                    }
        except Exception as e:
            print(f"Error al cargar información del análisis: {e}")
            pass
    
    context = {
        'analysis': analysis,
        'steady_states': steady_states,
        'analysis_info': analysis_info,
    }
    
    return render(request, 'steady_states_app/results.html', context)


def get_analysis_progress(request, analysis_id):
    """Endpoint JSON para obtener el progreso de un análisis."""
    analysis = get_object_or_404(AnalysisRun, id=analysis_id)
    
    return JsonResponse({
        'status': analysis.status,
        'progress_percent': analysis.progress_percent,
        'progress_message': analysis.progress_message,
        'total_combinations': analysis.total_combinations,
        'processed_combinations': analysis.processed_combinations,
        'started_at': analysis.started_at.isoformat() if analysis.started_at else None,
        'completed_at': analysis.completed_at.isoformat() if analysis.completed_at else None,
    })


def cancel_analysis(request, analysis_id):
    """Cancela un análisis que está ejecutándose."""
    analysis = get_object_or_404(AnalysisRun, id=analysis_id)
    
    if analysis.status == 'running':
        analysis.status = 'failed'
        analysis.completed_at = timezone.now()
        if analysis.started_at:
            time_elapsed = timezone.now() - analysis.started_at
            analysis.error_message = f"Análisis cancelado manualmente después de {time_elapsed}"
        else:
            analysis.error_message = "Análisis cancelado manualmente"
        analysis.progress_message = "Análisis cancelado"
        analysis.save()
        messages.info(request, 'Análisis cancelado exitosamente')
    else:
        messages.warning(request, 'El análisis no está ejecutándose, no se puede cancelar')
    
    return redirect('steady_states_app:run_analysis', analysis_id=analysis_id)


def reset_stuck_analyses(request):
    """Resetea análisis que llevan mucho tiempo ejecutándose."""
    stuck_threshold = timedelta(hours=6)
    stuck_analyses = AnalysisRun.objects.filter(
        status='running',
        started_at__lt=timezone.now() - stuck_threshold
    )
    
    count = 0
    for analysis in stuck_analyses:
        analysis.status = 'failed'
        analysis.completed_at = timezone.now()
        time_elapsed = timezone.now() - analysis.started_at
        analysis.error_message = f"Análisis cancelado automáticamente: llevaba ejecutándose más de {time_elapsed}"
        analysis.save()
        count += 1
    
    if count > 0:
        messages.success(request, f'Se resetearon {count} análisis atascados')
    else:
        messages.info(request, 'No se encontraron análisis atascados')
    
    return redirect('steady_states_app:analysis_list')


def run_complete_3d(analysis):
    """
    Ejecuta la búsqueda de estados estacionarios en (c, s, i).
    
    Condición R_c = R_s = R_i = 0 para c, s, i; resolución numérica tipo Newton-Raphson
    sobre la malla de semillas en el espacio (c, s, i).
    """
    if not MODULES_LOADED:
        raise ImportError(f"No se pudieron cargar los módulos necesarios. Error: {IMPORT_ERROR}")
    
    params = analysis.config_params
    
    # Construir semillas desde los parámetros del formulario
    seeds_c_vals = np.linspace(
        params.get('seeds_c_min', 0.01),
        params.get('seeds_c_max', 2.5),
        params.get('seeds_n_points', 3)
    )
    seeds_s_vals = np.linspace(
        params.get('seeds_s_min', 0.01),
        params.get('seeds_s_max', 2.5),
        params.get('seeds_n_points', 3)
    )
    seeds_i_vals = np.linspace(
        params.get('seeds_i_min', 0.01),
        params.get('seeds_i_max', 2.0),
        params.get('seeds_n_points', 3)
    )
    
    # Malla de semillas en (c, s, i)
    seeds = [
        (c, s, i)
        for c in seeds_c_vals
        for s in seeds_s_vals
        for i in seeds_i_vals
    ]
    
    # Parámetros de control adaptativo (opcional)
    use_control = params.get('use_adaptive_control', False)
    ku_val = params.get('ku', 0.2) if use_control else 0.0
    eps_val = params.get('eps_u', 1e-3) if use_control else 1e-3
    umax_val = params.get('u_max', None) if use_control else None
    
    # Escaneo sobre parámetros y semillas (c, s, i)
    df_raw = scan_grid_3d(
        params.get('rc_vals', [5.0, 6.0]),
        params.get('beta_vals', [5.0, 7.0]),
        params.get('delta_vals', [5.0, 7.0]),
        params.get('eta_vals', [3.0, 5.0]),
        params.get('rd_vals', [9.0, 11.0]),
        params.get('a_vals', [0.1]),
        seeds,
        mu_val=params.get('mu', 1),
        ku_val=ku_val,
        eps_val=eps_val,
        umax_val=umax_val
    )
    
    # Filtrar resultados físicos
    df_filt = filter_physical_3d(df_raw)
    
    # Guardar CSV y JSON
    csv_filename = f'steady_states_complete_3d_{analysis.id}.csv'
    json_filename = f'steady_states_complete_3d_{analysis.id}.json'
    django_results_dir = RESULTS_DIR / 'django_analyses'
    django_results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = django_results_dir / csv_filename
    json_path = django_results_dir / json_filename
    
    # Guardar CSV
    df_filt.to_csv(csv_path, index=False)
    print(f"CSV guardado en: {csv_path}")
    
    # Guardar JSON con información completa
    results_dict = {
        'analysis_id': analysis.id,
        'analysis_type': 'complete_3d',
        'raw_points': len(df_raw),
        'filtered_points': len(df_filt),
        'config_params': params,
        'method': 'Newton-Raphson en (c, s, i): R_c=R_s=R_i=0',
        'equations': 'Tres ecuaciones algebraicas de equilibrio (sin reducción 2D)',
        'results': df_filt.to_dict('records')
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False, default=str)
    print(f"JSON guardado en: {json_path}")
    
    # JSON auxiliar local (Allee) si el modal lo pide; el dashboard solo lee steady_states_full_run (SCENARIOS_JSON_PATH).
    if params.get('generate_scenarios', True) and len(df_filt) > 0:
        scenarios_file = Path(settings.SCENARIOS_JSON_WRITE_PATH)
        generate_scenarios_from_control_3d(
            df_filt,
            scenarios_file,
            max_scenarios=10,
            include_spatial_params=True
        )
    
    return {
        'summary': {
            'type': 'complete_3d',
            'message': 'Búsqueda de EE (c, s, i) completada',
            'raw_points': len(df_raw),
            'filtered_points': len(df_filt),
        },
        'dataframe': df_filt,
        'csv_path': csv_filename,
        'json_path': json_filename,
    }


def save_steady_states_to_db(analysis, df):
    """Guarda estados estacionarios del DataFrame en la base de datos."""
    for _, row in df.iterrows():
        SteadyState.objects.create(
            analysis_run=analysis,
            rc=row.get('rc', 0),
            rs=row.get('rs', 0),
            rd=row.get('rd', 0),
            alpha=row.get('alpha', 0),
            delta=row.get('delta', 0),
            beta=row.get('beta', 0),
            eta=row.get('eta', 0),
            mu=row.get('mu', 0),
            c_star=row.get('c_star', 0),
            s_star=row.get('s_star', 0),
            i_star=row.get('i_star', None),
            eig1_real=row.get('eig1_real', 0),
            eig1_imag=row.get('eig1_imag', 0),
            eig2_real=row.get('eig2_real', 0),
            eig2_imag=row.get('eig2_imag', 0),
            eig3_real=row.get('eig3_real', None),
            eig3_imag=row.get('eig3_imag', 0),
            unstable=row.get('unstable', False),
            max_real=row.get('max_real', row.get('eig1_real', 0)),
        )


def analysis_list(request):
    """Lista todos los análisis ejecutados."""
    analyses = AnalysisRun.objects.all().order_by('-created_at')
    
    context = {
        'analyses': analyses,
    }
    
    return render(request, 'steady_states_app/analysis_list.html', context)


def plot_results(request, analysis_id):
    """Genera gráficos de resultados como imágenes."""
    try:
        analysis = get_object_or_404(AnalysisRun, id=analysis_id)
        
        # Obtener parámetros de tamaño desde la URL (opcionales)
        width = float(request.GET.get('width', 8.75))
        height = float(request.GET.get('height', 5.25))
        dpi_val = int(request.GET.get('dpi', 56))
        
        # Validar parámetros
        width = max(3, min(20, width))  # Limitar entre 3 y 20 pulgadas
        height = max(2, min(15, height))  # Limitar entre 2 y 15 pulgadas
        dpi_val = max(30, min(200, dpi_val))  # Limitar entre 30 y 200 DPI
        
        # Intentar obtener datos del CSV primero (misma lógica que run_scenarios.py)
        df = None
        if analysis.csv_file_path:
            csv_filename = Path(analysis.csv_file_path).name
            # Buscar en RESULTS_DIR/django_analyses (mismo lugar donde se guardan)
            django_results_dir = RESULTS_DIR / 'django_analyses'
            csv_path = django_results_dir / csv_filename
            
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    print(f"CSV cargado desde {csv_path}: {len(df)} filas")
                except Exception as e:
                    print(f"Error leyendo CSV {csv_path}: {e}")
            else:
                print(f"CSV no encontrado en: {csv_path}")
        
        # Si no hay CSV, obtener datos de la base de datos
        if df is None or len(df) == 0:
            steady_states = analysis.steady_states.all()
            print(f"Estados estacionarios en BD: {steady_states.count()}")
            if steady_states.exists():
                data = []
                for ss in steady_states:
                    data.append({
                        'c_star': float(ss.c_star),
                        's_star': float(ss.s_star),
                        'i_star': float(ss.i_star) if ss.i_star is not None else 0.0,
                        'max_real': float(ss.max_real),
                        'unstable': bool(ss.unstable)
                    })
                df = pd.DataFrame(data)
                print(f"DataFrame creado desde BD: {len(df)} filas")
        
        if df is None or len(df) == 0:
            return HttpResponse(
                '<div class="alert alert-warning"><i class="bi bi-exclamation-triangle"></i> No hay datos disponibles para generar el gráfico.</div>',
                content_type='text/html'
            )
        
        # Crear gráfico
        try:
            # Tamaño de figura configurable desde parámetros URL
            fig, ax = plt.subplots(figsize=(width, height))
            
            # Verificar que tenemos las columnas necesarias
            if 'c_star' not in df.columns or 's_star' not in df.columns:
                return HttpResponse(
                    '<div class="alert alert-warning">Los datos no tienen las columnas necesarias (c_star, s_star).</div>',
                    content_type='text/html'
                )
            
            # Colorear según estabilidad si hay datos suficientes
            if 'unstable' in df.columns:
                stable_mask = ~df['unstable']
                unstable_mask = df['unstable']
                
                if stable_mask.any():
                    ax.scatter(df.loc[stable_mask, 'c_star'], 
                              df.loc[stable_mask, 's_star'], 
                              c='green', s=50, alpha=0.7, label='Estable', marker='o')
                if unstable_mask.any():
                    ax.scatter(df.loc[unstable_mask, 'c_star'], 
                              df.loc[unstable_mask, 's_star'], 
                              c='red', s=50, alpha=0.7, label='Inestable', marker='x')
                
                if stable_mask.any() or unstable_mask.any():
                    ax.legend()
            else:
                # Colorear según max_real si está disponible
                if 'max_real' in df.columns:
                    scatter = ax.scatter(df['c_star'], df['s_star'], 
                                       c=df['max_real'], 
                                       cmap='viridis', s=50, alpha=0.7)
                    plt.colorbar(scatter, ax=ax, label='max Re(λ)')
                else:
                    # Solo puntos sin color
                    ax.scatter(df['c_star'], df['s_star'], 
                              s=50, alpha=0.7, color='blue')
            
            ax.set_xlabel('c* (Concentración de células cancerosas)', fontsize=11)
            ax.set_ylabel('s* (Concentración de células sanas)', fontsize=11)
            ax.set_title(f'Estados Estacionarios - {analysis.name}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Convertir a imagen PNG con DPI configurable
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=dpi_val, bbox_inches='tight')
            buffer.seek(0)
            plt.close()
            
            # Devolver como imagen PNG directamente
            response = HttpResponse(buffer.getvalue(), content_type='image/png')
            return response
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error generando gráfico: {e}\n{error_trace}")
            # Crear una imagen de error simple
            try:
                fig, ax = plt.subplots(figsize=(width, height))
                ax.text(0.5, 0.5, f'Error al generar gráfico:\n{str(e)}', 
                       ha='center', va='center', fontsize=11, 
                       transform=ax.transAxes, wrap=True)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=dpi_val, bbox_inches='tight')
                buffer.seek(0)
                plt.close()
                return HttpResponse(buffer.getvalue(), content_type='image/png')
            except:
                # Si incluso el error falla, crear una imagen simple con matplotlib
                try:
                    fig, ax = plt.subplots(figsize=(width, height))
                    ax.text(0.5, 0.5, 'Error al generar gráfico', 
                           ha='center', va='center', fontsize=11, 
                           transform=ax.transAxes)
                    ax.axis('off')
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png', dpi=dpi_val, bbox_inches='tight')
                    buffer.seek(0)
                    plt.close()
                    return HttpResponse(buffer.getvalue(), content_type='image/png')
                except:
                    # Último recurso: devolver un PNG mínimo válido (1x1 píxel transparente)
                    # PNG mínimo válido en base64
                    minimal_png = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==')
                    return HttpResponse(minimal_png, content_type='image/png')
            
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error en plot_results: {e}\n{error_trace}")
        # Devolver una imagen de error
        try:
            # Obtener parámetros de tamaño desde la URL (opcionales)
            width = float(request.GET.get('width', 8.75))
            height = float(request.GET.get('height', 5.25))
            dpi_val = int(request.GET.get('dpi', 56))
            
            # Validar parámetros
            width = max(3, min(20, width))
            height = max(2, min(15, height))
            dpi_val = max(30, min(200, dpi_val))
            
            fig, ax = plt.subplots(figsize=(width, height))
            ax.text(0.5, 0.5, f'Error:\n{str(e)}', 
                   ha='center', va='center', fontsize=11, 
                   transform=ax.transAxes, wrap=True)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=dpi_val, bbox_inches='tight')
            buffer.seek(0)
            plt.close()
            return HttpResponse(buffer.getvalue(), content_type='image/png')
        except:
            # Fallback: crear imagen de error simple
            try:
                # Obtener parámetros de tamaño desde la URL (opcionales)
                width = float(request.GET.get('width', 8.75))
                height = float(request.GET.get('height', 5.25))
                dpi_val = int(request.GET.get('dpi', 56))
                
                # Validar parámetros
                width = max(3, min(20, width))
                height = max(2, min(15, height))
                dpi_val = max(30, min(200, dpi_val))
                
                fig, ax = plt.subplots(figsize=(width, height))
                ax.text(0.5, 0.5, 'Error al generar gráfico', 
                       ha='center', va='center', fontsize=11, 
                       transform=ax.transAxes)
                ax.axis('off')
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=dpi_val, bbox_inches='tight')
                buffer.seek(0)
                plt.close()
                return HttpResponse(buffer.getvalue(), content_type='image/png')
            except:
                # Último recurso: PNG mínimo válido (1x1 píxel transparente)
                minimal_png = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==')
                return HttpResponse(minimal_png, content_type='image/png')
