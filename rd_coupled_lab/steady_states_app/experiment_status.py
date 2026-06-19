"""
Estado de experimentos en disco (solo lectura), alineado con run_scenarios.check_scenario_status.

Usa la misma fórmula de matrices esperadas: expected_steps = int(T/dt)+1,
expected_matrices = expected_steps * 3 * nb, tolerancia baja ~90 %.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from django.conf import settings
from django.core.cache import cache

from .lab_paths import get_lab_results_root
from .scenarios_catalog import load_catalog_for_lab

logger = logging.getLogger(__name__)

# Clave de caché Django (ver views.dashboard)
EXPERIMENT_STATUS_CACHE_KEY = "experiment_status_payload_v2"
EXPERIMENT_STATUS_CACHE_TTL = 90  # segundos


def _merge_params(scenario: Dict[str, Any], common_params: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(common_params)
    out.update(scenario)
    return out


def _expected_matrices_and_correlations(
    scenario: Dict[str, Any], common_params: Dict[str, Any]
) -> Tuple[int, int, int]:
    """(expected_matrices, expected_correlations, expected_steps) como en run_scenarios."""
    p = _merge_params(scenario, common_params)
    T = float(p.get("T", 0.05))
    dt = float(p.get("dt", 0.001))
    nb = int(p.get("nb", 1))
    expected_steps = int(T / dt) + 1
    expected_matrices = expected_steps * 3 * nb
    expected_correlations = 6 * nb
    return expected_matrices, expected_correlations, expected_steps


def _substep(label: str, ok: bool, detail: str = "") -> Dict[str, Any]:
    return {"label": label, "ok": ok, "detail": detail}


def _stage(
    stage_id: str,
    label: str,
    status: str,
    detail: str = "",
    percent: float | None = None,
    steps: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    return {
        "id": stage_id,
        "label": label,
        "status": status,
        "detail": detail,
        "percent": percent,
        "steps": steps or [],
    }


def stage_satisfies_catalog_gate(status: str) -> bool:
    """Etapas complete o na desbloquean la siguiente en la UI del catálogo."""
    return status in ("complete", "na")


def annotate_catalog_gates(stages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Añade catalog_unlocked y blocked_by_stage_label por orden canónico del pipeline.
    """
    out: List[Dict[str, Any]] = []
    for i, st in enumerate(stages):
        prev_ok = all(stage_satisfies_catalog_gate(stages[j]["status"]) for j in range(i))
        blocked_label: str | None = None
        if not prev_ok:
            for j in range(i):
                if not stage_satisfies_catalog_gate(stages[j]["status"]):
                    blocked_label = stages[j]["label"]
                    break
        row = dict(st)
        row["catalog_unlocked"] = prev_ok
        row["blocked_by_stage_label"] = blocked_label
        out.append(row)
    return out


def _count_files(d: Path, pattern: str = "*") -> int:
    if not d.is_dir():
        return 0
    return sum(1 for _ in d.glob(pattern) if _.is_file())


def _probe_results_root(results_root: Path) -> Tuple[bool, str | None]:
    if not results_root.exists():
        return False, f"La ruta no existe: {results_root}"
    if not results_root.is_dir():
        return False, f"No es un directorio: {results_root}"
    try:
        next(results_root.iterdir())
    except StopIteration:
        return True, None  # vacío pero legible
    except OSError as e:
        return False, f"No se puede leer el directorio ({e}): {results_root}"
    return True, None


def build_scenario_stages(
    scenario_dir: Path,
    scenario: Dict[str, Any],
    common_params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Orden canónico del pipeline (la UI del catálogo exige completar o marcar N/A cada etapa
    antes de considerar desbloqueada la siguiente):

    1. simulation — matrices c, s, i
    2. images
    3. thermodynamics
    4. correlations
    5. nonequilibrium
    6. reciprocity
    """
    stages: List[Dict[str, Any]] = []
    exp_m, exp_corr, exp_steps = _expected_matrices_and_correlations(scenario, common_params)

    matrices_dir = scenario_dir / "matrices"
    matrix_n = len(list(matrices_dir.glob("matrix_*.txt"))) if matrices_dir.is_dir() else 0
    tolerance_low = exp_m * 0.9
    has_matrix_dir = matrices_dir.is_dir()
    step_m_dir = _substep("Directorio matrices/", has_matrix_dir, str(matrices_dir) if has_matrix_dir else "No existe matrices/")
    step_m_files = _substep("Archivos matrix_*.txt", matrix_n > 0, f"{matrix_n} archivo(s)")
    at_least_90_m = exp_m <= 0 or matrix_n >= tolerance_low
    step_m_quota = _substep(
        "Cobertura ≥90% del total esperado (T, dt, nb)",
        at_least_90_m,
        f"{matrix_n} / ~{exp_m} matrices; pasos temporales esperados ≈ {exp_steps}",
    )

    if not matrices_dir.exists():
        stages.append(
            _stage(
                "simulation",
                "Simulación (matrices c, s, i)",
                "missing",
                "No existe carpeta matrices/",
                percent=0.0,
                steps=[step_m_dir, step_m_files, step_m_quota],
            )
        )
    elif matrix_n == 0:
        stages.append(
            _stage(
                "simulation",
                "Simulación (matrices c, s, i)",
                "missing",
                "Sin archivos matrix_*.txt",
                percent=0.0,
                steps=[step_m_dir, step_m_files, step_m_quota],
            )
        )
    else:
        pct = min(100.0, (matrix_n / exp_m) * 100.0) if exp_m > 0 else 100.0
        if matrix_n < tolerance_low:
            st = "partial"
            det = f"{matrix_n} / ~{exp_m} matrices (pasos esperados ≈ {exp_steps})"
        else:
            st = "complete"
            det = f"{matrix_n} matrices (~{exp_m} esperadas, T/dt→pasos≈{exp_steps})"
        stages.append(
            _stage(
                "simulation",
                "Simulación (matrices c, s, i)",
                st,
                det,
                percent=round(pct, 1),
                steps=[step_m_dir, step_m_files, step_m_quota],
            )
        )

    images_dir = scenario_dir / "images"
    img_n = _count_files(images_dir)
    has_img_dir = images_dir.is_dir()
    step_i_dir = _substep("Directorio images/", has_img_dir, str(images_dir) if has_img_dir else "No existe images/")
    step_i_any = _substep("Al menos un archivo exportado", img_n > 0, f"{img_n} archivo(s)")
    ref_90_img = max(1, int(exp_steps * 0.9))
    img_at_ref = img_n >= ref_90_img
    step_i_ref = _substep(
        "Cobertura ~90% respecto a pasos temporales (referencia)",
        img_at_ref,
        f"{img_n} vs referencia ≈ {exp_steps} pasos (umbral ≥{ref_90_img})",
    )
    img_steps = [step_i_dir, step_i_any, step_i_ref]

    if img_n == 0 and not images_dir.exists():
        stages.append(_stage("images", "Imágenes", "missing", "Sin carpeta images/", steps=img_steps))
    elif img_n == 0:
        stages.append(_stage("images", "Imágenes", "missing", "Carpeta vacía", steps=img_steps))
    else:
        img_pct = min(100.0, (img_n / max(1, exp_steps)) * 100.0)
        if img_n >= max(1, int(exp_steps * 0.9)):
            stages.append(
                _stage(
                    "images",
                    "Imágenes",
                    "complete",
                    f"{img_n} archivo(s) (referencia ≈ {exp_steps} pasos)",
                    percent=round(img_pct, 1),
                    steps=img_steps,
                )
            )
        elif img_n >= 1:
            stages.append(
                _stage(
                    "images",
                    "Imágenes",
                    "partial",
                    f"{img_n} archivo(s); referencia ≈ {exp_steps} pasos",
                    percent=round(img_pct, 1),
                    steps=img_steps,
                )
            )
        else:
            stages.append(_stage("images", "Imágenes", "missing", "Sin imágenes", steps=img_steps))

    thermo = scenario_dir / "thermodynamics"
    thermo_txt = list(thermo.glob("*.txt")) if thermo.is_dir() else []
    thermo_imgs = _count_files(thermo / "images", "*") if (thermo / "images").is_dir() else 0
    key_names = {"free_energy_F_t.txt", "entropy_production_sigma_t.txt"}
    present_keys = {f.name for f in thermo_txt} & key_names
    has_thermo_dir = thermo.is_dir()
    step_th_dir = _substep("Directorio thermodynamics/", has_thermo_dir, str(thermo) if has_thermo_dir else "—")
    step_th_out = _substep(
        "Salidas .txt o figuras en thermodynamics/images/",
        len(thermo_txt) > 0 or thermo_imgs > 0,
        f"{len(thermo_txt)} .txt, {thermo_imgs} figura(s)",
    )
    step_th_keys = _substep(
        "Series clave F y σ (free_energy_F_t.txt, entropy_production_sigma_t.txt)",
        len(present_keys) >= 2,
        f"Presentes: {sorted(present_keys) or 'ninguno'}",
    )
    th_steps = [step_th_dir, step_th_out, step_th_keys]

    if not thermo.exists():
        stages.append(_stage("thermodynamics", "Termodinámica efectiva", "missing", "Sin carpeta thermodynamics/", steps=th_steps))
    elif len(thermo_txt) == 0 and thermo_imgs == 0:
        stages.append(_stage("thermodynamics", "Termodinámica efectiva", "missing", "Sin salidas .txt ni images/", steps=th_steps))
    elif len(present_keys) >= 2:
        stages.append(
            _stage(
                "thermodynamics",
                "Termodinámica efectiva",
                "complete",
                f"{len(thermo_txt)} .txt, {thermo_imgs} en thermodynamics/images/",
                steps=th_steps,
            )
        )
    elif len(thermo_txt) >= 1 or thermo_imgs > 0:
        stages.append(
            _stage(
                "thermodynamics",
                "Termodinámica efectiva",
                "partial",
                f"{len(thermo_txt)} .txt, {thermo_imgs} figuras",
                steps=th_steps,
            )
        )
    else:
        stages.append(_stage("thermodynamics", "Termodinámica efectiva", "missing", "Sin archivos reconocidos", steps=th_steps))

    corr_dir = scenario_dir / "correlations"
    corr_n = len(list(corr_dir.glob("corr_length_*.txt"))) if corr_dir.is_dir() else 0
    has_corr_dir = corr_dir.is_dir()
    step_co_pre = _substep("Prerrequisito: simulación con matrices", matrix_n > 0, f"matrix_*.txt: {matrix_n}")
    step_co_dir = _substep("Directorio correlations/", has_corr_dir, str(corr_dir) if has_corr_dir else "—")
    step_co_n = _substep(
        f"Archivos corr_length_*.txt ({corr_n} / {exp_corr})",
        corr_n >= exp_corr,
        f"Esperados {exp_corr} (6×nb)",
    )

    if matrix_n == 0:
        stages.append(
            _stage(
                "correlations",
                "Correlaciones",
                "na",
                "Sin matrices; correlaciones no aplica aún",
                steps=[
                    _substep("Prerrequisito: simulación con matrices", False, "Sin matrix_*.txt"),
                    step_co_dir,
                    step_co_n,
                ],
            )
        )
    elif not corr_dir.exists():
        stages.append(
            _stage(
                "correlations",
                "Correlaciones",
                "missing",
                "Sin carpeta correlations/",
                steps=[step_co_pre, step_co_dir, step_co_n],
            )
        )
    elif corr_n < exp_corr:
        stages.append(
            _stage(
                "correlations",
                "Correlaciones",
                "partial",
                f"{corr_n} / {exp_corr} corr_length_*.txt",
                steps=[step_co_pre, step_co_dir, step_co_n],
            )
        )
    else:
        stages.append(
            _stage(
                "correlations",
                "Correlaciones",
                "complete",
                f"{corr_n} archivos corr_length_*.txt (esperados {exp_corr})",
                steps=[step_co_pre, step_co_dir, step_co_n],
            )
        )

    ne_dir = scenario_dir / "nonequilibrium_plots"
    ne_n = _count_files(ne_dir, "*.png") + _count_files(ne_dir, "*.pdf")
    has_ne = ne_dir.is_dir()
    step_ne_dir = _substep("Directorio nonequilibrium_plots/", has_ne, str(ne_dir) if has_ne else "—")
    step_ne_files = _substep("Figuras .png o .pdf", ne_n > 0, f"{ne_n} archivo(s)")
    ne_steps = [step_ne_dir, step_ne_files]

    if ne_n == 0 and not ne_dir.exists():
        stages.append(_stage("nonequilibrium", "TdC espacial / flujos", "missing", "Sin nonequilibrium_plots/", steps=ne_steps))
    elif ne_n == 0:
        stages.append(_stage("nonequilibrium", "TdC espacial / flujos", "missing", "Carpeta vacía", steps=ne_steps))
    else:
        stages.append(_stage("nonequilibrium", "TdC espacial / flujos", "complete", f"{ne_n} figura(s)", steps=ne_steps))

    recip_patterns = list(scenario_dir.glob("**/reciprocity*.json")) + list(scenario_dir.glob("**/reciprocity*.txt"))
    step_rec_find = _substep(
        "Artefactos reciprocity*.json o reciprocity*.txt bajo el escenario",
        bool(recip_patterns),
        f"{len(recip_patterns)} hallado(s)" if recip_patterns else "Ninguno (opcional según CLI)",
    )
    if recip_patterns:
        stages.append(
            _stage(
                "reciprocity",
                "Reciprocidad (jacobiano)",
                "complete",
                f"{len(recip_patterns)} artefacto(s) reciprocity*",
                steps=[step_rec_find],
            )
        )
    else:
        stages.append(
            _stage(
                "reciprocity",
                "Reciprocidad (jacobiano)",
                "na",
                "El script CLI no guarda archivo fijo; añade reciprocity*.json si quieres marcar hecho",
                steps=[step_rec_find],
            )
        )

    return stages


def build_experiment_row_dict(
    name: str,
    scenario: Dict[str, Any],
    common: Dict[str, Any],
    results_root: Path,
) -> Dict[str, Any]:
    """
    Una fila del dashboard / API: escaneo de disco solo para este escenario.
    """
    disk_folder = ""
    if isinstance(scenario, dict):
        disk_folder = str(scenario.get("pipeline_folder") or "").strip()
    if disk_folder:
        sdir = results_root / disk_folder
    else:
        sdir = results_root / str(name)
    dir_exists = sdir.is_dir()
    stages = build_scenario_stages(sdir, scenario, common) if dir_exists else _missing_scenario_stages(name)
    n_complete = sum(1 for x in stages if x.get("status") == "complete")
    n_na = sum(1 for x in stages if x.get("status") == "na")
    return {
        "name": name,
        "pipeline_folder": disk_folder if disk_folder else "",
        "scenario_dir_exists": dir_exists,
        "scenario_dir": str(sdir),
        "stages": stages,
        "pipeline_complete_count": n_complete,
        "pipeline_na_count": n_na,
        "pipeline_total_stages": len(stages),
    }


# Abreviaturas para la micro-barra en tarjetas del dashboard (orden = build_scenario_stages).
_PIPELINE_STAGE_ABBREV: Dict[str, str] = {
    "simulation": "Sim",
    "images": "Img",
    "thermodynamics": "TdC",
    "correlations": "Corr",
    "nonequilibrium": "NE",
    "reciprocity": "Rec",
}


def _row_pipeline_all_done(stages: List[Dict[str, Any]]) -> bool:
    """True si cada etapa está completa o marcada N/A (desbloqueo canónico del catálogo)."""
    for st in stages or []:
        if st.get("status") not in ("complete", "na"):
            return False
    return bool(stages)


def merge_pipeline_into_dashboard_groups(
    groups: List[Dict[str, Any]], rows: List[Dict[str, Any]]
) -> None:
    """Añade ``eq['pipeline']`` a cada equilibrio según ``rows`` (nombre plano del catálogo)."""
    by_name = {r["name"]: r for r in rows if r.get("name")}
    for g in groups:
        for eq in g.get("equilibria") or []:
            if not isinstance(eq, dict):
                continue
            cn = eq.get("catalog_name")
            row = by_name.get(cn) if cn else None
            if not row:
                eq["pipeline"] = None
                continue
            stages = row.get("stages") or []
            slim = []
            for st in stages:
                sid = str(st.get("id") or "")
                slim.append(
                    {
                        "id": sid,
                        "label": str(st.get("label") or sid),
                        "abbr": _PIPELINE_STAGE_ABBREV.get(sid, sid[:3].upper() if sid else "?"),
                        "status": str(st.get("status") or "missing"),
                        "percent": st.get("percent"),
                    }
                )
            eq["pipeline"] = {
                "folder_ok": bool(row.get("scenario_dir_exists")),
                "disk_folder": str(row.get("pipeline_folder") or ""),
                "scenario_dir": str(row.get("scenario_dir") or ""),
                "complete_done": int(row.get("pipeline_complete_count") or 0),
                "na_done": int(row.get("pipeline_na_count") or 0),
                "total_stages": int(row.get("pipeline_total_stages") or 0),
                "all_done": _row_pipeline_all_done(stages),
                "stages": slim,
            }


def compute_pipeline_rollups(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Conteos por etapa del pipeline (escenarios = filas del catálogo con escaneo de disco).

    Cada etapa incluye ``segments`` con porcentajes para barras apiladas en la UI.
    """
    empty: Dict[str, Any] = {
        "n_scenarios": 0,
        "n_folder_present": 0,
        "n_pipeline_all_ok": 0,
        "stages": [],
    }
    if not rows:
        return empty

    n = len(rows)
    n_folder = sum(1 for r in rows if r.get("scenario_dir_exists"))
    n_all_ok = sum(
        1
        for r in rows
        if _row_pipeline_all_done(r.get("stages") or [])
    )

    first_stages = rows[0].get("stages") or []
    stage_ids = [str(s.get("id") or "") for s in first_stages]
    labels = {str(s.get("id") or ""): str(s.get("label") or s.get("id")) for s in first_stages}

    tallies: Dict[str, Dict[str, int]] = {
        sid: {"complete": 0, "partial": 0, "missing": 0, "na": 0} for sid in stage_ids if sid
    }
    for r in rows:
        for st in r.get("stages") or []:
            sid = str(st.get("id") or "")
            if sid not in tallies:
                continue
            status = str(st.get("status") or "missing")
            if status in tallies[sid]:
                tallies[sid][status] += 1

    status_order = ("complete", "partial", "missing", "na")
    css = {
        "complete": "bg-success",
        "partial": "bg-warning",
        "missing": "bg-danger",
        "na": "bg-secondary bg-opacity-50",
    }
    stages_out: List[Dict[str, Any]] = []
    for sid in stage_ids:
        if not sid:
            continue
        t = tallies.get(sid, {})
        segments: List[Dict[str, Any]] = []
        for key in status_order:
            count = int(t.get(key, 0))
            if count <= 0:
                continue
            pct = round(100.0 * count / n, 1) if n else 0.0
            segments.append(
                {
                    "key": key,
                    "n": count,
                    "pct": pct,
                    "css": css[key],
                    "label_es": {
                        "complete": "Completo",
                        "partial": "Parcial",
                        "missing": "Falta",
                        "na": "N/A",
                    }[key],
                }
            )
        stages_out.append(
            {
                "id": sid,
                "label": labels.get(sid, sid),
                "abbr": _PIPELINE_STAGE_ABBREV.get(sid, sid[:3].upper()),
                "complete": t.get("complete", 0),
                "partial": t.get("partial", 0),
                "missing": t.get("missing", 0),
                "na": t.get("na", 0),
                "n": n,
                "segments": segments,
            }
        )

    return {
        "n_scenarios": n,
        "n_folder_present": n_folder,
        "n_pipeline_all_ok": n_all_ok,
        "stages": stages_out,
    }


def _experiment_status_payload_start(results_root: Path) -> Dict[str, Any]:
    catalog_path = str(Path(settings.SCENARIOS_JSON_PATH))
    ok, err = _probe_results_root(results_root)
    payload: Dict[str, Any] = {
        "results_root": str(results_root),
        "results_root_ok": ok,
        "results_root_error": err,
        "results_root_empty": False,
        "scenarios_catalog_path": catalog_path,
        "scenarios_json_missing": False,
        "scenarios_json_error": None,
        "scenario_dashboard_groups": [],
        "rows": [],
        "scenario_names": [],
        "pipeline_rollups": None,
    }

    # Catálogo (Drive / steady_states_full_run.json): cargar siempre que exista el archivo.
    # Antes las tarjetas del dashboard quedaban vacías si RESULTS_DIR no era accesible aunque el JSON fuera válido.
    scenarios_path = Path(settings.SCENARIOS_JSON_PATH)
    if scenarios_path.exists():
        data, cat_err, dash_groups = load_catalog_for_lab(scenarios_path)
        if cat_err:
            logger.error("No se pudo leer el catálogo de escenarios: %s", cat_err)
            payload["scenarios_json_error"] = cat_err
        else:
            payload["scenario_dashboard_groups"] = dash_groups

            common = data.get("common_params") or {}
            if not isinstance(common, dict):
                common = {}
            raw_scenarios = data.get("scenarios") or []

            named: List[Tuple[str, Dict[str, Any]]] = []
            for sc in raw_scenarios:
                if not isinstance(sc, dict):
                    continue
                raw_name = sc.get("name")
                if not raw_name:
                    continue
                nm = str(raw_name).strip()
                if not nm:
                    continue
                named.append((nm, sc))

            payload["scenario_names"] = [n for n, _ in named]
            payload["_named_scenarios"] = named
            payload["_common_params"] = common
    else:
        payload["scenarios_json_missing"] = True

    if not ok:
        return payload

    try:
        empty = not any(results_root.iterdir())
    except OSError as e:
        payload["results_root_ok"] = False
        payload["results_root_error"] = str(e)
        return payload

    payload["results_root_empty"] = empty
    return payload


def collect_experiment_shell() -> Dict[str, Any]:
    """
    Metadatos rápidos sin escanear cada carpeta de escenario (solo catálogo JSON y raíz).
    Incluye scenario_names; rows queda vacío hasta cargar por fragmento / fila.
    """
    results_root = get_lab_results_root()
    payload = _experiment_status_payload_start(results_root)
    payload.pop("_named_scenarios", None)
    payload.pop("_common_params", None)
    return payload


def collect_experiment_status() -> Dict[str, Any]:
    """
    Escanea results_root y el catálogo (steady_states_full_run.json en la ruta configurada). No escribe en disco.
    """
    results_root = get_lab_results_root()
    payload = _experiment_status_payload_start(results_root)

    if not payload.get("results_root_ok") or payload.get("scenarios_json_missing") or payload.get("scenarios_json_error"):
        payload.pop("_named_scenarios", None)
        payload.pop("_common_params", None)
        payload["pipeline_rollups"] = compute_pipeline_rollups([])
        merge_pipeline_into_dashboard_groups(payload["scenario_dashboard_groups"], [])
        return payload

    named = payload.pop("_named_scenarios", [])
    common = payload.pop("_common_params", {})

    for name, sc in named:
        payload["rows"].append(build_experiment_row_dict(name, sc, common, results_root))

    payload["pipeline_rollups"] = compute_pipeline_rollups(payload["rows"])
    merge_pipeline_into_dashboard_groups(payload["scenario_dashboard_groups"], payload["rows"])

    return payload


def row_dict_for_scenario_name(scenario_name: str) -> Dict[str, Any] | None:
    """
    Construye la fila de un escenario leyendo el catálogo una vez.
    None si el nombre no existe o hay error de configuración previo al listado.
    """
    name = (scenario_name or "").strip()
    if not name:
        return None
    results_root = get_lab_results_root()
    payload = _experiment_status_payload_start(results_root)
    if not payload.get("results_root_ok") or payload.get("scenarios_json_missing") or payload.get("scenarios_json_error"):
        return None
    named = payload.get("_named_scenarios") or []
    common = payload.get("_common_params") or {}
    for nm, sc in named:
        if nm == name:
            return build_experiment_row_dict(nm, sc, common, results_root)
    return None


def merge_scenario_row_into_experiment_cache(row: Dict[str, Any]) -> None:
    """
    Inserta o actualiza una fila en el payload cacheado de experiment status (sin reescanear todos los escenarios).
    """
    if not row or not row.get("name"):
        return
    payload = cache.get(EXPERIMENT_STATUS_CACHE_KEY)
    if not isinstance(payload, dict):
        return
    rows = list(payload.get("rows") or [])
    nm = row["name"]
    idx = next((i for i, r in enumerate(rows) if r.get("name") == nm), None)
    if idx is not None:
        rows[idx] = row
    else:
        rows.append(row)
    payload["rows"] = rows
    payload["pipeline_rollups"] = compute_pipeline_rollups(rows)
    merge_pipeline_into_dashboard_groups(payload.get("scenario_dashboard_groups") or [], rows)
    cache.set(EXPERIMENT_STATUS_CACHE_KEY, payload, EXPERIMENT_STATUS_CACHE_TTL)


def _missing_scenario_stages(name: str) -> List[Dict[str, Any]]:
    return [
        _stage(
            "simulation",
            "Simulación (matrices c, s, i)",
            "missing",
            f"Sin carpeta para «{name}»",
            steps=[
                _substep("Directorio del escenario bajo RESULTS_DIR", False, f"Esperado: …/{name}/"),
                _substep("Directorio matrices/", False, "—"),
                _substep("Cobertura de matrices vs T, dt, nb", False, "—"),
            ],
        ),
        _stage(
            "images",
            "Imágenes",
            "missing",
            "—",
            steps=[
                _substep("Directorio images/", False, "—"),
                _substep("Al menos un archivo exportado", False, "—"),
                _substep("Cobertura ~90% vs pasos temporales", False, "—"),
            ],
        ),
        _stage(
            "thermodynamics",
            "Termodinámica efectiva",
            "missing",
            "—",
            steps=[
                _substep("Directorio thermodynamics/", False, "—"),
                _substep("Salidas .txt o figuras", False, "—"),
                _substep("Series F y σ", False, "—"),
            ],
        ),
        _stage(
            "correlations",
            "Correlaciones",
            "missing",
            "—",
            steps=[
                _substep("Prerrequisito: simulación con matrices", False, "—"),
                _substep("Directorio correlations/", False, "—"),
                _substep("Archivos corr_length_*.txt completos", False, "—"),
            ],
        ),
        _stage(
            "nonequilibrium",
            "TdC espacial / flujos",
            "missing",
            "—",
            steps=[
                _substep("Directorio nonequilibrium_plots/", False, "—"),
                _substep("Figuras .png o .pdf", False, "—"),
            ],
        ),
        _stage(
            "reciprocity",
            "Reciprocidad (jacobiano)",
            "na",
            "—",
            steps=[_substep("Artefactos reciprocity*", False, "N/A hasta tener carpeta de escenario")],
        ),
    ]
