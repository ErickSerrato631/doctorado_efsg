"""
Catálogo de escenarios: por defecto ``steady_states_full_run.json`` (Drive u otra ruta vía env).

Si el JSON tiene forma clásica ``{common_params, scenarios}`` (otro archivo apuntado por env),
también se normaliza a {common_params, scenarios} para el laboratorio. La lectura **no** usa
``Models/Allee/scenarios.json`` (resuelto en settings).

Para **generar** el JSON unificado en Drive a partir de ``Models/Allee/scenarios_v1.json`` y
``scenarios.json``: ``python manage.py merge_steady_catalog_to_drive``.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from django.conf import settings

# Bloque escenario+hoja de rejilla tipo ``strong_mu0_uNo_bajo_umbral_c0_s1_i0`` (Newton / hybrid).
_DASHBOARD_GROUP_BRANCH_SUFFIX_RE = re.compile(r"^.+_c\d+_s\d+_i\d+$")


def _slug_num(val: Any, decimals: int = 3) -> str:
    if val is None:
        return "na"
    try:
        x = float(val)
    except (TypeError, ValueError):
        return "na"
    if not math.isfinite(x):
        return "na"
    return f"{x:.{decimals}f}".replace(".", "p")


def steady_state_scenario_name(rec: Dict[str, Any]) -> str:
    at = str(rec.get("allee_type") or "WEAK").lower()
    if at not in ("weak", "strong"):
        at = "weak"
    mu = int(rec.get("mu") or 0)
    u_part = "uSi" if _use_adaptive_control_on(rec) else "uNo"
    try:
        a_val = float(rec.get("a") or 0.1)
    except (TypeError, ValueError):
        a_val = 0.1
    try:
        c_star = float(rec.get("c_star") or 0.0)
    except (TypeError, ValueError):
        c_star = 0.0
    um = "bajo_umbral" if c_star < a_val else "sobre_umbral"
    return f"{at}_mu{mu}_{u_part}_{um}"


def steady_state_equilibrium_slug(rec: Dict[str, Any]) -> str:
    return (
        f"rc{_slug_num(rec.get('rc'))}_b{_slug_num(rec.get('beta'))}"
        f"_d{_slug_num(rec.get('delta'))}_e{_slug_num(rec.get('eta'))}"
        f"_rd{_slug_num(rec.get('rd'))}_c{_slug_num(rec.get('c_star'))}"
        f"_s{_slug_num(rec.get('s_star'))}_i{_slug_num(rec.get('i_star'))}"
    )


def steady_state_equilibrium_name(rec: Dict[str, Any]) -> str:
    return f"{steady_state_scenario_name(rec)}_{steady_state_equilibrium_slug(rec)}"


def _default_common_params() -> Dict[str, str]:
    return {
        "rc": "5.84",
        "rs": "13.12",
        "rd": "10.92",
        "alpha": "10.22",
        "delta": "5.40",
        "beta": "7.6",
        "alle": "0.1",
        "gamma": "0.74",
        "eta": "5.08",
        "mu": "1",
        "ALLEE_TYPE": "WEAK",
        "USE_ADAPTIVE_CONTROL": "Y",
        "U_MAX": "0.5",
        "HILL_KC": "0.05",
        "HILL_NC": "2",
        "HILL_KI": "0.2",
        "HILL_NI": "2",
        "KU": "0.2",
        "EPS_U": "0.001",
        "D_c": "0.012",
        "D_s": "0.022",
        "D_i": "0.022",
        "dt": "0.001",
        "T": "2",
        "nodes_in_xaxis": "100",
        "nodes_in_yaxis": "100",
        "space_size": "4",
        "nb": "1",
        "sample_rate": "0.02",
        "SAVE_IMAGES": "Y",
    }


def _str_cell(v: Any, fallback: Optional[str] = None) -> str:
    if v is None:
        return fallback if fallback is not None else ""
    if isinstance(v, bool):
        return "Y" if v else "N"
    return str(v)


def _hill_on(rec: Dict[str, Any]) -> bool:
    hc = rec.get("hill_control")
    if isinstance(hc, bool):
        return hc
    if isinstance(hc, str):
        return hc.strip().upper() in ("Y", "TRUE", "1", "YES")
    return bool(hc)


def _use_adaptive_control_on(rec: Dict[str, Any]) -> bool:
    """Alineado con filas Newton / JSON: ``uSi`` vs ``uNo`` en el nombre corto."""
    v = rec.get("use_adaptive_control")
    if isinstance(v, str):
        return v.strip().upper() in ("Y", "TRUE", "1", "YES")
    return bool(v)


def _clean_steady_state_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(rec)
    out.pop("equilibrium_index", None)
    out.pop("equilibrium_slug", None)
    return out


def _row_to_scenario_dict(
    row: Dict[str, Any],
    common: Dict[str, str],
    *,
    pipeline_folder: Optional[str] = None,
) -> Dict[str, str]:
    r = _clean_steady_state_row(row)
    use_ctrl = _use_adaptive_control_on(r)
    use_hill = _hill_on(r)
    allee = str(r.get("allee_type") or "WEAK").upper()
    mu_val = int(r.get("mu") or 0)
    try:
        c_star = float(r.get("c_star") or 0.1)
        s_star = float(r.get("s_star") or 0.1)
        i_star = float(r.get("i_star") or 0.9)
    except (TypeError, ValueError):
        c_star, s_star, i_star = 0.1, 0.1, 0.9

    umax_src = r.get("umax_used")
    if umax_src is None:
        umax_src = r.get("umax")
    u_max_str: str
    if umax_src is not None and str(umax_src).strip() != "":
        u_max_str = _str_cell(umax_src, common.get("U_MAX", "0.5"))
    else:
        u_max_str = common.get("U_MAX", "0.5")

    scenario: Dict[str, str] = {
        "name": steady_state_equilibrium_name(r),
        "ALLEE_TYPE": allee,
        "mu": str(mu_val),
        "rc": _str_cell(r.get("rc"), common.get("rc", "5.84")),
        "beta": _str_cell(r.get("beta"), common.get("beta", "7.6")),
        "delta": _str_cell(r.get("delta"), common.get("delta", "5.40")),
        "eta": _str_cell(r.get("eta"), common.get("eta", "5.08")),
        "rd": _str_cell(r.get("rd"), common.get("rd", "10.92")),
        "a": _str_cell(r.get("a"), common.get("alle", "0.1")),
        "USE_ADAPTIVE_CONTROL": "Y" if use_ctrl else "N",
        "HILL_CONTROL": "Y" if use_hill else "N",
        "HILL_KC": common.get("HILL_KC", "0.05"),
        "HILL_NC": common.get("HILL_NC", "2"),
        "HILL_KI": common.get("HILL_KI", "0.2"),
        "HILL_NI": common.get("HILL_NI", "2"),
        "KU": _str_cell(r.get("ku"), common.get("KU", "0.2")),
        "EPS_U": _str_cell(r.get("eps_u"), common.get("EPS_U", "0.001")),
        "U_MAX": u_max_str,
        "C_INIT_MIN": str(max(0.01, c_star * 0.9)),
        "C_INIT_MAX": str(min(1.0, c_star * 1.1)),
        "S_INIT_MIN": str(max(0.01, s_star * 0.9)),
        "S_INIT_MAX": str(min(1.0, s_star * 1.1)),
        "I_INIT_MIN": str(max(0.01, i_star * 0.9)),
        "I_INIT_MAX": str(min(1.0, i_star * 1.1)),
    }
    if r.get("rs") is not None:
        scenario["rs"] = _str_cell(r.get("rs"), common.get("rs"))
    if r.get("alpha") is not None:
        scenario["alpha"] = _str_cell(r.get("alpha"), common.get("alpha"))
    if r.get("gamma") is not None:
        scenario["gamma"] = _str_cell(r.get("gamma"), common.get("gamma"))
    out = {k: v for k, v in scenario.items() if v is not None and v != ""}
    if pipeline_folder and str(pipeline_folder).strip():
        out["pipeline_folder"] = str(pipeline_folder).strip()
    return out


def _is_steady_states_full_run(data: Dict[str, Any]) -> bool:
    if not isinstance(data, dict):
        return False
    if data.get("weak_grid") is not None or data.get("strong_corner") is not None:
        return True
    meta = data.get("meta")
    if isinstance(meta, dict) and meta.get("source") in (
        "steady_states.full_pipeline",
        "steady_states.corner_strong_only",
    ):
        return True
    return False


def _iter_group_steady_state_pairs(
    steady_states_key: str, section: Optional[Dict[str, Any]]
) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Para cada grupo bajo steady_states_filtered, emite (grupo_padre, fila_equilibrio).
    El nombre corto ``_dashboard_block_short_label(grupo)`` coincide con RESULTS_DIR/<carpeta>/.
    """
    if not section or not isinstance(section, dict):
        return
    groups = section.get(steady_states_key)
    if not isinstance(groups, list):
        return
    for g in groups:
        if not isinstance(g, dict):
            continue
        for st in g.get("steady_states") or []:
            if isinstance(st, dict):
                yield g, st


def pipeline_results_subdir_for_scenario(
    normalized_catalog: Dict[str, Any], catalog_scenario_name: str
) -> str:
    """
    Subcarpeta real bajo RESULTS_DIR para un escenario del catálogo normalizado.

    ``scenarios[].name`` puede ser único y largo; ``pipeline_folder`` alinea rutas de disco
    (p. ej. ``strong_mu0_uSi_bajo_umbral`` o ``strong_mu0_uNo_bajo_umbral_c0_s1_i0``).
    """
    nm = (catalog_scenario_name or "").strip()
    if not nm:
        return nm
    for s in normalized_catalog.get("scenarios") or []:
        if isinstance(s, dict) and str(s.get("name") or "").strip() == nm:
            pf = (s.get("pipeline_folder") or "").strip()
            return pf if pf else nm
    return nm


def flatten_full_run_to_scenarios_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convierte steady_states_full_run.json al shape {common_params, scenarios}."""
    common = _default_common_params()
    ordered: Dict[str, Dict[str, str]] = {}

    weak = data.get("weak_grid")
    if isinstance(weak, dict):
        for g, st in _iter_group_steady_state_pairs("steady_states_filtered", weak):
            disk = _dashboard_block_short_label(g)
            scen = _row_to_scenario_dict(st, common, pipeline_folder=disk)
            ordered.setdefault(scen["name"], scen)

    corner = data.get("strong_corner")
    if isinstance(corner, dict):
        for g, st in _iter_group_steady_state_pairs("steady_states_filtered", corner):
            disk = _dashboard_block_short_label(g)
            scen = _row_to_scenario_dict(st, common, pipeline_folder=disk)
            ordered.setdefault(scen["name"], scen)

    return {"common_params": common, "scenarios": list(ordered.values())}


def normalize_scenarios_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    if _is_steady_states_full_run(raw):
        return flatten_full_run_to_scenarios_payload(raw)
    common = raw.get("common_params")
    scenarios = raw.get("scenarios")
    if not isinstance(common, dict):
        common = {}
    if not isinstance(scenarios, list):
        scenarios = []
    return {"common_params": common, "scenarios": scenarios}


def _equilibrium_display_row(st: Dict[str, Any]) -> Dict[str, Any]:
    base = {k: v for k, v in st.items() if k not in ("equilibrium_index", "equilibrium_slug")}
    return {
        "equilibrium_index": st.get("equilibrium_index"),
        "equilibrium_slug": st.get("equilibrium_slug") or "",
        "catalog_name": steady_state_equilibrium_name(base),
        "c_star": st.get("c_star"),
        "s_star": st.get("s_star"),
        "i_star": st.get("i_star"),
    }


def _dashboard_block_short_label(group: Dict[str, Any]) -> str:
    """
    Título único por bloque para el dashboard.

    El nombre con sufijo ``_c0_s1_i0`` no es redundant con el sólo ``strong_mu0_uNo_bajo_umbral``:
    cada uno marca un **punto estacionario distinto** (p. ej. (0,1,0) vs (1,0,0)).

    Por tanto, si ``name`` del bloque lleva ese sufijo, se muestra integramente.
    Para bloques sólo tipo v1 (sin rama codificada en el ``name``), se usa canon v1 /
    ``scenario_json_name``.
    """
    nm = str(group.get("name") or "").strip()
    if nm and _DASHBOARD_GROUP_BRANCH_SUFFIX_RE.match(nm):
        return nm
    vc = group.get("v1_canonical_name")
    if vc is not None and str(vc).strip():
        return str(vc).strip()
    states = group.get("steady_states") or []
    if states and isinstance(states[0], dict):
        sj = states[0].get("scenario_json_name")
        if sj is not None and str(sj).strip():
            return str(sj).strip()
    return nm


def _groups_from_nested_section(section_label: str, nested_list: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(nested_list, list):
        return out
    for g in nested_list:
        if not isinstance(g, dict):
            continue
        short = _dashboard_block_short_label(g) or (g.get("name") or "").strip()
        states = g.get("steady_states") or []
        equilibria: List[Dict[str, Any]] = []
        for st in states:
            if isinstance(st, dict):
                equilibria.append(_equilibrium_display_row(st))
        if not short and not equilibria:
            continue
        out.append(
            {
                "section": section_label,
                "short_name": short or "(sin nombre)",
                "n_steady_states": len(equilibria),
                "equilibria": equilibria,
            }
        )
    return out


def _nested_steady_state_blocks(section: Optional[Dict[str, Any]]) -> Any:
    """
    Lista de bloques {name, steady_states,...} para pintar tarjetas del dashboard.
    Prefiere ``steady_states_filtered``; si falta o viene vacía, usa ``all``
    (muchas exportaciones solo rellenan ``all``).
    """
    if not section or not isinstance(section, dict):
        return None
    nested = section.get("steady_states_filtered")
    if not nested:
        nested = section.get("all")
    return nested


def build_dashboard_scenario_groups(raw: Dict[str, Any], normalized: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Tarjetas del dashboard: un bloque por (sección × escenario corto) con lista de equilibrios.
    """
    if _is_steady_states_full_run(raw):
        groups: List[Dict[str, Any]] = []
        weak = raw.get("weak_grid")
        if isinstance(weak, dict):
            groups.extend(
                _groups_from_nested_section("WEAK · filtrado físico", _nested_steady_state_blocks(weak))
            )
        corner = raw.get("strong_corner")
        if isinstance(corner, dict):
            groups.extend(
                _groups_from_nested_section("STRONG · filtrado físico", _nested_steady_state_blocks(corner))
            )
        return groups
    groups_classic: List[Dict[str, Any]] = []
    for s in normalized.get("scenarios") or []:
        if not isinstance(s, dict):
            continue
        nm = (s.get("name") or "").strip()
        if not nm:
            continue
        groups_classic.append(
            {
                "section": "Catálogo clásico",
                "short_name": nm,
                "n_steady_states": 1,
                "equilibria": [
                    {
                        "equilibrium_index": 0,
                        "equilibrium_slug": "",
                        "catalog_name": nm,
                        "c_star": None,
                        "s_star": None,
                        "i_star": None,
                    }
                ],
            }
        )
    return groups_classic


def load_catalog_for_lab(path: Path) -> Tuple[Dict[str, Any], Optional[str], List[Dict[str, Any]]]:
    """
    Lee el JSON del catálogo: payload normalizado (pipeline/SymPy), error si lo hay,
    y grupos para tarjetas del dashboard (escenario + equilibrios).
    """
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except OSError as e:
        return {}, str(e), []
    except json.JSONDecodeError as e:
        return {}, str(e), []
    if not isinstance(raw, dict):
        return {}, "El catálogo no es un objeto JSON", []
    try:
        data = normalize_scenarios_payload(raw)
        groups = build_dashboard_scenario_groups(raw, data)
    except Exception as e:
        return {}, str(e), []
    return data, None, groups


def load_normalized_catalog(path: Path) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Lee JSON desde path y devuelve (payload normalizado, error).
    error es None si todo OK.
    """
    data, err, _ = load_catalog_for_lab(path)
    return data, err


def get_catalog_path() -> Path:
    return Path(settings.SCENARIOS_JSON_PATH).expanduser().resolve()


def scenario_names_from_normalized(data: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for s in data.get("scenarios") or []:
        if not isinstance(s, dict):
            continue
        name = (s.get("name") or "").strip()
        if name:
            out.append(name)
    return out


def get_all_scenario_names(path: Optional[Path] = None) -> List[str]:
    p = path or get_catalog_path()
    if not p.is_file():
        return []
    data, err = load_normalized_catalog(p)
    if err:
        raise ValueError(err)
    return scenario_names_from_normalized(data)
