"""
Fusiona ``scenarios_v1.json`` + ``scenarios.json`` (Állee / Models) en un único
``steady_states_full_run.json`` (forma ``weak_grid`` / ``strong_corner``) para publicarlo en Drive.

Los bloques híbridos conservan ``name`` como ``…_c0_s1_i0`` (``scenarios.json``); añaden
``v1_canonical_name`` y ``scenario_json_name`` por equilibrio para alinear con v1.

La app Django solo **lee** el archivo configurado en ``SCENARIOS_JSON_PATH`` / montaje Drive;
esta utilidad **escribe** el JSON unificado en la ruta de salida indicada.
"""

from __future__ import annotations

import copy
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from steady_states_app.scenarios_catalog import (
    _default_common_params,
    _is_steady_states_full_run,
    steady_state_equilibrium_name,
    steady_state_equilibrium_slug,
)

# ``strong_mu0_uNo_bajo_umbral_c0_s1_i0`` → prefijo alineado con ``scenarios_v1.json``.
_BRANCH_NAME_SUFFIX_RE = re.compile(r"^(?P<base>.+)_c\d+_s\d+_i\d+$")


def _derive_v1_canonical_name(group: Dict[str, Any]) -> str:
    """
    Nombre corto alineado con ``scenarios_v1.json`` (p. ej. ``strong_mu0_uNo_bajo_umbral``).
    Preferimos ``scenario_json_name`` del primer equilibrio; si no existe, despoja ``…_c0_s1_i0`` del ``name``.
    """
    sts = group.get("steady_states") or []
    nm_blk = str(group.get("name") or "").strip()
    if sts and isinstance(sts[0], dict):
        sj = sts[0].get("scenario_json_name")
        if sj is not None and str(sj).strip():
            return str(sj).strip()
    m = _BRANCH_NAME_SUFFIX_RE.match(nm_blk)
    if m:
        return str(m.group("base")).strip()
    return nm_blk


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"No es un objeto JSON: {path}")
    return data


def _parse_float(x: Any, default: float = 0.0) -> float:
    if x is None or x == "":
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def build_merged_common(main: Dict[str, Any], v1: Dict[str, Any]) -> Dict[str, str]:
    """common_params unificado: por defecto del laboratorio, luego main, luego v1 (v1 gana choques)."""
    out: Dict[str, str] = {k: str(v) for k, v in _default_common_params().items()}
    mc = main.get("common_params")
    if isinstance(mc, dict):
        for k, v in mc.items():
            out[str(k)] = str(v)
    vc = v1.get("common_params")
    if isinstance(vc, dict):
        for k, v in vc.items():
            out[str(k)] = str(v)
    return out


def _merge_group_seen_key(g: Dict[str, Any]) -> str:
    """
    Clave única por bloque al fusionar. No debe colapsar el bloque Newton ``…_c0_s1_i0`` con otro
    punto para el mismo escenario parametrizacional en ``scenarios_v1`` (ej. otro estado (c*,s*,i*)).
    """
    nm = str(g.get("name") or "").strip()
    if nm and _BRANCH_NAME_SUFFIX_RE.match(nm):
        return nm
    sts = g.get("steady_states") or []
    if sts and isinstance(sts[0], dict):
        try:
            return steady_state_equilibrium_name(sts[0])
        except Exception:
            pass
    return _derive_v1_canonical_name(g) or nm


def _collect_merge_seen_keys(groups: List[Dict[str, Any]]) -> Set[str]:
    seen: Set[str] = set()
    for g in groups:
        if not isinstance(g, dict):
            continue
        sn = _merge_group_seen_key(g)
        if sn:
            seen.add(sn)
    return seen


def _is_hybrid_lab_scenarios(obj: Dict[str, Any]) -> bool:
    if _is_steady_states_full_run(obj):
        return False
    sc = obj.get("scenarios")
    return isinstance(sc, list) and len(sc) > 0


def hybrid_main_to_full_run(main: Dict[str, Any]) -> Dict[str, Any]:
    """Convierte ``{ common_params, scenarios [, steady_states_filtered] }`` a forma full-run."""
    sc_groups = copy.deepcopy(main.get("scenarios") or [])
    filt = main.get("steady_states_filtered")
    if isinstance(filt, list) and filt:
        filtered_copy = copy.deepcopy(filt)
    else:
        filtered_copy = copy.deepcopy(sc_groups)

    strong_corner: Dict[str, Any] = {
        "meta": {
            "scan_kind": "merged_from_models_scenarios_json",
            "note": "Generado desde Models/Allee/scenarios.json (+ escenarios sintéticos si faltan en v1).",
        },
        "all": sc_groups,
        "steady_states_filtered": filtered_copy,
    }

    wg = main.get("weak_grid")
    weak_grid_out = copy.deepcopy(wg) if wg is not None else None

    return {
        "meta": copy.deepcopy(main.get("meta") or {}),
        "weak_grid": weak_grid_out,
        "strong_corner": strong_corner,
    }


def _synthetic_group(short_name: str, scen: Dict[str, Any], common: Dict[str, str]) -> Dict[str, Any]:
    flat: Dict[str, str] = dict(common)
    for k, v in scen.items():
        if k == "name":
            continue
        if isinstance(v, (dict, list)):
            continue
        flat[str(k)] = str(v)

    def gf(key: str, default: float) -> float:
        return _parse_float(flat.get(key), default)

    c_star = (gf("C_INIT_MIN", 0.05) + gf("C_INIT_MAX", 0.05)) / 2.0
    s_star = (gf("S_INIT_MIN", 0.05) + gf("S_INIT_MAX", 0.05)) / 2.0
    i_star = (gf("I_INIT_MIN", 0.05) + gf("I_INIT_MAX", 0.05)) / 2.0

    use_adaptive = str(flat.get("USE_ADAPTIVE_CONTROL", "N")).upper() == "Y"
    hill_raw = flat.get("HILL_CONTROL")
    if hill_raw is None or str(hill_raw).strip() == "":
        hill_on = use_adaptive
    else:
        hill_on = str(hill_raw).upper() == "Y"
    allee_type = str(flat.get("ALLEE_TYPE", "WEAK")).upper()
    mu_f = gf("mu", 0.0)

    rec_core: Dict[str, Any] = {
        "scenario_json_name": short_name,
        "target_branch": "c0_s1_i0",
        "c_star": c_star,
        "s_star": s_star,
        "i_star": i_star,
        "residual_l2": None,
        "max_real": None,
        "unstable": None,
        "near_c0_s1_i1": False,
        "mu": mu_f,
        "allee_type": allee_type,
        "use_adaptive_control": use_adaptive,
        "control_mode": (
            "hill" if hill_on else ("min_adaptive" if use_adaptive else "none")
        ),
        "hill_control": hill_on,
        "a": gf("alle", gf("a", 0.1)),
        "rc": gf("rc", 5.84),
        "rs": gf("rs", 13.12),
        "rd": gf("rd", 10.92),
        "alpha": gf("alpha", 10.22),
        "beta": gf("beta", 7.6),
        "delta": gf("delta", 5.4),
        "eta": gf("eta", 5.08),
        "gamma": gf("gamma", 0.74),
        "ku": gf("KU", 0.2) if use_adaptive else 0.0,
        "eps_u": gf("EPS_U", 0.001),
    }
    um_raw = flat.get("U_MAX")
    rec_core["umax"] = float(um_raw) if um_raw not in (None, "") else None

    slug = steady_state_equilibrium_slug(rec_core)
    long_name = steady_state_equilibrium_name(rec_core)
    row = {
        **rec_core,
        "equilibrium_index": 0,
        "equilibrium_slug": slug,
    }
    # ``name`` del bloque = etiqueta v1 (coherente con steady_states.py); el nombre plano del equilibrio sigue en la fila.
    return {"name": short_name, "n_steady_states": 1, "steady_states": [row]}


def canonialize_strong_corner_v1_names(strong_corner: Dict[str, Any]) -> None:
    """
    Cuando viene de ``scenarios.json``, el bloque usa ``name`` híbrido ``…_c0_s1_i0``: se **conserva**
    ese literal para que siga apareciendo en el JSON.

    Para alinear con ``scenarios_v1.json`` / SymPy / tarjetas, se asegura ``scenario_json_name``
    en cada equilibrio (etiqueta corta tipo ``strong_mu0_uNo_bajo_umbral``).

    Sin sufijo `_c*_s*_i*` (p. ej. export corner Newton), ``name`` pasa al canónico v1 corto como antes.
    """
    for key in ("all", "steady_states_filtered"):
        lst = strong_corner.get(key)
        if not isinstance(lst, list):
            continue
        for g in lst:
            if not isinstance(g, dict):
                continue
            old_nm = str(g.get("name") or "").strip()
            label = _derive_v1_canonical_name(g)
            if not label:
                continue
            g.pop("hybrid_block_name", None)
            if old_nm and _BRANCH_NAME_SUFFIX_RE.match(old_nm):
                g["name"] = old_nm
                g["v1_canonical_name"] = label
            else:
                g["name"] = label
                g.pop("v1_canonical_name", None)
            for st in g.get("steady_states") or []:
                if not isinstance(st, dict):
                    continue
                cur = st.get("scenario_json_name")
                if cur is None or str(cur).strip() == "":
                    st["scenario_json_name"] = label


def add_missing_v1_scenarios(
    strong_corner: Dict[str, Any],
    v1: Dict[str, Any],
    merged_common: Dict[str, str],
) -> int:
    """Añade bloques sintéticos para filas de v1 sin datos en ``all`` / ``steady_states_filtered``."""
    all_list = strong_corner.setdefault("all", [])
    filt_list = strong_corner.setdefault("steady_states_filtered", [])
    if not isinstance(all_list, list):
        all_list = []
        strong_corner["all"] = all_list
    if not isinstance(filt_list, list):
        filt_list = []
        strong_corner["steady_states_filtered"] = filt_list

    seen = _collect_merge_seen_keys(all_list) | _collect_merge_seen_keys(filt_list)

    added = 0
    for s in v1.get("scenarios") or []:
        if not isinstance(s, dict):
            continue
        nm = str(s.get("name") or "").strip()
        if not nm:
            continue
        grp = _synthetic_group(nm, s, merged_common)
        ksyn = _merge_group_seen_key(grp)
        # Coexistencia Newton ``*_c*_s*_i*`` con fila v1 mismo ``nm``: claves físicas distintas.
        if ksyn in seen:
            continue
        gcopy = copy.deepcopy(grp)
        all_list.append(gcopy)
        filt_list.append(copy.deepcopy(gcopy))
        seen.add(ksyn)
        added += 1
    return added


def merge_catalog_sources(main: Dict[str, Any], v1: Dict[str, Any], *, source_paths: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Devuelve un dict tipo ``steady_states_full_run.json`` listo para serializar.

    ``main``: habitualmente ``Models/Allee/scenarios.json`` (híbrido con ``scenarios`` + ``steady_states_filtered``)
    o ya un ``steady_states_full_run`` completo.
    ``v1``: ``Models/Allee/scenarios_v1.json`` (definiciones cortas).
    """
    merged_common = build_merged_common(main, v1)

    if _is_steady_states_full_run(main):
        base = copy.deepcopy(main)
    elif _is_hybrid_lab_scenarios(main):
        base = hybrid_main_to_full_run(main)
    else:
        raise ValueError(
            "scenarios.json no tiene forma reconocida: "
            "se esperaba steady_states_full_run (weak_grid/strong_corner) "
            "o bien {common_params, scenarios[, steady_states_filtered]}."
        )

    strong = base.get("strong_corner")
    if not isinstance(strong, dict):
        strong = {}
        base["strong_corner"] = strong

    add_missing_v1_scenarios(strong, v1, merged_common)
    canonialize_strong_corner_v1_names(strong)

    meta_in = base.get("meta") if isinstance(base.get("meta"), dict) else {}
    meta_out = {**meta_in}
    meta_out["merged_at"] = datetime.now(timezone.utc).isoformat()
    if source_paths:
        meta_out["merge_sources"] = dict(source_paths)
    base["meta"] = meta_out

    return base


def write_catalog_json(path: Path, data: Dict[str, Any], *, dry_run: bool = False) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


def merge_paths_and_write(
    path_v1: Path,
    path_main: Path,
    output: Path,
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    v1 = load_json(path_v1)
    main = load_json(path_main)
    merged = merge_catalog_sources(
        main,
        v1,
        source_paths={"scenarios_v1": str(path_v1.resolve()), "scenarios": str(path_main.resolve())},
    )
    write_catalog_json(output, merged, dry_run=dry_run)
    return merged
