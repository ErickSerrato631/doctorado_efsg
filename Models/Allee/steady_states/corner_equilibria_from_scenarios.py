"""
Equilibrios 3D (c,s,i) cercanos a las esquinas (0,1,0) y (0,1,1) para **cada**
escenario de un ``scenarios.json``.

Reutiliza la misma construcción de ecuaciones que ``extract_steady_states_from_scenarios``
(Hill / min-adaptativo / sin control). Semillas densas alrededor de ambas esquinas,
Newton–Raphson, deduplicación y clasificación por cajas disjuntas en *i*.

Salida principal: JSON con la misma forma que ``scenarios_v1.json`` (``common_params`` +
``scenarios``), añadiendo en cada escenario claves ``EQ_C0S1I0_*`` y ``EQ_C0S1I1_*`` (valores
en cadena, estilo escenarios). Archivo por defecto en la raíz de ``Allee/``: ``corner_equilibria_scenarios.json``.

Además, por defecto se **actualiza** ``steady_states_full_run.json`` en
``Resultados Paper/estados_estacionarios/`` (Drive o ``RESULTS_DIR``); si no hay montaje,
se usa ``Allee/estados_estacionarios/``. Usa ``--skip-full-run-json`` para no tocarlo.

También se actualiza por defecto ``Allee/scenarios.json`` con ``common_params`` + ``scenarios``
+ ``steady_states_filtered`` (misma estructura híbrida). ``--skip-scenarios-json`` lo desactiva;
``--scenarios-json-out`` cambia la ruta.

Ejemplo (WSL, ajusta la ruta si tu usuario es distinto)::

    wsl -e bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && conda activate fenicsx-env && cd '/mnt/c/Users/Erick Serrato/Documents/Doctorado/doctorado_efsg-main/Models/Allee' && python steady_states/corner_equilibria_from_scenarios.py --scenarios-file scenarios_v1.json"

Por defecto se crea ``Allee/corner_equilibria_scenarios.json`` (misma carpeta que ``scenarios.json``).

Desde ``Allee`` sin WSL::

    python steady_states/corner_equilibria_from_scenarios.py
    python steady_states/corner_equilibria_from_scenarios.py --scenarios-file scenarios_v1.json --out-json corner_equilibria_scenarios.json
    python steady_states/corner_equilibria_from_scenarios.py --out-csv steady_states/corners.csv
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import sympy as sp

_ALLEE_ROOT = Path(__file__).resolve().parent.parent
if str(_ALLEE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALLEE_ROOT))

from steady_states import (
    newton_root_3d,
    steady_state_equilibrium_slug,
    a,
    rc,
    rs,
    rd,
    alpha,
    delta,
    beta,
    gamma,
    eta,
    mu,
    ku,
    eps_u,
    umax,
    kc_h,
    nc_h,
    ki_h,
    ni_h,
    c_3d,
    s_3d,
    i_3d,
    build_equations_3d as ss_build_equations_3d,
)
from steady_states.steady_states import save_steady_states_full_run_json
from steady_states.extract_steady_states_from_scenarios import (
    build_equations_3d_min_adaptive,
    load_scenarios,
    scenario_uses_hill_control,
)

# Clasificación “cerca de la esquina” (c pequeña, s alta); *i* separa las dos ramas.
_CMAX = 0.18
_SMIN = 0.82
_I_LOW_MAX = 0.28   # rama tipo (0,1,0)
_I_HIGH_MIN = 0.72  # rama tipo (0,1,1)

# Objetivos para elegir la mejor raíz si hay varias en la misma rama
_T_I0 = np.array([0.0, 1.0, 0.0], dtype=float)
_T_I1 = np.array([0.0, 1.0, 1.0], dtype=float)


def _seeds_near_c0_s1_i0() -> List[Tuple[float, float, float]]:
    """Semillas hacia tumor ausente, sanos altos, inmunidad baja (esquina i≈0)."""
    pts: List[Tuple[float, float, float]] = [
        (0.0, 0.999, 1e-5),
        (1e-10, 0.995, 1e-4),
        (0.0, 0.99, 0.01),
        (0.01, 0.98, 0.02),
        (1e-8, 0.995, 0.05),
        (0.0, 0.97, 0.08),
        (0.02, 0.99, 0.01),
        (1e-6, 1.0, 1e-6),
    ]
    return _dedupe_seeds(pts)


def _seeds_near_c0_s1_i1() -> List[Tuple[float, float, float]]:
    """Semillas hacia (0,1,1) (alineado con ``default_seeds_3d_control_hill`` / extractor)."""
    pts: List[Tuple[float, float, float]] = [
        (0.0, 1.0, 1.0),
        (1e-10, 1.0, 1.0),
        (0.0, 0.99, 1.0),
        (0.0, 1.0, 0.99),
        (1e-8, 0.995, 0.995),
        (0.0, 0.98, 1.02),
        (1e-6, 0.99, 0.99),
        (0.0, 0.999, 0.999),
    ]
    return _dedupe_seeds(pts)


def _all_corner_seeds() -> List[Tuple[float, float, float]]:
    return _dedupe_seeds(_seeds_near_c0_s1_i0() + _seeds_near_c0_s1_i1())


def _dedupe_seeds(
    seeds: List[Tuple[float, float, float]], tol: float = 1e-9
) -> List[Tuple[float, float, float]]:
    kept: List[Tuple[float, float, float]] = []
    for p in seeds:
        if any(
            np.linalg.norm(np.array(p, dtype=float) - np.array(q, dtype=float)) < tol
            for q in kept
        ):
            continue
        kept.append(p)
    return kept


def build_lambdify_for_scenario(
    common_params: Dict[str, Any], scenario: Dict[str, Any]
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Construye (f, Jsym, meta) con la misma lógica que ``calculate_steady_state_for_scenario``.
    ``meta`` resume ramas de control para el CSV.
    """
    mu_val = float(scenario.get("mu", common_params.get("mu", "1")))
    allee_type = scenario.get("ALLEE_TYPE", "WEAK")
    use_control = scenario.get("USE_ADAPTIVE_CONTROL", "N") == "Y"
    use_hill = scenario_uses_hill_control(scenario)

    a_val = float(common_params.get("a", "0.1"))
    rc_val = float(scenario.get("rc", common_params.get("rc", "6.5")))
    beta_val = float(scenario.get("beta", common_params.get("beta", "3")))
    delta_val = float(scenario.get("delta", common_params.get("delta", "9")))
    eta_val = float(scenario.get("eta", common_params.get("eta", "1")))
    rd_val = float(scenario.get("rd", common_params.get("rd", "14")))

    ku_val = float(scenario.get("KU", common_params.get("KU", "0.2"))) if use_control else 0.0
    eps_val = float(scenario.get("EPS_U", common_params.get("EPS_U", "0.02"))) if use_control else 1e-3
    umax_val = float(scenario.get("U_MAX", common_params.get("U_MAX", "1.0"))) if use_control else None

    params_dict: Dict[Any, Any] = {
        a: a_val,
        rc: rc_val,
        rs: float(common_params.get("rs", "13.12")),
        rd: rd_val,
        alpha: float(common_params.get("alpha", "10.22")),
        delta: delta_val,
        beta: beta_val,
        gamma: float(common_params.get("gamma", "0.74")),
        eta: eta_val,
        mu: mu_val,
    }

    if use_hill:
        params_dict[kc_h] = float(scenario.get("HILL_KC", common_params.get("HILL_KC", "0.05")))
        params_dict[nc_h] = float(scenario.get("HILL_NC", common_params.get("HILL_NC", "2")))
        params_dict[ki_h] = float(scenario.get("HILL_KI", common_params.get("HILL_KI", "0.2")))
        params_dict[ni_h] = float(scenario.get("HILL_NI", common_params.get("HILL_NI", "2")))
        params_dict[umax] = float(scenario.get("U_MAX", common_params.get("U_MAX", "0.5")))
        Fc, Fs, Fi = ss_build_equations_3d(allee_type, True)
        control_mode = "hill"
    elif use_control:
        params_dict[ku] = ku_val
        params_dict[eps_u] = eps_val
        params_dict[umax] = umax_val if umax_val is not None else sp.oo
        Fc, Fs, Fi = build_equations_3d_min_adaptive(allee_type)
        control_mode = "min_adaptive"
    else:
        params_dict[ku] = 0.0
        params_dict[eps_u] = 1e-3
        params_dict[umax] = sp.oo
        Fc, Fs, Fi = ss_build_equations_3d(allee_type, False)
        control_mode = "none"

    pcur_sub = {k: v for k, v in params_dict.items() if v is not None}
    Fc_eval = Fc.subs(pcur_sub)
    Fs_eval = Fs.subs(pcur_sub)
    Fi_eval = Fi.subs(pcur_sub)
    F_vec = sp.Matrix([Fc_eval, Fs_eval, Fi_eval])
    Jsym = F_vec.jacobian([c_3d, s_3d, i_3d])
    f = sp.lambdify((c_3d, s_3d, i_3d), F_vec, modules="numpy")

    rs_val = float(common_params.get("rs", "13.12"))
    al_v = float(common_params.get("alpha", "10.22"))
    ga_v = float(common_params.get("gamma", "0.74"))
    if use_hill:
        umax_numeric = float(params_dict.get(umax, 0.5))
    elif use_control and umax_val is not None:
        umax_numeric = float(umax_val)
    else:
        umax_numeric = None

    meta = {
        "mu": mu_val,
        "allee_type": str(allee_type).upper(),
        "use_adaptive_control": use_control,
        "control_mode": control_mode,
        "hill_control": bool(use_hill),
        "a": a_val,
        "rc": rc_val,
        "rs": rs_val,
        "rd": rd_val,
        "alpha": al_v,
        "beta": beta_val,
        "delta": delta_val,
        "eta": eta_val,
        "gamma": ga_v,
        "ku": ku_val,
        "eps_u": eps_val,
        "umax": umax_numeric,
    }
    return f, Jsym, meta


def _is_near_c0_s1_i0(c: float, s: float, i: float) -> bool:
    return (
        c <= _CMAX
        and s >= _SMIN
        and i <= _I_LOW_MAX
        and i >= 0.0
    )


def _is_near_c0_s1_i1(c: float, s: float, i: float) -> bool:
    return c <= _CMAX and s >= _SMIN and i >= _I_HIGH_MIN


def flatten_scenario_for_kinetics(
    scenario: Dict[str, Any], common_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convierte un bloque ``{name, steady_states:[{...cinética...}]}`` en dict plano
    compatible con ``build_lambdify_for_scenario`` / ``scenario_uses_hill_control``.
    Si ya es estilo ``scenarios_v1`` (``ALLEE_TYPE`` o ``C_INIT_MIN`` en raíz), devuelve copia.
    """
    if not isinstance(scenario, dict):
        return {}
    if scenario.get("ALLEE_TYPE") is not None or scenario.get("C_INIT_MIN") is not None:
        return dict(scenario)
    sts = scenario.get("steady_states")
    if not isinstance(sts, list) or len(sts) == 0:
        return dict(scenario)
    ss0 = sts[0]
    flat: Dict[str, Any] = {"name": str(scenario.get("name", ""))}
    flat["ALLEE_TYPE"] = str(ss0.get("allee_type", "WEAK")).upper()
    flat["mu"] = str(ss0.get("mu", common_params.get("mu", "1")))
    flat["rc"] = str(ss0.get("rc", common_params.get("rc", "5.84")))
    flat["beta"] = str(ss0.get("beta", common_params.get("beta", "7.6")))
    flat["delta"] = str(ss0.get("delta", common_params.get("delta", "5.40")))
    flat["eta"] = str(ss0.get("eta", common_params.get("eta", "5.08")))
    flat["rd"] = str(ss0.get("rd", common_params.get("rd", "10.92")))
    flat["a"] = str(ss0.get("a", common_params.get("a", "0.1")))
    uac = bool(ss0.get("use_adaptive_control"))
    flat["USE_ADAPTIVE_CONTROL"] = "Y" if uac else "N"
    if bool(ss0.get("hill_control")) and uac:
        for k in ("HILL_KC", "HILL_NC", "HILL_KI", "HILL_NI"):
            if k in common_params:
                flat[k] = str(common_params[k])
        um = ss0.get("umax")
        flat["U_MAX"] = str(um) if um is not None else str(common_params.get("U_MAX", "0.5"))
    elif uac:
        flat["KU"] = str(ss0.get("ku", common_params.get("KU", "0.2")))
        flat["EPS_U"] = str(ss0.get("eps_u", common_params.get("EPS_U", "0.001")))
        um = ss0.get("umax")
        if um is not None:
            flat["U_MAX"] = str(um)
        elif "U_MAX" in common_params:
            flat["U_MAX"] = str(common_params["U_MAX"])
    return flat


def collect_corner_equilibria_for_scenario(
    common_params: Dict[str, Any],
    scenario: Dict[str, Any],
    *,
    seeds: Optional[List[Tuple[float, float, float]]] = None,
    dedupe_tol: float = 1e-3,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Devuelve (lista_i0, lista_i1) de dicts con equilibrios encontrados en cada rama.
    Cada lista puede tener 0 o 1 elemento (el más cercano al objetivo de la rama).
    """
    scenario_eff = flatten_scenario_for_kinetics(scenario, common_params)
    name = str(scenario.get("name", scenario_eff.get("name", "")))
    f, Jsym, smeta = build_lambdify_for_scenario(common_params, scenario_eff)
    seed_list = seeds if seeds is not None else _all_corner_seeds()

    raw: List[Tuple[float, float, float]] = []
    for seed in seed_list:
        try:
            r = newton_root_3d(f, Jsym, seed)
        except Exception:
            continue
        if r is None:
            continue
        cx, sy, iz = float(r[0]), float(r[1]), float(r[2])
        if not (np.isfinite(cx) and np.isfinite(sy) and np.isfinite(iz)):
            continue
        if cx < -1e-6 or sy < -1e-6 or iz < -1e-6:
            continue
        cx = max(cx, 0.0)
        sy = max(sy, 0.0)
        # i* = 0 es válido en la rama (0,1,0); solo descartar negativos claros.
        if abs(iz) < 1e-12:
            iz = 0.0
        elif iz < 0.0:
            continue
        if any(
            np.linalg.norm([cx - px, sy - py, iz - pz]) < dedupe_tol for px, py, pz in raw
        ):
            continue
        raw.append((cx, sy, iz))

    candidates_i0: List[Tuple[float, float, float]] = []
    candidates_i1: List[Tuple[float, float, float]] = []
    for cx, sy, iz in raw:
        if _is_near_c0_s1_i0(cx, sy, iz):
            candidates_i0.append((cx, sy, iz))
        if _is_near_c0_s1_i1(cx, sy, iz):
            candidates_i1.append((cx, sy, iz))

    def _best(
        pts: List[Tuple[float, float, float]], target: np.ndarray, label: str
    ) -> Optional[Dict[str, Any]]:
        if not pts:
            return None
        best_pt = min(pts, key=lambda p: np.linalg.norm(np.array(p, dtype=float) - target))
        cx, sy, iz = best_pt
        Fv = np.array(f(cx, sy, iz), dtype=float).ravel()
        res = float(np.linalg.norm(Fv))
        Jnum = np.array(Jsym.subs({c_3d: cx, s_3d: sy, i_3d: iz}), dtype=float)
        eigs = np.linalg.eigvals(Jnum)
        max_re = float(max(ev.real for ev in eigs))
        e_sorted = sorted([complex(z) for z in eigs], key=lambda z: z.real, reverse=True)
        while len(e_sorted) < 3:
            e_sorted.append(0j)
        eig_fields: Dict[str, float] = {}
        for j, ev in enumerate(e_sorted[:3], start=1):
            eig_fields[f"eig{j}_real"] = float(ev.real)
            eig_fields[f"eig{j}_imag"] = float(ev.imag)
        return {
            "scenario_json_name": name,
            "target_branch": label,
            "c_star": cx,
            "s_star": sy,
            "i_star": iz,
            "residual_l2": res,
            "max_real": max_re,
            "unstable": bool(max_re > 0),
            "near_c0_s1_i1": bool(label == "c0_s1_i1"),
            **eig_fields,
            **smeta,
        }

    out_i0: List[Dict[str, Any]] = []
    out_i1: List[Dict[str, Any]] = []
    b0 = _best(candidates_i0, _T_I0, "c0_s1_i0")
    b1 = _best(candidates_i1, _T_I1, "c0_s1_i1")
    if b0 is not None:
        out_i0.append(b0)
    if b1 is not None:
        out_i1.append(b1)
    return out_i0, out_i1


def _corner_scenario_block_name(
    base: str, steady_inner: List[Dict[str, Any]]
) -> str:
    """
    Nombre del bloque en salida: ``<base>_c0_s1_i0``, ``…_c0_s1_i1`` o ``…_c0_s1_i0_c0_s1_i1``
    según las ramas presentes en ``steady_inner`` (orden i0, luego i1).
    No duplica el sufijo si ``base`` ya termina con ese mismo sufijo.
    """
    branches: List[str] = []
    for st in steady_inner:
        b = st.get("target_branch")
        if isinstance(b, str) and b and b not in branches:
            branches.append(b)
    if not branches:
        return base
    suffix = "_".join(branches)
    if base.endswith("_" + suffix):
        return base
    return f"{base}_{suffix}"


def _fmt_scenario_value(x: Any) -> str:
    """Formatea números / booleanos como cadenas legibles (estilo scenarios_v1)."""
    if x is None:
        return ""
    if isinstance(x, (bool, np.bool_)):
        return "Y" if x else "N"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(xf):
        return ""
    return f"{xf:.12g}".rstrip("0").rstrip(".") if "." in f"{xf:.12g}" else f"{xf:.12g}"


def _empty_eq_fields(prefix: str) -> Dict[str, str]:
    return {
        f"{prefix}_FOUND": "N",
        f"{prefix}_C": "",
        f"{prefix}_S": "",
        f"{prefix}_I": "",
        f"{prefix}_RESIDUAL_L2": "",
        f"{prefix}_MAX_REAL": "",
        f"{prefix}_UNSTABLE": "",
        f"{prefix}_CONTROL_MODE": "",
    }


def _inner_equilibrium_record(rec: Dict[str, Any], eq_idx: int) -> Dict[str, Any]:
    """Fila compatible con ``steady_states_full_run.json`` → ``strong_corner.*.steady_states``."""
    slug = f"{rec['target_branch']}_{steady_state_equilibrium_slug(rec)}"
    row: Dict[str, Any] = {"equilibrium_index": eq_idx, "equilibrium_slug": slug}
    for k, v in rec.items():
        row[k] = v
    return row


def _nested_filter_target_branch(
    nested: List[Dict[str, Any]], branch: str
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for block in nested:
        inner = [x for x in block["steady_states"] if x.get("target_branch") == branch]
        if inner:
            out.append(
                {
                    "name": block["name"],
                    "n_steady_states": len(inner),
                    "steady_states": inner,
                }
            )
    return out


def _resolve_steady_states_run_dir(local_only: bool) -> Path:
    """``…/Resultados Paper/estados_estacionarios`` (Drive o RESULTS_DIR) o respaldo local."""
    if local_only:
        p = _ALLEE_ROOT / "estados_estacionarios"
        p.mkdir(parents=True, exist_ok=True)
        return p
    try:
        from utils_paths import ensure_steady_states_results_dir_ready

        return ensure_steady_states_results_dir_ready()
    except RuntimeError as e:
        print(
            f"[!] No se pudo usar Drive/RESULTS_DIR para steady_states_full_run.json ({e}). "
            "Escribiendo en Allee/estados_estacionarios/."
        )
        p = _ALLEE_ROOT / "estados_estacionarios"
        p.mkdir(parents=True, exist_ok=True)
        return p


def write_steady_states_full_run_for_corners(
    scenarios_path: Path,
    nested_all: List[Dict[str, Any]],
    common_params: Dict[str, Any],
    *,
    full_run_local_only: bool,
) -> Path:
    """
    Actualiza ``steady_states_full_run.json`` (solo bloque ``strong_corner``; ``weak_grid`` null),
    en ``estados_estacionarios`` bajo Drive/RESULTS_DIR o en ``Allee/estados_estacionarios/``.
    """
    near_only = _nested_filter_target_branch(nested_all, "c0_s1_i1")
    umax_ref = float(common_params.get("U_MAX", "0.5"))
    corner_meta = {
        "allee_type": "MIXED",
        "scan_kind": "scenarios_json_corner_newton",
        "source_script": "steady_states/corner_equilibria_from_scenarios.py",
        "scenarios_file": str(scenarios_path.resolve()),
        "umax_hill": umax_ref,
        "n_scenario_blocks": len(nested_all),
        "n_blocks_with_equilibria": sum(1 for b in nested_all if b.get("n_steady_states", 0) > 0),
        "n_near_corner_blocks": len(near_only),
        "near_corner_criteria_note": (
            "near_corner_only lista solo target_branch=c0_s1_i1 (caja i alta); "
            "c0_s1_i0 queda en all / steady_states_filtered."
        ),
        "name_pattern": (
            "name = <escenario origen>_<rama(s)>: c0_s1_i0 y/o c0_s1_i1 según steady_states; "
            "equilibrium_slug = target_branch + '_' + slug rc_rs_b_…_c_s_i"
        ),
    }
    corner_payload: Dict[str, Any] = {
        "meta": corner_meta,
        "all": nested_all,
        "steady_states_filtered": nested_all,
        "near_corner_only": near_only,
    }
    global_meta = {
        "source": "steady_states.corner_equilibria_from_scenarios",
        "created": datetime.now().isoformat(),
        "scenarios_file": str(scenarios_path.resolve()),
        "parts": ["strong_corner"],
        "weak_grid_note": "omitido (equilibrios solo desde scenarios.json + Newton 3D)",
        "full_run_json_local_only": bool(full_run_local_only),
    }
    out_dir = _resolve_steady_states_run_dir(full_run_local_only)
    out_path = out_dir / "steady_states_full_run.json"
    save_steady_states_full_run_json(
        out_path, global_meta, None, None, None, corner_payload
    )
    print(f"steady_states_full_run.json actualizado: {out_path}")
    return out_path


def write_allee_scenarios_json_hybrid(
    dest: Path,
    common_params: Dict[str, Any],
    nested_all: List[Dict[str, Any]],
    *,
    preserve_extra_root_keys: bool = True,
) -> None:
    """
    Actualiza ``Allee/scenarios.json`` (o ``dest``) con la estructura híbrida habitual:

    - ``common_params``
    - ``scenarios``: lista de bloques ``{name, n_steady_states, steady_states, EQ_*...}``
    - ``steady_states_filtered``: misma lista que ``scenarios`` (espejo; como en tu JSON actual).

    Si el archivo ya existe, conserva sus escenarios y agrega solo los análogos faltantes,
    uno por cada steady state encontrado. Esto hace idempotente el comando de esquinas:
    correrlo varias veces no duplica ``*_c0_s1_i0`` / ``*_c0_s1_i1``.
    """
    extra: Dict[str, Any] = {}
    existing_scenarios: List[Dict[str, Any]] = []
    if preserve_extra_root_keys and dest.exists():
        try:
            prev = json.loads(dest.read_text(encoding="utf-8"))
            if isinstance(prev, dict):
                reserved = {"common_params", "scenarios", "steady_states_filtered"}
                for k, v in prev.items():
                    if k not in reserved:
                        extra[k] = v
                raw_scenarios = prev.get("scenarios")
                if isinstance(raw_scenarios, list):
                    existing_scenarios = [
                        x for x in raw_scenarios if isinstance(x, dict)
                    ]
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    existing_names = {
        str(s.get("name", "")).strip()
        for s in existing_scenarios
        if str(s.get("name", "")).strip()
    }
    all_names = set(existing_names)
    generated_scenarios: List[Dict[str, Any]] = []

    for block in nested_all:
        steady_states = [
            x for x in (block.get("steady_states") or []) if isinstance(x, dict)
        ]
        if not steady_states:
            continue
        for ss in steady_states:
            base_name = str(
                ss.get("scenario_json_name") or block.get("name") or "scenario"
            )
            branch = str(ss.get("target_branch") or "").strip()
            if branch and not base_name.endswith("_" + branch):
                out_name = f"{base_name}_{branch}"
            else:
                out_name = base_name

            if out_name in existing_names:
                continue

            if out_name in all_names:
                slug = str(ss.get("equilibrium_slug") or ss.get("target_branch") or "")
                if not slug:
                    slug = f"eq{ss.get('equilibrium_index', len(all_names))}"
                out_name = f"{base_name}_{slug}"

            suffix_i = 1
            unique_name = out_name
            while unique_name in all_names:
                suffix_i += 1
                unique_name = f"{out_name}__{suffix_i}"
            out_name = unique_name
            all_names.add(out_name)

            generated_scenarios.append(
                {
                    "name": out_name,
                    "n_steady_states": 1,
                    "steady_states": [copy.deepcopy(ss)],
                }
            )

    added_scenarios = generated_scenarios
    final_scenarios = copy.deepcopy(existing_scenarios) + copy.deepcopy(added_scenarios)

    root: Dict[str, Any] = {**extra}
    root["common_params"] = dict(common_params)
    root["scenarios"] = final_scenarios
    root["steady_states_filtered"] = copy.deepcopy(final_scenarios)

    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(root, f, indent=2, ensure_ascii=False, default=str)
    print(
        f"scenarios.json (híbrido) actualizado: {dest} "
        f"({len(existing_scenarios)} existentes, {len(added_scenarios)} agregados)"
    )


def _eq_record_to_fields(prefix: str, rec: Dict[str, Any]) -> Dict[str, str]:
    return {
        f"{prefix}_FOUND": "Y",
        f"{prefix}_C": _fmt_scenario_value(rec.get("c_star")),
        f"{prefix}_S": _fmt_scenario_value(rec.get("s_star")),
        f"{prefix}_I": _fmt_scenario_value(rec.get("i_star")),
        f"{prefix}_RESIDUAL_L2": _fmt_scenario_value(rec.get("residual_l2")),
        f"{prefix}_MAX_REAL": _fmt_scenario_value(rec.get("max_real")),
        f"{prefix}_UNSTABLE": "Y" if rec.get("unstable") else "N",
        f"{prefix}_CONTROL_MODE": str(rec.get("control_mode", "")),
    }


def build_corner_enriched_payload(
    common_params: Dict[str, Any],
    scenarios: List[Dict[str, Any]],
    *,
    rows_accumulator: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], List[str], List[Dict[str, Any]]]:
    """
    Construye ``{"common_params": ..., "scenarios": [...]}`` como en scenarios_v1,
    copiando cada escenario y añadiendo claves ``EQ_C0S1I0_*`` / ``EQ_C0S1I1_*``.

    También devuelve ``nested_for_full_run`` (lista de bloques ``name`` + ``steady_states``)
    para ``steady_states_full_run.json`` → ``strong_corner``. El ``name`` de cada bloque en
    salida lleva sufijo ``_c0_s1_i0`` / ``_c0_s1_i1`` / ``_c0_s1_i0_c0_s1_i1`` según ramas
    encontradas; ``scenario_json_name`` dentro de cada ``steady_states[*]`` conserva el
    nombre del escenario en el JSON de entrada.
    """
    missing: List[str] = []
    out_scenarios: List[Dict[str, Any]] = []
    nested_for_full_run: List[Dict[str, Any]] = []

    for i, sc in enumerate(scenarios, 1):
        nm = sc.get("name", f"scenario_{i}")
        i0_list, i1_list = collect_corner_equilibria_for_scenario(common_params, sc)
        if not i0_list and not i1_list:
            missing.append(nm)

        steady_inner: List[Dict[str, Any]] = []
        eq_idx = 0
        for lst in (i0_list, i1_list):
            if lst:
                steady_inner.append(_inner_equilibrium_record(lst[0], eq_idx))
                eq_idx += 1

        out_nm = _corner_scenario_block_name(nm, steady_inner)
        print(f"  [{i}/{len(scenarios)}] {out_nm} ...", flush=True)

        merged: Dict[str, Any] = dict(sc)
        merged["name"] = out_nm
        merged["n_steady_states"] = len(steady_inner)
        merged["steady_states"] = copy.deepcopy(steady_inner)
        p0, p1 = "EQ_C0S1I0", "EQ_C0S1I1"
        if i0_list:
            merged.update(_eq_record_to_fields(p0, i0_list[0]))
            if rows_accumulator is not None:
                rows_accumulator.append(i0_list[0])
        else:
            merged.update(_empty_eq_fields(p0))
        if i1_list:
            merged.update(_eq_record_to_fields(p1, i1_list[0]))
            if rows_accumulator is not None:
                rows_accumulator.append(i1_list[0])
        else:
            merged.update(_empty_eq_fields(p1))

        out_scenarios.append(merged)

        nested_for_full_run.append(
            {
                "name": out_nm,
                "n_steady_states": len(steady_inner),
                "steady_states": copy.deepcopy(steady_inner),
            }
        )

    payload = {
        "common_params": dict(common_params),
        "scenarios": out_scenarios,
    }
    return payload, missing, nested_for_full_run


def run_all_scenarios(
    scenarios_path: Path,
    out_json: Path,
    out_csv: Optional[Path] = None,
    *,
    write_full_run_json: bool = True,
    full_run_local_only: bool = False,
    write_scenarios_json: bool = True,
    scenarios_json_out: Optional[Path] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    common, scenarios = load_scenarios(scenarios_path)
    rows: List[Dict[str, Any]] = []
    payload, missing, nested_fr = build_corner_enriched_payload(
        common, scenarios, rows_accumulator=rows
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nJSON escrito: {out_json}  ({len(payload['scenarios'])} escenarios)")
    if rows:
        vc = pd.DataFrame(rows)["target_branch"].value_counts()
        print("Por rama:", {str(k): int(v) for k, v in vc.items()})
    if missing:
        print(f"Sin equilibrio en ninguna de las dos cajas ({len(missing)} escenarios):")
        for m in missing[:30]:
            print(f"  - {m}")
        if len(missing) > 30:
            print(f"  ... y {len(missing) - 30} más")

    if write_full_run_json:
        write_steady_states_full_run_for_corners(
            scenarios_path,
            nested_fr,
            common,
            full_run_local_only=full_run_local_only,
        )
    else:
        print("steady_states_full_run.json omitido (--skip-full-run-json).")

    if write_scenarios_json:
        sj = scenarios_json_out if scenarios_json_out is not None else (_ALLEE_ROOT / "scenarios.json")
        write_allee_scenarios_json_hybrid(sj, common, nested_fr, preserve_extra_root_keys=True)
    else:
        print("scenarios.json omitido (--skip-scenarios-json).")

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"CSV escrito: {out_csv}  ({len(rows)} filas)")
    return payload, df


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Newton 3D: equilibrios cerca de (0,1,0) y (0,1,1) por escenario."
    )
    ap.add_argument(
        "--scenarios-file",
        type=Path,
        default=_ALLEE_ROOT / "scenarios.json",
        help="Ruta a scenarios.json (o scenarios_v1.json)",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=_ALLEE_ROOT / "corner_equilibria_scenarios.json",
        help=(
            "Salida JSON (misma forma que scenarios_v1 + EQ_C0S1I0_* / EQ_C0S1I1_*). "
            "Por defecto: raíz Allee/corner_equilibria_scenarios.json (junto a scenarios.json)."
        ),
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Si se indica, escribe también un CSV agregado por rama encontrada",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Legado: equivalente a --out-csv",
    )
    ap.add_argument(
        "--skip-full-run-json",
        action="store_true",
        help="No escribir steady_states_full_run.json (Drive/RESULTS_DIR o respaldo local).",
    )
    ap.add_argument(
        "--full-run-json-local-only",
        action="store_true",
        help="Forzar steady_states_full_run.json solo en Allee/estados_estacionarios/ (sin Drive).",
    )
    ap.add_argument(
        "--skip-scenarios-json",
        action="store_true",
        help="No escribir scenarios.json (estructura híbrida common_params+scenarios+steady_states_filtered).",
    )
    ap.add_argument(
        "--scenarios-json-out",
        type=Path,
        default=None,
        help="Ruta del scenarios.json híbrido a escribir (por defecto: Allee/scenarios.json).",
    )
    args = ap.parse_args()
    if not args.scenarios_file.is_file():
        print(f"No existe el archivo: {args.scenarios_file}", file=sys.stderr)
        sys.exit(1)
    csv_path = args.out_csv or args.out
    print(f"Escenarios: {args.scenarios_file}")
    run_all_scenarios(
        args.scenarios_file,
        args.out_json,
        out_csv=csv_path,
        write_full_run_json=not bool(args.skip_full_run_json),
        full_run_local_only=bool(args.full_run_json_local_only),
        write_scenarios_json=not bool(args.skip_scenarios_json),
        scenarios_json_out=args.scenarios_json_out,
    )


if __name__ == "__main__":
    main()
