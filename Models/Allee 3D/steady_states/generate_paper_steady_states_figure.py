"""
Generate the paper Fig. 1 phase portrait with all available steady-state markers.

This script keeps the simplex-style view used by the manuscript figure, but it
collects red markers from the scenario files, CSV exports, and the raw steady
state catalog so filtered-out roots can be shown again in the (c, s) projection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import sympy as sp
from scipy.optimize import root

_ALLEE_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _ALLEE_ROOT.parent.parent
if str(_ALLEE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALLEE_ROOT))

from model_equations import c_sym, s_sym
from model_parameters import ModelParameters
from steady_states.generate_phase_planes import (
    SIMPLEX_PHASE_PLANE_C_RANGE,
    SIMPLEX_PHASE_PLANE_S_RANGE,
    build_reduced_model_2d_sympy_strong,
    plot_nullclines_2d,
)


DEFAULT_SCENARIO_NAME = "strong_mu1_uSi_bajo_umbral_c0_s1_i0"
DEFAULT_OUTPUT = _PROJECT_ROOT / "Paper" / "figures" / "steady_states.png"
DEFAULT_TABLE_OUTPUT = _PROJECT_ROOT / "Paper" / "figures" / "steady_states_i_table.tex"


def _as_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return 0.0 if abs(out) < 1e-12 else out


def _iter_points_from_json_obj(obj) -> Iterable[tuple[float, float, float | None]]:
    if isinstance(obj, dict):
        if "c_star" in obj and "s_star" in obj:
            c_val = _as_float(obj.get("c_star"))
            s_val = _as_float(obj.get("s_star"))
            i_val = _as_float(obj.get("i_star"))
            if c_val is not None and s_val is not None:
                yield (c_val, s_val, i_val)
        for value in obj.values():
            yield from _iter_points_from_json_obj(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_points_from_json_obj(item)


def _iter_points_from_json(path: Path) -> Iterable[tuple[float, float, float | None]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    yield from _iter_points_from_json_obj(data)


def _iter_points_from_csv(path: Path) -> Iterable[tuple[float, float, float | None]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            c_val = _as_float(row.get("c_star"))
            s_val = _as_float(row.get("s_star"))
            i_val = _as_float(row.get("i_star"))
            if c_val is not None and s_val is not None:
                yield (c_val, s_val, i_val)


def _dedupe_projected_points(
    points: Iterable[tuple[float, float, float | None]],
) -> list[tuple[float, float, float | None]]:
    unique: list[tuple[float, float, float | None]] = []
    min_dist = 0.025
    for c_val, s_val, i_val in points:
        c_plot = round(c_val, 3)
        s_plot = round(s_val, 3)
        if any(math.hypot(c_plot - c_old, s_plot - s_old) < min_dist for c_old, s_old, _ in unique):
            continue
        unique.append((c_plot, s_plot, i_val))
    return sorted(unique, key=lambda p: (p[0], p[1], -1.0 if p[2] is None else p[2]))


def reduced_i_value(c_val: float, s_val: float, params: ModelParameters) -> float:
    """Algebraic i(c, s) used by the reduced two-dimensional system."""
    numerator = (
        2 * params.rd
        + 2 * params.delta * s_val**2
        - c_val**2 * (2 * params.eta + params.beta * params.mu)
    )
    return numerator / (2 * params.rd)


def collect_steady_state_points() -> list[tuple[float, float, float | None]]:
    """Collect projected steady states from all local catalog sources."""
    csv_sources = [
        _ALLEE_ROOT / "steady_states_scenarios.csv",
        _ALLEE_ROOT / "results_steady_states_run" / "steady_states_scenarios.csv",
    ]
    csv_sources.extend((_ALLEE_ROOT / "results_steady_states_run").glob("*/steady_states_scenarios.csv"))

    json_sources = [
        _ALLEE_ROOT / "scenarios.json",
        _ALLEE_ROOT / "steady_states_catalog.json",
        _ALLEE_ROOT / "estados_estacionarios" / "steady_states_full_run.json",
    ]

    points: list[tuple[float, float, float | None]] = []
    for path in csv_sources:
        points.extend(_iter_points_from_csv(path))
    for path in json_sources:
        points.extend(_iter_points_from_json(path))

    return _dedupe_projected_points(points)


def _get_float(params: dict, key: str, default: float = 0.0) -> float:
    value = params.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_reference_parameters(scenario_name: str = DEFAULT_SCENARIO_NAME) -> ModelParameters:
    scenarios_path = _ALLEE_ROOT / "scenarios.json"
    with scenarios_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    common = data.get("common_params", {})
    scenarios = data.get("scenarios", [])
    scenario = next((item for item in scenarios if item.get("name") == scenario_name), None)
    if scenario is None and scenarios:
        scenario = scenarios[0]
    combined = {**common, **(scenario or {})}

    return ModelParameters(
        rc=_get_float(combined, "rc"),
        rs=_get_float(combined, "rs"),
        rd=_get_float(combined, "rd"),
        alpha=_get_float(combined, "alpha"),
        delta=_get_float(combined, "delta"),
        beta=_get_float(combined, "beta"),
        a=_get_float(combined, "alle", _get_float(combined, "a", 0.1)),
        gamma=_get_float(combined, "gamma"),
        eta=_get_float(combined, "eta"),
        mu=_get_float(combined, "mu"),
        allee_type="STRONG",
        use_adaptive_control=str(combined.get("USE_ADAPTIVE_CONTROL", "N")).upper() == "Y",
        control_uses_hill=str(combined.get("HILL_CONTROL", "N")).upper() == "Y",
        ku=_get_float(combined, "KU", 0.2),
        eps_u=_get_float(combined, "EPS_U", 1e-3),
        u_max=_get_float(combined, "U_MAX") if combined.get("U_MAX") else None,
    )


def find_reduced_2d_roots(
    f1,
    f2,
    params: ModelParameters,
    c_range: tuple[float, float] = SIMPLEX_PHASE_PLANE_C_RANGE,
    s_range: tuple[float, float] = SIMPLEX_PHASE_PLANE_S_RANGE,
) -> list[tuple[float, float, float]]:
    """Find stationary points of the reduced 2D phase-plane system."""

    roots: list[tuple[float, float, float]] = []
    c_grid = np.linspace(c_range[0], c_range[1], 13)
    s_grid = np.linspace(s_range[0], s_range[1], 13)

    def system(x):
        return [float(f1(x[0], x[1])), float(f2(x[0], x[1]))]

    for c0 in c_grid:
        for s0 in s_grid:
            try:
                sol = root(system, [c0, s0], method="hybr")
            except Exception:
                continue
            if not sol.success:
                continue

            c_val, s_val = (_as_float(sol.x[0]), _as_float(sol.x[1]))
            if c_val is None or s_val is None:
                continue
            if not (c_range[0] - 0.05 <= c_val <= c_range[1] + 0.05):
                continue
            if not (s_range[0] - 0.05 <= s_val <= s_range[1] + 0.05):
                continue
            if abs(float(f1(c_val, s_val))) > 1e-6 or abs(float(f2(c_val, s_val))) > 1e-6:
                continue
            if any(math.hypot(c_val - c_old, s_val - s_old) < 1e-4 for c_old, s_old, _ in roots):
                continue
            roots.append((c_val, s_val, reduced_i_value(c_val, s_val, params)))

    return sorted(roots, key=lambda p: (p[0], p[1]))


def write_i_table(points: list[tuple[float, float, float | None]], table_path: Path = DEFAULT_TABLE_OUTPUT) -> None:
    def fmt(value: float) -> str:
        value = 0.0 if abs(value) < 5e-4 else value
        return f"{value:.3f}"

    table_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{@{}ccc@{}}",
        "\\toprule",
        "$c^*$ & $s^*$ & $i^*$ \\\\",
        "\\midrule",
    ]
    for c_val, s_val, i_val in points:
        i_text = "---" if i_val is None else fmt(i_val)
        lines.append(f"{fmt(c_val)} & {fmt(s_val)} & {i_text} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    table_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"OK: wrote {table_path}")


def generate_paper_steady_states_figure(
    output_path: Path = DEFAULT_OUTPUT,
    table_path: Path = DEFAULT_TABLE_OUTPUT,
) -> None:
    params = load_reference_parameters()
    f1_sym, f2_sym = build_reduced_model_2d_sympy_strong(params)
    f1 = sp.lambdify((c_sym, s_sym), f1_sym, modules="numpy")
    f2 = sp.lambdify((c_sym, s_sym), f2_sym, modules="numpy")
    points = _dedupe_projected_points(
        [*collect_steady_state_points(), *find_reduced_2d_roots(f1, f2, params)]
    )
    if not points:
        raise RuntimeError("No steady-state points found in local CSV/JSON sources or reduced 2D roots.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_nullclines_2d(
        f1,
        f2,
        c_range=SIMPLEX_PHASE_PLANE_C_RANGE,
        s_range=SIMPLEX_PHASE_PLANE_S_RANGE,
        mu_val=float(params.mu),
        allee_type=params.allee_type,
        title="Steady States of the System",
        save_path=output_path,
        steady_states_points=points,
        simplex_corner_view=True,
    )
    write_i_table(points, table_path)
    print(f"OK: wrote {output_path}")
    print(f"Projected steady-state markers: {len(points)}")
    for c_val, s_val, i_val in points:
        i_txt = "nan" if i_val is None else f"{i_val:.6g}"
        print(f"  c={c_val:.6g}, s={s_val:.6g}, i={i_txt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Paper Fig. 1 steady-state phase portrait.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--table-output", type=Path, default=DEFAULT_TABLE_OUTPUT)
    args = parser.parse_args()
    generate_paper_steady_states_figure(args.output, args.table_output)


if __name__ == "__main__":
    main()
