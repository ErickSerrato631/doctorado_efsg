"""
Generate paper-ready time series figures for the nonequilibrium thermodynamics
postprocess.

The script focuses on the positive diffusive dissipation proxy

    Sigma_diss(t) = int_Omega sum_a D_a |grad phi_a|^2 dA, phi_a in {c, s, i},

plus its per-field contributions and the effective chemical potentials already
implemented in termodynamics/calculate_thermodynamic_properties.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

_ALLEE_ROOT = Path(__file__).resolve().parent.parent
if str(_ALLEE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALLEE_ROOT))

from model_parameters import load_from_scenarios_json  # noqa: E402
from termodynamics.calculate_thermodynamic_properties import (  # noqa: E402
    calculate_chemical_potentials,
    calculate_entropy_and_dissipation_integrals,
    get_available_times,
    load_field_matrix,
)
from utils_paths import get_results_dir, get_scenario_dir  # noqa: E402


DEFAULT_SCENARIOS_FILE = _ALLEE_ROOT / "scenarios.json"
DEFAULT_SCENARIOS = [
    "strong_mu0_uNo_bajo_umbral_c0_s1_i0",
    "strong_mu0_uSi_bajo_umbral_c0_s1_i0",
    "strong_mu1_uNo_bajo_umbral_c0_s1_i0",
    "strong_mu1_uSi_bajo_umbral_c0_s1_i0",
]
DEFAULT_OUTPUT_SUBDIR = Path("figures") / "nonequilibrium_time"
DEFAULT_SERIES_SUBDIR = Path("nonequilibrium_time_series")
DEFAULT_T_MAX = 1.0
DEFAULT_SAMPLE_DT = 0.005
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 2.0
PAPER_FIGSIZE = (8.6, 5.2)
PAPER_FIGSIZE_WITH_LEGEND = (10.4, 5.2)
PAPER_DPI = 450
PAPER_FACE = "#fbfbfd"
PAPER_GRID = "#d7dce2"
PAPER_TEXT = "#222831"
PAPER_CURVE_STYLES = {
    ("0", "uNo"): {"color": "#0072B2", "linestyle": "-"},
    ("0", "uSi"): {"color": "#56B4E9", "linestyle": (0, (5, 2))},
    ("1", "uNo"): {"color": "#D55E00", "linestyle": "-"},
    ("1", "uSi"): {"color": "#CC79A7", "linestyle": (0, (5, 2))},
}
FIELD_STYLES = {
    "c": {"color": "#0072B2", "linestyle": "-"},
    "s": {"color": "#009E73", "linestyle": (0, (5, 2))},
    "i": {"color": "#D55E00", "linestyle": (0, (2, 2))},
}
CSV_COLUMNS = [
    "t",
    "Sigma_diss_total",
    "Sigma_diss_c",
    "Sigma_diss_s",
    "Sigma_diss_i",
    "mu_c_avg",
    "mu_s_avg",
    "mu_i_avg",
    "Sigma_mu_total",
    "Sigma_mu_c",
    "Sigma_mu_s",
    "Sigma_mu_i",
]
CSV_HEADER = ",".join(CSV_COLUMNS)


@dataclass
class ScenarioSeries:
    name: str
    label: str
    time: np.ndarray
    sigma_total: np.ndarray
    sigma_c: np.ndarray
    sigma_s: np.ndarray
    sigma_i: np.ndarray
    mu_c: np.ndarray
    mu_s: np.ndarray
    mu_i: np.ndarray
    sigma_mu_total: np.ndarray
    sigma_mu_c: np.ndarray
    sigma_mu_s: np.ndarray
    sigma_mu_i: np.ndarray
    requested_t_max: float
    available_t_max: float
    clipped_to_t_max: bool


def _scenario_tags(name: str) -> tuple[str, str]:
    mu = "1" if "_mu1_" in name else "0"
    u_tag = "uSi" if "_uSi_" in name else "uNo"
    return mu, u_tag


def _scenario_label(name: str) -> str:
    mu, u_tag = _scenario_tags(name)
    control = "u on" if u_tag == "uSi" else "u off"
    return f"mu{mu}, {control}"


def _style_for(name: str) -> Dict[str, str]:
    return PAPER_CURVE_STYLES.get(_scenario_tags(name), {"color": "black", "linestyle": "-"})


def _apply_paper_rc() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "axes.titleweight": "semibold",
        "figure.dpi": 120,
        "savefig.bbox": "tight",
    })


def _apply_paper_axis_style(ax, t_max: float) -> None:
    ax.set_facecolor(PAPER_FACE)
    ax.set_xlim(0.0, float(t_max))
    ax.grid(True, which="major", color=PAPER_GRID, linewidth=0.85, alpha=0.75)
    ax.grid(True, which="minor", color=PAPER_GRID, linewidth=0.45, alpha=0.35)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", labelsize=10, colors=PAPER_TEXT, length=4, width=0.8)
    ax.tick_params(axis="both", which="minor", colors=PAPER_TEXT, length=2.5, width=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#6b7280")
        ax.spines[side].set_linewidth(0.9)
    ax.xaxis.label.set_color(PAPER_TEXT)
    ax.yaxis.label.set_color(PAPER_TEXT)


def _plot_and_legend_axes():
    fig, (ax, legend_ax) = plt.subplots(
        1,
        2,
        figsize=PAPER_FIGSIZE_WITH_LEGEND,
        gridspec_kw={"width_ratios": [4.8, 1.05]},
        layout="constrained",
    )
    legend_ax.axis("off")
    return fig, ax, legend_ax


def _style_legend(ax, legend_ax, *, ncol: int = 1) -> None:
    handles, labels = ax.get_legend_handles_labels()
    leg = legend_ax.legend(
        handles=handles,
        labels=labels,
        loc="center",
        fontsize=8.4,
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        borderpad=0.65,
        labelspacing=0.55,
        handlelength=3.0,
        ncol=ncol,
    )
    if leg is None:
        return
    frame = leg.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("#cfd6df")
    frame.set_linewidth(0.8)


def _style_factor_legends(ax, legend_ax, series_list: Sequence[ScenarioSeries]) -> None:
    scenario_handles = []
    for series in series_list:
        style = _style_for(series.name)
        scenario_handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle="-",
                linewidth=2.0,
                label=series.label,
            )
        )

    field_handles = []
    for field, label in (
        ("c", r"$c$ field"),
        ("s", r"$s$ field"),
        ("i", r"$i$ field"),
    ):
        field_handles.append(
            Line2D(
                [0],
                [0],
                color="#333333",
                linestyle=FIELD_STYLES[field]["linestyle"],
                linewidth=2.0,
                label=label,
            )
        )

    scenario_legend = legend_ax.legend(
        handles=scenario_handles,
        title="Scenario",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.98),
        fontsize=8.1,
        title_fontsize=8.4,
        frameon=True,
        fancybox=True,
        framealpha=0.94,
        borderpad=0.65,
        labelspacing=0.55,
        handlelength=3.0,
    )
    field_legend = legend_ax.legend(
        handles=field_handles,
        title="Line style",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.48),
        fontsize=8.1,
        title_fontsize=8.4,
        frameon=True,
        fancybox=True,
        framealpha=0.94,
        borderpad=0.65,
        labelspacing=0.55,
        handlelength=3.0,
    )
    legend_ax.add_artist(scenario_legend)

    for legend in (scenario_legend, field_legend):
        frame = legend.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("#cfd6df")
        frame.set_linewidth(0.8)


def _available_times_with_retries(
    scenario_dir: Path,
    field: str,
    block: int,
    retry_attempts: int,
    retry_delay: float,
) -> List[float]:
    attempts = max(1, int(retry_attempts))
    for attempt in range(1, attempts + 1):
        try:
            return get_available_times(scenario_dir, field, block)
        except OSError as exc:
            print(
                f"WARN: I/O error listing times for field {field} "
                f"(attempt {attempt}/{attempts}): {exc}",
                flush=True,
            )
            if attempt < attempts:
                time.sleep(max(0.0, float(retry_delay)))
    return []


def _times_with_all_fields(
    scenario_dir: Path,
    block: int,
    retry_attempts: int,
    retry_delay: float,
) -> List[float]:
    time_sets = []
    for field in ("c", "s", "i"):
        times = _available_times_with_retries(
            scenario_dir, field, block, retry_attempts, retry_delay
        )
        time_sets.append({round(float(t), 6) for t in times})
    if not time_sets:
        return []
    common = set.intersection(*time_sets)
    return sorted(common)


def _sample_times_on_grid(times: Sequence[float], t_max: float, sample_dt: Optional[float]) -> List[float]:
    if sample_dt is None or sample_dt <= 0:
        return [float(t) for t in times if float(t) <= float(t_max) + 1e-9]
    if not times:
        return []

    available = np.asarray(sorted(float(t) for t in times), dtype=float)
    grid = np.arange(0.0, float(t_max) + 0.5 * float(sample_dt), float(sample_dt))
    sampled: List[float] = []
    seen: set[float] = set()
    tol = max(float(sample_dt) * 0.25, 1e-9)

    for target in grid:
        idx = int(np.argmin(np.abs(available - target)))
        nearest = float(available[idx])
        if abs(nearest - float(target)) <= tol and nearest <= float(t_max) + 1e-9:
            key = round(nearest, 6)
            if key not in seen:
                seen.add(key)
                sampled.append(nearest)
    return sampled


def _series_csv_path(series_dir: Path, scenario_name: str) -> Path:
    return series_dir / f"{scenario_name}_noneq_time_series.csv"


def _time_key(t: float) -> float:
    return round(float(t), 6)


def _ensure_series_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or path.stat().st_size == 0:
        path.write_text(CSV_HEADER + "\n", encoding="utf-8")


def _load_existing_series_rows(path: Path) -> Dict[float, List[float]]:
    rows: Dict[float, List[float]] = {}
    if not path.is_file():
        return rows
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text or text.startswith("t,"):
                continue
            try:
                values = [float(x) for x in text.split(",")]
            except ValueError:
                print(f"WARN: ignoring malformed checkpoint row {line_no} in {path.name}")
                continue
            if len(values) != len(CSV_COLUMNS):
                print(f"WARN: ignoring incomplete checkpoint row {line_no} in {path.name}")
                continue
            rows[_time_key(values[0])] = values
    return rows


def _append_series_row(path: Path, values: Sequence[float]) -> None:
    _ensure_series_csv(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(",".join(f"{float(v):.18e}" for v in values) + "\n")
        f.flush()


def _series_from_rows(
    scenario_name: str,
    rows: Dict[float, List[float]],
    requested_t_max: float,
    available_t_max: Optional[float] = None,
) -> Optional[ScenarioSeries]:
    selected = [row for key, row in sorted(rows.items()) if key <= float(requested_t_max) + 1e-9]
    if not selected:
        return None
    arr = np.asarray(selected, dtype=float)
    inferred_available_t_max = float(np.max(arr[:, 0]))
    if available_t_max is None or not np.isfinite(float(available_t_max)):
        available_t_max = inferred_available_t_max
    else:
        available_t_max = max(float(available_t_max), inferred_available_t_max)

    return ScenarioSeries(
        name=scenario_name,
        label=_scenario_label(scenario_name),
        time=arr[:, 0],
        sigma_total=arr[:, 1],
        sigma_c=arr[:, 2],
        sigma_s=arr[:, 3],
        sigma_i=arr[:, 4],
        mu_c=arr[:, 5],
        mu_s=arr[:, 6],
        mu_i=arr[:, 7],
        sigma_mu_total=arr[:, 8],
        sigma_mu_c=arr[:, 9],
        sigma_mu_s=arr[:, 10],
        sigma_mu_i=arr[:, 11],
        requested_t_max=float(requested_t_max),
        available_t_max=float(available_t_max),
        clipped_to_t_max=float(available_t_max) > float(requested_t_max) + 1e-9,
    )


def _load_field_matrix_with_retries(
    scenario_dir: Path,
    field_name: str,
    t: float,
    block: int,
    retry_attempts: int,
    retry_delay: float,
) -> Optional[np.ndarray]:
    matrices_dir = scenario_dir / "matrices"
    npz_path = matrices_dir / f"matrix_{field_name}_{float(t):.3f}_nb_{block}.npz"
    txt_path = matrices_dir / f"matrix_{field_name}_{float(t):.3f}_nb_{block}.txt"
    attempts = max(1, int(retry_attempts))

    for attempt in range(1, attempts + 1):
        try:
            if not npz_path.exists() and not txt_path.exists():
                return None
            return load_field_matrix(scenario_dir, field_name, t, block)
        except OSError as exc:
            print(
                f"WARN: I/O error reading {npz_path.name if npz_path.exists() else txt_path.name} "
                f"(attempt {attempt}/{attempts}): {exc}",
                flush=True,
            )
            if attempt < attempts:
                time.sleep(max(0.0, float(retry_delay)))
        except Exception as exc:
            print(f"WARN: failed reading {npz_path if npz_path.exists() else txt_path}: {exc}", flush=True)
            return None
    print(f"WARN: giving up on {npz_path.name if npz_path.exists() else txt_path.name} after {attempts} attempts", flush=True)
    return None


def _safe_fraction(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.zeros_like(numerator, dtype=float)
    mask = np.abs(denominator) > 1e-30
    out[mask] = numerator[mask] / denominator[mask]
    return out


def calculate_series_for_scenario(
    scenario_name: str,
    scenarios_file: Path,
    block: int,
    max_times: Optional[int],
    t_max: float,
    sample_dt: Optional[float],
    series_dir: Path,
    retry_attempts: int,
    retry_delay: float,
    results_dir: Optional[Path] = None,
    scenario_dir: Optional[Path] = None,
) -> Optional[ScenarioSeries]:
    if scenario_dir is not None:
        scenario_path = scenario_dir
    elif results_dir is not None:
        scenario_path = results_dir / scenario_name
    else:
        scenario_path = get_scenario_dir(scenario_name, base_dir=_ALLEE_ROOT)
    csv_path = _series_csv_path(series_dir, scenario_name)
    rows = _load_existing_series_rows(csv_path)
    matrices_dir = scenario_path / "matrices"
    if not matrices_dir.is_dir():
        if rows:
            print(
                f"  checkpoint-only: using {len(rows)} row(s) from {csv_path.name} "
                f"because matrices are unavailable",
                flush=True,
            )
            return _series_from_rows(scenario_name, rows, requested_t_max=t_max)
        print(f"WARN: missing matrices directory for {scenario_name}: {matrices_dir}")
        return None

    all_times = _times_with_all_fields(
        scenario_path, block, retry_attempts=retry_attempts, retry_delay=retry_delay
    )
    times = _sample_times_on_grid(all_times, t_max=t_max, sample_dt=sample_dt)
    if max_times is not None and max_times > 0 and len(times) > max_times:
        idx = np.linspace(0, len(times) - 1, max_times).round().astype(int)
        times = [times[int(i)] for i in sorted(set(idx.tolist()))]
    if not times:
        if rows:
            print(
                f"  checkpoint-only: using {len(rows)} row(s) from {csv_path.name} "
                f"because no complete matrix time grid was found",
                flush=True,
            )
            return _series_from_rows(scenario_name, rows, requested_t_max=t_max)
        print(f"WARN: no complete c/s/i time points in [0, {t_max:g}] for {scenario_name}")
        return None
    available_t_max = float(max(all_times)) if all_times else float("nan")
    completed = set(rows)
    if completed:
        print(f"  checkpoint: {len(completed)} row(s) already in {csv_path.name}", flush=True)

    params = load_from_scenarios_json(
        scenarios_file,
        scenario_name=scenario_name,
        load_spatial_params=True,
    )

    for n, t in enumerate(times, start=1):
        tk = _time_key(t)
        if tk in completed:
            if n == 1 or n % 100 == 0 or n == len(times):
                print(f"  {scenario_name}: checkpoint skip {n}/{len(times)} (t={t:.3f})")
            continue

        c = _load_field_matrix_with_retries(
            scenario_path, "c", t, block, retry_attempts, retry_delay
        )
        s = _load_field_matrix_with_retries(
            scenario_path, "s", t, block, retry_attempts, retry_delay
        )
        i_field = _load_field_matrix_with_retries(
            scenario_path, "i", t, block, retry_attempts, retry_delay
        )
        if c is None or s is None or i_field is None:
            print(f"WARN: skipping incomplete time t={t:.3f} for {scenario_name}", flush=True)
            continue
        if c.shape != s.shape or c.shape != i_field.shape:
            print(f"WARN: shape mismatch at t={t:.6f} for {scenario_name}")
            continue

        nx = c.shape[0]
        dx = float(params.space_size or 1.0) / max(nx - 1, 1)
        mu_data = calculate_chemical_potentials(c, s, i_field, params)
        detail = calculate_entropy_and_dissipation_integrals(
            mu_data["mu_c"],
            mu_data["mu_s"],
            mu_data["mu_i"],
            c,
            s,
            i_field,
            params,
            dx,
        )

        row = [
            float(t),
            detail["int_diss_total"],
            detail["int_diss_c"],
            detail["int_diss_s"],
            detail["int_diss_i"],
            float(np.mean(mu_data["mu_c"])),
            float(np.mean(mu_data["mu_s"])),
            float(np.mean(mu_data["mu_i"])),
            detail["int_mu_total"],
            detail["int_mu_c"],
            detail["int_mu_s"],
            detail["int_mu_i"],
        ]
        rows[tk] = row
        completed.add(tk)
        _append_series_row(csv_path, row)

        if n == 1 or n % 100 == 0 or n == len(times):
            print(f"  {scenario_name}: processed {n}/{len(times)} times (t={t:.3f})")

    if not rows:
        print(f"WARN: no valid rows produced for {scenario_name}")
        return None
    return _series_from_rows(
        scenario_name,
        rows,
        requested_t_max=t_max,
        available_t_max=available_t_max,
    )


def save_series_csv(series: ScenarioSeries, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{series.name}_noneq_time_series.csv"
    data = np.column_stack(
        [
            series.time,
            series.sigma_total,
            series.sigma_c,
            series.sigma_s,
            series.sigma_i,
            series.mu_c,
            series.mu_s,
            series.mu_i,
            series.sigma_mu_total,
            series.sigma_mu_c,
            series.sigma_mu_s,
            series.sigma_mu_i,
        ]
    )
    header = (
        "t,Sigma_diss_total,Sigma_diss_c,Sigma_diss_s,Sigma_diss_i,"
        "mu_c_avg,mu_s_avg,mu_i_avg,Sigma_mu_total,Sigma_mu_c,Sigma_mu_s,Sigma_mu_i"
    )
    np.savetxt(path, data, delimiter=",", header=header, comments="")
    return path


def _short_scenario(name: str) -> str:
    return name.replace("_bajo_umbral_c0_s1_i0", "").replace("strong_", "")


def _plot_availability_note(ax, missing: Sequence[str], incomplete: Sequence[ScenarioSeries]) -> None:
    notes = []
    if missing:
        notes.append("Missing: " + ", ".join(_short_scenario(name) for name in missing))
    if incomplete:
        notes.append(
            "Incomplete to T: "
            + ", ".join(f"{_short_scenario(s.name)} (t={float(np.max(s.time)):.3f})" for s in incomplete)
        )
    if not notes:
        return
    ax.text(
        0.01,
        0.02,
        "\n".join(notes),
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        alpha=0.75,
    )


def _incomplete_series(series_list: Sequence[ScenarioSeries], t_max: float) -> List[ScenarioSeries]:
    return [s for s in series_list if float(np.max(s.time)) < float(t_max) - 1e-9]


def _mark_series_end(ax, series: ScenarioSeries, y: float, t_max: float) -> None:
    t_end = float(np.max(series.time))
    if t_end >= float(t_max) - 1e-9:
        return
    style = _style_for(series.name)
    ax.axvline(t_end, color=style["color"], linewidth=0.8, linestyle=":", alpha=0.75)
    ax.scatter([t_end], [y], color=style["color"], s=18, zorder=5)


def plot_sigma_total(
    series_list: Sequence[ScenarioSeries], missing: Sequence[str], out_dir: Path, t_max: float
) -> Path:
    fig, ax, legend_ax = _plot_and_legend_axes()
    for series in series_list:
        style = _style_for(series.name)
        ax.semilogy(
            series.time,
            series.sigma_total,
            label=series.label,
            linewidth=2.0,
            **style,
        )
        _mark_series_end(ax, series, float(series.sigma_total[-1]), t_max)
    ax.set_xlabel("Time $t$")
    ax.set_ylabel(r"$\Sigma_{\mathrm{diss}}(t)=\int\sum_a D_a|\nabla\phi_a|^2\,dA$")
    ax.set_title("Integrated diffusive dissipation")
    _apply_paper_axis_style(ax, t_max)
    _style_legend(ax, legend_ax)
    _plot_availability_note(ax, missing, _incomplete_series(series_list, t_max))
    out = out_dir / "noneq_sigma_diss_comparison.png"
    fig.patch.set_facecolor("white")
    fig.savefig(out, dpi=PAPER_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def plot_sigma_mu_total(
    series_list: Sequence[ScenarioSeries], missing: Sequence[str], out_dir: Path, t_max: float
) -> Path:
    fig, ax, legend_ax = _plot_and_legend_axes()
    for series in series_list:
        style = _style_for(series.name)
        ax.semilogy(
            series.time,
            series.sigma_mu_total,
            label=series.label,
            linewidth=2.0,
            **style,
        )
        _mark_series_end(ax, series, float(series.sigma_mu_total[-1]), t_max)
    ax.set_xlabel("Time $t$")
    ax.set_ylabel(r"$\Sigma_{\mu}(t)=\int\sum_a D_a|\nabla\mu_a|^2\,dA$")
    ax.set_title("Effective-potential gradient dissipation")
    _apply_paper_axis_style(ax, t_max)
    _style_legend(ax, legend_ax)
    _plot_availability_note(ax, missing, _incomplete_series(series_list, t_max))
    out = out_dir / "noneq_sigma_mu_comparison.png"
    fig.patch.set_facecolor("white")
    fig.savefig(out, dpi=PAPER_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def plot_sigma_by_field(
    series_list: Sequence[ScenarioSeries], missing: Sequence[str], out_dir: Path, t_max: float
) -> List[Path]:
    outputs: List[Path] = []
    fields = [
        ("c", r"$\Sigma_c(t)=\int D_c|\nabla c|^2\,dA$", "sigma_c", "Cancer-field diffusive dissipation"),
        ("s", r"$\Sigma_s(t)=\int D_s|\nabla s|^2\,dA$", "sigma_s", "Healthy-tissue diffusive dissipation"),
        ("i", r"$\Sigma_i(t)=\int D_i|\nabla i|^2\,dA$", "sigma_i", "Immune-field diffusive dissipation"),
    ]
    for field, ylabel, attr, title in fields:
        fig, ax, legend_ax = _plot_and_legend_axes()
        for series in series_list:
            scenario_style = _style_for(series.name)
            ax.semilogy(
                series.time,
                getattr(series, attr),
                label=series.label,
                linewidth=2.0,
                alpha=0.95,
                **scenario_style,
            )
            _mark_series_end(ax, series, float(getattr(series, attr)[-1]), t_max)
        ax.set_xlabel("Time $t$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _apply_paper_axis_style(ax, t_max)
        _style_legend(ax, legend_ax)
        _plot_availability_note(ax, missing, _incomplete_series(series_list, t_max))
        out = out_dir / f"noneq_sigma_{field}_comparison.png"
        fig.patch.set_facecolor("white")
        fig.savefig(out, dpi=PAPER_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        outputs.append(out)
    return outputs


def plot_mu_effective(
    series_list: Sequence[ScenarioSeries], missing: Sequence[str], out_dir: Path, t_max: float
) -> List[Path]:
    outputs: List[Path] = []
    fields = [
        ("c", r"$\langle\mu_c^{\mathrm{eff}}\rangle$", "mu_c", "Effective cancer potential"),
        ("s", r"$\langle\mu_s^{\mathrm{eff}}\rangle$", "mu_s", "Effective healthy-tissue potential"),
        ("i", r"$\langle\mu_i^{\mathrm{eff}}\rangle$", "mu_i", "Effective immune potential"),
    ]
    for field, ylabel, attr, title in fields:
        fig, ax, legend_ax = _plot_and_legend_axes()
        for series in series_list:
            scenario_style = _style_for(series.name)
            ax.plot(
                series.time,
                getattr(series, attr),
                label=series.label,
                linewidth=2.0,
                alpha=0.95,
                **scenario_style,
            )
            _mark_series_end(ax, series, float(getattr(series, attr)[-1]), t_max)
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":", alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Time $t$")
        ax.set_ylabel(ylabel)
        _apply_paper_axis_style(ax, t_max)
        _style_legend(ax, legend_ax)
        _plot_availability_note(ax, missing, _incomplete_series(series_list, t_max))
        out = out_dir / f"noneq_mu_{field}_effective_comparison.png"
        fig.patch.set_facecolor("white")
        fig.savefig(out, dpi=PAPER_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        outputs.append(out)
    return outputs


def plot_dissipation_fractions(
    series_list: Sequence[ScenarioSeries], missing: Sequence[str], out_dir: Path, t_max: float
) -> List[Path]:
    outputs: List[Path] = []
    fields = [
        ("c", r"$\Sigma_c/\Sigma_{\mathrm{diss}}$", "sigma_c", "Cancer fraction of diffusive dissipation"),
        ("s", r"$\Sigma_s/\Sigma_{\mathrm{diss}}$", "sigma_s", "Healthy-tissue fraction of diffusive dissipation"),
        ("i", r"$\Sigma_i/\Sigma_{\mathrm{diss}}$", "sigma_i", "Immune fraction of diffusive dissipation"),
    ]
    for field, ylabel, attr, title in fields:
        fig, ax, legend_ax = _plot_and_legend_axes()
        for series in series_list:
            scenario_style = _style_for(series.name)
            fraction = _safe_fraction(getattr(series, attr), series.sigma_total)
            ax.plot(
                series.time,
                fraction,
                label=series.label,
                linewidth=2.0,
                alpha=0.95,
                **scenario_style,
            )
            _mark_series_end(ax, series, float(fraction[-1]), t_max)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(title)
        ax.set_xlabel("Time $t$")
        ax.set_ylabel(ylabel)
        _apply_paper_axis_style(ax, t_max)
        _style_legend(ax, legend_ax)
        _plot_availability_note(ax, missing, _incomplete_series(series_list, t_max))
        out = out_dir / f"noneq_dissipation_fraction_{field}_comparison.png"
        fig.patch.set_facecolor("white")
        fig.savefig(out, dpi=PAPER_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        outputs.append(out)
    return outputs


def write_summary(
    series_list: Sequence[ScenarioSeries],
    missing: Sequence[str],
    out_dir: Path,
    series_dir: Path,
    results_dir: Optional[Path],
) -> Path:
    payload = {
        "available_scenarios": [
            {
                "name": s.name,
                "n_times": int(len(s.time)),
                "t_min": float(np.min(s.time)),
                "t_max": float(np.max(s.time)),
                "requested_t_max": float(s.requested_t_max),
                "available_t_max": float(s.available_t_max),
                "complete_to_requested_t_max": bool(np.max(s.time) >= s.requested_t_max - 1e-9),
                "clipped_to_requested_t_max": bool(s.clipped_to_t_max),
                "sigma_diss_min": float(np.min(s.sigma_total)),
                "sigma_diss_max": float(np.max(s.sigma_total)),
                "sigma_diss_final": float(s.sigma_total[-1]),
                "sigma_mu_min": float(np.min(s.sigma_mu_total)),
                "sigma_mu_max": float(np.max(s.sigma_mu_total)),
                "sigma_mu_final": float(s.sigma_mu_total[-1]),
                "max_field_sum_error": float(
                    np.max(np.abs((s.sigma_c + s.sigma_s + s.sigma_i) - s.sigma_total))
                ),
            }
            for s in series_list
        ],
        "missing_scenarios": list(missing),
        "source_results_dir": str(results_dir) if results_dir is not None else None,
        "figures_dir": str(out_dir),
        "series_dir": str(series_dir),
    }
    path = series_dir / "noneq_time_figures_summary.json"
    series_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def generate_figures(
    series_list: Sequence[ScenarioSeries],
    missing: Sequence[str],
    out_dir: Path,
    t_max: float,
) -> List[Path]:
    _apply_paper_rc()
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        plot_sigma_total(series_list, missing, out_dir, t_max),
        plot_sigma_mu_total(series_list, missing, out_dir, t_max),
    ]
    outputs.extend(plot_sigma_by_field(series_list, missing, out_dir, t_max))
    outputs.extend(plot_mu_effective(series_list, missing, out_dir, t_max))
    outputs.extend(plot_dissipation_fractions(series_list, missing, out_dir, t_max))
    return outputs


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate nonequilibrium thermodynamics time figures for the paper."
    )
    parser.add_argument("--scenarios-file", type=Path, default=DEFAULT_SCENARIOS_FILE)
    parser.add_argument("--scenario", action="append", dest="scenarios", default=None)
    parser.add_argument("--block", type=int, default=1)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Explicit root containing scenario result folders. Use this for mounted Google Drive.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--series-dir", type=Path, default=None)
    parser.add_argument("--T", type=float, default=DEFAULT_T_MAX, dest="t_max")
    parser.add_argument(
        "--sample-dt",
        type=float,
        default=DEFAULT_SAMPLE_DT,
        help="Sampling interval for output time series. Use <=0 to keep every available time.",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=DEFAULT_RETRY_ATTEMPTS,
        help="Number of attempts for each matrix read from remote storage.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=DEFAULT_RETRY_DELAY,
        help="Seconds to wait between retry attempts after an I/O error.",
    )
    parser.add_argument(
        "--max-times",
        type=int,
        default=None,
        help="Optional downsampling for quick checks. By default all common c/s/i times are used.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    scenarios = args.scenarios or DEFAULT_SCENARIOS
    scenarios_file = args.scenarios_file.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve() if args.results_dir is not None else None
    auto_results_dir = results_dir if results_dir is not None else get_results_dir(_ALLEE_ROOT)
    out_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (auto_results_dir / DEFAULT_OUTPUT_SUBDIR).resolve()
    )
    series_dir = (
        args.series_dir.expanduser().resolve()
        if args.series_dir is not None
        else (auto_results_dir / DEFAULT_SERIES_SUBDIR).resolve()
    )
    t_max = float(args.t_max)

    series_list: List[ScenarioSeries] = []
    missing: List[str] = []

    print(f"Scenarios file: {scenarios_file}")
    print(f"Results source: {auto_results_dir}")
    print(f"Figures output: {out_dir}")
    print(f"Series output: {series_dir}")
    print(f"Time window: 0 <= t <= {t_max:g}")
    print(f"Sampling dt: {float(args.sample_dt):g}")
    print(f"Matrix read retries: {int(args.retry_attempts)} attempt(s), delay={float(args.retry_delay):g}s")

    for scenario in scenarios:
        print(f"\n== {scenario} ==")
        series = calculate_series_for_scenario(
            scenario,
            scenarios_file=scenarios_file,
            block=args.block,
            max_times=args.max_times,
            t_max=t_max,
            sample_dt=float(args.sample_dt),
            series_dir=series_dir,
            retry_attempts=int(args.retry_attempts),
            retry_delay=float(args.retry_delay),
            results_dir=results_dir,
        )
        if series is None:
            missing.append(scenario)
            continue
        csv_path = save_series_csv(series, series_dir)
        print(f"  wrote {csv_path}")
        series_list.append(series)

    if not series_list:
        print("ERROR: no scenarios with valid data were found.")
        return 1

    figure_paths = generate_figures(series_list, missing, out_dir, t_max)
    summary_path = write_summary(series_list, missing, out_dir, series_dir, results_dir)

    print("\nGenerated figures:")
    for path in figure_paths:
        print(f"  {path}")
    print(f"Summary: {summary_path}")
    if missing:
        print("Missing local data for: " + ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
