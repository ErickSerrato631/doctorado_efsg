"""
correlation_comparison.py

Builds publication figures for spatial correlation lengths ξ(t) from
`correlation_fourier.py` outputs:
  correlations/corr_length_real_inverse_nb_<block>_<field1>_<field2>.txt

Paper pipeline (STRONG_BAJO_ONLY): one panel per correlation type with the four
strong-Allee, below-threshold scenarios (μ ∈ {0,1} × adaptive control off/on).
Each base PNG is also saved as _loglog and _semilogx variants.

Output:
- PNGs: corr_grid_<corr>.png (+ _loglog, _semilogx)
- summary: correlation_summary.json / correlation_summary.csv

Terminology (thesis / journal): μ = variational morphological selector;
u = Hill-type adaptive immunotherapy control (uNo ≡ 0, uSi > 0).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

_ALLEE_ROOT = Path(__file__).resolve().parent.parent
if str(_ALLEE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALLEE_ROOT))

import numpy as np

# Forzar backend no-interactivo para WSL/headless (evita errores de Qt/Wayland)
os.environ.setdefault("MPLBACKEND", "Agg")


CORR_TYPES = ["c_c", "s_s", "i_i", "c_s", "c_i", "s_i"]
# Autocorrelaciones: el grid μ debe comparar u=0 vs u>0 (no solo sin control)
AUTO_CORR_TYPES = frozenset({"c_c", "s_s", "i_i"})
# Variantes de ejes guardadas junto a cada PNG base
PLOT_SCALES = ("linear", "loglog", "semilogx")
PAPER_FIGSIZE = (8.6, 5.2)
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


def scaled_out_path(path: Path, scale: str) -> Path:
    if scale == "linear":
        return path
    return path.with_name(f"{path.stem}_{scale}{path.suffix}")


def _scale_suptitle_suffix(scale: str) -> str:
    if scale == "loglog":
        return " [log-log]"
    if scale == "semilogx":
        return " [semilog t]"
    return ""


def _apply_axis_scale(ax, scale: str, t_plot_max: float, t_min_pos: float = 0.05) -> None:
    """Configura ejes lineales, log-log o semilog (log t, ξ lineal)."""
    if scale == "linear":
        ax.set_xlim(0.0, t_plot_max)
        return
    t_left = max(float(t_min_pos), 1e-4)
    t_right = max(t_left * 1.01, float(t_plot_max))
    if scale in ("loglog", "semilogx"):
        ax.set_xscale("log")
        ax.set_xlim(t_left, t_right)
    if scale == "loglog":
        ax.set_yscale("log")
        ymin, ymax = ax.get_ylim()
        if ymax > ymin > 0:
            ax.set_ylim(max(ymin, 1e-6), ymax * 1.05)


def _apply_paper_axis_style(ax) -> None:
    ax.set_facecolor(PAPER_FACE)
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
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.xaxis.label.set_color(PAPER_TEXT)
    ax.yaxis.label.set_color(PAPER_TEXT)


def _style_legend(ax) -> None:
    leg = ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        fontsize=8.4,
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        borderpad=0.65,
        labelspacing=0.55,
        handlelength=3.0,
    )
    if leg is None:
        return
    frame = leg.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("#cfd6df")
    frame.set_linewidth(0.8)


def _apply_paper_rc(plt) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "axes.titleweight": "semibold",
        "figure.dpi": 120,
        "savefig.bbox": "tight",
    })
# Para alinear con termodinámica: solo Strong Allee, bajo umbral (4 escenarios)
STRONG_BAJO_ONLY = True  # True = paper: excluir weak y sobre_umbral
ALLEE_TYPES = ["STRONG"] if STRONG_BAJO_ONLY else ["WEAK", "STRONG"]
UMBRAL_TYPES = ["bajo"] if STRONG_BAJO_ONLY else ["bajo", "sobre"]
# Cuadrícula paper (0,1,0) — solo estos cuatro bajo scenarios.json / RESULTS_DIR
STRONG_BAJO_PAPER_SCENARIOS = frozenset({
    "strong_mu0_uNo_bajo_umbral_c0_s1_i0",
    "strong_mu0_uSi_bajo_umbral_c0_s1_i0",
    "strong_mu1_uNo_bajo_umbral_c0_s1_i0",
    "strong_mu1_uSi_bajo_umbral_c0_s1_i0",
})


def obtener_tipo_correlacion(file_path: str) -> str:
    """
    Extrae el tipo de correlación del nombre del archivo.
    
    Formato esperado:
      corr_length_real_inverse_nb_<block>_<field1>_<field2>.txt
      e.g. corr_length_real_inverse_nb_1_c_s.txt
    """
    base = Path(file_path).name.replace(".txt", "")
    if "_nb_" not in base:
        return "Desconocido"

    try:
        suffix = base.split("_nb_", 1)[1]  # "<block>_<field1>_<field2>"
        parts = suffix.split("_")
        if len(parts) >= 3:
            # parts[0] es block
            return f"{parts[1]}_{parts[2]}"
    except Exception:
        pass
    return "Desconocido"


def _default_results_dir(base_dir: Path) -> Path:
    """Usa detección automática (Drive si está montado)."""
    try:
        from utils_paths import get_results_dir  # local import para evitar problemas si no está
        return get_results_dir(base_dir)
    except Exception as exc:
        raise RuntimeError(
            "No hay directorio de resultados en Drive. Monta Google Drive con "
            "mount_google_drive.sh o define RESULTS_DIR apuntando a Drive."
        ) from exc


def setup_directories(
    base_dir: Optional[Path],
    results_dir: Optional[Path],
    output_dir: Optional[Path],
) -> Tuple[Path, Path, Path]:
    """
    Returns: (BASE_DIR, RESULTS_DIR, OUTPUT_DIR)
    """
    BASE_DIR = Path(base_dir) if base_dir else _ALLEE_ROOT
    RESULTS_DIR = Path(results_dir) if results_dir else _default_results_dir(BASE_DIR)
    OUTPUT_DIR = Path(output_dir) if output_dir else (RESULTS_DIR / "comparisons" / "correlation_grids")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return BASE_DIR, RESULTS_DIR, OUTPUT_DIR


def load_scenarios_from_json(scenarios_file: Path) -> Tuple[Dict, List[Dict]]:
    with open(scenarios_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("common_params", {}) or {}, data.get("scenarios", []) or []


def _infer_umbral_from_name(name: str) -> str:
    if "bajo_umbral" in name:
        return "bajo"
    if "sobre_umbral" in name:
        return "sobre"
    return "na"


def _infer_u_from_name(name: str) -> str:
    # compat: nombres incluyen uNo/uSi
    if "_uSi_" in name or name.endswith("_uSi_bajo_umbral") or name.endswith("_uSi_sobre_umbral"):
        return "uSi"
    if "_uNo_" in name or name.endswith("_uNo_bajo_umbral") or name.endswith("_uNo_sobre_umbral"):
        return "uNo"
    return "uNA"


def _infer_allee_from_name(name: str) -> str:
    lower = name.lower()
    if lower.startswith("strong_") or "_strong_" in lower:
        return "STRONG"
    if lower.startswith("weak_") or "_weak_" in lower:
        return "WEAK"
    return "na"


def _infer_mu_from_name(name: str) -> str:
    """Variational parameter μ from scenario name (mu0/mu1)."""
    if "_mu1_" in name or name.startswith("strong_mu1") or name.startswith("weak_mu1"):
        return "1"
    if "_mu0_" in name or name.startswith("strong_mu0") or name.startswith("weak_mu0"):
        return "0"
    return "na"


def _mu_variational_label(mu: str) -> str:
    """Variational morphological selector μ (not the adaptive control field u)."""
    return rf"$\mu = {mu}$"


def _u_adaptive_label(u_tag: str) -> str:
    """Adaptive Hill immunotherapy control."""
    if u_tag == "uSi":
        return "adaptive control on (Hill, $u>0$)"
    if u_tag == "uNo":
        return "no adaptive control ($u=0$)"
    return u_tag


def _series_label(mu: str, u_tag: str) -> str:
    return f"{_mu_variational_label(mu)}, {_u_adaptive_label(u_tag)}"


# Backward-compatible aliases (tests / external imports)
_mu_thermo_label = _mu_variational_label
_u_control_label = _u_adaptive_label


def _corr_kind_name(corr_type: str) -> str:
    return "Autocorrelation" if corr_type in AUTO_CORR_TYPES else "Cross-correlation"


def _corr_field_notation(corr_type: str) -> str:
    return {
        "c_c": r"$c_c$",
        "s_s": r"$s_s$",
        "i_i": r"$i_i$",
        "c_s": r"$c$–$s$",
        "c_i": r"$c$–$i$",
        "s_i": r"$s$–$i$",
    }.get(corr_type, corr_type)


def _figure_suptitle(corr_type: str) -> str:
    return (
        f"{_corr_kind_name(corr_type)} {_corr_field_notation(corr_type)}: "
        r"variational $\mu$ and Hill adaptive control"
    )


def _steady_state_row(scenario: Dict) -> Dict:
    """Primera fila de steady_states (formato scenarios.json actual)."""
    raw = scenario.get("steady_states")
    if isinstance(raw, list) and raw:
        row = raw[0]
        return row if isinstance(row, dict) else {}
    return {}


def scenario_tags(name: str, scenario: Dict) -> Dict[str, str]:
    """
    Tags para agrupar/filtrar sin mezclar mecanismos.
    Lee campos planos o steady_states[0]; el nombre hace de respaldo.
    """
    ss = _steady_state_row(scenario)

    allee_raw = (
        scenario.get("ALLEE_TYPE")
        or scenario.get("allee_type")
        or ss.get("ALLEE_TYPE")
        or ss.get("allee_type")
    )
    if allee_raw is not None:
        allee = str(allee_raw).upper()
    else:
        inferred = _infer_allee_from_name(name)
        allee = inferred if inferred != "na" else "WEAK"

    mu_raw = scenario.get("mu", ss.get("mu"))
    if mu_raw is not None:
        mu = str(int(float(mu_raw)))
    else:
        inferred_mu = _infer_mu_from_name(name)
        mu = inferred_mu if inferred_mu != "na" else "0"

    use_u = scenario.get("USE_ADAPTIVE_CONTROL", ss.get("use_adaptive_control"))
    if use_u is None:
        use_u = scenario.get("use_adaptive_control")
    if isinstance(use_u, bool):
        use_adaptive = use_u
    else:
        use_adaptive = str(use_u or "N").upper() in ("Y", "YES", "TRUE", "1")
    u_tag = "uSi" if use_adaptive else "uNo"
    u_from_name = _infer_u_from_name(name)
    if u_from_name in ("uSi", "uNo"):
        u_tag = u_from_name

    umbral = _infer_umbral_from_name(name)

    return {"name": name, "allee": allee, "mu": mu, "u": u_tag, "umbral": umbral}


def scenario_correlation_files(scenario_dir: Path, block: int) -> Dict[str, Path]:
    """
    Devuelve dict corr_type -> file_path para un escenario.
    """
    corr_dir = scenario_dir / "correlations"
    if not corr_dir.exists():
        return {}
    out: Dict[str, Path] = {}
    for fp in corr_dir.glob(f"corr_length_real_inverse_nb_{block}_*.txt"):
        corr_type = obtener_tipo_correlacion(str(fp))
        if corr_type != "Desconocido":
            out[corr_type] = fp
    return out


def load_corr_series(path: Path, t_max: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(-1, 2)
    t = data[:, 0].astype(float)
    xi = data[:, 1].astype(float)
    m = np.isfinite(t) & np.isfinite(xi)
    if t_max is not None:
        m &= t <= float(t_max) + 1e-9
    t, xi = t[m], xi[m]
    order = np.argsort(t)
    t, xi = t[order], xi[order]
    # asegurar tiempos únicos
    t_u, idx = np.unique(t, return_index=True)
    return t_u, xi[idx]


def resolve_T(args: argparse.Namespace, common_params: Dict) -> float:
    if getattr(args, "t_max", None) is not None:
        return float(args.t_max)
    if os.getenv("T") is not None:
        return float(os.getenv("T"))
    return float(common_params.get("T", 1.0))


def resample_to_grid(t: np.ndarray, xi: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    # fuera de dominio => NaN para que no sesgue nanmean
    return np.interp(t_grid, t, xi, left=np.nan, right=np.nan)


@dataclass(frozen=True)
class AggResult:
    t: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    n_eff: np.ndarray
    n_series: int


def aggregate_files(
    files: List[Path], t_grid: np.ndarray, t_clip: Optional[float] = None,
) -> Optional[AggResult]:
    if not files:
        return None
    series = []
    for fp in files:
        try:
            t, xi = load_corr_series(fp, t_max=t_clip)
            series.append(resample_to_grid(t, xi, t_grid))
        except Exception:
                    continue
    if not series:
        return None
    stack = np.vstack(series)  # (n, T)
    n_eff = np.sum(np.isfinite(stack), axis=0)
    mean = np.full(stack.shape[1], np.nan, dtype=float)
    std = np.full(stack.shape[1], np.nan, dtype=float)
    valid = n_eff > 0
    if np.any(valid):
        mean[valid] = np.nanmean(stack[:, valid], axis=0)
        std[valid] = np.nanstd(stack[:, valid], axis=0)
    return AggResult(t=t_grid, mean=mean, std=std, n_eff=n_eff, n_series=stack.shape[0])


def _t_samples(t: np.ndarray, dt: float = 0.25) -> np.ndarray:
    """Muestreo cada dt en t (default 0.25 para errores)."""
    t_min, t_max = float(np.nanmin(t)), float(np.nanmax(t))
    ts = np.arange(0.0, t_max + 0.005, dt)
    return ts[(ts >= t_min) & (ts <= t_max)]


def _positive_mask(t: np.ndarray, y: np.ndarray, scale: str) -> np.ndarray:
    m = np.isfinite(t) & np.isfinite(y)
    if scale != "linear":
        m &= (t > 0) & (y > 0)
    return m


def _plot_experimental_points(
    ax, t: np.ndarray, mean: np.ndarray, color: str, label: str, scale: str = "linear",
) -> None:
    """Mean correlation length as a continuous curve, without point markers."""
    m = _positive_mask(t, mean, scale)
    if not np.any(m):
        return
    t_p, y_p = t[m], mean[m]
    ax.plot(t_p, y_p, color=color, linewidth=1.6, label=label, zorder=4, alpha=0.9)


def _plot_uncertainty_band(
    ax, t: np.ndarray, mean: np.ndarray, std: np.ndarray, color: str, scale: str = "linear",
) -> None:
    """Optional ±1σ band, without point markers."""
    if np.nanmax(std) < 1e-10:
        return
    ylo, yhi = mean - std, mean + std
    m = _positive_mask(t, mean, scale) & (ylo > 0 if scale != "linear" else True) & (yhi > 0 if scale != "linear" else True)
    if not np.any(m):
        return
    t, ylo, yhi = t[m], ylo[m], yhi[m]
    ax.fill_between(t, ylo, yhi, color=color, alpha=0.18, interpolate=True)


def fit_power_law(
    t: np.ndarray,
    y: np.ndarray,
    exponent: Optional[float],
    tmin: float,
    tmax: float,
) -> Optional[Dict]:
    """
    Ajuste y ≈ exp(alpha) * t^m en ventana [tmin, tmax].
    - Si exponent != None, fija m=exponent y estima alpha.
    - Si exponent == None, estima m y alpha por regresión lineal en log-log.
    """
    msk = np.isfinite(t) & np.isfinite(y) & (t > 0) & (y > 0) & (t >= tmin) & (t <= tmax)
    tt = t[msk]
    yy = y[msk]
    if tt.size < 5:
        return None
    lx = np.log(tt)
    ly = np.log(yy)

    if exponent is None:
        # ly ~ a + m*lx
        m, a = np.polyfit(lx, ly, 1)
    else:
        m = float(exponent)
        a = float(np.mean(ly - m * lx))

    pred = a + m * lx
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - float(np.mean(ly))) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "m": float(m),
        "alpha": float(a),
        "r2": float(r2),
        "tmin": float(tmin),
        "tmax": float(tmax),
        "model": f"xi(t) ~ exp({a:.6f}) * t^{m:.6f}",
    }


def _pretty_allee(a: str) -> str:
    return "Weak" if a.upper() == "WEAK" else "Strong"


def _format_power_law_label(series_label: str, fit: Dict) -> str:
    """Legend entry for power-law fit ξ(t) ~ exp(α) t^m."""
    m = float(fit["m"])
    A = math.exp(float(fit["alpha"]))
    r2 = fit.get("r2")
    m_txt = f"{m:.3f}".rstrip("0").rstrip(".")
    if abs(A - 1.0) < 0.05:
        law = rf"$\xi \propto t^{{{m_txt}}}$"
    else:
        law = rf"$\xi \approx {A:.3g}\,t^{{{m_txt}}}$"
    if r2 is not None and math.isfinite(float(r2)):
        return f"{series_label} ({law}, $R^2={float(r2):.3f}$)"
    return f"{series_label} ({law})"


def _plot_power_law_fit(
    ax,
    agg: AggResult,
    color: str,
    data_label: str,
    fit_exponent: Optional[float],
    fit_tmin: float,
    fit_tmax: float,
    show_std: bool,
    scale: str = "linear",
    linestyle: object = "-",
) -> Optional[Dict]:
    """Power-law approximation only, fitted in [fit_tmin, fit_tmax]."""
    fit = fit_power_law(agg.t, agg.mean, fit_exponent, fit_tmin, fit_tmax)
    if fit is None:
        return None

    t_mask = (
        np.isfinite(agg.t)
        & np.isfinite(agg.mean)
        & (agg.t > 0)
        & (agg.t >= fit_tmin)
        & (agg.t <= fit_tmax)
    )
    if scale == "loglog":
        t_mask &= agg.mean > 0
    t_fit = agg.t[t_mask]
    if t_fit.size < 2:
        return fit
    fit_y = np.exp(fit["alpha"]) * np.power(t_fit, fit["m"])
    if scale == "loglog":
        m_line = fit_y > 0
        t_fit, fit_y = t_fit[m_line], fit_y[m_line]
        if t_fit.size < 2:
            return fit
    ax.plot(
        t_fit,
        fit_y,
        color=color,
        linestyle=linestyle,
        linewidth=2.6,
        solid_capstyle="round",
        dash_capstyle="round",
        alpha=0.96,
        label=_format_power_law_label(data_label, fit),
    )
    return fit


def _save_correlation_figure(
    fig,
    axes: Iterable,
    out_path: Path,
    suptitle: str,
    t_plot_max: float,
    scale: str,
    fit_tmin: float,
) -> None:
    import matplotlib.pyplot as plt

    for ax in axes:
        _apply_axis_scale(ax, scale, t_plot_max, t_min_pos=fit_tmin)
        _apply_paper_axis_style(ax)
        _style_legend(ax)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        suptitle + _scale_suptitle_suffix(scale),
        fontsize=14,
        fontweight="semibold",
        color=PAPER_TEXT,
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(scaled_out_path(out_path, scale), dpi=PAPER_DPI, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_all_scales(
    plot_fn: Callable[..., List[Dict]],
    out_path: Path,
    fit_tmin: float,
    **kwargs,
) -> List[Dict]:
    """Genera PNG lineal, _loglog y _semilogx; el summary solo usa la pasada lineal."""
    summary: List[Dict] = []
    for scale in PLOT_SCALES:
        rows = plot_fn(out_path=out_path, scale=scale, fit_tmin=fit_tmin, **kwargs)
        if scale == "linear":
            summary = rows
        print(f"    {scaled_out_path(out_path, scale).name}")
    return summary


def plot_grid_mu(
    corr_type: str,
    data_by_stratum: Dict[Tuple[str, str, str], AggResult],
    out_path: Path,
    fit_exponent: Optional[float],
    fit_tmin: float,
    fit_tmax: float,
    show_std: bool,
    t_plot_max: float = 1.0,
    scale: str = "linear",
) -> List[Dict]:
    """
    Si STRONG_BAJO_ONLY: 1 subplot (Strong | bajo umbral), μ=0 vs μ=1.
    Si no: 2x2 (weak,bajo) (weak,sobre) (strong,bajo) (strong,sobre).
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Falta 'matplotlib'. Instala dependencias (p.ej. `pip install matplotlib`)"
        ) from e

    if STRONG_BAJO_ONLY:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        axes_grid = [[ax]]
        suptitle = f"Correlación {corr_type}: Strong, bajo umbral — μᵢ (potencial químico)"
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        axes_grid = axes
        suptitle = f"Correlación {corr_type}: comparación μ (weak/strong × bajo/sobre)"

    summary_rows: List[Dict] = []

    for r, allee in enumerate(ALLEE_TYPES):
        for c, umbral in enumerate(UMBRAL_TYPES):
            ax = axes_grid[r][c] if not STRONG_BAJO_ONLY else axes_grid[0][0]
            ax.set_title(f"{_pretty_allee(allee)} | {umbral} umbral")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.set_xlabel("t")
            ax.set_ylabel("ξ(t)")

            for mu, color in [("0", "C1"), ("1", "C0")]:
                key = (allee, umbral, mu)
                agg = data_by_stratum.get(key)
                if agg is None:
                    continue
                data_label = _mu_thermo_label(mu)
                fit = _plot_power_law_fit(
                    ax, agg, color, data_label, fit_exponent, fit_tmin, fit_tmax, show_std, scale=scale,
                )
                if fit is not None:
                    summary_rows.append({
                        "comparison": "mu_thermo",
                        "corr_type": corr_type,
                        "allee": allee,
                        "umbral": umbral,
                        "mu": mu,
                        "u": "mixed",
                        "n_series": agg.n_series,
                        **fit,
                    })

            ax.legend(fontsize=8)
            if STRONG_BAJO_ONLY:
                break
        if STRONG_BAJO_ONLY:
            break

    axes_flat = [axes_grid[r][c] for r in range(len(axes_grid)) for c in range(len(axes_grid[0]))]
    _save_correlation_figure(fig, axes_flat, out_path, suptitle, t_plot_max, scale, fit_tmin)
    return summary_rows


def plot_grid_autocorr_u_control(
    corr_type: str,
    data_by_stratum: Dict[Tuple[str, str, str], AggResult],
    out_path: Path,
    fit_exponent: Optional[float],
    fit_tmin: float,
    fit_tmax: float,
    show_std: bool,
    t_plot_max: float = 1.0,
    scale: str = "linear",
) -> List[Dict]:
    """
    Autocorrelaciones: 1×2 paneles por potencial químico μᵢ; en cada panel compara
    control adaptativo u = 0 (Sin) vs u > 0 (Con / Hill).
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Falta 'matplotlib'. Instala dependencias (p.ej. `pip install matplotlib`)"
        ) from e

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    summary_rows: List[Dict] = []
    u_styles = [("uNo", "C1"), ("uSi", "C0")]

    for c, mu in enumerate(["0", "1"]):
        ax = axes[c]
        ax.set_title(_mu_thermo_label(mu))
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_xlabel("t")
        ax.set_ylabel("ξ(t)")

        for u_tag, color in u_styles:
            key = ("STRONG", mu, u_tag)
            agg = data_by_stratum.get(key)
            if agg is None:
                continue
            fit = _plot_power_law_fit(
                ax,
                agg,
                color,
                _u_control_label(u_tag),
                fit_exponent,
                fit_tmin,
                fit_tmax,
                show_std,
                scale=scale,
            )
            if fit is not None:
                summary_rows.append({
                    "comparison": "autocorr_u_control",
                    "corr_type": corr_type,
                    "allee": "STRONG",
                    "umbral": "bajo",
                    "mu": mu,
                    "u": u_tag,
                    "n_series": agg.n_series,
                    **fit,
                })

        ax.legend(fontsize=8)

    suptitle = f"Autocorrelación {corr_type}: control adaptativo u (Sin vs Con Hill), por μᵢ"
    _save_correlation_figure(fig, axes, out_path, suptitle, t_plot_max, scale, fit_tmin)
    return summary_rows


def plot_grid_four_cases(
    corr_type: str,
    data_by_stratum: Dict[Tuple[str, str, str], AggResult],
    out_path: Path,
    fit_exponent: Optional[float],
    fit_tmin: float,
    fit_tmax: float,
    show_std: bool,
    t_plot_max: float = 1.0,
    scale: str = "linear",
) -> List[Dict]:
    """Single panel: four strong-Allee below-threshold scenarios (μ × adaptive control)."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Falta 'matplotlib'. Instala dependencias (p.ej. `pip install matplotlib`)"
        ) from e

    _apply_paper_rc(plt)
    fig, ax = plt.subplots(1, 1, figsize=PAPER_FIGSIZE)
    summary_rows: List[Dict] = []

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\xi(t)$")

    for mu in ["0", "1"]:
        for u_tag in ["uNo", "uSi"]:
            key = ("STRONG", mu, u_tag)
            agg = data_by_stratum.get(key)
            if agg is None:
                continue
            style = PAPER_CURVE_STYLES[(mu, u_tag)]
            fit = _plot_power_law_fit(
                ax,
                agg,
                style["color"],
                _series_label(mu, u_tag),
                fit_exponent,
                fit_tmin,
                fit_tmax,
                show_std,
                scale=scale,
                linestyle=style["linestyle"],
            )
            if fit is not None:
                summary_rows.append({
                    "comparison": "four_cases",
                    "corr_type": corr_type,
                    "allee": "STRONG",
                    "umbral": "bajo",
                    "mu": mu,
                    "u": u_tag,
                    "n_series": agg.n_series,
                    **fit,
                })

    _save_correlation_figure(fig, [ax], out_path, _figure_suptitle(corr_type), t_plot_max, scale, fit_tmin)
    return summary_rows


# Alias for older call sites
plot_grid_autocorr_all_cases = plot_grid_four_cases


def plot_grid_u(
    corr_type: str,
    data_by_stratum: Dict[Tuple[str, str, str], AggResult],
    out_path: Path,
    fit_exponent: Optional[float],
    fit_tmin: float,
    fit_tmax: float,
    show_std: bool,
    t_plot_max: float = 1.0,
    scale: str = "linear",
) -> List[Dict]:
    """
    Si STRONG_BAJO_ONLY: 1x2 (Sin control: μ=0 vs μ=1 | Con control: μ=0 vs μ=1).
    Si no: 2x2 (weak,μ=0) (weak,μ=1) (strong,μ=0) (strong,μ=1) con uNo vs uSi.
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Falta 'matplotlib'. Instala dependencias (p.ej. `pip install matplotlib`)"
        ) from e

    if STRONG_BAJO_ONLY:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        suptitle = f"Correlación {corr_type}: control u (Sin vs Con) — en cada panel μᵢ"
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        suptitle = f"Correlación {corr_type}: comparación u (weak/strong × μ) en bajo umbral"

    summary_rows: List[Dict] = []

    if STRONG_BAJO_ONLY:
        # Panel 0: sin control u | Panel 1: con control Hill — en cada uno μᵢ=0 vs μᵢ=1
        for c, u_tag in enumerate(["uNo", "uSi"]):
            ax = axes[c]
            ax.set_title(_u_control_label(u_tag))
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.set_xlabel("t")
            ax.set_ylabel("ξ(t)")

            for mu, color in [("0", "C1"), ("1", "C0")]:
                key = ("STRONG", mu, u_tag)
                agg = data_by_stratum.get(key)
                if agg is None:
                    continue
                data_label = _mu_thermo_label(mu)
                fit = _plot_power_law_fit(
                    ax, agg, color, data_label, fit_exponent, fit_tmin, fit_tmax, show_std, scale=scale,
                )
                if fit is not None:
                    summary_rows.append({
                        "comparison": "u_vs_mu",
                        "corr_type": corr_type,
                        "allee": "STRONG",
                        "umbral": "bajo",
                        "mu": mu,
                        "u": u_tag,
                        "n_series": agg.n_series,
                        **fit,
                    })

            ax.legend(fontsize=8)
    else:
        for r, allee in enumerate(ALLEE_TYPES):
            for c, mu in enumerate(["0", "1"]):
                ax = axes[r, c]
                ax.set_title(f"{_pretty_allee(allee)} | μ={mu} | bajo umbral")
                ax.grid(True, linestyle="--", alpha=0.35)
                ax.set_xlabel("t")
                ax.set_ylabel("ξ(t)")

                for u_tag, color in [("uNo", "C2"), ("uSi", "C3")]:
                    key = (allee, mu, u_tag)
                    agg = data_by_stratum.get(key)
                    if agg is None:
                        continue
                    data_label = u_tag
                    fit = _plot_power_law_fit(
                        ax, agg, color, data_label, fit_exponent, fit_tmin, fit_tmax, show_std, scale=scale,
                    )
                    if fit is not None:
                        summary_rows.append({
                            "comparison": "u",
                            "corr_type": corr_type,
                            "allee": allee,
                            "umbral": "bajo",
                            "mu": mu,
                            "u": u_tag,
                            "n_series": agg.n_series,
                            **fit,
                        })

                ax.legend(fontsize=8)

    axes_flat = np.atleast_1d(axes).ravel().tolist()
    _save_correlation_figure(fig, axes_flat, out_path, suptitle, t_plot_max, scale, fit_tmin)
    return summary_rows


def _copy_if_requested(src: Path, dst_dir: Optional[Path]) -> None:
    if not dst_dir:
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    try:
        # copy2 intenta copiar metadatos/xattrs; en mounts FUSE (rclone) puede fallar con Errno 5.
        shutil.copy2(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _copy_all_scale_variants(src: Path, dst_dir: Optional[Path]) -> None:
    for scale in PLOT_SCALES:
        _copy_if_requested(scaled_out_path(src, scale), dst_dir)


def write_summary(summary_rows: List[Dict], out_dir: Path) -> None:
    json_path = out_dir / "correlation_summary.json"
    csv_path = out_dir / "correlation_summary.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)

    # CSV con headers estables
    headers = [
        "comparison", "corr_type", "allee", "umbral", "mu", "u", "n_series",
        "m", "alpha", "r2", "tmin", "tmax", "model",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: row.get(k, "") for k in headers})


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera figuras estructuradas de ξ(t) desde correlations/*.txt")
    parser.add_argument("--base-dir", type=str, default=None, help="Directorio base Allee (default: padre de correlations/)")
    parser.add_argument("--results-dir", type=str, default=None, help="Directorio de resultados (default: auto-detecta Drive o local)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directorio de salida (default: RESULTS_DIR/comparisons/correlation_grids)")
    parser.add_argument("--paper-figures-dir", type=str, default=None, help="Si se define, copia PNGs a Paper/figures")
    parser.add_argument("--tesis-images-dir", type=str, default=None, help="Si se define, copia PNGs a Tesis/images")
    parser.add_argument(
        "--T",
        type=float,
        default=None,
        dest="t_max",
        help="Tiempo máximo para rejilla y ejes (default: --T / env T / common_params.T en scenarios.json)",
    )
    parser.add_argument("--block", type=int, default=1, help="Bloque nb a usar (default: 1)")
    parser.add_argument("--corr-types", type=str, nargs="*", default=CORR_TYPES, help="Correlaciones a procesar (default: todas)")
    parser.add_argument("--no-grids", action="store_true", help="Skip correlation-length figure generation")
    parser.add_argument(
        "--no-mu-grids",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-u-grids",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fit-exponent",
        type=float,
        default=-1.0,
        help="Exponente fijo para fit. Default: -1, estima m libremente en log-log.",
    )
    parser.add_argument("--fit-tmin", type=float, default=0.05, help="t mínimo para fit (default: 0.05)")
    parser.add_argument("--fit-tmax", type=float, default=1.0, help="t máximo para fit (default: 1.0)")
    parser.add_argument("--no-std", action="store_true", help="No sombrear ±1σ en curvas agregadas")
    args = parser.parse_args()

    base_dir, results_dir, out_dir = setup_directories(
        Path(args.base_dir) if args.base_dir else None,
        Path(args.results_dir) if args.results_dir else None,
        Path(args.output_dir) if args.output_dir else None,
    )

    scenarios_file = base_dir / "scenarios.json"
    if not scenarios_file.exists():
        print(f"✗ No se encontró scenarios.json en: {scenarios_file}")
        return 2

    common_params, scenarios = load_scenarios_from_json(scenarios_file)
    dt = float(os.getenv("dt", common_params.get("dt", 0.001)))
    T = resolve_T(args, common_params)
    t_grid = np.arange(0.0, T + 0.5 * dt, dt)
    print(f"✓ T (ventana correlaciones): {T}  (dt={dt})")

    paper_dir = Path(args.paper_figures_dir) if args.paper_figures_dir else None
    tesis_dir = Path(args.tesis_images_dir) if args.tesis_images_dir else None

    # Descubrir escenarios disponibles en results_dir
    available = []
    for s in scenarios:
        name = s.get("name")
        if not name:
            continue
        sdir = results_dir / name
        if not (sdir / "correlations").exists():
            continue
        tags = scenario_tags(name, s)
        if STRONG_BAJO_ONLY and name not in STRONG_BAJO_PAPER_SCENARIOS:
            continue
        if STRONG_BAJO_ONLY and (tags["allee"] != "STRONG" or tags["umbral"] != "bajo"):
            continue
        files = scenario_correlation_files(sdir, block=args.block)
        if not files:
            continue
        available.append((tags, files))

    print(f"✓ RESULTS_DIR: {results_dir}")
    print(f"✓ OUTPUT_DIR:  {out_dir}")
    print(f"✓ Escenarios con correlaciones disponibles: {len(available)}")
    if not available:
        print("✗ No hay escenarios con correlations/*.txt. Revisa RESULTS_DIR o ejecuta correlation_fourier.py.")
        return 1

    fit_exponent = None if args.fit_exponent < 0 else float(args.fit_exponent)
    show_std = not args.no_std

    summary_rows: List[Dict] = []

    corr_types = [c for c in args.corr_types if c in CORR_TYPES]
    missing = [c for c in args.corr_types if c not in CORR_TYPES]
    if missing:
        print(f"⚠ Se ignoraron corr-types desconocidos: {missing}")

    skip_grids = args.no_grids or args.no_mu_grids or args.no_u_grids

    for corr in corr_types:
        if skip_grids:
            continue

        data_four: Dict[Tuple[str, str, str], AggResult] = {}
        for allee in ALLEE_TYPES:
            for mu in ["0", "1"]:
                for u_tag in ["uNo", "uSi"]:
                    files: List[Path] = []
                    for tags, fdict in available:
                        if tags["allee"] != allee or tags["umbral"] != "bajo":
                            continue
                        if tags["mu"] != mu or tags["u"] != u_tag:
                            continue
                        fp = fdict.get(corr)
                        if fp is not None:
                            files.append(fp)
                    agg = aggregate_files(files, t_grid, t_clip=T)
                    if agg is not None:
                        data_four[(allee, mu, u_tag)] = agg

        out_png = out_dir / f"corr_grid_{corr}.png"
        print(f"  ✓ {out_png.stem}")
        rows = plot_all_scales(
            plot_grid_four_cases,
            out_png,
            float(args.fit_tmin),
            corr_type=corr,
            data_by_stratum=data_four,
            fit_exponent=fit_exponent,
            fit_tmax=min(float(args.fit_tmax), T),
            show_std=show_std,
            t_plot_max=T,
        )
        summary_rows.extend(rows)
        _copy_all_scale_variants(out_png, paper_dir)
        _copy_all_scale_variants(out_png, tesis_dir)

    write_summary(summary_rows, out_dir)
    print(f"✓ Summary: {out_dir / 'correlation_summary.json'}")
    print(f"✓ Summary: {out_dir / 'correlation_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
