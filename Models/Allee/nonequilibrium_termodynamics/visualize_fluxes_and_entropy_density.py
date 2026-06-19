"""
visualize_fluxes_and_entropy_density.py

Visualiza gradientes ||∇φ||, magnitudes de corriente difusiva ||J_a|| = D_a||∇φ||,
densidad σ⁺ = Σ_a D_a||∇φ_a||² y quiver opcional de J_c, J_s, J_i sobre c, s, i.

Si el escenario tiene control adaptativo, genera mapa de u y quiver de ∇u sobre u:
``u_Hill_t_*`` / ``quiver_JHill_t_*`` (ley Hill) o ``u_minadapt_t_*`` / ``quiver_Ju_t_*`` (uSi, ku·c/(i+ε)).
Se escriben en ``<RESULTS_DIR o Drive>/<escenario>/nonequilibrium_plots/`` (misma ruta que quiver_Jc).

Por defecto:
- Escenarios: Models/Allee/scenarios_v1.json
- Carpeta de datos: get_scenario_dir(nombre) → RESULTS_DIR o Drive (ver utils_paths)
- Salida: <scenario_dir>/nonequilibrium_plots/*.png

Las figuras PNG se escriben en cuanto termina cada tiempo (no se acumulan todas en RAM).
Con --sigma-csv, sigma_plus_integral_vs_time.csv también se va escribiendo al paso (append).

**Checkpoint (por defecto activo):** en ``nonequilibrium_plots/.visualize_fluxes_checkpoint.json`` se guardan los
tiempos ya graficados; si interrumpes el proceso y relanzas el mismo comando, solo procesa tiempos pendientes.
``--no-checkpoint`` desactiva lectura/escritura y borra un checkpoint previo al inicio.
``--fresh-visualize`` borra checkpoint y (si aplica) el CSV para regenerar desde cero.

Ejemplo (desde Models/Allee):

  python nonequilibrium_termodynamics/visualize_fluxes_and_entropy_density.py \\
      --scenario strong_mu0_uNo_bajo_umbral --time 0.5 1.0

  python nonequilibrium_termodynamics/visualize_fluxes_and_entropy_density.py \\
      --scenario-dir "C:/ruta/local/al/escenario" --time 1.0

  # Todos los escenarios del JSON, todos los tiempos con matriz c:
  python nonequilibrium_termodynamics/visualize_fluxes_and_entropy_density.py \\
      --scenarios scenarios_v1.json --all-scenarios --all-times --sigma-csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

_ALLEE_ROOT = Path(__file__).resolve().parent.parent
if str(_ALLEE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALLEE_ROOT))

from model_parameters import load_from_scenarios_json  # noqa: E402
from termodynamics.calculate_thermodynamic_properties import (  # noqa: E402
    calculate_gradient_2d,
    get_available_times,
    load_field_matrix,
)
from utils_paths import get_scenario_dir  # noqa: E402

DEFAULT_SCENARIOS = _ALLEE_ROOT / "scenarios_v1.json"
CHECKPOINT_FILENAME = ".visualize_fluxes_checkpoint.json"
CHECKPOINT_VERSION = 14

# Campos escalares: negro (bajo) → blanco (alto); flechas del quiver en color aparte
_CMAP_FIELD = "gray"
# Por debajo de esto |J| es numéricamente nulo (no hay dirección)
_QUIVER_MAG_MIN_ABS = 1e-40
# Por encima de esto se usa magnitud real; por debajo, quiver de dirección normalizada
_QUIVER_VISUAL_FULL_MIN = 1e-8
# Longitud de flecha normalizada ≈ esta fracción del paso de malla (evita tapizar en verde)
_QUIVER_NORM_LEN_FRAC = 0.38


def _scalar_color_limits(arr: np.ndarray) -> Tuple[float, float]:
    """
    Escala data_range: negro = mínimo local, blanco = máximo local en ese tiempo.
    Evita vmin=0 fijo cuando s≈1 (todo el mapa en blanco).
    """
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1e-9
    return vmin, vmax


def _log_field_color_scale(label: str, t_str: str, arr: np.ndarray) -> Tuple[float, float]:
    vmin, vmax = _scalar_color_limits(arr)
    print(
        f"  [{label} @ t={t_str}] min={vmin:.6g} max={vmax:.6g} "
        f"-> escala color [{vmin:.6g}, {vmax:.6g}]",
        flush=True,
    )
    return vmin, vmax


def _style_colorbar(cbar, vmin: float, vmax: float) -> None:
    """
    Ticks legibles para artículo: sin offset tipo ``1e-5+1.2245`` en la barra.
    """
    span = float(vmax) - float(vmin)
    if not np.isfinite(span) or span <= 0:
        return
    ax = cbar.ax
    max_abs = max(abs(vmin), abs(vmax), span)
    if max_abs < 1e-3 or max_abs >= 1e4:
        fmt = mticker.ScalarFormatter(useMathText=True)
        fmt.set_useOffset(False)
        fmt.set_scientific(True)
        fmt.set_powerlimits((-3, 4))
    elif span < 0.05:
        fmt = mticker.FormatStrFormatter("%.4f")
    else:
        fmt = mticker.FormatStrFormatter("%.3g")
    ax.yaxis.set_major_formatter(fmt)


def _attach_colorbar(
    fig,
    im,
    ax,
    label: str,
    vmin: float,
    vmax: float,
) -> None:
    cbar = fig.colorbar(im, ax=ax, label=label, shrink=0.94, pad=0.03)
    _style_colorbar(cbar, vmin, vmax)


def _prepare_quiver_components(
    Jx: np.ndarray,
    Jy: np.ndarray,
    bg_vmin: Optional[float],
    bg_vmax: Optional[float],
) -> Tuple[str, float, np.ndarray, np.ndarray]:
    """
    Modo de quiver:
    - ``full``: magnitud real (|J| suficiente para verse con autoscale).
    - ``normalized``: dirección unitaria + ``scale`` fijo (p. ej. u~1e-17 en Hill t=1).
    - ``skip``: |J|≡0.
    """
    mag = np.hypot(Jx, Jy)
    jmax = float(np.nanmax(mag))
    if not np.isfinite(jmax) or jmax <= _QUIVER_MAG_MIN_ABS:
        return "skip", 0.0, Jx, Jy

    if jmax >= _QUIVER_VISUAL_FULL_MIN:
        return "full", jmax, Jx, Jy

    eps = max(jmax * 1e-3, 1e-50)
    Ux = np.zeros_like(Jx, dtype=float)
    Uy = np.zeros_like(Jy, dtype=float)
    mask = mag > eps
    if not np.any(mask):
        return "skip", jmax, Jx, Jy
    Ux[mask] = Jx[mask] / mag[mask]
    Uy[mask] = Jy[mask] / mag[mask]
    return "normalized", jmax, Ux, Uy


def _quiver_norm_scale(space_size: float, nx: int, ny: int, skip: int) -> float:
    """``scale`` de matplotlib: mayor → flechas más cortas. ~40 % del paso entre nodos."""
    sk = max(1, int(skip))
    step_x = float(space_size) / max((nx - 1) / sk, 1.0)
    step_y = float(space_size) / max((ny - 1) / sk, 1.0)
    grid_step = min(step_x, step_y)
    arrow_len = _QUIVER_NORM_LEN_FRAC * grid_step
    return max(1.0 / arrow_len, 1.0)


def _time_checkpoint_key(t: float) -> str:
    """Misma convención que el sufijo de los PNG (tres decimales)."""
    return f"{float(t):.3f}"


def _checkpoint_matches(data: dict, args: argparse.Namespace) -> bool:
    if int(data.get("version", 0)) != CHECKPOINT_VERSION:
        return False
    if int(data.get("block", -1)) != int(args.block):
        return False
    if bool(data.get("no_quiver", False)) != bool(args.no_quiver):
        return False
    if bool(data.get("sigma_csv", False)) != bool(args.sigma_csv):
        return False
    if bool(data.get("control_plots", True)) != bool(not args.no_control_plots):
        return False
    return True


def _save_visualize_checkpoint(ck_path: Path, completed: Set[str], args: argparse.Namespace) -> None:
    payload = {
        "version": CHECKPOINT_VERSION,
        "block": int(args.block),
        "no_quiver": bool(args.no_quiver),
        "sigma_csv": bool(args.sigma_csv),
        "control_plots": bool(not args.no_control_plots),
        "completed": sorted(completed, key=float),
    }
    ck_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = ck_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(ck_path)


def mesh_extent(ny: int, nx: int, space_size: float) -> Tuple[float, float, float, float]:
    """extent para imshow: (left, right, bottom, top) en unidades físicas."""
    return (0.0, space_size, 0.0, space_size)


def grad_magnitude(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    return np.sqrt(gx * gx + gy * gy)


def sigma_plus_density(
    c: np.ndarray,
    s: np.ndarray,
    i: np.ndarray,
    Dc: float,
    Ds: float,
    Di: float,
    dx: float,
) -> np.ndarray:
    gcx, gcy = calculate_gradient_2d(c, dx)
    gsx, gsy = calculate_gradient_2d(s, dx)
    gix, giy = calculate_gradient_2d(i, dx)
    return (
        Dc * (gcx * gcx + gcy * gcy)
        + Ds * (gsx * gsx + gsy * gsy)
        + Di * (gix * gix + giy * giy)
    )


def sigma_plus_integral(sigma: np.ndarray, dx: float) -> float:
    return float(np.sum(sigma) * dx * dx)


def compute_control_field_hill(c: np.ndarray, i: np.ndarray, params) -> Tuple[np.ndarray, str]:
    """
    u tipo Hill — misma ley que ``cancer_dynamics.py`` cuando USE_ADAPTIVE_CONTROL=Y.
    """
    if not params.use_adaptive_control:
        raise ValueError("compute_control_field_hill requiere use_adaptive_control=True")
    i_pos = np.maximum(i, 0.0)
    u_max = float(params.u_max if params.u_max is not None else 1.0)
    nc = float(params.hill_nc)
    ni = float(params.hill_ni)
    kc = float(params.hill_kc)
    ki = float(params.hill_ki)
    c_pow = np.power(c, nc)
    kc_pow = kc**nc
    h_act = c_pow / (kc_pow + c_pow + 1e-16)
    i_pow = np.power(i_pos, ni)
    ki_pow = ki**ni
    h_inh = ki_pow / (ki_pow + i_pow + 1e-16)
    u = u_max * h_act * h_inh
    u = np.clip(np.nan_to_num(u, nan=0.0), 0.0, u_max)
    label = r"$u = u_{\max}\,H_{\mathrm{act}}(c)\,H_{\mathrm{inh}}(i)$ (Hill, sim.)"
    return u, label


def compute_control_field_u(c: np.ndarray, i: np.ndarray, params) -> Tuple[np.ndarray, str]:
    """
    Campo de control u(c,i) según JSON (Hill o min-adaptativo uSi).

    Returns:
        (u_grid, etiqueta_ley) para títulos de figura.
    """
    if not params.use_adaptive_control:
        raise ValueError("compute_control_field_u requiere use_adaptive_control=True")

    i_pos = np.maximum(i, 0.0)
    if params.control_uses_hill:
        u_max = float(params.u_max if params.u_max is not None else 1.0)
        nc = float(params.hill_nc)
        ni = float(params.hill_ni)
        kc = float(params.hill_kc)
        ki = float(params.hill_ki)
        c_pow = np.power(c, nc)
        kc_pow = kc**nc
        h_act = c_pow / (kc_pow + c_pow + 1e-16)
        i_pow = np.power(i_pos, ni)
        ki_pow = ki**ni
        h_inh = ki_pow / (ki_pow + i_pow + 1e-16)
        u = u_max * h_act * h_inh
        u = np.clip(np.nan_to_num(u, nan=0.0), 0.0, u_max)
        label = (
            r"$u = u_{\max}\,H_{\mathrm{act}}(c)\,H_{\mathrm{inh}}(i)$ (Hill)"
        )
    else:
        ku = float(params.ku)
        eps_u = float(params.eps_u)
        u_raw = ku * c / (i_pos + eps_u)
        if params.u_max is not None:
            u = np.minimum(u_raw, float(params.u_max))
            cap = f", cap $u_{{\\max}}$={params.u_max:g}"
        else:
            u = u_raw
            cap = ""
        u = np.clip(np.nan_to_num(u, nan=0.0), 0.0, None)
        label = rf"$u = k_u c/(i+\varepsilon)$ (min-adapt.){cap}"
    return u, label


def plot_control_scalar(
    u: np.ndarray,
    extent: Tuple[float, float, float, float],
    out_path: Path,
    dpi: int,
    title: str,
    cmap: str = _CMAP_FIELD,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2), layout="constrained")
    vmin, vmax = _scalar_color_limits(u)
    im = ax.imshow(
        u,
        origin="lower",
        extent=extent,
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    _attach_colorbar(fig, im, ax, "u", vmin, vmax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"    [control] {out_path.name}", flush=True)


def plot_three_panels(
    arrays: Tuple[np.ndarray, np.ndarray, np.ndarray],
    titles: Tuple[str, str, str],
    cbar_label: str,
    fig_title: str,
    extent: Tuple[float, float, float, float],
    out_path: Path,
    dpi: int,
    share_vmax: bool = True,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), layout="constrained")
    vmax = max(float(np.nanmax(a)) for a in arrays) if share_vmax else None
    for ax, arr, tit in zip(axes, arrays, titles):
        panel_vmax = vmax if vmax is not None else float(np.nanmax(arr))
        im = ax.imshow(
            arr,
            origin="lower",
            extent=extent,
            aspect="equal",
            vmin=0.0,
            vmax=panel_vmax,
            cmap="viridis",
        )
        ax.set_title(tit)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        _attach_colorbar(fig, im, ax, cbar_label, 0.0, panel_vmax)
    fig.suptitle(fig_title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_quiver_flux_on_scalar(
    scalar_bg: np.ndarray,
    Jx: np.ndarray,
    Jy: np.ndarray,
    space_size: float,
    skip: int,
    extent: Tuple[float, float, float, float],
    out_path: Path,
    dpi: int,
    title: str,
    quiver_color: str = "coral",
    bg_cmap: str = _CMAP_FIELD,
    bg_vmin: Optional[float] = None,
    bg_vmax: Optional[float] = None,
    cbar_label: Optional[str] = None,
) -> None:
    """Quiver de un flujo (Jx, Jy) sobre un mapa de fondo del campo escalar asociado."""
    ny, nx = scalar_bg.shape
    xx = np.linspace(0.0, space_size, nx)
    yy = np.linspace(0.0, space_size, ny)
    X, Y = np.meshgrid(xx, yy)
    sk = max(1, int(skip))
    fig, ax = plt.subplots(figsize=(6.4, 5.5), layout="constrained")
    imshow_kw: dict = {
        "origin": "lower",
        "extent": extent,
        "aspect": "equal",
        "cmap": bg_cmap,
    }
    if cbar_label is not None:
        imshow_kw["alpha"] = 1.0
        if bg_vmin is not None:
            imshow_kw["vmin"] = bg_vmin
        if bg_vmax is not None:
            imshow_kw["vmax"] = bg_vmax
    else:
        imshow_kw["alpha"] = 0.85
    im = ax.imshow(scalar_bg, **imshow_kw)
    c_vmin = float(bg_vmin) if bg_vmin is not None else float(np.nanmin(scalar_bg))
    c_vmax = float(bg_vmax) if bg_vmax is not None else float(np.nanmax(scalar_bg))
    if cbar_label is not None:
        _attach_colorbar(fig, im, ax, cbar_label, c_vmin, c_vmax)

    mode, jmax, Qx, Qy = _prepare_quiver_components(Jx, Jy, bg_vmin, bg_vmax)
    if mode == "skip":
        print(
            f"    [quiver] {out_path.name}: |J|≡0; solo mapa escalar",
            flush=True,
        )
    else:
        if mode == "normalized":
            q_scale = _quiver_norm_scale(space_size, nx, ny, sk)
            print(
                f"    [quiver] {out_path.name}: |J|_max={jmax:.3e} → dirección (norm.), "
                f"scale={q_scale:.2f}",
                flush=True,
            )
            q_width = 0.004
            q_zorder = 10
            sk_plot = max(sk, int(sk * 1.5))
        else:
            q_scale = None
            q_width = 0.004
            q_zorder = 5
            sk_plot = sk
        ax.quiver(
            X[::sk_plot, ::sk_plot],
            Y[::sk_plot, ::sk_plot],
            Qx[::sk_plot, ::sk_plot],
            Qy[::sk_plot, ::sk_plot],
            angles="xy",
            scale_units="xy",
            scale=q_scale,
            color=quiver_color,
            width=q_width,
            zorder=q_zorder,
            pivot="mid",
            headwidth=3.5,
            headlength=4.0,
            headaxislength=3.5,
        )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"    [quiver] {out_path.name}", flush=True)


def process_one_time(
    scenario_dir: Path,
    params,
    t: float,
    block: int,
    out_dir: Path,
    dpi: int,
    quiver_skip: int,
    do_quiver: bool,
    do_control_plots: bool = True,
) -> Optional[float]:
    c = load_field_matrix(scenario_dir, "c", t, block)
    s = load_field_matrix(scenario_dir, "s", t, block)
    i = load_field_matrix(scenario_dir, "i", t, block)
    if c is None or s is None or i is None:
        print(f"WARN: Faltan matrices c,s,i en t={t:.3f} (block={block})")
        return None

    if c.shape != s.shape or c.shape != i.shape:
        print(f"WARN: Formas distintas c{s.shape} s{s.shape} i{i.shape}")
        return None

    ny, nx = c.shape
    space_size = float(params.space_size or 1.0)
    dx = space_size / max(nx - 1, 1)

    Dc = float(params.D_c or 0.0)
    Ds = float(params.D_s or 0.0)
    Di = float(params.D_i or 0.0)

    gcx, gcy = calculate_gradient_2d(c, dx)
    gsx, gsy = calculate_gradient_2d(s, dx)
    gix, giy = calculate_gradient_2d(i, dx)

    mag_c = grad_magnitude(gcx, gcy)
    mag_s = grad_magnitude(gsx, gsy)
    mag_i = grad_magnitude(gix, giy)

    Jcx, Jcy = -Dc * gcx, -Dc * gcy
    Jsx, Jsy = -Ds * gsx, -Ds * gsy
    Jix, Jiy = -Di * gix, -Di * giy

    mag_Jc = grad_magnitude(Jcx, Jcy)
    mag_Js = grad_magnitude(Jsx, Jsy)
    mag_Ji = grad_magnitude(Jix, Jiy)

    sigma = sigma_plus_density(c, s, i, Dc, Ds, Di, dx)
    sigma_tot = sigma_plus_integral(sigma, dx)

    ext = mesh_extent(ny, nx, space_size)
    ts = f"{t:.3f}"

    plot_three_panels(
        (mag_c, mag_s, mag_i),
        (r"$\|\nabla c\|$", r"$\|\nabla s\|$", r"$\|\nabla i\|$"),
        r"$|\nabla\phi|$",
        rf"Gradient magnitudes ($t={ts}$)",
        ext,
        out_dir / f"grad_mag_t_{ts}.png",
        dpi,
    )
    plot_three_panels(
        (mag_Jc, mag_Js, mag_Ji),
        (r"$\|\mathbf{J}_c\|$", r"$\|\mathbf{J}_s\|$", r"$\|\mathbf{J}_i\|$"),
        r"$|\mathbf{J}|$",
        rf"Diffusive flux magnitudes ($t={ts}$)",
        ext,
        out_dir / f"J_mag_t_{ts}.png",
        dpi,
    )

    fig, ax = plt.subplots(figsize=(6.0, 5.0), layout="constrained")
    im = ax.imshow(sigma, origin="lower", extent=ext, aspect="equal", cmap="magma")
    ax.set_title(
        r"$\sigma^{+}=\sigma_{\mathrm{diss,tot}}$" + rf" ($t={ts}$)"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    sig_vmin, sig_vmax = _scalar_color_limits(sigma)
    _attach_colorbar(fig, im, ax, r"$\sigma^+$", sig_vmin, sig_vmax)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"sigma_plus_t_{ts}.png", dpi=dpi)
    plt.close(fig)

    if do_quiver:
        c_vmin, c_vmax = _log_field_color_scale("c", ts, c)
        plot_quiver_flux_on_scalar(
            c,
            Jcx,
            Jcy,
            space_size,
            quiver_skip,
            ext,
            out_dir / f"quiver_Jc_t_{ts}.png",
            dpi,
            title=rf"$\mathbf{{J}}_c$ on $c$ ($t={ts}$)",
            quiver_color="coral",
            bg_cmap=_CMAP_FIELD,
            bg_vmin=c_vmin,
            bg_vmax=c_vmax,
            cbar_label="c",
        )
        s_vmin, s_vmax = _log_field_color_scale("s", ts, s)
        plot_quiver_flux_on_scalar(
            s,
            Jsx,
            Jsy,
            space_size,
            quiver_skip,
            ext,
            out_dir / f"quiver_Js_t_{ts}.png",
            dpi,
            title=rf"$\mathbf{{J}}_s$ on $s$ ($t={ts}$)",
            quiver_color="mediumseagreen",
            bg_cmap=_CMAP_FIELD,
            bg_vmin=s_vmin,
            bg_vmax=s_vmax,
            cbar_label="s",
        )
        i_vmin, i_vmax = _log_field_color_scale("i", ts, i)
        plot_quiver_flux_on_scalar(
            i,
            Jix,
            Jiy,
            space_size,
            quiver_skip,
            ext,
            out_dir / f"quiver_Ji_t_{ts}.png",
            dpi,
            title=rf"$\mathbf{{J}}_i$ on $i$ ($t={ts}$)",
            quiver_color="cornflowerblue",
            bg_cmap=_CMAP_FIELD,
            bg_vmin=i_vmin,
            bg_vmax=i_vmax,
            cbar_label="i",
        )

    if do_control_plots and params.use_adaptive_control:
        # Ley Hill = la que integra cancer_dynamics.py (también en escenarios uSi del JSON)
        u_field, u_law = compute_control_field_hill(c, i, params)
        u_vmin, u_vmax = _log_field_color_scale("u (Hill)", ts, u_field)
        print(f"    [control] law: {u_law}", flush=True)
        plot_control_scalar(
            u_field,
            ext,
            out_dir / f"u_Hill_t_{ts}.png",
            dpi,
            title=rf"$u$ (adaptive control) ($t={ts}$)",
            cmap=_CMAP_FIELD,
        )
        if do_quiver:
            gux, guy = calculate_gradient_2d(u_field, dx)
            plot_quiver_flux_on_scalar(
                u_field,
                gux,
                guy,
                space_size,
                quiver_skip,
                ext,
                out_dir / f"quiver_JHill_t_{ts}.png",
                dpi,
                title=rf"$\nabla u$ on $u$ ($t={ts}$)",
                quiver_color="yellowgreen",
                bg_cmap=_CMAP_FIELD,
                bg_vmin=u_vmin,
                bg_vmax=u_vmax,
                cbar_label="u",
            )

    print(f"  t={ts}: integral_sigma_plus ~ {sigma_tot:.6g}  ->  {out_dir}")
    return sigma_tot


def _scenario_names_from_json(path: Path) -> List[str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out: List[str] = []
    for s in data.get("scenarios", []) or []:
        if isinstance(s, dict) and s.get("name"):
            out.append(str(s["name"]))
    return out


def _execute_visualization(
    args: argparse.Namespace,
    scenario_name: Optional[str],
    scenario_dir: Path,
) -> int:
    """
    Genera figuras para un escenario. Devuelve 0 si hubo al menos un tiempo procesado, 1 si no.
    """
    if not scenario_dir.is_dir():
        print(f"WARN: No existe carpeta de escenario: {scenario_dir}")
        return 1

    scenarios_path = args.scenarios.expanduser().resolve()
    params = load_from_scenarios_json(
        scenarios_path,
        scenario_name=scenario_name,
        load_spatial_params=True,
    )
    if params.D_c is None or params.space_size is None:
        print("WARN: scenarios JSON sin D_c/D_s/D_i o space_size; revisa common_params y escenario.")
    out_dir = (args.out_dir or (scenario_dir / "nonequilibrium_plots")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Datos: {scenario_dir}", flush=True)
    print(f"  Salida (Drive/local): {out_dir}", flush=True)
    if params.use_adaptive_control:
        json_law = "Hill (JSON)" if params.control_uses_hill else "min-adapt. (JSON uSi)"
        print(
            f"  Control adaptativo: {json_law}; figuras u/quiver con ley Hill (cancer_dynamics)",
            flush=True,
        )
    elif not args.no_control_plots:
        print(
            "  WARN: control desactivado en parámetros del escenario; "
            "no se generan quiver_JHill / u_Hill (revise steady_states o USE_ADAPTIVE_CONTROL).",
            flush=True,
        )
    ck_path = out_dir / CHECKPOINT_FILENAME
    csv_path: Optional[Path] = (
        (out_dir / "sigma_plus_integral_vs_time.csv") if args.sigma_csv else None
    )

    if args.fresh_visualize:
        if ck_path.is_file():
            ck_path.unlink()
            print(f"Checkpoint reiniciado ({ck_path.name} eliminado).", flush=True)
        if csv_path is not None and csv_path.is_file():
            csv_path.unlink()
            print(f"CSV anterior eliminado: {csv_path.name}", flush=True)

    if args.no_checkpoint and ck_path.is_file():
        ck_path.unlink()
        print(f"Sin checkpoint: eliminado {ck_path.name}", flush=True)

    completed: Set[str] = set()
    if not args.no_checkpoint and ck_path.is_file():
        try:
            data = json.loads(ck_path.read_text(encoding="utf-8"))
            if _checkpoint_matches(data, args):
                completed = {str(x) for x in data.get("completed", [])}
                if completed:
                    print(
                        f"Checkpoint: {len(completed)} tiempo(s) ya graficados; se reanuda.",
                        flush=True,
                    )
            else:
                print(
                    "WARN: checkpoint obsoleto (cambiaron --block / --no-quiver / --sigma-csv / control); se ignora.",
                    flush=True,
                )
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            print(f"WARN: checkpoint ilegible ({exc}); se ignora.", flush=True)

    if args.all_times:
        times: List[float] = get_available_times(scenario_dir, "c", args.block)
        if not times:
            print("WARN: No hay matrices matrix_c_* en", scenario_dir / "matrices")
            return 1
    elif args.time is not None and len(args.time) > 0:
        times = list(args.time)
    else:
        avail = get_available_times(scenario_dir, "c", args.block)
        if not avail:
            print("WARN: No hay matrices; use --time explícito")
            return 1
        times = [avail[-1]]
        print(f"Usando último tiempo disponible: {times[0]:.3f}")

    times_all = list(times)
    if not args.no_checkpoint and completed:
        times = [t for t in times if _time_checkpoint_key(t) not in completed]
        if len(times) < len(times_all):
            print(f"Tiempos pendientes: {len(times)} de {len(times_all)}.", flush=True)
        if not times:
            print("Checkpoint: nada pendiente para este escenario.", flush=True)
            if csv_path is not None and csv_path.is_file():
                print(f"OK {csv_path}", flush=True)
            return 0

    if (
        not args.no_checkpoint
        and completed
        and csv_path is not None
        and not csv_path.is_file()
    ):
        print(
            "WARN: hay checkpoint pero no existe el CSV; solo se añadirán filas de los tiempos pendientes.",
            flush=True,
        )

    if args.sigma_csv and csv_path is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        resume_csv = (
            not args.no_checkpoint
            and not args.fresh_visualize
            and bool(completed)
            and csv_path.is_file()
        )
        if resume_csv:
            print(f"CSV: añadiendo filas a {csv_path.name}", flush=True)
        else:
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("# t\tSigma_plus_integral\n")

    n_ok = 0
    for t in times:
        sig = process_one_time(
            scenario_dir,
            params,
            t,
            args.block,
            out_dir,
            args.dpi,
            args.quiver_skip,
            do_quiver=not args.no_quiver,
            do_control_plots=not args.no_control_plots,
        )
        if sig is not None:
            n_ok += 1
            if not args.no_checkpoint:
                completed.add(_time_checkpoint_key(t))
                _save_visualize_checkpoint(ck_path, completed, args)
        if sig is not None and csv_path is not None:
            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(f"{t:.6f}\t{sig:.10e}\n")

    if csv_path is not None and csv_path.is_file():
        print(f"OK {csv_path}", flush=True)
    return 0 if n_ok > 0 else 1


def run_cli() -> None:
    p = argparse.ArgumentParser(
        description="Gradientes, flujos difusivos y densidad sigma+ desde matrices del escenario."
    )
    p.add_argument(
        "--scenarios",
        type=Path,
        default=DEFAULT_SCENARIOS,
        help="JSON de escenarios (default: scenarios_v1.json en Allee)",
    )
    p.add_argument("--scenario", type=str, default=None, help="Nombre del escenario (campo name en JSON)")
    p.add_argument(
        "--scenario-dir",
        type=Path,
        default=None,
        help="Carpeta del escenario con subcarpeta matrices/ (anula get_scenario_dir)",
    )
    p.add_argument(
        "--time",
        type=float,
        nargs="*",
        default=None,
        help="Tiempos a graficar. Si se omite, se usa el último tiempo disponible en matrices/",
    )
    p.add_argument("--all-times", action="store_true", help="Procesar todos los tiempos con matriz c")
    p.add_argument("--block", type=int, default=1, help="Índice de bloque nb en el nombre de archivo")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Salida (default: <scenario_dir>/nonequilibrium_plots)",
    )
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--quiver-skip", type=int, default=4, help="Submuestreo de quiver (cada n nodos)")
    p.add_argument("--no-quiver", action="store_true", help="No generar figura quiver")
    p.add_argument(
        "--no-control-plots",
        action="store_true",
        help="No generar u_Hill/u_minadapt ni quiver_JHill/quiver_Ju (solo J_c, J_s, J_i)",
    )
    p.add_argument(
        "--sigma-csv",
        action="store_true",
        help="Escribir sigma_plus_integral_vs_time.csv en out_dir (cabecera + una fila por tiempo al vuelo)",
    )
    p.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Recorre todos los escenarios del JSON (omitir --scenario / --scenario-dir).",
    )
    p.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Desactiva reanudación: no lee/escribe checkpoint y borra .visualize_fluxes_checkpoint.json al inicio.",
    )
    p.add_argument(
        "--fresh-visualize",
        action="store_true",
        help="Borra checkpoint y sigma_plus_integral_vs_time.csv en out_dir antes de procesar (regeneración limpia).",
    )
    args = p.parse_args()

    if args.all_scenarios:
        if args.scenario_dir is not None:
            p.error("--all-scenarios no es compatible con --scenario-dir")
        if args.scenario:
            p.error("--all-scenarios no se combina con --scenario")
        scenarios_path = args.scenarios.expanduser().resolve()
        names = _scenario_names_from_json(scenarios_path)
        if not names:
            print("WARN: No hay escenarios en", scenarios_path)
            sys.exit(1)
        failed: List[str] = []
        for name in names:
            print(f"\n========== {name} ==========")
            sdir = get_scenario_dir(name, base_dir=_ALLEE_ROOT)
            rc = _execute_visualization(args, name, sdir)
            if rc != 0:
                failed.append(name)
        if failed:
            print(f"\nResumen: fallidos u omitidos ({len(failed)}): {', '.join(failed)}")
        sys.exit(0 if len(failed) < len(names) else 1)

    if args.scenario_dir is not None:
        scenario_dir = args.scenario_dir.expanduser().resolve()
        scenario_name = args.scenario
    else:
        if not args.scenario:
            p.error("Se requiere --scenario si no se pasa --scenario-dir (o usa --all-scenarios)")
        scenario_name = args.scenario
        scenario_dir = get_scenario_dir(scenario_name, base_dir=_ALLEE_ROOT)

    sys.exit(_execute_visualization(args, scenario_name, scenario_dir))


if __name__ == "__main__":
    run_cli()
