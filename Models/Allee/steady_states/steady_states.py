"""
steady_states.py

Análisis de estados estacionarios **3D** $(c,s,i)$: construcción simbólica, Newton–Raphson,
barridos `scan_grid_3d` (rejilla factorial que incluye `rs`, `alpha`, `gamma` además de `rc`…`a`),
filtros `filter_physical_3d` (concentraciones + cota en max Re λ; estables e inestables) y generación opcional de `scenarios.json`
(CLI: por defecto pipeline WEAK+STRONG y JSON unificado en Resultados Paper/estados_estacionarios; `--mode corner-strong` solo STRONG; `--legacy-split-outputs` y `--local-only` opcionales).

Puede ejecutarse desde línea de comandos o importarse como módulo.
"""

import sys
from pathlib import Path

# Ejecución como script: python steady_states/steady_states.py (requiere Allee en sys.path)
if __package__ is None:
    _allee_root = Path(__file__).resolve().parent.parent
    if str(_allee_root) not in sys.path:
        sys.path.insert(0, str(_allee_root))

import sympy as sp
import numpy as np
import pandas as pd
from itertools import product
from datetime import datetime
import argparse
import json
from typing import Optional, Tuple, List, Dict, Any
from collections import OrderedDict
from model_parameters import create_scenarios_json


# ============================================================================
# 1. Símbolos simbólicos y parámetros base
# ============================================================================

rc, rs, rd, alpha, delta, beta, a, gamma, eta, mu = sp.symbols(
    'rc rs rd alpha delta beta a gamma eta mu'
)

c_3d, s_3d, i_3d = sp.symbols('c s i')
ku, eps_u, umax = sp.symbols('ku eps_u umax')
# Parámetros Hill para control (Opción A)
kc_h, nc_h, ki_h, ni_h = sp.symbols('kc_h nc_h ki_h ni_h')

params_base_3d = {
    rc: 5.84,
    rs: 13.12,
    rd: 10.92,
    alpha: 10.22,
    delta: 5.40,
    beta: 7.6,
    a: 0.1,
    gamma: 0.74,
    eta: 5.08,
    mu: 1,
    ku: 0.2,
    eps_u: 1e-3,
    umax: None,
    kc_h: 0.05,
    nc_h: 2.0,
    ki_h: 0.2,
    ni_h: 2.0,
}


# ============================================================================
# 2. Definición de ecuaciones del modelo
# ============================================================================

def build_equations_3d(allee_type: str = 'WEAK', include_hill_control: bool = True):
    """
    Construye las ecuaciones del modelo completo 3D (c, s, i).

    Args:
        allee_type: 'WEAK' o 'STRONG' (misma forma que en model_equations / extract).
        include_hill_control: Si True, F_i incluye u = umax * H_act(c) * H_inh(i). Si False, sin término de control.
    
    Returns:
        Tuple de expresiones simbólicas (F_c, F_s, F_i).
    """
    c, s, i = c_3d, s_3d, i_3d
    
    at = (allee_type or 'WEAK').upper()
    if at == 'STRONG':
        alle_term = rc * c * (1 - c) * ((c - a) / (1 - a))
    elif at == 'WEAK':
        alle_term = rc * c * (c - a) * (1 - c)
    else:
        raise ValueError("allee_type debe ser 'WEAK' o 'STRONG'")
    
    if include_hill_control:
        i_pos = sp.Max(i, 0)
        h_act = (c**nc_h) / (kc_h**nc_h + c**nc_h)
        h_inh = (ki_h**ni_h) / (ki_h**ni_h + i_pos**ni_h)
        u_ctrl = umax * h_act * h_inh
    else:
        u_ctrl = sp.Integer(0)
    
    F_c = alle_term - c * (alpha * s**2 + beta * i**2) - mu * c * (gamma * s**2 + eta * i**2)
    F_s = rs * s * (1 - s) - gamma * c**2 * s + delta * i**2 * s - (mu * alpha * c**2 * s) / 2
    F_i = rd * i * (1 - i) + delta * i * s**2 - eta * c**2 * i - (mu * beta * c**2 * i) / 2 + u_ctrl
    
    return F_c, F_s, F_i


# ============================================================================
# 3. Funciones de construcción numérica
# ============================================================================

def build_numeric_3d(
    params_override: Optional[Dict] = None,
    *,
    allee_type: str = 'WEAK',
    include_hill_control: bool = True,
):
    """
    Construye funciones numéricas y Jacobiano para el modelo 3D (Hill opcional, Allee weak/strong).
    
    Args:
        params_override: Diccionario con valores a sobrescribir en params_base_3d
        allee_type: 'WEAK' o 'STRONG'
        include_hill_control: incluir o no el término Hill en F_i
        
    Returns:
        Tuple (f, J, pcur) donde f es función vectorial numérica, J es el Jacobiano simbólico,
        y pcur son los parámetros usados
    """
    pcur = params_base_3d.copy()
    if params_override:
        pcur.update(params_override)
    
    # Evitar problemas de sympify cuando umax=None: usar infinito simbólico (solo relevante con Hill)
    if include_hill_control and pcur.get(umax, None) is None:
        pcur[umax] = sp.oo
    
    Fc, Fs, Fi = build_equations_3d(allee_type, include_hill_control)
    # SymPy no admite subs(..., None); params_base_3d trae umax=None si no hay Hill ni override
    pcur_sub = {k: v for k, v in pcur.items() if v is not None}
    Fc_eval = Fc.subs(pcur_sub)
    Fs_eval = Fs.subs(pcur_sub)
    Fi_eval = Fi.subs(pcur_sub)
    
    F_vec = sp.Matrix([Fc_eval, Fs_eval, Fi_eval])
    J = F_vec.jacobian([c_3d, s_3d, i_3d])
    f = sp.lambdify((c_3d, s_3d, i_3d), F_vec, modules='numpy')
    
    return f, J, pcur


# ============================================================================
# 4. Método de Newton-Raphson
# ============================================================================

def newton_root_3d(f, Jsym, x0: Tuple[float, float, float],
                   tol: float = 1e-8, max_iter: int = 80, cond_max: float = 1e12):
    """
    Encuentra raíz del sistema 3D usando Newton-Raphson.
    
    Args:
        f: Función vectorial numérica del sistema
        Jsym: Jacobiano simbólico
        x0: Tupla con valores iniciales (x0, y0, z0)
        tol: Tolerancia para convergencia
        max_iter: Número máximo de iteraciones
        cond_max: Condición máxima del Jacobiano
        
    Returns:
        Tuple (x, y, z) con la solución o None si no converge
    """
    x, y, z = map(float, x0)
    
    for _ in range(max_iter):
        Fv = np.array(f(x, y, z), dtype=float).reshape(-1)
        Jnum = np.array(Jsym.subs({c_3d: x, s_3d: y, i_3d: z}), dtype=float)
        
        if not np.isfinite(Fv).all() or not np.isfinite(Jnum).all():
            return None
        
        if np.linalg.cond(Jnum) > cond_max:
            return None
        
        try:
            delta = np.linalg.solve(Jnum, Fv)
        except Exception:
            return None
        
        x -= delta[0]
        y -= delta[1]
        z -= delta[2]
        
        if np.linalg.norm(delta) < tol:
            return (x, y, z)
    
    return None


# ============================================================================
# 5. Escaneo sistemático de parámetros
# ============================================================================

# Criterio heurístico “esquina” (tumor bajo / s,i altos), coherente con scan_corner_strong_allee_hill_mu.
_CORNER_C_MAX = 0.12
_CORNER_S_MIN = 0.88
_CORNER_I_MIN = 0.88


def control_3d_parameter_mesh() -> Tuple[
    List[float], List[float], List[float], List[float], List[float], List[float],
    List[float], List[float], List[float],
]:
    """
    Rejilla factorial (rc, beta, delta, eta, rd, a, rs, alpha, gamma) compartida por los barridos
    WEAK y STRONG del pipeline control-3d.

    rs, alpha y gamma se añaden respecto a versiones anteriores (antes fijos en params_base_3d)
    para explorar coexistencias y la región c bajo / s,i altos con más variedad paramétrica.
    """
    rc_vals = [5.0, 6.0]
    beta_vals = [5.0, 7.0]
    delta_vals = [5.0, 7.0]
    eta_vals = [3.0, 5.0]
    rd_vals = [9.0, 11.0]
    a_vals = [0.1]
    # Alrededor de params_base_3d: rs=13.12, alpha=10.22, gamma=0.74
    rs_vals = [12.0, 14.0]
    alpha_vals = [9.0, 11.0]
    gamma_vals = [0.6, 0.9]
    return (
        rc_vals,
        beta_vals,
        delta_vals,
        eta_vals,
        rd_vals,
        a_vals,
        rs_vals,
        alpha_vals,
        gamma_vals,
    )


def scan_grid_3d(
    rc_vals: List[float],
    beta_vals: List[float],
    delta_vals: List[float],
    eta_vals: List[float],
    rd_vals: List[float],
    a_vals: List[float],
    rs_vals: List[float],
    alpha_vals: List[float],
    gamma_vals: List[float],
    seeds: List[Tuple[float, float, float]],
    mu_val: float = 1,
    ku_val: float = 0.2,
    eps_val: float = 1e-3,
    umax_val: Optional[float] = None,
    allee_type: str = "WEAK",
    include_hill_control: bool = True,
) -> pd.DataFrame:
    """
    Escanea una rejilla de parámetros para encontrar estados estacionarios 3D con control Hill.
    
    Args:
        rc_vals, beta_vals, delta_vals, eta_vals, rd_vals, a_vals, rs_vals, alpha_vals, gamma_vals:
            Listas de valores de parámetros (producto completo).
        seeds: Lista de tuplas (x0, y0, z0) con valores iniciales
        mu_val: Valor de mu a usar
        ku_val, eps_val: se guardan en el DataFrame (histórico); el kinetics Hill no los usa
        umax_val: Valor máximo del control Hill (None → inf simbólico si include_hill_control)
        allee_type: 'WEAK' o 'STRONG'
        include_hill_control: si False, sin término Hill en F_i
        
    Returns:
        DataFrame con los resultados del escaneo
    """
    rows = []

    for a_v, rc_v, b_v, d_v, e_v, rd_v, rs_v, al_v, ga_v in product(
        a_vals,
        rc_vals,
        beta_vals,
        delta_vals,
        eta_vals,
        rd_vals,
        rs_vals,
        alpha_vals,
        gamma_vals,
    ):
        f, Jsym, _ = build_numeric_3d(
            {
                a: a_v,
                rc: rc_v,
                beta: b_v,
                delta: d_v,
                eta: e_v,
                rd: rd_v,
                rs: rs_v,
                alpha: al_v,
                gamma: ga_v,
                mu: mu_val,
                ku: ku_val,
                eps_u: eps_val,
                umax: umax_val,
            },
            allee_type=allee_type,
            include_hill_control=include_hill_control,
        )
        sols = []
        
        for x0, y0, z0 in seeds:
            r = newton_root_3d(f, Jsym, (x0, y0, z0))
            if r is None:
                continue
            
            cx, sy, iz = r
            if not np.isfinite(cx) or not np.isfinite(sy) or not np.isfinite(iz):
                continue
            
            # Evitar duplicados
            if any(np.linalg.norm([cx - px, sy - py, iz - pz]) < 1e-3 for px, py, pz in sols):
                continue
            
            # Calcular autovalores
            Jnum = np.array(Jsym.subs({c_3d: cx, s_3d: sy, i_3d: iz}), dtype=float)
            eigs = np.linalg.eigvals(Jnum)
            max_re = float(max(ev.real for ev in eigs))
            near_corner = bool(
                cx < _CORNER_C_MAX and sy > _CORNER_S_MIN and iz > _CORNER_I_MIN
            )
            umax_used: Optional[float]
            if include_hill_control and umax_val is not None:
                try:
                    umax_used = float(umax_val) if np.isfinite(float(umax_val)) else None
                except (TypeError, ValueError):
                    umax_used = None
            else:
                umax_used = None

            rows.append({
                'mu': mu_val,
                'allee_type': allee_type,
                'hill_control': include_hill_control,
                'a': a_v,
                'rc': rc_v,
                'rs': rs_v,
                'beta': b_v,
                'alpha': al_v,
                'delta': d_v,
                'eta': e_v,
                'gamma': ga_v,
                'rd': rd_v,
                'ku': ku_val, 'eps_u': eps_val, 'umax': umax_val,
                'umax_used': umax_used,
                'c_star': float(cx), 's_star': float(sy), 'i_star': float(iz),
                'eig1_real': float(eigs[0].real), 'eig1_imag': float(eigs[0].imag),
                'eig2_real': float(eigs[1].real), 'eig2_imag': float(eigs[1].imag),
                'eig3_real': float(eigs[2].real), 'eig3_imag': float(eigs[2].imag),
                'max_real': max_re,
                'near_c0_s1_i1': near_corner,
                'unstable': bool(any(ev.real > 0 for ev in eigs)),
            })
            sols.append((cx, sy, iz))
    
    if not rows:
        return pd.DataFrame(
            columns=[
                'mu', 'allee_type', 'hill_control', 'a', 'rc', 'rs', 'beta', 'alpha', 'delta', 'eta', 'gamma', 'rd',
                'ku', 'eps_u', 'umax', 'umax_used', 'c_star', 's_star', 'i_star',
                'eig1_real', 'eig1_imag', 'eig2_real', 'eig2_imag',
                'eig3_real', 'eig3_imag', 'max_real', 'near_c0_s1_i1', 'unstable',
            ]
        )
    return pd.DataFrame(rows)


def scan_strong_grid_mu_hill(
    ku_val: float = 0.2,
    eps_val: float = 1e-3,
    umax_val: Optional[float] = 0.5,
    sweep_mu_hill: bool = True,
    mu_single: float = 1.0,
    seeds: Optional[List[Tuple[float, float, float]]] = None,
) -> pd.DataFrame:
    """
    Allee STRONG sobre la **misma rejilla factorial** que ``control_3d_parameter_mesh``,
    barriendo μ y Hill como en la parte WEAK del pipeline.
    """
    (
        rc_vals,
        beta_vals,
        delta_vals,
        eta_vals,
        rd_vals,
        a_vals,
        rs_vals,
        alpha_vals,
        gamma_vals,
    ) = control_3d_parameter_mesh()
    seeds = seeds if seeds is not None else default_seeds_3d_control_hill()
    chunks: List[pd.DataFrame] = []
    if sweep_mu_hill:
        mus: Tuple[float, ...] = (0.0, 1.0)
        hills: Tuple[bool, ...] = (True, False)
    else:
        mus = (float(mu_single),)
        hills = (True,)
    for mu_v in mus:
        for hill_on in hills:
            u_use = umax_val if hill_on else None
            print(f"  bloque STRONG (rejilla): mu={mu_v}, hill_control={hill_on}")
            chunks.append(
                scan_grid_3d(
                    rc_vals,
                    beta_vals,
                    delta_vals,
                    eta_vals,
                    rd_vals,
                    a_vals,
                    rs_vals,
                    alpha_vals,
                    gamma_vals,
                    seeds,
                    mu_val=mu_v,
                    ku_val=ku_val,
                    eps_val=eps_val,
                    umax_val=u_use,
                    allee_type="STRONG",
                    include_hill_control=hill_on,
                )
            )
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def scan_corner_strong_allee_hill_mu(
    mu_vals: Tuple[float, ...] = (0.0, 1.0),
    hill_on_vals: Tuple[bool, ...] = (True, False),
    params_override: Optional[Dict] = None,
    seeds: Optional[List[Tuple[float, float, float]]] = None,
    umax_hill: float = 0.5,
    c_corner_max: float = 0.12,
    s_corner_min: float = 0.88,
    i_corner_min: float = 0.88,
) -> pd.DataFrame:
    """
    Busca equilibrios 3D con **Allee STRONG**, semillas hacia **c≈0, s≈1, i≈1**, para cada
    combinación de **μ** en ``mu_vals`` y **con / sin control Hill** en ``hill_on_vals``.

    Usa los parámetros de ``params_base_3d`` salvo lo que se pase en ``params_override``.
    Sin Hill: ``include_hill_control=False`` (no entra término en F_i). Con Hill: ``umax=umax_hill``.
    """
    seeds = seeds if seeds is not None else default_seeds_3d_control_hill()
    base = params_base_3d.copy()
    if params_override:
        base.update(params_override)

    rows: List[Dict[str, Any]] = []

    for mu_v in mu_vals:
        for hill_on in hill_on_vals:
            pcur = dict(base)
            pcur[mu] = mu_v
            if hill_on:
                pcur[umax] = umax_hill
            f, Jsym, _ = build_numeric_3d(
                pcur,
                allee_type='STRONG',
                include_hill_control=hill_on,
            )
            sols: List[Tuple[float, float, float]] = []
            for x0, y0, z0 in seeds:
                r = newton_root_3d(f, Jsym, (x0, y0, z0))
                if r is None:
                    continue
                cx, sy, iz = r
                if not np.isfinite(cx) or not np.isfinite(sy) or not np.isfinite(iz):
                    continue
                if any(np.linalg.norm([cx - px, sy - py, iz - pz]) < 1e-3 for px, py, pz in sols):
                    continue
                Jnum = np.array(Jsym.subs({c_3d: cx, s_3d: sy, i_3d: iz}), dtype=float)
                eigs = np.linalg.eigvals(Jnum)
                max_re = float(max(ev.real for ev in eigs))
                near = bool(cx < c_corner_max and sy > s_corner_min and iz > i_corner_min)
                rows.append({
                    'mu': float(mu_v),
                    'allee_type': 'STRONG',
                    'hill_control': hill_on,
                    'umax_used': float(umax_hill) if hill_on else None,
                    'a': float(pcur.get(a, base[a])),
                    'rc': float(pcur.get(rc, base[rc])),
                    'rs': float(pcur.get(rs, base[rs])),
                    'rd': float(pcur.get(rd, base[rd])),
                    'alpha': float(pcur.get(alpha, base[alpha])),
                    'beta': float(pcur.get(beta, base[beta])),
                    'delta': float(pcur.get(delta, base[delta])),
                    'gamma': float(pcur.get(gamma, base[gamma])),
                    'eta': float(pcur.get(eta, base[eta])),
                    'c_star': float(cx),
                    's_star': float(sy),
                    'i_star': float(iz),
                    'near_c0_s1_i1': near,
                    'eig1_real': float(eigs[0].real),
                    'eig1_imag': float(eigs[0].imag),
                    'eig2_real': float(eigs[1].real),
                    'eig2_imag': float(eigs[1].imag),
                    'eig3_real': float(eigs[2].real),
                    'eig3_imag': float(eigs[2].imag),
                    'max_real': max_re,
                    'unstable': bool(max_re > 0),
                })
                sols.append((cx, sy, iz))

    if not rows:
        return pd.DataFrame(
            columns=[
                'mu', 'allee_type', 'hill_control', 'umax_used', 'a', 'rc', 'rs', 'rd',
                'alpha', 'beta', 'delta', 'gamma', 'eta', 'c_star', 's_star', 'i_star',
                'near_c0_s1_i1', 'eig1_real', 'eig1_imag', 'eig2_real', 'eig2_imag',
                'eig3_real', 'eig3_imag', 'max_real', 'unstable',
            ]
        )
    return pd.DataFrame(rows)


def _dedupe_seed_points(
    seeds: List[Tuple[float, float, float]], tol: float = 1e-9
) -> List[Tuple[float, float, float]]:
    """Elimina semillas casi duplicadas (misma distancia euclídea < tol)."""
    kept: List[Tuple[float, float, float]] = []
    for p in seeds:
        if any(
            np.linalg.norm(np.array(p, dtype=float) - np.array(q, dtype=float)) < tol
            for q in kept
        ):
            continue
        kept.append(p)
    return kept


def seeds_unit_cube_grid(n: int) -> List[Tuple[float, float, float]]:
    """
    Rejilla de semillas en el cubo ``[0, 1]^3`` para c, s, i (``np.linspace(0, 1, n)`` por eje).

    Con ``n`` puntos por eje hay ``n**3`` semillas → coste de Newton multiplicado por eso
    en cada combinación de parámetros del barrido.

    Raises:
        ValueError: si ``n < 2`` (hace falta al menos los extremos 0 y 1).
    """
    if n < 2:
        raise ValueError("seeds_unit_cube_grid: n debe ser >= 2 para cubrir [0, 1].")
    xs = np.linspace(0.0, 1.0, n)
    out: List[Tuple[float, float, float]] = []
    for c in xs:
        for s in xs:
            for i in xs:
                out.append((float(c), float(s), float(i)))
    return out


def default_seeds_3d_control_hill(
    *,
    grid_n: Optional[int] = None,
    include_legacy: bool = True,
) -> List[Tuple[float, float, float]]:
    """
    Semillas para Newton 3D (control Hill en build_equations_3d).

    Por defecto (sin kwargs): puntos interiores habituales más la esquina c≈0, s≈1, i≈1.

    Si ``grid_n`` es un entero ``>= 2``, se añade (o solo se usa) una rejilla en ``[0,1]^3``
    con ``grid_n`` valores por eje vía ``seeds_unit_cube_grid``.

    Args:
        grid_n: Si no es None y ``>= 2``, incluye la rejilla ``[0,1]^3``.
        include_legacy: Si True y hay ``grid_n``, se unen rejilla + semillas legacy (deduplicadas).
            Si False y hay ``grid_n``, solo la rejilla.
    """
    interior = [
        (0.2, 0.2, 0.2),
        (0.5, 0.4, 0.4),
        (0.9, 0.6, 0.3),
        (1.1, 0.2, 0.8),
    ]
    near_c0_s1_i1 = [
        (0.0, 1.0, 1.0),
        (1e-10, 1.0, 1.0),
        (0.0, 0.99, 1.0),
        (0.0, 1.0, 0.99),
        (1e-8, 0.995, 0.995),
        (0.0, 0.98, 1.02),
        (1e-6, 0.99, 0.99),
    ]
    legacy = interior + near_c0_s1_i1
    if grid_n is None:
        return legacy
    if grid_n < 2:
        raise ValueError("default_seeds_3d_control_hill: grid_n debe ser None o un entero >= 2.")
    grid = seeds_unit_cube_grid(grid_n)
    if include_legacy:
        return _dedupe_seed_points(legacy + grid)
    return grid


def seeds_meta_note(*, grid_n: Optional[int], grid_only: bool) -> str:
    """Texto para meta JSON / logs según configuración de semillas."""
    if grid_n is not None and grid_n >= 2:
        ntot = grid_n**3
        base = (
            f"Rejilla (c,s,i) en [0,1] con numpy.linspace ({grid_n} puntos por eje → {ntot} semillas)"
        )
        return base + (" solamente." if grid_only else "; además semillas legacy (interior + c~0,s~1,i~1).")
    return (
        "Solo semillas legacy: interior + cerca de c~0, s~1, i~1 "
        "(ver default_seeds_3d_control_hill sin grid_n)."
    )


def _slug_num(val: Any, decimals: int = 3) -> str:
    """Número seguro para slugs de nombre (punto decimal → p)."""
    if val is None:
        return "na"
    try:
        x = float(val)
    except (TypeError, ValueError):
        return "na"
    if not np.isfinite(x):
        return "na"
    return f"{x:.{decimals}f}".replace(".", "p")


def steady_state_scenario_name(rec: Dict[str, Any]) -> str:
    """
    Identificador corto del escenario (sin parámetros de rejilla ni punto (c*,s*,i*)).

    Patrón: ``weak|strong_mu{0|1}_uSi|uNo_bajo_umbral|sobre_umbral``.
    Varios equilibrios pueden compartir el mismo escenario (misma rejilla, distintas raíces).
    """
    at = str(rec.get("allee_type") or "WEAK").lower()
    if at not in ("weak", "strong"):
        at = "weak"
    mu = int(rec.get("mu") or 0)
    hc = rec.get("hill_control")
    u_part = "uSi" if hc else "uNo"
    a_val = float(rec.get("a") or 0.1)
    c_star = float(rec.get("c_star") or 0.0)
    um = "bajo_umbral" if c_star < a_val else "sobre_umbral"
    return f"{at}_mu{mu}_{u_part}_{um}"


def steady_state_equilibrium_slug(rec: Dict[str, Any]) -> str:
    """Sufijo único por punto de rejilla y estado (rc…, rs…, alpha…, gamma…, c*, s*, i*)."""
    return (
        f"rc{_slug_num(rec.get('rc'))}_rs{_slug_num(rec.get('rs'))}"
        f"_b{_slug_num(rec.get('beta'))}_al{_slug_num(rec.get('alpha'))}"
        f"_d{_slug_num(rec.get('delta'))}_e{_slug_num(rec.get('eta'))}"
        f"_ga{_slug_num(rec.get('gamma'))}_rd{_slug_num(rec.get('rd'))}"
        f"_c{_slug_num(rec.get('c_star'))}_s{_slug_num(rec.get('s_star'))}"
        f"_i{_slug_num(rec.get('i_star'))}"
    )


def steady_state_equilibrium_name(rec: Dict[str, Any]) -> str:
    """
    Nombre plano completo (escenario + slug), alineado con ``scenarios_v1.json``.

    - uSi / uNo: control Hill activo o no (``hill_control``).
    - bajo_umbral / sobre_umbral: ``c_star < a`` vs no (umbral Allee del propio escaneo).
    """
    return f"{steady_state_scenario_name(rec)}_{steady_state_equilibrium_slug(rec)}"


def dataframe_to_named_steady_state_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Filas listas para JSON con campo ``name`` único (colisiones → sufijo __1, __2, ...)."""
    raw_list = df.replace({np.nan: None}).to_dict(orient="records")
    seen_counts: Dict[str, int] = {}
    out: List[Dict[str, Any]] = []
    for rec in raw_list:
        base = steady_state_equilibrium_name(rec)
        n = seen_counts.get(base, 0)
        seen_counts[base] = n + 1
        name = base if n == 0 else f"{base}__{n}"
        out.append({"name": name, **rec})
    return out


def dataframe_to_nested_scenario_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Agrupa por ``steady_state_scenario_name`` (nombre corto de escenario).

    Cada elemento: ``name``, ``n_steady_states``, ``steady_states`` (lista de filas con
    ``equilibrium_index``, ``equilibrium_slug`` y el resto de columnas del DataFrame).
    Colisiones de slug dentro del mismo escenario → sufijo ``__1``, ``__2``, ...
    """
    raw_list = df.replace({np.nan: None}).to_dict(orient="records")
    buckets: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()
    for rec in raw_list:
        key = steady_state_scenario_name(rec)
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(rec)
    out: List[Dict[str, Any]] = []
    for scenario_name, recs in buckets.items():
        slug_counts: Dict[str, int] = {}
        inner: List[Dict[str, Any]] = []
        for idx, rec in enumerate(recs):
            slug_base = steady_state_equilibrium_slug(rec)
            n = slug_counts.get(slug_base, 0)
            slug_counts[slug_base] = n + 1
            slug = slug_base if n == 0 else f"{slug_base}__{n}"
            inner.append({"equilibrium_index": idx, "equilibrium_slug": slug, **rec})
        out.append(
            {
                "name": scenario_name,
                "n_steady_states": len(inner),
                "steady_states": inner,
            }
        )
    return out


def save_steady_states_catalog_json(
    df_raw: pd.DataFrame,
    df_filt: pd.DataFrame,
    output_path: Path,
    meta: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> None:
    """
    Guarda todos los equilibrios encontrados en JSON para filtrado manual (jq, scripts, IDE).

    Estructura:
      - steady_states_raw: escenarios anidados (``name`` corto + ``steady_states``)
      - steady_states_filtered: igual, tras filter_physical_3d
      - meta: parámetros del escaneo
    """
    payload = {
        "meta": meta or {},
        "steady_states_raw": dataframe_to_nested_scenario_records(df_raw),
        "steady_states_filtered": dataframe_to_nested_scenario_records(df_filt),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    if verbose:
        print(f"Catálogo JSON (filtra tú las filas): {output_path}")


def build_corner_strong_payload(df: pd.DataFrame, umax_hill: float) -> Dict[str, Any]:
    """Serializa bloque STRONG del JSON unificado (rejilla o legado params fijos)."""
    df_filt_phys = filter_physical_3d(df) if len(df) else df
    if len(df) and "near_c0_s1_i1" in df.columns:
        near_df = df[df["near_c0_s1_i1"]]
    else:
        near_df = df.iloc[0:0]
    return {
        "meta": {
            "allee_type": "STRONG",
            "scan_kind": "parameter_grid",
            "parameter_mesh": (
                "control_3d_parameter_mesh: rc,beta,delta,eta,rd,a,rs,alpha,gamma (misma rejilla que WEAK)"
            ),
            "mu_values": [0.0, 1.0],
            "hill_modes": [True, False],
            "umax_hill": umax_hill,
            "near_corner_criteria": {
                "c_max": _CORNER_C_MAX,
                "s_min": _CORNER_S_MIN,
                "i_min": _CORNER_I_MIN,
            },
            "n_all": len(df),
            "n_filtered_physical": len(df_filt_phys),
            "n_near_corner": len(near_df),
            "note": "near_c0_s1_i1 es heuristico; STRONG usa rejilla factorial como WEAK.",
            "name_pattern": (
                "Cada objeto tiene ``name`` = escenario corto (steady_state_scenario_name); "
                "cada equilibrio en ``steady_states`` lleva ``equilibrium_slug`` (rc_b_…_c_s_i). "
                "Nombre plano completo = name + '_' + equilibrium_slug (ver steady_state_equilibrium_name)."
            ),
        },
        "all": dataframe_to_nested_scenario_records(df),
        "steady_states_filtered": dataframe_to_nested_scenario_records(df_filt_phys),
        "near_corner_only": dataframe_to_nested_scenario_records(near_df),
    }


def save_steady_states_full_run_json(
    output_path: Path,
    global_meta: Dict[str, Any],
    weak_meta: Optional[Dict[str, Any]],
    df_raw: Optional[pd.DataFrame],
    df_filt: Optional[pd.DataFrame],
    corner_payload: Optional[Dict[str, Any]],
) -> None:
    """Un solo JSON: meta global, bloque WEAK (rejilla) y bloque STRONG (misma rejilla)."""
    weak_grid = None
    if weak_meta is not None and df_raw is not None and df_filt is not None:
        weak_grid = {
            "meta": weak_meta,
            "steady_states_raw": dataframe_to_nested_scenario_records(df_raw),
            "steady_states_filtered": dataframe_to_nested_scenario_records(df_filt),
        }
    payload = {
        "meta": global_meta,
        "weak_grid": weak_grid,
        "strong_corner": corner_payload,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    print(f"JSON unificado (pipeline completo): {output_path}")


def merge_nested_steady_states_into_scenarios_json(
    output_file: Path,
    common_params: Dict[str, Any],
    nested_blocks: List[Dict[str, Any]],
) -> None:
    """
    Fusiona en ``scenarios.json`` una entrada por cada estado estacionario encontrado.

    Conserva escenarios existentes y agrega solo nombres faltantes. Para estados
    generales usa ``<scenario_name>_<equilibrium_slug>``; para ramas etiquetadas
    con ``target_branch`` usa también ese prefijo dentro del slug si ya viene así.
    """
    existing: Dict[str, Any] = {}
    existing_scenarios: List[Dict[str, Any]] = []
    if output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                existing = loaded
                raw = loaded.get("scenarios")
                if isinstance(raw, list):
                    existing_scenarios = [x for x in raw if isinstance(x, dict)]
        except (OSError, json.JSONDecodeError, TypeError):
            existing = {}
            existing_scenarios = []

    existing_names = {
        str(sc.get("name", "")).strip()
        for sc in existing_scenarios
        if str(sc.get("name", "")).strip()
    }
    all_names = set(existing_names)
    generated: List[Dict[str, Any]] = []

    for block in nested_blocks:
        base_name = str(block.get("name") or "scenario")
        steady_states = [
            ss for ss in (block.get("steady_states") or []) if isinstance(ss, dict)
        ]
        for ss in steady_states:
            slug = str(ss.get("equilibrium_slug") or "").strip()
            if not slug:
                slug = steady_state_equilibrium_slug(ss)
            out_name = f"{base_name}_{slug}"
            if out_name in existing_names:
                continue

            unique_name = out_name
            suffix_i = 1
            while unique_name in all_names:
                suffix_i += 1
                unique_name = f"{out_name}__{suffix_i}"
            all_names.add(unique_name)

            generated.append(
                {
                    "name": unique_name,
                    "n_steady_states": 1,
                    "steady_states": [dict(ss)],
                }
            )

    root = {
        k: v
        for k, v in existing.items()
        if k not in {"common_params", "scenarios", "steady_states_filtered"}
    }
    merged_common = dict(existing.get("common_params") or {})
    merged_common.update(common_params)
    final_scenarios = existing_scenarios + generated

    root["common_params"] = merged_common
    root["scenarios"] = final_scenarios
    root["steady_states_filtered"] = list(final_scenarios)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(root, f, indent=2, ensure_ascii=False, default=str)

    print(
        f"scenarios.json actualizado desde steady_states: {output_file} "
        f"({len(existing_scenarios)} existentes, {len(generated)} agregados)"
    )


def default_spatial_common_params(umax_value: Optional[float]) -> Dict[str, str]:
    """Parámetros comunes compatibles con ``run_scenarios.py`` para escenarios derivados."""
    return {
        "a": "0.1",
        "alle": "0.1",
        "gamma": "0.74",
        "alpha": "10.22",
        "rs": "13.12",
        "rc": "5.84",
        "delta": "5.40",
        "eta": "5.08",
        "beta": "7.6",
        "rd": "10.92",
        "D_c": "0.012",
        "D_s": "0.022",
        "D_i": "0.022",
        "dt": "0.001",
        "T": "1",
        "nodes_in_xaxis": "100",
        "nodes_in_yaxis": "100",
        "space_size": "8",
        "nb": "1",
        "sample_rate": "0.02",
        "inner_max_iter": "3",
        "inner_tol": "0.001",
        "SAVE_IMAGES": "Y",
        "MONITOR_MEMORY": "Y",
        "MEMORY_CLEANUP_INTERVAL": "5",
        "MEMORY_WARNING_THRESHOLD_MB": "0",
        "MEMORY_WARNING_THRESHOLD_PCT": "80",
        "SOLVER_RECREATE_INTERVAL": "25",
        "ENABLE_CHECKPOINT": "Y",
        "CHECKPOINT_INTERVAL": "500",
        "CHECKPOINT_MEMORY_THRESHOLD_PCT": "80",
        "CHECKPOINT_RESTART_THRESHOLD_PCT": "85",
        "mu": "1",
        "ALLEE_TYPE": "STRONG",
        "USE_ADAPTIVE_CONTROL": "N",
        "U_MAX": str(umax_value if umax_value is not None else 0.5),
        "HILL_KC": "0.05",
        "HILL_NC": "2",
        "HILL_KI": "0.2",
        "HILL_NI": "2",
        "KU": "0.2",
        "EPS_U": "0.001",
    }


# 6. Filtrado físico
# ============================================================================

def filter_physical_3d(df: pd.DataFrame, c_max: float = 1.5, s_max: float = 1.2,
                       i_min: float = 0.01, i_max: float = 1.5, re_max: float = 120) -> pd.DataFrame:
    """
    Filtra resultados 3D según criterios físicos (concentraciones) y acotación del espectro.

    Acepta equilibrios **linealmente estables** (max(Re λ) ≤ 0) e **inestables**
    (max(Re λ) > 0). Solo se excluyen puntos con max(Re λ) ≥ re_max (inestabilidad extrema
    o ruido numérico).
    
    Args:
        df: DataFrame con resultados del escaneo
        c_max, s_max: Valores máximos permitidos para c* y s*
        i_min, i_max: Rango permitido para i*
        re_max: Cota superior estricta para max(Re λ); estables cumplen max_real < re_max de forma trivial
        
    Returns:
        DataFrame filtrado
    """
    if df.empty:
        return df.copy()
    df = df.copy()
    df['max_real'] = df[['eig1_real', 'eig2_real', 'eig3_real']].max(axis=1)
    return df[
        (df['c_star'] > 0) & (df['s_star'] > 0) & (df['i_star'] > i_min) & (df['i_star'] < i_max) &
        (df['max_real'] < re_max) &
        (df['c_star'] < c_max) & (df['s_star'] < s_max)
    ]


# ============================================================================
# 7. Utilidades de directorios
# ============================================================================

def resolve_run_dir(mu_val: Optional[float] = None, regime_folder: str = "Weak_Allee",
                    subfolder: Optional[str] = None, use_date: bool = False) -> Path:
    """
    Resuelve el directorio de salida para guardar resultados.
    
    Args:
        mu_val: Valor de mu (si se proporciona, se usa en la ruta)
        regime_folder: Nombre de la carpeta del régimen
        subfolder: Subcarpeta adicional (para análisis con control)
        use_date: Si True, agrega fecha al nombre del directorio
        
    Returns:
        Path al directorio de salida
    """
    root = Path.cwd()
    fs_root = None
    
    # Buscar directorio filesystem
    for p in [root] + list(root.parents):
        candidate = p / "filesystem"
        if candidate.exists():
            fs_root = candidate
            break
    
    if fs_root is None:
        fs_root = root / "filesystem"
        fs_root.mkdir(parents=True, exist_ok=True)
    
    # Construir ruta según parámetros
    if subfolder:
        out_dir = fs_root / subfolder
    elif mu_val is not None:
        out_dir = fs_root / f"mu{int(mu_val)}" / regime_folder
    else:
        out_dir = fs_root / regime_folder
    
    if use_date:
        date_tag = datetime.now().strftime("%Y-%m-%d")
        out_dir = out_dir / date_tag
    
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ============================================================================
# 8. Análisis sistemático 3D (CLI y módulo)
# ============================================================================

def main_control_3d(mu_val: float = 1, ku_val: float = 0.2, eps_val: float = 1e-3,
                   umax_val: Optional[float] = 0.5, save_results: bool = True,
                   generate_scenarios: bool = False,
                   catalog_json_path: Optional[Path] = None,
                   persist_legacy: bool = True,
                   sweep_weak_mu_hill: bool = True,
                   seeds: Optional[List[Tuple[float, float, float]]] = None,
                   seeds_meta_note_str: Optional[str] = None):
    """
    Realiza analisis sistematico 3D con control adaptativo (equivalente a steady_states_story_control.ipynb).

    Args:
        mu_val: Valor de mu cuando sweep_weak_mu_hill es False; ignorado si sweep_weak_mu_hill es True
        ku_val: Intensidad del control adaptativo
        eps_val: Parametro epsilon del control
        umax_val: Valor maximo del control Hill cuando hay Hill (None para sin limite)
        save_results: Si True, permite escrituras y escenarios segun flags
        generate_scenarios: Si True, fusiona escenarios en scenarios.json desde df filtrado (solo filas con Hill)
        catalog_json_path: Ruta del JSON catalogo; default: raiz Allee / steady_states_catalog.json
        persist_legacy: Si True y save_results, CSV bajo filesystem/ y steady_states_catalog.json
        sweep_weak_mu_hill: Si True, barrido WEAK Allee con mu en {0,1} y Hill on/off (4 bloques)
        seeds: Semillas Newton (c,s,i); si None, ``default_seeds_3d_control_hill()``.
        seeds_meta_note_str: Texto opcional para ``weak_meta['seeds_note']``.
    """
    print("=" * 60)
    print("Analisis sistematico de estados estacionarios 3D con control (Allee WEAK)")
    if sweep_weak_mu_hill:
        print(f"Barrido WEAK: mu en {{0, 1}} x Hill on/off; ku = {ku_val}, eps_u = {eps_val}, umax (con Hill) = {umax_val}")
    else:
        print(f"mu = {mu_val}, ku = {ku_val}, eps_u = {eps_val}, umax = {umax_val} (solo con Hill)")
    print("=" * 60)

    (
        rc_vals,
        beta_vals,
        delta_vals,
        eta_vals,
        rd_vals,
        a_vals,
        rs_vals,
        alpha_vals,
        gamma_vals,
    ) = control_3d_parameter_mesh()

    if seeds is None:
        seeds = default_seeds_3d_control_hill()
        default_seeds_note = seeds_meta_note(grid_n=None, grid_only=False)
    else:
        default_seeds_note = f"Semillas suministradas al llamador (n={len(seeds)})."
    seeds_note = (
        seeds_meta_note_str if seeds_meta_note_str is not None else default_seeds_note
    )

    run_dir = resolve_run_dir(subfolder="with_control_weak", use_date=False)
    allee_root = Path(__file__).resolve().parent.parent

    print("\nEjecutando escaneo WEAK...")
    if sweep_weak_mu_hill:
        chunks: List[pd.DataFrame] = []
        for mu_v in (0.0, 1.0):
            for hill_on in (True, False):
                u_use = umax_val if hill_on else None
                print(f"  bloque WEAK: mu={mu_v}, hill_control={hill_on}")
                chunks.append(
                    scan_grid_3d(
                        rc_vals,
                        beta_vals,
                        delta_vals,
                        eta_vals,
                        rd_vals,
                        a_vals,
                        rs_vals,
                        alpha_vals,
                        gamma_vals,
                        seeds,
                        mu_val=mu_v,
                        ku_val=ku_val,
                        eps_val=eps_val,
                        umax_val=u_use,
                        allee_type="WEAK",
                        include_hill_control=hill_on,
                    )
                )
        df_raw = pd.concat(chunks, ignore_index=True)
        n_cells = (
            len(a_vals)
            * len(rc_vals)
            * len(beta_vals)
            * len(delta_vals)
            * len(eta_vals)
            * len(rd_vals)
            * len(rs_vals)
            * len(alpha_vals)
            * len(gamma_vals)
        )
        weak_meta = {
            "source": "steady_states.main_control_3d",
            "allee_type": "WEAK",
            "sweep_weak_mu_hill": True,
            "mu_swept": [0.0, 1.0],
            "hill_swept": [True, False],
            "ku": ku_val,
            "eps_u": eps_val,
            "umax_with_hill": umax_val,
            "parameter_mesh": {
                "rc": rc_vals,
                "beta": beta_vals,
                "delta": delta_vals,
                "eta": eta_vals,
                "rd": rd_vals,
                "a": a_vals,
                "rs": rs_vals,
                "alpha": alpha_vals,
                "gamma": gamma_vals,
            },
            "n_parameter_cells": n_cells,
            "run_dir": str(run_dir),
            "seeds_note": seeds_note,
            "n_seeds": len(seeds),
        }
    else:
        df_raw = scan_grid_3d(
            rc_vals,
            beta_vals,
            delta_vals,
            eta_vals,
            rd_vals,
            a_vals,
            rs_vals,
            alpha_vals,
            gamma_vals,
            seeds,
            mu_val=mu_val,
            ku_val=ku_val,
            eps_val=eps_val,
            umax_val=umax_val,
            allee_type="WEAK",
            include_hill_control=True,
        )
        n_cells = (
            len(a_vals)
            * len(rc_vals)
            * len(beta_vals)
            * len(delta_vals)
            * len(eta_vals)
            * len(rd_vals)
            * len(rs_vals)
            * len(alpha_vals)
            * len(gamma_vals)
        )
        weak_meta = {
            "source": "steady_states.main_control_3d",
            "allee_type": "WEAK",
            "sweep_weak_mu_hill": False,
            "mu": mu_val,
            "ku": ku_val,
            "eps_u": eps_val,
            "umax": umax_val,
            "hill_control_fixed": True,
            "parameter_mesh": {
                "rc": rc_vals,
                "beta": beta_vals,
                "delta": delta_vals,
                "eta": eta_vals,
                "rd": rd_vals,
                "a": a_vals,
                "rs": rs_vals,
                "alpha": alpha_vals,
                "gamma": gamma_vals,
            },
            "n_parameter_cells": n_cells,
            "run_dir": str(run_dir),
            "seeds_note": seeds_note,
            "n_seeds": len(seeds),
        }

    df_filt = filter_physical_3d(df_raw)

    weak_meta["n_raw"] = len(df_raw)
    weak_meta["n_filtered"] = len(df_filt)

    print(f"raw: {len(df_raw)}, filtered: {len(df_filt)}")
    if len(df_raw) == 0:
        print(
            "  Nota: ninguna raiz convergio. Si usaste umax=None, prueba un tope finito, p. ej. "
            "--umax 0.5 (lambdify con infinito simbolico a veces deja Newton sin soluciones)."
        )

    if save_results and persist_legacy:
        df_raw.to_csv(run_dir / "steady_states_control_weak_raw.csv", index=False)
        df_filt.to_csv(run_dir / "steady_states_control_weak_filtered.csv", index=False)
        print(f"\nResultados guardados en {run_dir}")
        json_path = catalog_json_path or (allee_root / "steady_states_catalog.json")
        save_steady_states_catalog_json(df_raw, df_filt, json_path, meta=weak_meta)
        save_steady_states_catalog_json(
            df_raw,
            df_filt,
            run_dir / "steady_states_catalog.json",
            meta={**weak_meta},
            verbose=False,
        )
        print(f"  (copia en run_dir) {run_dir / 'steady_states_catalog.json'}")

    if len(df_filt) > 0:
        print(
            f"\nPrimeros resultados filtrados (pandas head() = 5 filas de {len(df_filt)}; "
            f"raw sin filtro físico: {len(df_raw)} filas):"
        )
        print(df_filt.head())
    else:
        print("\nNo se encontraron puntos que pasen los filtros fisicos.")

    if save_results and generate_scenarios and len(df_filt) > 0:
        df_for_scen = df_filt
        if "hill_control" in df_filt.columns:
            df_for_scen = df_filt[df_filt["hill_control"]].copy()
        if len(df_for_scen) == 0:
            print(
                "\nNo hay filas WEAK con control Hill para generar escenarios "
                "(con barrido mu x Hill, --generate-scenarios usa solo hill_control=True)."
            )
        else:
            scenarios_file = allee_root / "scenarios.json"
            print("\nGenerando scenarios.json desde estados estacionarios con control (solo filas con Hill)...")
            generate_scenarios_from_control_3d(
                df_for_scen,
                scenarios_file,
                max_scenarios=10,
                include_spatial_params=True,
            )

    return df_raw, df_filt, weak_meta


# ============================================================================
# 9. Generación de scenarios.json desde barrido 3D con control
# ============================================================================

def generate_scenarios_from_control_3d(
    df_filt: pd.DataFrame,
    output_file: Path,
    common_params: Optional[Dict] = None,
    max_scenarios: int = 10,
    include_spatial_params: bool = True
) -> None:
    """
    Genera scenarios.json basado en estados estacionarios 3D con control.
    
    Args:
        df_filt: DataFrame con resultados filtrados de análisis 3D con control
        output_file: Ruta al archivo scenarios.json
        common_params: Parámetros comunes (si None, usa valores por defecto)
        max_scenarios: Número máximo de escenarios a generar
        include_spatial_params: Si True, incluye parámetros espaciales
    """
    if len(df_filt) == 0:
        print("⚠ No hay resultados filtrados para generar escenarios")
        return
    
    # Parámetros comunes por defecto
    if common_params is None:
        common_params = {
            'rc': '5.84',
            'rs': '13.12',
            'rd': '10.92',
            'alpha': '10.22',
            'delta': '5.40',
            'beta': '7.6',
            'alle': '0.1',
            'gamma': '0.74',
            'eta': '5.08',
            'mu': '1',
            'ALLEE_TYPE': 'WEAK',
            'USE_ADAPTIVE_CONTROL': 'Y',
            'U_MAX': '0.5',
            # Control Hill (Opción A) - defaults
            'HILL_KC': '0.05',
            'HILL_NC': '2',
            'HILL_KI': '0.2',
            'HILL_NI': '2',
            # Mantener KU/EPS_U por compatibilidad (ya no se usan si control=Hill)
            'KU': '0.2',
            'EPS_U': '0.001',
        }
        
        if include_spatial_params:
            common_params.update({
                'D_c': '0.012',
                'D_s': '0.022',
                'D_i': '0.022',
                'dt': '0.001',
                'T': '2',
                'nodes_in_xaxis': '100',
                'nodes_in_yaxis': '100',
                'space_size': '4',
                'nb': '1',
                'sample_rate': '0.02',
                'SAVE_IMAGES': 'Y',
            })
    
    # Crear escenarios
    scenarios = []
    
    for idx, row in df_filt.head(max_scenarios).iterrows():
        mu_val = int(row.get('mu', 1))
        c_star_val = row.get('c_star', 0)
        scenario_name = f"weak_mu{mu_val}_control_c{c_star_val:.3f}_ku{row.get('ku', 0.2):.1f}"
        scenario_name = scenario_name.replace('.', 'p')
        
        scenario = {
            'name': scenario_name,
            'ALLEE_TYPE': 'WEAK',
            'mu': str(mu_val),
            'rc': str(row.get('rc', common_params.get('rc', '5.84'))),
            'rs': str(row.get('rs', common_params.get('rs', '13.12'))),
            'beta': str(row.get('beta', common_params.get('beta', '7.6'))),
            'alpha': str(row.get('alpha', common_params.get('alpha', '10.22'))),
            'delta': str(row.get('delta', common_params.get('delta', '5.40'))),
            'eta': str(row.get('eta', common_params.get('eta', '5.08'))),
            'gamma': str(row.get('gamma', common_params.get('gamma', '0.74'))),
            'rd': str(row.get('rd', common_params.get('rd', '10.92'))),
            'a': str(row.get('a', common_params.get('alle', '0.1'))),
            'USE_ADAPTIVE_CONTROL': 'Y',
            # Control Hill (Opción A) - usar defaults (o valores de common_params si se sobrescriben)
            'HILL_KC': str(common_params.get('HILL_KC', '0.05')),
            'HILL_NC': str(common_params.get('HILL_NC', '2')),
            'HILL_KI': str(common_params.get('HILL_KI', '0.2')),
            'HILL_NI': str(common_params.get('HILL_NI', '2')),
            # KU/EPS_U se mantienen por compatibilidad con scripts antiguos
            'KU': str(row.get('ku', common_params.get('KU', 0.2))),
            'EPS_U': str(row.get('eps_u', common_params.get('EPS_U', 1e-3))),
            'U_MAX': str(row.get('umax', common_params.get('U_MAX', 0.5))) if row.get('umax') else common_params.get('U_MAX', '0.5'),
        }
        
        # Condiciones iniciales
        c_star = row.get('c_star', 0.1)
        s_star = row.get('s_star', 0.1)
        i_star = row.get('i_star', 0.9)
        
        scenario.update({
            'C_INIT_MIN': str(max(0.01, c_star * 0.9)),
            'C_INIT_MAX': str(min(1.0, c_star * 1.1)),
            'S_INIT_MIN': str(max(0.01, s_star * 0.9)),
            'S_INIT_MAX': str(min(1.0, s_star * 1.1)),
            'I_INIT_MIN': str(max(0.01, i_star * 0.9)),
            'I_INIT_MAX': str(min(1.0, i_star * 1.1)),
        })
        
        # Eliminar None values
        scenario = {k: v for k, v in scenario.items() if v is not None}
        scenarios.append(scenario)
    
    create_scenarios_json(output_file, common_params, scenarios, overwrite=False)
    
    print(f"\n✓ Generados {len(scenarios)} escenarios con control desde estados estacionarios")
    print(f"  Archivo: {output_file}")


# ============================================================================
# 10. Interfaz CLI
# ============================================================================

def main():
    """Funcion principal para ejecucion desde linea de comandos."""
    from utils_paths import ensure_steady_states_results_dir_ready

    parser = argparse.ArgumentParser(
        description="Analisis de estados estacionarios para modelos de dinamica de cancer"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["control-3d", "corner-strong"],
        default="control-3d",
        help="Por defecto: pipeline WEAK (rejilla) + STRONG (misma rejilla); un JSON en Resultados Paper. "
        "corner-strong: solo STRONG en rejilla (mu x Hill completos).",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=1,
        help="Parte WEAK: mu fijo solo si usas --no-sweep-weak (default: 1). Con barrido completo, se ignora.",
    )
    parser.add_argument(
        "--no-sweep-weak",
        action="store_true",
        help="Desactiva barrido WEAK mu en {0,1} x Hill on/off; usa solo --mu y siempre Hill (mas rapido).",
    )
    parser.add_argument("--no-save", action="store_true", help="No escribir archivos de salida")
    parser.add_argument("--ku", type=float, default=0.2, help="Intensidad del control adaptativo (default: 0.2)")
    parser.add_argument("--eps-u", type=float, default=1e-3, help="Parametro epsilon del control (default: 1e-3)")
    parser.add_argument(
        "--umax",
        type=float,
        default=0.5,
        help="Tope u_max en control Hill (default: 0.5). Parte STRONG usa este valor en ramas con Hill.",
    )
    parser.add_argument(
        "--no-umax-limit",
        action="store_true",
        help="Sin tope u_max en la parte WEAK (infinito simbolico). La parte STRONG sigue usando --umax.",
    )
    parser.add_argument(
        "--generate-scenarios",
        action="store_true",
        help="Fusiona scenarios.json desde el df filtrado WEAK (como antes).",
    )
    parser.add_argument(
        "--legacy-split-outputs",
        action="store_true",
        help="Ademas del JSON unificado: CSV + steady_states_catalog.json (WEAK) y steady_states_corner_strong_mu.json.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Escribir en Allee/estados_estacionarios/ sin exigir Drive ni RESULTS_DIR.",
    )
    parser.add_argument(
        "--skip-scenarios-json",
        action="store_true",
        help="No fusionar los estados estacionarios encontrados en Allee/scenarios.json.",
    )
    parser.add_argument(
        "--seed-grid-n",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Si N>=2, añade semillas en el cubo [0,1]^3 para (c,s,i) con numpy.linspace(0,1,N) "
            "por eje (N**3 puntos; coste de CPU alto). Por defecto se mantienen también las semillas "
            "legacy salvo --seed-grid-only."
        ),
    )
    parser.add_argument(
        "--seed-grid-only",
        action="store_true",
        help="Con --seed-grid-n: usar solo la rejilla [0,1]^3, sin semillas legacy.",
    )

    args = parser.parse_args()
    if args.seed_grid_n is not None and args.seed_grid_n < 2:
        parser.error("--seed-grid-n debe ser >= 2 (u omite el flag).")
    if args.seed_grid_only and args.seed_grid_n is None:
        parser.error("--seed-grid-only requiere --seed-grid-n >= 2.")

    scan_seeds = default_seeds_3d_control_hill(
        grid_n=args.seed_grid_n,
        include_legacy=not bool(args.seed_grid_only),
    )
    scan_seeds_note = seeds_meta_note(
        grid_n=args.seed_grid_n, grid_only=bool(args.seed_grid_only)
    )
    save_results = not args.no_save
    allee_root = Path(__file__).resolve().parent.parent

    if args.mode == "control-3d":
        umax_cli = None if args.no_umax_limit else args.umax
        df_raw, df_filt, weak_meta = main_control_3d(
            mu_val=args.mu,
            ku_val=args.ku,
            eps_val=args.eps_u,
            umax_val=umax_cli,
            save_results=save_results,
            generate_scenarios=args.generate_scenarios,
            persist_legacy=bool(args.legacy_split_outputs),
            sweep_weak_mu_hill=not bool(args.no_sweep_weak),
            seeds=scan_seeds,
            seeds_meta_note_str=scan_seeds_note,
        )
        print("\n" + "=" * 60)
        print("Parte STRONG (misma rejilla que WEAK), mu x Hill como en WEAK")
        print("=" * 60)
        df_strong = scan_strong_grid_mu_hill(
            ku_val=args.ku,
            eps_val=args.eps_u,
            umax_val=umax_cli,
            sweep_mu_hill=not bool(args.no_sweep_weak),
            mu_single=args.mu,
            seeds=scan_seeds,
        )
        print(f"STRONG (rejilla): {len(df_strong)} filas en DataFrame raw (una por equilibrio distinto × celda paramétrica).")
        if len(df_strong) > 0:
            nh = int(df_strong["near_c0_s1_i1"].sum()) if "near_c0_s1_i1" in df_strong.columns else 0
            print(
                f"  Filas con bandera near_c0_s1_i1=True (solo etiqueta; NO recorta el barrido): {nh} de {len(df_strong)}"
            )
            print(
                "  En JSON: strong_corner.all = todos los hallazgos; steady_states_filtered = tras filter_physical_3d; "
                "near_corner_only = solo las que cumplen near_c0_s1_i1."
            )
            cols = [
                "mu", "hill_control", "rc", "rs", "beta", "alpha", "delta", "eta", "gamma", "rd",
                "c_star", "s_star", "i_star", "near_c0_s1_i1", "max_real",
            ]
            cols = [c for c in cols if c in df_strong.columns]
            nshow = min(20, len(df_strong))
            print(f"  Vista previa ({nshow} de {len(df_strong)} filas):")
            print(df_strong[cols].head(20).to_string())
        corner_payload = build_corner_strong_payload(df_strong, args.umax)

        if save_results and args.legacy_split_outputs:
            out_corner = allee_root / "steady_states_corner_strong_mu.json"
            with open(out_corner, "w", encoding="utf-8") as f:
                json.dump(corner_payload, f, indent=2, ensure_ascii=False, default=str)
            print(f"\nJSON legacy STRONG: {out_corner}")

        if save_results:
            global_meta = {
                "source": "steady_states.full_pipeline",
                "created": datetime.now().isoformat(),
                "umax_cli": args.umax,
                "mu_weak_cli": args.mu,
                "sweep_weak_mu_hill": not bool(args.no_sweep_weak),
                "weak_umax_limit_off": bool(args.no_umax_limit),
                "legacy_split_outputs": bool(args.legacy_split_outputs),
                "local_only": bool(args.local_only),
                "seed_grid_n": args.seed_grid_n,
                "seed_grid_only": bool(args.seed_grid_only),
                "n_seeds": len(scan_seeds),
                "parts": ["weak_grid", "strong_corner"],
                "note": (
                    "weak_grid: WEAK en rejilla; strong_corner: STRONG en la misma rejilla factorial. "
                    "sweep_weak_mu_hill aplica a WEAK y STRONG salvo --no-sweep-weak."
                ),
            }
            if args.local_only:
                out_dir = allee_root / "estados_estacionarios"
            else:
                try:
                    out_dir = ensure_steady_states_results_dir_ready()
                except RuntimeError as e:
                    print(f"\nError de ruta de resultados: {e}")
                    print("Usa --local-only para guardar en Allee/estados_estacionarios/ o define RESULTS_DIR / monta Drive.")
                    raise SystemExit(1) from e
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "steady_states_full_run.json"
            save_steady_states_full_run_json(
                out_path,
                global_meta,
                weak_meta,
                df_raw,
                df_filt,
                corner_payload,
            )
            if not args.skip_scenarios_json:
                nested_for_scenarios: List[Dict[str, Any]] = []
                if len(df_filt) > 0:
                    nested_for_scenarios.extend(dataframe_to_nested_scenario_records(df_filt))
                if corner_payload is not None:
                    nested_for_scenarios.extend(
                        corner_payload.get("steady_states_filtered") or []
                    )
                merge_nested_steady_states_into_scenarios_json(
                    allee_root / "scenarios.json",
                    default_spatial_common_params(umax_cli),
                    nested_for_scenarios,
                )
            else:
                print("scenarios.json omitido (--skip-scenarios-json).")

    elif args.mode == "corner-strong":
        umax_cli = None if args.no_umax_limit else args.umax
        df = scan_strong_grid_mu_hill(
            ku_val=args.ku,
            eps_val=args.eps_u,
            umax_val=umax_cli,
            sweep_mu_hill=True,
            mu_single=args.mu,
            seeds=scan_seeds,
        )
        print(f"STRONG solo (rejilla, mu en {{0,1}} x Hill on/off): {len(df)} filas raw.")
        if len(df) > 0:
            nh = int(df["near_c0_s1_i1"].sum()) if "near_c0_s1_i1" in df.columns else 0
            print(f"  near_c0_s1_i1=True (etiqueta, no filtro del barrido): {nh} de {len(df)}")
            cols = [
                "mu", "hill_control", "rc", "rs", "beta", "alpha", "delta", "eta", "gamma", "rd",
                "c_star", "s_star", "i_star", "near_c0_s1_i1", "max_real",
            ]
            cols = [c for c in cols if c in df.columns]
            nshow = min(20, len(df))
            print(f"  Vista previa ({nshow} de {len(df)} filas):")
            print(df[cols].head(20).to_string())
        corner_payload = build_corner_strong_payload(df, args.umax)
        if save_results:
            global_meta = {
                "source": "steady_states.corner_strong_only",
                "created": datetime.now().isoformat(),
                "umax_cli": args.umax,
                "legacy_split_outputs": bool(args.legacy_split_outputs),
                "local_only": bool(args.local_only),
                "seed_grid_n": args.seed_grid_n,
                "seed_grid_only": bool(args.seed_grid_only),
                "n_seeds": len(scan_seeds),
                "parts": ["strong_corner"],
            }
            if args.local_only:
                out_dir = allee_root / "estados_estacionarios"
            else:
                try:
                    out_dir = ensure_steady_states_results_dir_ready()
                except RuntimeError as e:
                    print(f"\nError de ruta de resultados: {e}")
                    print("Usa --local-only para guardar en Allee/estados_estacionarios/ o define RESULTS_DIR / monta Drive.")
                    raise SystemExit(1) from e
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "steady_states_full_run.json"
            save_steady_states_full_run_json(
                out_path,
                global_meta,
                None,
                None,
                None,
                corner_payload,
            )
            if not args.skip_scenarios_json:
                merge_nested_steady_states_into_scenarios_json(
                    allee_root / "scenarios.json",
                    default_spatial_common_params(umax_cli),
                    corner_payload.get("steady_states_filtered") or [],
                )
            else:
                print("scenarios.json omitido (--skip-scenarios-json).")
            if args.legacy_split_outputs:
                out_json = allee_root / "steady_states_corner_strong_mu.json"
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(corner_payload, f, indent=2, ensure_ascii=False, default=str)
                print(f"\nJSON legacy STRONG: {out_json}")


if __name__ == "__main__":
    main()

