"""
calculate_thermodynamic_properties.py

Calcula propiedades termodinámicas del modelo:
1. Energía Libre F[c,s,i]
2. Potenciales químicos μ_c, μ_s, μ_i (del funcional usado en este módulo)
3. Producción de entropía / disipación por tiempo:
   - integral de sum_a D_a|grad mu_a|^2 (total y por campo)
   - tasa de disipación difusiva integral de sum_a D_a|grad phi_a|^2 (total y por campo; TdC, J=-D grad phi)

Por defecto lee scenarios_v1.json. Recorre todos los tiempos con matrices c,s,i disponibles.
Salida: thermodynamics/entropy_production_by_field_t.txt, gráficas por campo, etc.
Los resultados se guardan en Google Drive si está montado, o localmente en caso contrario.

Durante el bucle temporal, las series .txt se escriben al paso (append) y en memoria solo se
mantienen escalares por tiempo (F, σ, desglose); no se acumulan las matrices 2D de μ.

Checkpoints / reanudación (mismas convenciones que cancer_dynamics / .env):
- ENABLE_CHECKPOINT=Y|N (default Y)
- CHECKPOINT_INTERVAL (default 500): guardar progreso cada N índices de tiempo procesados
- CHECKPOINT_MEMORY_THRESHOLD_PCT (default 80): si la RAM *del sistema* (no espacio en Drive) supera
  el umbral, se fuerza guardado adicional, como máximo cada THERMO_MEM_CHECKPOINT_MIN_STEPS pasos
  (default 100) para no escribir JSON en cada iteración.
Estado en thermodynamics/checkpoints/thermo_checkpoint_latest.json; al reanudar se reutilizan los .txt.

Nota: el porcentaje de memoria es RAM local (psutil); el espacio libre en Google Drive no reduce ese valor.
"""

import os
import sys
from pathlib import Path

_ALLEE_ROOT = Path(__file__).resolve().parent.parent
if str(_ALLEE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALLEE_ROOT))

import numpy as np
import json
import argparse
from typing import Dict, List, Tuple, Optional
from scipy import integrate

# Matplotlib para gráficas
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt

# Importar utilidades del proyecto
from utils_paths import get_scenario_dir, get_results_dir
from model_parameters import load_from_scenarios_json, ModelParameters
from model_equations import reaction_term_allee_weak, reaction_term_allee_strong


# ============================================================================
# Funciones de carga de datos
# ============================================================================

def load_field_matrix(scenario_dir: Path, field_name: str, time: float, block: int = 1) -> Optional[np.ndarray]:
    """
    Carga una matriz de campo desde el directorio del escenario.
    
    Args:
        scenario_dir: Directorio del escenario
        field_name: Nombre del campo ('c', 's', o 'i')
        time: Tiempo de la simulación
        block: Número de bloque (default: 1)
    
    Returns:
        Matriz numpy con los valores del campo, o None si no se encuentra
    """
    matrices_dir = scenario_dir / 'matrices'
    if not matrices_dir.exists():
        return None
    
    # Formato: matrix_{field_name}_{time:.3f}_nb_{block}.txt
    time_str = f"{time:.3f}"
    filename = f"matrix_{field_name}_{time_str}_nb_{block}.txt"
    filepath = matrices_dir / filename
    
    if not filepath.exists():
        return None
    
    try:
        field = np.loadtxt(filepath, dtype=float)
        return field
    except Exception as e:
        print(f"⚠ Error al cargar {filepath}: {e}")
        return None


def get_available_times(scenario_dir: Path, field_name: str = 'c', block: int = 1) -> List[float]:
    """
    Obtiene la lista de tiempos disponibles para un escenario.
    
    Args:
        scenario_dir: Directorio del escenario
        field_name: Nombre del campo (default: 'c')
        block: Número de bloque (default: 1)
    
    Returns:
        Lista de tiempos (floats) ordenados
    """
    matrices_dir = scenario_dir / 'matrices'
    if not matrices_dir.exists():
        return []
    
    times = []
    pattern = f"matrix_{field_name}_*_nb_{block}.txt"
    
    for filepath in matrices_dir.glob(pattern):
        # Extraer tiempo del nombre: matrix_c_0.001_nb_1.txt -> 0.001
        parts = filepath.stem.split('_')
        if len(parts) >= 3:
            try:
                time = float(parts[2])
                times.append(time)
            except ValueError:
                continue
    
    return sorted(times)


def load_all_field_matrices(scenario_dir: Path, field_name: str, times: List[float], block: int = 1) -> Dict[float, np.ndarray]:
    """
    Carga todas las matrices de un campo para múltiples tiempos.
    
    Args:
        scenario_dir: Directorio del escenario
        field_name: Nombre del campo ('c', 's', o 'i')
        times: Lista de tiempos a cargar
        block: Número de bloque (default: 1)
    
    Returns:
        Diccionario {tiempo: matriz}
    """
    matrices = {}
    for t in times:
        matrix = load_field_matrix(scenario_dir, field_name, t, block)
        if matrix is not None:
            matrices[t] = matrix
    return matrices


# ============================================================================
# Funciones de cálculo de gradientes
# ============================================================================

def calculate_gradient_2d(field: np.ndarray, dx: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula el gradiente 2D de un campo usando diferencias finitas.
    
    Args:
        field: Matriz 2D del campo
        dx: Espaciado espacial
    
    Returns:
        Tupla (grad_x, grad_y) con las componentes del gradiente
    """
    ny, nx = field.shape
    
    # Gradiente en x (dirección de columnas)
    grad_x = np.zeros_like(field)
    grad_x[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2 * dx)
    # Bordes
    grad_x[:, 0] = (field[:, 1] - field[:, 0]) / dx
    grad_x[:, -1] = (field[:, -1] - field[:, -2]) / dx
    
    # Gradiente en y (dirección de filas)
    grad_y = np.zeros_like(field)
    grad_y[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2 * dx)
    # Bordes
    grad_y[0, :] = (field[1, :] - field[0, :]) / dx
    grad_y[-1, :] = (field[-1, :] - field[-2, :]) / dx
    
    return grad_x, grad_y


def calculate_gradient_magnitude_squared(field: np.ndarray, dx: float) -> np.ndarray:
    """
    Calcula |∇φ|² para un campo 2D.
    
    Args:
        field: Matriz 2D del campo
        dx: Espaciado espacial
    
    Returns:
        Matriz con |∇φ|² en cada punto
    """
    grad_x, grad_y = calculate_gradient_2d(field, dx)
    return grad_x**2 + grad_y**2


# ============================================================================
# Cálculo de Energía Libre F[c,s,i]
# ============================================================================

def calculate_allee_integral(c: np.ndarray, rc: float, a: float, allee_type: str) -> np.ndarray:
    """
    Calcula la integral ∫[0→c] r_c·c'·f_Allee(c') dc' numéricamente.
    
    Para Weak Allee: f_Allee(c) = (c - a)·(1 - c)
    Para Strong Allee: f_Allee(c) = (1 - c)·((c - a)/(1 - a))
    
    Args:
        c: Matriz de concentración de cáncer
        rc: Tasa de crecimiento
        a: Parámetro de Allee
        allee_type: 'WEAK' o 'STRONG'
    
    Returns:
        Matriz con el valor de la integral en cada punto
    """
    result = np.zeros_like(c)
    epsilon = 1e-10
    
    # Vectorizar el cálculo para mejor rendimiento
    c_flat = c.flatten()
    result_flat = np.zeros_like(c_flat)
    
    if allee_type == 'WEAK':
        # Para Weak Allee: ∫[0→c] rc·c'·(c' - a)·(1 - c') dc'
        # Integración numérica vectorizada
        for idx, c_val in enumerate(c_flat):
            c_val = max(epsilon, min(c_val, 1.0 - epsilon))
            if c_val > epsilon:
                c_points = np.linspace(epsilon, c_val, 100)
                integrand = rc * c_points * (c_points - a) * (1 - c_points)
                result_flat[idx] = integrate.trapz(integrand, c_points)
    else:  # STRONG
        # Para Strong Allee: ∫[0→c] rc·c'·(1 - c')·((c' - a)/(1 - a)) dc'
        for idx, c_val in enumerate(c_flat):
            c_val = max(epsilon, min(c_val, 1.0 - epsilon))
            if c_val < a:
                result_flat[idx] = 0.0  # Por debajo del umbral
            else:
                c_points = np.linspace(a, c_val, 100)
                integrand = rc * c_points * (1 - c_points) * ((c_points - a) / (1 - a))
                result_flat[idx] = integrate.trapz(integrand, c_points)
    
    result = result_flat.reshape(c.shape)
    return result


def calculate_free_energy(
    c: np.ndarray,
    s: np.ndarray,
    i: np.ndarray,
    params: ModelParameters,
    dx: float
) -> Dict[str, np.ndarray]:
    """
    Calcula el funcional de energía libre F[c,s,i] y sus componentes.
    
    Args:
        c, s, i: Matrices 2D de los campos
        params: Parámetros del modelo
        dx: Espaciado espacial
    
    Returns:
        Diccionario con:
        - 'F_local': Términos locales
        - 'F_gradient': Términos de gradiente
        - 'F_coupling': Términos de acoplamiento
        - 'F_total': Energía libre total
    """
    # Términos locales
    # Integral de Allee
    allee_integral = calculate_allee_integral(c, params.rc, params.a, params.allee_type)
    
    # Términos de crecimiento logístico
    growth_s = (params.rs / 2) * s**2 * (1 - s / 2)
    growth_i = (params.rd / 2) * i**2 * (1 - i / 2)
    
    F_local = -allee_integral + growth_s + growth_i
    
    # Términos de gradiente
    grad_c_sq = calculate_gradient_magnitude_squared(c, dx)
    grad_s_sq = calculate_gradient_magnitude_squared(s, dx)
    grad_i_sq = calculate_gradient_magnitude_squared(i, dx)
    
    F_gradient = (params.D_c / 2) * grad_c_sq + (params.D_s / 2) * grad_s_sq + (params.D_i / 2) * grad_i_sq
    
    # Términos de acoplamiento base
    F_coupling = (
        (params.alpha / 2) * c * s**2 +
        (params.beta / 2) * c * i**2 +
        (params.gamma / 2) * c**2 * s +
        (params.eta / 2) * c**2 * i +
        (params.delta / 2) * s**2 * i**2
    )
    
    # Términos adicionales cuando μ = 1
    if params.mu > 0:
        F_coupling += (
            (params.alpha * params.mu / 4) * c**2 * s**2 +
            (params.beta * params.mu / 4) * c**2 * i**2 +
            (params.gamma * params.mu / 2) * c * s**2 +
            (params.eta * params.mu / 2) * c * i**2
        )
    
    F_total = F_local + F_gradient + F_coupling
    
    return {
        'F_local': F_local,
        'F_gradient': F_gradient,
        'F_coupling': F_coupling,
        'F_total': F_total
    }


# ============================================================================
# Cálculo de Potenciales Químicos
# ============================================================================

def calculate_chemical_potentials(
    c: np.ndarray,
    s: np.ndarray,
    i: np.ndarray,
    params: ModelParameters
) -> Dict[str, np.ndarray]:
    """
    Calcula los potenciales químicos μ_c, μ_s, μ_i.
    
    Estos son las variaciones del funcional de energía libre:
    μ_c = -δF/δc, μ_s = -δF/δs, μ_i = -δF/δi
    
    Args:
        c, s, i: Matrices 2D de los campos
        params: Parámetros del modelo
    
    Returns:
        Diccionario con 'mu_c', 'mu_s', 'mu_i'
    """
    # Término de Allee
    if params.allee_type == 'WEAK':
        allee_term = params.rc * c * (c - params.a) * (1 - c)
    else:  # STRONG
        allee_term = params.rc * c * (1 - c) * ((c - params.a) / (1 - params.a))
    
    # Potencial químico del cáncer μ_c
    mu_c = (
        allee_term -
        (params.alpha * s**2 + params.beta * i**2) -
        params.mu * (params.gamma * s**2 + params.eta * i**2)
    )
    
    if params.mu > 0:
        mu_c -= (params.alpha * params.mu / 2) * c * s**2
        mu_c -= (params.beta * params.mu / 2) * c * i**2
    
    # Potencial químico de células sanas μ_s
    mu_s = (
        params.rs * s * (1 - s) -
        params.gamma * c**2 -
        (params.alpha * params.mu / 2) * c**2 * s +
        params.delta * i**2 * s
    )
    
    # Potencial químico del sistema inmune μ_i
    mu_i = (
        params.rd * i * (1 - i) -
        params.eta * c**2 -
        (params.beta * params.mu / 2) * c**2 * i +
        params.delta * s**2 * i
    )
    
    return {
        'mu_c': mu_c,
        'mu_s': mu_s,
        'mu_i': mu_i
    }


# ============================================================================
# Cálculo de Producción de Entropía
# ============================================================================

def calculate_entropy_production(
    mu_c: np.ndarray,
    mu_s: np.ndarray,
    mu_i: np.ndarray,
    params: ModelParameters,
    dx: float
) -> np.ndarray:
    """
    Calcula la producción de entropía σ = ∫ [D_c·(∇μ_c)² + D_s·(∇μ_s)² + D_i·(∇μ_i)²] dx
    
    Args:
        mu_c, mu_s, mu_i: Matrices de potenciales químicos
        params: Parámetros del modelo
        dx: Espaciado espacial
    
    Returns:
        Matriz con la producción de entropía en cada punto
    """
    # Calcular gradientes de potenciales químicos
    grad_mu_c_sq = calculate_gradient_magnitude_squared(mu_c, dx)
    grad_mu_s_sq = calculate_gradient_magnitude_squared(mu_s, dx)
    grad_mu_i_sq = calculate_gradient_magnitude_squared(mu_i, dx)
    
    # Producción de entropía
    sigma = (
        params.D_c * grad_mu_c_sq +
        params.D_s * grad_mu_s_sq +
        params.D_i * grad_mu_i_sq
    )
    
    return sigma


def integrate_spatial_density(density: np.ndarray, dx: float) -> float:
    """Integral 2D: sum * dx^2."""
    return float(np.sum(density) * dx * dx)


def calculate_entropy_and_dissipation_integrals(
    mu_c: np.ndarray,
    mu_s: np.ndarray,
    mu_i: np.ndarray,
    c: np.ndarray,
    s: np.ndarray,
    i_field: np.ndarray,
    params: ModelParameters,
    dx: float,
) -> Dict[str, float]:
    """
    Integrales espaciales por tiempo.

    - Disipación difusiva tipo TdC (flujo J_a = -D_a grad phi_a, fuerza ~ grad phi_a):
      sigma_diss_a = D_a |grad phi_a|^2, tasa de disipación local (proxy >= 0).

    - Término con potenciales del funcional usado en este módulo:
      sigma_mu_a = D_a |grad mu_a|^2 (coherente con calculate_entropy_production).
    """
    if params.D_c is None or params.D_s is None or params.D_i is None:
        raise ValueError("params requiere D_c, D_s, D_i")

    gmc = calculate_gradient_magnitude_squared(mu_c, dx)
    gms = calculate_gradient_magnitude_squared(mu_s, dx)
    gmi = calculate_gradient_magnitude_squared(mu_i, dx)
    int_mu_c = integrate_spatial_density(params.D_c * gmc, dx)
    int_mu_s = integrate_spatial_density(params.D_s * gms, dx)
    int_mu_i = integrate_spatial_density(params.D_i * gmi, dx)

    gcc = calculate_gradient_magnitude_squared(c, dx)
    gss = calculate_gradient_magnitude_squared(s, dx)
    gii = calculate_gradient_magnitude_squared(i_field, dx)
    int_diss_c = integrate_spatial_density(params.D_c * gcc, dx)
    int_diss_s = integrate_spatial_density(params.D_s * gss, dx)
    int_diss_i = integrate_spatial_density(params.D_i * gii, dx)

    return {
        "int_mu_c": int_mu_c,
        "int_mu_s": int_mu_s,
        "int_mu_i": int_mu_i,
        "int_mu_total": int_mu_c + int_mu_s + int_mu_i,
        "int_diss_c": int_diss_c,
        "int_diss_s": int_diss_s,
        "int_diss_i": int_diss_i,
        "int_diss_total": int_diss_c + int_diss_s + int_diss_i,
    }


# ============================================================================
# Funciones de guardado
# ============================================================================

def _get_mu_avg_spatial(mu_entry: Dict, field: str) -> float:
    """
    Promedio espacial de μ_c, μ_s o μ_i.
    Acepta entradas compactas (mu_*_avg) o matrices completas (mu_*).
    """
    avg_key = f"mu_{field}_avg"
    if avg_key in mu_entry:
        return float(mu_entry[avg_key])
    return float(np.mean(mu_entry[f"mu_{field}"]))


def init_incremental_thermo_files(thermo_dir: Path) -> None:
    """Crea thermodynamics/ y deja cabeceras en los .txt (sobrescribe si existen)."""
    thermo_dir.mkdir(parents=True, exist_ok=True)
    F_file = thermo_dir / "free_energy_F_t.txt"
    with open(F_file, "w", encoding="utf-8") as f:
        f.write("# tiempo\tF_total\tF_local\tF_gradient\tF_coupling\n")
    sigma_file = thermo_dir / "entropy_production_sigma_t.txt"
    with open(sigma_file, "w", encoding="utf-8") as f:
        f.write("# tiempo\tsigma_mu_total\n")
    sigma_by_field = thermo_dir / "entropy_production_by_field_t.txt"
    with open(sigma_by_field, "w", encoding="utf-8") as f:
        f.write(
            "# t\tint_D_grad_mu_sq_total\tint_D_grad_mu_sq_c\tint_D_grad_mu_sq_s\tint_D_grad_mu_sq_i\t"
            "int_diss_phi_total\tint_diss_phi_c\tint_diss_phi_s\tint_diss_phi_i\n"
        )
    mu_file = thermo_dir / "chemical_potentials_mu_t.txt"
    with open(mu_file, "w", encoding="utf-8") as f:
        f.write("# tiempo\tmu_c_promedio\tmu_s_promedio\tmu_i_promedio\n")


def append_incremental_thermo_row(
    thermo_dir: Path,
    t: float,
    F_data: Dict,
    sigma_total: float,
    sigma_detail: Dict,
    mu_c_avg: float,
    mu_s_avg: float,
    mu_i_avg: float,
) -> None:
    """Añade una fila por tiempo a cada serie en thermodynamics/ (modo append)."""
    F_file = thermo_dir / "free_energy_F_t.txt"
    with open(F_file, "a", encoding="utf-8") as f:
        f.write(
            f"{t:.6f}\t{F_data['F_total']:.10e}\t{F_data['F_local']:.10e}\t"
            f"{F_data['F_gradient']:.10e}\t{F_data['F_coupling']:.10e}\n"
        )
    sigma_file = thermo_dir / "entropy_production_sigma_t.txt"
    with open(sigma_file, "a", encoding="utf-8") as f:
        f.write(f"{t:.6f}\t{sigma_total:.10e}\n")
    sigma_by_field = thermo_dir / "entropy_production_by_field_t.txt"
    with open(sigma_by_field, "a", encoding="utf-8") as f:
        d = sigma_detail
        f.write(
            f"{t:.6f}\t{d['int_mu_total']:.10e}\t{d['int_mu_c']:.10e}\t{d['int_mu_s']:.10e}\t"
            f"{d['int_mu_i']:.10e}\t{d['int_diss_total']:.10e}\t{d['int_diss_c']:.10e}\t"
            f"{d['int_diss_s']:.10e}\t{d['int_diss_i']:.10e}\n"
        )
    mu_file = thermo_dir / "chemical_potentials_mu_t.txt"
    with open(mu_file, "a", encoding="utf-8") as f:
        f.write(f"{t:.6f}\t{mu_c_avg:.10e}\t{mu_s_avg:.10e}\t{mu_i_avg:.10e}\n")


# ============================================================================
# Checkpoints / reanudación (patrón ENABLE_CHECKPOINT + CHECKPOINT_INTERVAL)
# ============================================================================

THERMO_CHECKPOINT_VERSION = 1
THERMO_CHECKPOINT_FILENAME = "thermo_checkpoint_latest.json"


def _thermo_time_key(t: float) -> float:
    """Clave estable para comparar tiempos con los escritos en .txt (6 decimales)."""
    return round(float(t), 6)


def _get_memory_percentage() -> float:
    try:
        import psutil  # type: ignore

        return float(psutil.virtual_memory().percent)
    except Exception:
        return 0.0


def thermo_checkpoint_dir(thermo_dir: Path) -> Path:
    return thermo_dir / "checkpoints"


def thermo_checkpoint_path(thermo_dir: Path) -> Path:
    return thermo_checkpoint_dir(thermo_dir) / THERMO_CHECKPOINT_FILENAME


def _read_thermo_txt_noncomment_lines(path: Path) -> List[str]:
    if not path.is_file():
        return []
    lines: List[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                lines.append(s)
    return lines


def load_thermo_results_from_txt(thermo_dir: Path) -> Optional[Dict]:
    """
    Reconstruye results (escalares por t) desde los cuatro .txt incrementales.
    Devuelve None si faltan archivos o hay desalineación entre columnas/tiempos.
    """
    F_file = thermo_dir / "free_energy_F_t.txt"
    sigma_file = thermo_dir / "entropy_production_sigma_t.txt"
    sigma_by_field = thermo_dir / "entropy_production_by_field_t.txt"
    mu_file = thermo_dir / "chemical_potentials_mu_t.txt"
    lf = _read_thermo_txt_noncomment_lines(F_file)
    ls = _read_thermo_txt_noncomment_lines(sigma_file)
    lb = _read_thermo_txt_noncomment_lines(sigma_by_field)
    lm = _read_thermo_txt_noncomment_lines(mu_file)
    n = len(lf)
    if n == 0 or len(ls) != n or len(lb) != n or len(lm) != n:
        return None

    results: Dict = {"times": [], "F": {}, "sigma": {}, "sigma_detail": {}, "mu": {}}
    for i in range(n):
        pf = lf[i].split("\t")
        ps = ls[i].split("\t")
        pb = lb[i].split("\t")
        pm = lm[i].split("\t")
        if len(pf) < 5 or len(ps) < 2 or len(pb) < 9 or len(pm) < 4:
            return None
        t_f = float(pf[0])
        t_s = float(ps[0])
        t_b = float(pb[0])
        t_m = float(pm[0])
        if (
            abs(t_f - t_s) > 1e-5
            or abs(t_f - t_b) > 1e-5
            or abs(t_f - t_m) > 1e-5
        ):
            return None
        t = t_f
        results["F"][t] = {
            "F_total": float(pf[1]),
            "F_local": float(pf[2]),
            "F_gradient": float(pf[3]),
            "F_coupling": float(pf[4]),
        }
        results["sigma"][t] = float(ps[1])
        results["sigma_detail"][t] = {
            "int_mu_total": float(pb[1]),
            "int_mu_c": float(pb[2]),
            "int_mu_s": float(pb[3]),
            "int_mu_i": float(pb[4]),
            "int_diss_total": float(pb[5]),
            "int_diss_c": float(pb[6]),
            "int_diss_s": float(pb[7]),
            "int_diss_i": float(pb[8]),
        }
        results["mu"][t] = {
            "mu_c_avg": float(pm[1]),
            "mu_s_avg": float(pm[2]),
            "mu_i_avg": float(pm[3]),
        }
    results["times"] = sorted(results["F"].keys())
    return results


def _thermo_checkpoint_valid_for_run(
    data: Dict,
    scenario_name: str,
    block: int,
    scenarios_resolved: Path,
    times: List[float],
) -> bool:
    if int(data.get("version", 0)) != THERMO_CHECKPOINT_VERSION:
        return False
    if data.get("scenario_name") != scenario_name:
        return False
    if int(data.get("block", -1)) != int(block):
        return False
    try:
        ck_scen = Path(data.get("scenarios_file", "")).resolve()
    except Exception:
        return False
    if ck_scen != scenarios_resolved:
        return False
    ck_times = data.get("times")
    if not isinstance(ck_times, list) or len(ck_times) != len(times):
        return False
    for a, b in zip(ck_times, times):
        if abs(float(a) - float(b)) > 1e-9:
            return False
    return True


def save_thermo_checkpoint(
    thermo_dir: Path,
    scenario_name: str,
    block: int,
    scenarios_file: Path,
    times: List[float],
    results: Dict,
) -> None:
    ck_dir = thermo_checkpoint_dir(thermo_dir)
    ck_dir.mkdir(parents=True, exist_ok=True)
    completed = sorted(results["F"].keys(), key=float)
    payload = {
        "version": THERMO_CHECKPOINT_VERSION,
        "scenario_name": scenario_name,
        "block": int(block),
        "scenarios_file": str(scenarios_file.resolve()),
        "times": [float(x) for x in times],
        "completed_times": [float(x) for x in completed],
    }
    path = thermo_checkpoint_path(thermo_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_thermo_checkpoint(thermo_dir: Path) -> Optional[Dict]:
    path = thermo_checkpoint_path(thermo_dir)
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def remove_thermo_checkpoint(thermo_dir: Path) -> None:
    path = thermo_checkpoint_path(thermo_dir)
    if path.is_file():
        path.unlink()


def plot_thermodynamic_properties(
    results: Dict,
    scenario_name: str,
    thermo_dir: Path
) -> None:
    """
    Genera gráficas de las propiedades termodinámicas.
    
    Args:
        results: Diccionario con resultados calculados
        scenario_name: Nombre del escenario
        thermo_dir: Directorio donde guardar las gráficas
    """
    times = sorted(results['times'])
    
    # Extraer datos para gráficas
    F_total = [results['F'][t]['F_total'] for t in times]
    F_local = [results['F'][t]['F_local'] for t in times]
    F_gradient = [results['F'][t]['F_gradient'] for t in times]
    F_coupling = [results['F'][t]['F_coupling'] for t in times]
    sigma_values = [results['sigma'][t] for t in times]
    mu_c_avg = [_get_mu_avg_spatial(results['mu'][t], 'c') for t in times]
    mu_s_avg = [_get_mu_avg_spatial(results['mu'][t], 's') for t in times]
    mu_i_avg = [_get_mu_avg_spatial(results['mu'][t], 'i') for t in times]
    
    # Crear directorio de imágenes
    images_dir = thermo_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Gráfica de Energía Libre F(t)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, F_total, 'b-', linewidth=2, label='F_total')
    ax.plot(times, F_local, 'r--', linewidth=1.5, alpha=0.7, label='F_local')
    ax.plot(times, F_gradient, 'g--', linewidth=1.5, alpha=0.7, label='F_gradient')
    ax.plot(times, F_coupling, 'm--', linewidth=1.5, alpha=0.7, label='F_coupling')
    ax.set_xlabel('Tiempo t', fontsize=12)
    ax.set_ylabel('Energía Libre F', fontsize=12)
    ax.set_title(f'Energía Libre F(t) - {scenario_name}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    F_plot_file = images_dir / 'free_energy_F_t.png'
    plt.savefig(F_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Gráfica guardada: {F_plot_file}")
    
    # 2. Gráfica de Producción de Entropía σ(t) — total D|grad mu|^2
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, sigma_values, 'r-', linewidth=2, label=r'$\int \sum_a D_a|\nabla\mu_a|^2 dA$')
    ax.set_xlabel('Tiempo t', fontsize=12)
    ax.set_ylabel(r'Producción (integral)', fontsize=12)
    ax.set_title(f'Producción de entropía (potencial mu) - {scenario_name}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    sigma_plot_file = images_dir / 'entropy_production_sigma_t.png'
    plt.savefig(sigma_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Gráfica guardada: {sigma_plot_file}")
    
    # 2b. Disipación difusiva por campo: D_a |grad phi_a|^2 integrado
    if results.get('sigma_detail'):
        diss_c = [results['sigma_detail'][t]['int_diss_c'] for t in times]
        diss_s = [results['sigma_detail'][t]['int_diss_s'] for t in times]
        diss_i = [results['sigma_detail'][t]['int_diss_i'] for t in times]
        diss_tot = [results['sigma_detail'][t]['int_diss_total'] for t in times]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(times, diss_tot, 'k-', linewidth=2, label='total')
        ax.plot(times, diss_c, '--', linewidth=1.5, label=r'$D_c|\nabla c|^2$')
        ax.plot(times, diss_s, '--', linewidth=1.5, label=r'$D_s|\nabla s|^2$')
        ax.plot(times, diss_i, '--', linewidth=1.5, label=r'$D_i|\nabla i|^2$')
        ax.set_xlabel('Tiempo t', fontsize=12)
        ax.set_ylabel(r'Tasa de disipación (integral)', fontsize=12)
        ax.set_title(f'Disipación difusiva por campo - {scenario_name}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        diss_plot = images_dir / 'entropy_dissipation_by_field_t.png'
        plt.savefig(diss_plot, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Gráfica guardada: {diss_plot}")
        
        mu_c_int = [results['sigma_detail'][t]['int_mu_c'] for t in times]
        mu_s_int = [results['sigma_detail'][t]['int_mu_s'] for t in times]
        mu_i_int = [results['sigma_detail'][t]['int_mu_i'] for t in times]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(times, sigma_values, 'k-', linewidth=2, label=r'total $D|\nabla\mu|^2$')
        ax.plot(times, mu_c_int, '--', linewidth=1.5, label=r'$D_c|\nabla\mu_c|^2$')
        ax.plot(times, mu_s_int, '--', linewidth=1.5, label=r'$D_s|\nabla\mu_s|^2$')
        ax.plot(times, mu_i_int, '--', linewidth=1.5, label=r'$D_i|\nabla\mu_i|^2$')
        ax.set_xlabel('Tiempo t', fontsize=12)
        ax.set_ylabel(r'Integral', fontsize=12)
        ax.set_title(f'Producción con grad(mu) por campo - {scenario_name}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        mu_sigma_plot = images_dir / 'entropy_production_mu_by_field_t.png'
        plt.savefig(mu_sigma_plot, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Gráfica guardada: {mu_sigma_plot}")
    
    # 3. Gráfica de Potenciales Químicos μ(t)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, mu_c_avg, 'r-', linewidth=2, label='μ_c (cáncer)')
    ax.plot(times, mu_s_avg, 'g-', linewidth=2, label='μ_s (células sanas)')
    ax.plot(times, mu_i_avg, 'b-', linewidth=2, label='μ_i (sistema inmune)')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Tiempo t', fontsize=12)
    ax.set_ylabel('Potencial Químico μ (promedio espacial)', fontsize=12)
    ax.set_title(f'Potenciales Químicos μ(t) - {scenario_name}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    mu_plot_file = images_dir / 'chemical_potentials_mu_t.png'
    plt.savefig(mu_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Gráfica guardada: {mu_plot_file}")
    
    # 4. Gráfica combinada: F y σ en subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Subplot 1: Energía Libre
    ax1.plot(times, F_total, 'b-', linewidth=2, label='F_total')
    ax1.plot(times, F_local, 'r--', linewidth=1.5, alpha=0.7, label='F_local')
    ax1.plot(times, F_gradient, 'g--', linewidth=1.5, alpha=0.7, label='F_gradient')
    ax1.plot(times, F_coupling, 'm--', linewidth=1.5, alpha=0.7, label='F_coupling')
    ax1.set_xlabel('Tiempo t', fontsize=11)
    ax1.set_ylabel('Energía Libre F', fontsize=11)
    ax1.set_title(f'Energía Libre F(t) - {scenario_name}', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Producción de Entropía
    ax2.plot(times, sigma_values, 'r-', linewidth=2)
    ax2.set_xlabel('Tiempo t', fontsize=11)
    ax2.set_ylabel('Producción de Entropía σ', fontsize=11)
    ax2.set_title(f'Producción de Entropía σ(t) - {scenario_name}', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    combined_plot_file = images_dir / 'thermodynamic_properties_combined.png'
    plt.savefig(combined_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Gráfica guardada: {combined_plot_file}")


def save_thermodynamic_results(
    results: Dict,
    scenario_dir: Path,
    scenario_name: str,
    generate_plots: bool = True,
    incremental_series_on_disk: bool = False,
) -> None:
    """
    Guarda los resultados termodinámicos en el directorio del escenario.
    
    Args:
        results: Diccionario con resultados calculados
        scenario_dir: Directorio del escenario
        scenario_name: Nombre del escenario
        generate_plots: Si True, genera gráficas (default: True)
        incremental_series_on_disk: Si True, los .txt de series ya están escritos al paso;
            solo se generan resumen JSON y gráficas.
    """
    # Crear subdirectorio thermodynamics
    thermo_dir = scenario_dir / 'thermodynamics'
    thermo_dir.mkdir(parents=True, exist_ok=True)
    
    times = results['times']
    
    if not incremental_series_on_disk:
        # Guardar energía libre F(t)
        F_file = thermo_dir / 'free_energy_F_t.txt'
        with open(F_file, 'w') as f:
            f.write("# tiempo\tF_total\tF_local\tF_gradient\tF_coupling\n")
            for t in times:
                F_data = results['F'][t]
                f.write(f"{t:.6f}\t{F_data['F_total']:.10e}\t"
                       f"{F_data['F_local']:.10e}\t{F_data['F_gradient']:.10e}\t"
                       f"{F_data['F_coupling']:.10e}\n")
        print(f"✓ Guardado: {F_file}")
        
        # Guardar producción de entropía σ(t) — total = integral D|grad mu|^2 (compatibilidad)
        sigma_file = thermo_dir / 'entropy_production_sigma_t.txt'
        with open(sigma_file, 'w') as f:
            f.write("# tiempo\tsigma_mu_total\n")
            for t in times:
                sigma_total = results['sigma'][t]
                f.write(f"{t:.6f}\t{sigma_total:.10e}\n")
        print(f"✓ Guardado: {sigma_file}")
        
        # Desglose por campo y por tiempo: mu (variacional) y disipación difusiva D|grad phi|^2
        sigma_by_field = thermo_dir / 'entropy_production_by_field_t.txt'
        with open(sigma_by_field, 'w', encoding='utf-8') as f:
            f.write(
                "# t\tint_D_grad_mu_sq_total\tint_D_grad_mu_sq_c\tint_D_grad_mu_sq_s\tint_D_grad_mu_sq_i\t"
                "int_diss_phi_total\tint_diss_phi_c\tint_diss_phi_s\tint_diss_phi_i\n"
            )
            for t in times:
                d = results.get('sigma_detail', {}).get(t)
                if d is None:
                    continue
                f.write(
                    f"{t:.6f}\t{d['int_mu_total']:.10e}\t{d['int_mu_c']:.10e}\t{d['int_mu_s']:.10e}\t"
                    f"{d['int_mu_i']:.10e}\t{d['int_diss_total']:.10e}\t{d['int_diss_c']:.10e}\t"
                    f"{d['int_diss_s']:.10e}\t{d['int_diss_i']:.10e}\n"
                )
        print(f"✓ Guardado: {sigma_by_field}")
        
        # Guardar potenciales químicos promedio espacial
        mu_file = thermo_dir / 'chemical_potentials_mu_t.txt'
        with open(mu_file, 'w') as f:
            f.write("# tiempo\tmu_c_promedio\tmu_s_promedio\tmu_i_promedio\n")
            for t in times:
                mu_data = results['mu'][t]
                mu_c_avg = _get_mu_avg_spatial(mu_data, 'c')
                mu_s_avg = _get_mu_avg_spatial(mu_data, 's')
                mu_i_avg = _get_mu_avg_spatial(mu_data, 'i')
                f.write(f"{t:.6f}\t{mu_c_avg:.10e}\t{mu_s_avg:.10e}\t{mu_i_avg:.10e}\n")
        print(f"✓ Guardado: {mu_file}")
    else:
        print("✓ Series temporales ya escritas al paso en thermodynamics/*.txt (sin reescritura)")
    
    # Guardar resumen en JSON
    last_t = times[-1]
    summary = {
        'scenario_name': scenario_name,
        'num_times': len(times),
        'time_range': [float(min(times)), float(max(times))],
        'F_final': {
            'F_total': float(results['F'][last_t]['F_total']),
            'F_local': float(results['F'][last_t]['F_local']),
            'F_gradient': float(results['F'][last_t]['F_gradient']),
            'F_coupling': float(results['F'][last_t]['F_coupling'])
        },
        'sigma_mu_total_final': float(results['sigma'][last_t]),
        'sigma_final': float(results['sigma'][last_t]),
        'sigma_by_field_final': results.get('sigma_detail', {}).get(last_t),
        'mu_final': {
            'mu_c_avg': float(_get_mu_avg_spatial(results['mu'][last_t], 'c')),
            'mu_s_avg': float(_get_mu_avg_spatial(results['mu'][last_t], 's')),
            'mu_i_avg': float(_get_mu_avg_spatial(results['mu'][last_t], 'i')),
        }
    }
    
    summary_file = thermo_dir / 'thermodynamic_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✓ Guardado: {summary_file}")
    
    # Generar gráficas si está habilitado
    if generate_plots:
        try:
            plot_thermodynamic_properties(results, scenario_name, thermo_dir)
        except Exception as e:
            print(f"⚠ Advertencia: Error al generar gráficas: {e}")
            print("  Continuando sin gráficas...")


# ============================================================================
# Función principal de cálculo
# ============================================================================

def calculate_thermodynamic_properties(
    scenario_name: str,
    scenarios_file: Optional[Path] = None,
    base_dir: Optional[Path] = None,
    block: int = 1,
    use_checkpoint: Optional[bool] = None,
    fresh_thermo: bool = False,
) -> Dict:
    """
    Calcula las propiedades termodinámicas para un escenario.
    
    Args:
        scenario_name: Nombre del escenario
        scenarios_file: Ruta al archivo scenarios.json (None = buscar en directorio actual)
        base_dir: Directorio base (None = usar utils_paths)
        block: Número de bloque (default: 1)
        use_checkpoint: None = leer ENABLE_CHECKPOINT del entorno (default Y); False desactiva
        fresh_thermo: True = ignorar checkpoint y truncar series .txt (empezar de cero)
    
    Returns:
        Diccionario con resultados calculados
    """
    print(f"\n{'='*60}")
    print(f"Calculando propiedades termodinámicas para: {scenario_name}")
    print(f"{'='*60}")
    
    # Obtener directorio del escenario (base local por defecto: raíz Allee, no esta carpeta)
    effective_base = base_dir if base_dir is not None else _ALLEE_ROOT
    scenario_dir = get_scenario_dir(scenario_name, effective_base)
    print(f"Directorio del escenario: {scenario_dir}")
    
    if not scenario_dir.exists():
        raise FileNotFoundError(f"Directorio del escenario no existe: {scenario_dir}")
    
    # Cargar parámetros (prioridad scenarios_v1.json)
    if scenarios_file is None:
        scenarios_file = _ALLEE_ROOT / 'scenarios_v1.json'
        if not scenarios_file.exists():
            scenarios_file = _ALLEE_ROOT / 'scenarios.json'
        if not scenarios_file.exists():
            scenarios_file = _ALLEE_ROOT / 'scenarios copy.json'
    
    if not scenarios_file.exists():
        raise FileNotFoundError(f"Archivo scenarios.json no encontrado: {scenarios_file}")
    
    params = load_from_scenarios_json(scenarios_file, scenario_name, load_spatial_params=True)
    print(f"Parámetros cargados: μ={params.mu}, Allee={params.allee_type}")
    
    # Obtener tiempos disponibles
    times = get_available_times(scenario_dir, 'c', block)
    if not times:
        raise ValueError(f"No se encontraron matrices para el escenario {scenario_name}")
    
    print(f"Tiempos encontrados: {len(times)} (desde {min(times):.3f} hasta {max(times):.3f})")
    
    # Calcular espaciado espacial
    space_size = params.space_size
    # Asumir que las matrices tienen el mismo tamaño que nodes_in_xaxis x nodes_in_yaxis
    # Necesitamos cargar una matriz para obtener el tamaño
    sample_matrix = load_field_matrix(scenario_dir, 'c', times[0], block)
    if sample_matrix is None:
        raise ValueError(f"No se pudo cargar matriz de ejemplo para {scenario_name}")
    
    ny, nx = sample_matrix.shape
    dx = space_size / (nx - 1)  # Aproximación del espaciado
    
    print(f"Tamaño de matrices: {ny}x{nx}, dx ≈ {dx:.6f}")
    
    scenarios_resolved = scenarios_file.resolve()
    thermo_dir = scenario_dir / 'thermodynamics'
    if use_checkpoint is None:
        enable_checkpoint = os.getenv("ENABLE_CHECKPOINT", "Y").upper() == "Y"
    else:
        enable_checkpoint = bool(use_checkpoint)
    checkpoint_interval = max(1, int(os.getenv("CHECKPOINT_INTERVAL", "500")))
    checkpoint_mem_pct = float(os.getenv("CHECKPOINT_MEMORY_THRESHOLD_PCT", "80"))
    mem_ckpt_min_steps = max(1, int(os.getenv("THERMO_MEM_CHECKPOINT_MIN_STEPS", "100")))
    last_ckpt_index = 0
    
    results: Dict = {
        "times": times,
        "F": {},
        "sigma": {},
        "sigma_detail": {},
        "mu": {},
    }
    completed_rounded: set = set()
    
    if fresh_thermo:
        remove_thermo_checkpoint(thermo_dir)
        init_incremental_thermo_files(thermo_dir)
        if enable_checkpoint:
            print("  [Checkpoint termo] Modo --fresh-thermo: series reiniciadas, checkpoint eliminado")
    elif enable_checkpoint:
        ck_raw = load_thermo_checkpoint(thermo_dir)
        if ck_raw and _thermo_checkpoint_valid_for_run(
            ck_raw, scenario_name, block, scenarios_resolved, times
        ):
            loaded = load_thermo_results_from_txt(thermo_dir)
            n_ck = len(ck_raw.get("completed_times") or [])
            if loaded and len(loaded["F"]) > 0:
                if len(loaded["F"]) != n_ck:
                    print(
                        f"  [Checkpoint termo] Aviso: filas en .txt ({len(loaded['F'])}) ≠ "
                        f"completed_times en JSON ({n_ck}); se confía en los .txt"
                    )
                results = loaded
                completed_rounded = {_thermo_time_key(tk) for tk in results["F"]}
                print(
                    f"  [Checkpoint termo] Reanudando: {len(completed_rounded)} tiempos ya en disco "
                    f"(intervalo guardado cada {checkpoint_interval} índices, mem umbral {checkpoint_mem_pct}%)"
                )
            else:
                print("  [Checkpoint termo] JSON válido pero .txt vacíos o corruptos; reinicio series")
                remove_thermo_checkpoint(thermo_dir)
                init_incremental_thermo_files(thermo_dir)
        else:
            if ck_raw:
                print("  [Checkpoint termo] Checkpoint no coincide con escenario/tiempos; se elimina JSON")
                remove_thermo_checkpoint(thermo_dir)
            times_keys = {_thermo_time_key(x) for x in times}
            orphan = load_thermo_results_from_txt(thermo_dir)
            if (
                orphan
                and len(orphan["F"]) > 0
                and all(_thermo_time_key(tf) in times_keys for tf in orphan["F"])
            ):
                results = orphan
                completed_rounded = {_thermo_time_key(tk) for tk in results["F"]}
                if len(completed_rounded) < len(times):
                    print(
                        "  [Checkpoint termo] Reanudación desde .txt sin JSON válido "
                        f"({len(completed_rounded)} tiempos ya escritos)"
                    )
            else:
                init_incremental_thermo_files(thermo_dir)
    else:
        init_incremental_thermo_files(thermo_dir)
        print("  [Checkpoint termo] Desactivado (ENABLE_CHECKPOINT=N o use_checkpoint=False)")
    
    # Calcular para cada tiempo
    for i, t in enumerate(times):
        tk = _thermo_time_key(t)
        if tk in completed_rounded:
            if i > 0 and i % checkpoint_interval == 0:
                print(f"  (reanudado) índice {i}/{len(times)} — tiempos ya en disco, siguiendo…")
            continue
        
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Procesando tiempo {t:.3f} ({i+1}/{len(times)})...")
        
        # Cargar matrices
        c = load_field_matrix(scenario_dir, 'c', t, block)
        s = load_field_matrix(scenario_dir, 's', t, block)
        i_field = load_field_matrix(scenario_dir, 'i', t, block)
        
        if c is None or s is None or i_field is None:
            print(f"  ⚠ Saltando tiempo {t:.3f}: matrices incompletas")
            continue
        
        # Calcular energía libre
        F_data = calculate_free_energy(c, s, i_field, params, dx)
        F_total = np.sum(F_data['F_total']) * dx * dx  # Integración espacial
        F_local = np.sum(F_data['F_local']) * dx * dx
        F_gradient = np.sum(F_data['F_gradient']) * dx * dx
        F_coupling = np.sum(F_data['F_coupling']) * dx * dx
        
        results['F'][t] = {
            'F_total': F_total,
            'F_local': F_local,
            'F_gradient': F_gradient,
            'F_coupling': F_coupling
        }
        
        # Calcular potenciales químicos (solo en RAM para este t)
        mu_data = calculate_chemical_potentials(c, s, i_field, params)
        
        # Producción de entropía: integral D|grad mu|^2 (total) + desglose por campo;
        # disipación difusiva D|grad phi|^2 por especie (TdC, flujo x fuerza con phi)
        sigma_detail = calculate_entropy_and_dissipation_integrals(
            mu_data['mu_c'],
            mu_data['mu_s'],
            mu_data['mu_i'],
            c,
            s,
            i_field,
            params,
            dx,
        )
        results['sigma'][t] = sigma_detail['int_mu_total']
        results['sigma_detail'][t] = sigma_detail
        
        mu_c_avg = float(np.mean(mu_data['mu_c']))
        mu_s_avg = float(np.mean(mu_data['mu_s']))
        mu_i_avg = float(np.mean(mu_data['mu_i']))
        results['mu'][t] = {
            'mu_c_avg': mu_c_avg,
            'mu_s_avg': mu_s_avg,
            'mu_i_avg': mu_i_avg,
        }
        append_incremental_thermo_row(
            thermo_dir,
            t,
            results['F'][t],
            results['sigma'][t],
            sigma_detail,
            mu_c_avg,
            mu_s_avg,
            mu_i_avg,
        )
        completed_rounded.add(tk)
        
        if enable_checkpoint:
            mem_pct = _get_memory_percentage()
            periodic_ckpt = i > 0 and i % checkpoint_interval == 0
            mem_ckpt = (
                mem_pct >= checkpoint_mem_pct
                and mem_pct > 0
                and (i - last_ckpt_index) >= mem_ckpt_min_steps
            )
            should_ckpt = periodic_ckpt or mem_ckpt
            if should_ckpt:
                save_thermo_checkpoint(
                    thermo_dir, scenario_name, block, scenarios_resolved, times, results
                )
                last_ckpt_index = i
                reason = []
                if periodic_ckpt:
                    reason.append(f"cada {checkpoint_interval} índices")
                if mem_ckpt:
                    reason.append(f"RAM {mem_pct:.1f}% (≥{checkpoint_mem_pct}%)")
                print(
                    f"  [Checkpoint termo] Guardado índice {i}, t={t:.4f}"
                    + (f" — {', '.join(reason)}" if reason else "")
                )
    
    # Solo tiempos con datos completos (evita desalinear series guardadas)
    results['times'] = sorted(results['F'].keys())
    if not results['times']:
        raise ValueError(
            f"No se procesó ningún tiempo con matrices c,s,i completas en {scenario_dir}"
        )
    
    print(f"\n✓ Cálculo completado para {scenario_name}")
    
    if enable_checkpoint:
        remove_thermo_checkpoint(thermo_dir)
        print("  [Checkpoint termo] Completado: thermo_checkpoint_latest.json eliminado")
    
    # Resumen JSON y gráficas (series .txt ya escritas al paso)
    save_thermodynamic_results(
        results,
        scenario_dir,
        scenario_name,
        generate_plots=True,
        incremental_series_on_disk=True,
    )
    
    return results


# ============================================================================
# Comparaciones entre escenarios
# ============================================================================

def calculate_comparisons(
    all_results: Dict[str, Dict],
    scenarios_file: Path
) -> Dict:
    """
    Calcula comparaciones automáticas entre escenarios.
    
    Args:
        all_results: Diccionario {scenario_name: results}
        scenarios_file: Ruta al archivo scenarios.json
    
    Returns:
        Diccionario con comparaciones
    """
    comparisons = {}
    
    # Cargar información de escenarios
    with open(scenarios_file, 'r', encoding='utf-8') as f:
        scenarios_data = json.load(f)
    
    scenarios = scenarios_data.get('scenarios', [])
    scenario_dict = {s['name']: s for s in scenarios}
    
    # Función auxiliar para extraer características del escenario
    def get_scenario_features(name: str) -> Dict:
        scenario = scenario_dict.get(name, {})
        return {
            'allee_type': scenario.get('ALLEE_TYPE', 'WEAK'),
            'mu': float(scenario.get('mu', 0)),
            'use_control': scenario.get('USE_ADAPTIVE_CONTROL', 'N') == 'Y',
            'beta': float(scenario.get('beta', 0)),
            'rd': float(scenario.get('rd', 0)),
            'c_init_min': float(scenario.get('C_INIT_MIN', 0)),
            'c_init_max': float(scenario.get('C_INIT_MAX', 0))
        }
    
    # Comparación 1: Efecto de μ (μ=0 vs μ=1)
    for scenario_name, results in all_results.items():
        features = get_scenario_features(scenario_name)
        
        # Buscar escenario equivalente con μ diferente
        for other_name, other_results in all_results.items():
            if other_name == scenario_name:
                continue
            
            other_features = get_scenario_features(other_name)
            
            # Comparar si solo difieren en μ
            if (features['allee_type'] == other_features['allee_type'] and 
                features['use_control'] == other_features['use_control'] and
                features['mu'] != other_features['mu'] and
                abs(features['beta'] - other_features['beta']) < 0.1 and
                abs(features['rd'] - other_features['rd']) < 0.1):
                
                key = f"mu_comparison_{features['allee_type']}_{'uY' if features['use_control'] else 'uN'}"
                if key not in comparisons:
                    comparisons[key] = {}
                
                times = results['times']
                final_time = times[-1] if times else None
                
                if final_time and final_time in results['F'] and final_time in other_results['F']:
                    F_mu0 = results['F'][final_time]['F_total'] if features['mu'] == 0 else other_results['F'][final_time]['F_total']
                    F_mu1 = results['F'][final_time]['F_total'] if features['mu'] == 1 else other_results['F'][final_time]['F_total']
                    
                    comparisons[key][f"{scenario_name}_vs_{other_name}"] = {
                        'Delta_F_mu': float(F_mu1 - F_mu0),
                        'F_mu0': float(F_mu0),
                        'F_mu1': float(F_mu1),
                        'sigma_mu0': float(results['sigma'][final_time] if features['mu'] == 0 else other_results['sigma'][final_time]),
                        'sigma_mu1': float(results['sigma'][final_time] if features['mu'] == 1 else other_results['sigma'][final_time])
                    }
    
    # Comparación 2: Efecto del tipo de Allee (WEAK vs STRONG)
    for scenario_name, results in all_results.items():
        features = get_scenario_features(scenario_name)
        
        for other_name, other_results in all_results.items():
            if other_name == scenario_name:
                continue
            
            other_features = get_scenario_features(other_name)
            
            # Comparar si solo difieren en tipo de Allee
            if (features['allee_type'] != other_features['allee_type'] and
                features['mu'] == other_features['mu'] and
                features['use_control'] == other_features['use_control']):
                
                key = f"allee_comparison_mu{int(features['mu'])}_{'uY' if features['use_control'] else 'uN'}"
                if key not in comparisons:
                    comparisons[key] = {}
                
                times = results['times']
                final_time = times[-1] if times else None
                
                if final_time and final_time in results['F'] and final_time in other_results['F']:
                    F_weak = results['F'][final_time]['F_total'] if features['allee_type'] == 'WEAK' else other_results['F'][final_time]['F_total']
                    F_strong = results['F'][final_time]['F_total'] if features['allee_type'] == 'STRONG' else other_results['F'][final_time]['F_total']
                    
                    comparisons[key][f"{scenario_name}_vs_{other_name}"] = {
                        'Delta_F_allee': float(F_strong - F_weak),
                        'F_weak': float(F_weak),
                        'F_strong': float(F_strong)
                    }
    
    # Comparación 3: Efecto del control adaptativo
    for scenario_name, results in all_results.items():
        features = get_scenario_features(scenario_name)
        
        for other_name, other_results in all_results.items():
            if other_name == scenario_name:
                continue
            
            other_features = get_scenario_features(other_name)
            
            # Comparar si solo difieren en control
            if (features['allee_type'] == other_features['allee_type'] and
                features['mu'] == other_features['mu'] and
                features['use_control'] != other_features['use_control']):
                
                key = f"control_comparison_{features['allee_type']}_mu{int(features['mu'])}"
                if key not in comparisons:
                    comparisons[key] = {}
                
                times = results['times']
                final_time = times[-1] if times else None
                
                if final_time and final_time in results['F'] and final_time in other_results['F']:
                    F_no = results['F'][final_time]['F_total'] if not features['use_control'] else other_results['F'][final_time]['F_total']
                    F_yes = results['F'][final_time]['F_total'] if features['use_control'] else other_results['F'][final_time]['F_total']
                    
                    comparisons[key][f"{scenario_name}_vs_{other_name}"] = {
                        'Delta_F_control': float(F_yes - F_no),
                        'F_no_control': float(F_no),
                        'F_with_control': float(F_yes)
                    }
    
    return comparisons


# ============================================================================
# Interfaz de línea de comandos
# ============================================================================

def main():
    """Función principal con interfaz de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Calcula propiedades termodinámicas (F, σ, μ) para escenarios de simulación'
    )
    parser.add_argument(
        '--scenario',
        type=str,
        help='Nombre del escenario específico a procesar'
    )
    parser.add_argument(
        '--scenarios',
        nargs='+',
        help='Lista de nombres de escenarios a procesar'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Procesar todos los escenarios en scenarios.json'
    )
    parser.add_argument(
        '--scenarios-file',
        type=str,
        help='Ruta al JSON de escenarios (default: scenarios_v1.json si existe, si no scenarios.json)'
    )
    parser.add_argument(
        '--block',
        type=int,
        default=1,
        help='Número de bloque a procesar (default: 1)'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        help='Directorio base para buscar resultados (default: usar utils_paths)'
    )
    parser.add_argument(
        '--no-checkpoint',
        action='store_true',
        help='Desactivar checkpoints/reanudación (equivale a ENABLE_CHECKPOINT=N)',
    )
    parser.add_argument(
        '--fresh-thermo',
        action='store_true',
        help='Truncar series en thermodynamics/*.txt y borrar checkpoint termo (empezar de cero)',
    )
    
    args = parser.parse_args()
    
    # Determinar archivo de escenarios
    if args.scenarios_file:
        scenarios_file = Path(args.scenarios_file)
    else:
        scenarios_file = _ALLEE_ROOT / 'scenarios_v1.json'
        if not scenarios_file.exists():
            scenarios_file = _ALLEE_ROOT / 'scenarios.json'
        if not scenarios_file.exists():
            scenarios_file = _ALLEE_ROOT / 'scenarios copy.json'
    
    if not scenarios_file.exists():
        print(f"✗ Error: No se encontró scenarios.json en {scenarios_file}")
        sys.exit(1)
    
    # Cargar lista de escenarios
    with open(scenarios_file, 'r', encoding='utf-8') as f:
        scenarios_data = json.load(f)
    
    all_scenarios = [s['name'] for s in scenarios_data.get('scenarios', [])]
    
    # Determinar qué escenarios procesar
    if args.all:
        scenarios_to_process = all_scenarios
    elif args.scenarios:
        scenarios_to_process = args.scenarios
    elif args.scenario:
        scenarios_to_process = [args.scenario]
    else:
        print("✗ Error: Debes especificar --scenario, --scenarios, o --all")
        parser.print_help()
        sys.exit(1)
    
    # Validar escenarios
    invalid = [s for s in scenarios_to_process if s not in all_scenarios]
    if invalid:
        print(f"✗ Error: Escenarios no encontrados: {invalid}")
        print(f"  Escenarios disponibles: {all_scenarios}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Procesando {len(scenarios_to_process)} escenario(s)")
    print(f"{'='*60}\n")
    
    # Procesar cada escenario
    all_results = {}
    base_dir = Path(args.base_dir) if args.base_dir else _ALLEE_ROOT
    
    for scenario_name in scenarios_to_process:
        try:
            results = calculate_thermodynamic_properties(
                scenario_name,
                scenarios_file,
                base_dir,
                args.block,
                use_checkpoint=False if args.no_checkpoint else None,
                fresh_thermo=args.fresh_thermo,
            )
            all_results[scenario_name] = results
        except Exception as e:
            print(f"✗ Error procesando {scenario_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Calcular comparaciones si hay múltiples escenarios
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("Calculando comparaciones entre escenarios...")
        print(f"{'='*60}\n")
        
        comparisons = calculate_comparisons(all_results, scenarios_file)
        
        # Guardar comparaciones en el directorio de resultados
        results_dir = get_results_dir(base_dir)
        comparison_file = results_dir / 'thermodynamic_comparison.json'
        
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparisons, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Comparaciones guardadas en: {comparison_file}")
    
    print(f"\n{'='*60}")
    print("✓ Proceso completado")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

