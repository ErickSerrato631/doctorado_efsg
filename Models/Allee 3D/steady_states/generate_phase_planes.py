"""
Script para generar phase planes y nullclines (FIG. 2 y FIG. 3)
basándose en los escenarios de scenarios.json.
"""

import json
import os
import sys
from pathlib import Path

_ALLEE_ROOT = Path(__file__).resolve().parent.parent
if str(_ALLEE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALLEE_ROOT))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import sympy as sp
import pandas as pd
from scipy.optimize import fsolve

# Importar funciones necesarias
try:
    import json
    from model_parameters import ModelParameters
    from model_equations import c_sym, s_sym, rc_sym, rs_sym, rd_sym, alpha_sym, delta_sym, beta_sym, a_sym, gamma_sym, eta_sym, mu_sym
    import sympy as sp
except ImportError as e:
    print(f"Error importando módulos: {e}")
    print("Asegúrate de tener Models/Allee en PYTHONPATH o ejecutar con cwd Allee.")
    sys.exit(1)


def build_reduced_model_2d_sympy_strong(params: ModelParameters):
    """
    Construye el modelo reducido 2D para Strong Allee.
    Similar a build_reduced_model_2d_sympy pero con término de Allee fuerte.
    """
    c, s = c_sym, s_sym
    
    # Término interno para i*
    inner_expr = 2 * rd_sym + 2 * s**2 * delta_sym - c**2 * (2 * eta_sym + beta_sym * mu_sym)
    
    # F1 con término de Allee fuerte: rc * c * (1 - c) * ((c - a) / (1 - a))
    # En el modelo reducido esto se convierte en: -4 * rc * c * (1 - c) * ((c - a) / (1 - a))
    allee_term_strong = -4 * rc_sym * c * (1 - c) * ((c - a_sym) / (1 - a_sym))
    
    F1 = (sp.Rational(1, 4)) * c * (
        allee_term_strong - 4 * s**2 * alpha_sym -
        (beta_sym * inner_expr**2) / rd_sym**2 -
        4 * mu_sym * (s**2 * gamma_sym + (eta_sym * inner_expr**2) / (4 * rd_sym**2))
    )
    
    # F2 es igual para Weak y Strong
    F2 = (sp.Rational(1, 4)) * s * (
        -4 * rs_sym * (-1 + s) - 4 * c**2 * gamma_sym - 2 * c**2 * alpha_sym * mu_sym +
        (delta_sym * inner_expr**2) / rd_sym**2
    )
    
    # Sustituir valores numéricos
    subs_dict = {
        rc_sym: params.rc,
        rs_sym: params.rs,
        rd_sym: params.rd,
        alpha_sym: params.alpha,
        delta_sym: params.delta,
        beta_sym: params.beta,
        a_sym: params.a,
        gamma_sym: params.gamma,
        eta_sym: params.eta,
        mu_sym: params.mu,
    }
    
    F1 = F1.subs(subs_dict)
    F2 = F2.subs(subs_dict)
    
    return F1, F2


def _parse_phase_plane_pad() -> tuple[float, float]:
    """Lee padding desde entorno: ALLEE_PHASE_PLANE_PAD (default 5) y opcionales _PAD_C / _PAD_S."""
    default = 5.0
    raw = os.environ.get("ALLEE_PHASE_PLANE_PAD")
    if raw is None or str(raw).strip() == "":
        pad = default
    else:
        try:
            pad = float(raw)
        except ValueError:
            pad = default
    pad_c = os.environ.get("ALLEE_PHASE_PLANE_PAD_C")
    pad_s = os.environ.get("ALLEE_PHASE_PLANE_PAD_S")
    try:
        pad_c_f = float(pad_c) if pad_c is not None else pad
    except ValueError:
        pad_c_f = pad
    try:
        pad_s_f = float(pad_s) if pad_s is not None else pad
    except ValueError:
        pad_s_f = pad
    return pad_c_f, pad_s_f


def _env_truthy(key: str) -> bool:
    v = os.environ.get(key)
    if v is None or str(v).strip() == "":
        return False
    return str(v).strip().lower() in ("1", "y", "yes", "true", "on")


def _max_csv_equilibrium_markers() -> int:
    """Máximo de puntos del CSV a marcar cuando hay intersecciones de nullclinas (default 32)."""
    raw = os.environ.get("ALLEE_PHASE_PLANE_MAX_CSV_MARKERS", "32")
    try:
        n = int(raw)
        return max(1, min(256, n))
    except ValueError:
        return 32


def simplex_corner_points_3d() -> list[tuple[float, float, float]]:
    """Vértices (1,0,0), (0,1,0), (0,0,1) del simplex en (c,s,i)."""
    return [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]


# Vista fija (c,s) al marcar solo los vértices del simplex en el plano reducido
SIMPLEX_PHASE_PLANE_C_RANGE = (-0.5, 2.0)
SIMPLEX_PHASE_PLANE_S_RANGE = (-0.5, 2.0)


def phase_plane_ranges_from_equilibria(
    equilibrium_points,
    default_c_range: tuple[float, float] = (0, 1.5),
    default_s_range: tuple[float, float] = (0, 1.2),
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Bounding box de todos los (c*, s*) más padding (env ``ALLEE_PHASE_PLANE_PAD``, default 5).
    Por defecto se permiten c,s negativos (panorama amplio; p. ej. equilibrio en (0,0) → ~[-5,5]²).
    Para forzar solo primer cuadrante: ``ALLEE_PHASE_PLANE_CLIP_NONNEGATIVE=1``.
    """
    if not equilibrium_points:
        return default_c_range, default_s_range

    c_vals: list[float] = []
    s_vals: list[float] = []
    for point in equilibrium_points:
        if len(point) >= 2:
            c_vals.append(float(point[0]))
            s_vals.append(float(point[1]))

    if not c_vals:
        return default_c_range, default_s_range

    pad_c, pad_s = _parse_phase_plane_pad()
    c_min = min(c_vals) - pad_c
    c_max = max(c_vals) + pad_c
    s_min = min(s_vals) - pad_s
    s_max = max(s_vals) + pad_s

    clip_nonneg = os.environ.get("ALLEE_PHASE_PLANE_CLIP_NONNEGATIVE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if clip_nonneg:
        c_min = max(0.0, c_min)
        s_min = max(0.0, s_min)

    min_span_c = 0.15
    min_span_s = 0.15
    if c_max - c_min < min_span_c:
        mid = (c_min + c_max) / 2
        c_min, c_max = mid - min_span_c / 2, mid + min_span_c / 2
        if clip_nonneg:
            c_min = max(0.0, c_min)
    if s_max - s_min < min_span_s:
        mid = (s_min + s_max) / 2
        s_min, s_max = mid - min_span_s / 2, mid + min_span_s / 2
        if clip_nonneg:
            s_min = max(0.0, s_min)

    return (c_min, c_max), (s_min, s_max)


def _adaptive_n_points(c_range: tuple[float, float], s_range: tuple[float, float], base: int = 100) -> int:
    """Más resolución de malla cuando la ventana es pequeña (streamplot más estable)."""
    env_n = os.environ.get("ALLEE_PHASE_PLANE_N_POINTS")
    if env_n:
        try:
            n = int(env_n)
            return max(50, min(400, n))
        except ValueError:
            pass
    wc = c_range[1] - c_range[0]
    ws = s_range[1] - s_range[0]
    wmin = max(min(wc, ws), 1e-9)
    if wmin < 0.35:
        return max(base, 150)
    if wmin < 0.65:
        return max(base, 130)
    return base


def plot_nullclines_2d(
    f1,
    f2,
    c_range: tuple = (0, 1.5),
    s_range: tuple = (0, 1.2),
    n_points: int = 100,
    mu_val: float = 0,
    allee_type: str = "WEAK",
    title: str = None,
    save_path: Path = None,
    steady_states_points=None,
    *,
    simplex_corner_view: bool = False,
):
    """
    Visualiza campo vectorial (y, salvo vista simplex, nullclinas F1=0, F2=0) para el modelo 2D.
    
    Args:
        f1, f2: Funciones numéricas del sistema
        c_range, s_range: Rangos para graficar
        n_points: Número de puntos en cada dimensión
        mu_val: Valor de mu para el título
        allee_type: 'WEAK' o 'STRONG'
        title: Título personalizado (opcional)
        save_path: Ruta donde guardar la figura (opcional)
        steady_states_points: Lista de tuplas (c, s) o (c, s, i) con puntos de equilibrio a marcar (opcional)
        simplex_corner_view: Si True, solo streamlines y ``steady_states_points``; no se dibujan
            curvas F1=0 ni F2=0 ni intersecciones/malla de equilibrios. Los puntos son proyección
            de vértices 3D en el plano (c,s).
    """
    n_points = _adaptive_n_points(c_range, s_range, n_points)

    # Crear malla inicial con rangos por defecto (más amplios para encontrar intersecciones)
    # Usaremos una malla amplia primero para encontrar todas las intersecciones
    C, S = np.meshgrid(
        np.linspace(c_range[0], c_range[1], n_points),
        np.linspace(s_range[0], s_range[1], n_points)
    )
    
    # Evaluar funciones en la malla
    U = np.zeros_like(C)
    V = np.zeros_like(C)
    
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            try:
                U[i, j] = float(f1(C[i, j], S[i, j]))
                V[i, j] = float(f2(C[i, j], S[i, j]))
            except:
                U[i, j] = np.nan
                V[i, j] = np.nan
    
    # Filtrar valores no finitos
    mask_finite = np.isfinite(U) & np.isfinite(V)
    
    # Calcular curvas nulas e intersecciones aproximadas
    f1_vals = np.abs(U)
    f2_vals = np.abs(V)
    tolerance = 0.01
    mask_eq = mask_finite & (f1_vals < tolerance) & (f2_vals < tolerance)
    eq_points = np.column_stack((C[mask_eq], S[mask_eq]))
    
    # Gráfica
    plt.figure(figsize=(10, 9))
    
    # Campo vectorial (solo donde los valores son finitos)
    C_finite = C[mask_finite]
    S_finite = S[mask_finite]
    U_finite = U[mask_finite]
    V_finite = V[mask_finite]
    
    strm = None
    if len(C_finite) > 0:
        # Malla más fina y mayor density = más líneas y flechas
        step = max(1, min(n_points // 52, n_points - 1))
        if step >= n_points:
            step = max(1, n_points - 1)

        C_sub = C[::step, ::step]
        S_sub = S[::step, ::step]
        U_sub = U[::step, ::step]
        V_sub = V[::step, ::step]

        stream_color = "#2d4a66"
        stream_density = 4.25
        stream_lw = 1.55

        # Verificar que tenemos al menos 2x2 puntos
        if C_sub.shape[0] >= 2 and C_sub.shape[1] >= 2:
            strm = plt.streamplot(
                C_sub,
                S_sub,
                U_sub,
                V_sub,
                color=stream_color,
                linewidth=stream_lw,
                density=stream_density,
                arrowstyle="->",
                arrowsize=2.35,
            )
        else:
            plt.quiver(
                C_sub, S_sub, U_sub, V_sub, color=stream_color, scale=22, width=0.004
            )
    
    # Curvas nulas (F1=0, F2=0): no se dibujan en vista simplex — solo campo vectorial y vértices.
    contour_f1 = None
    contour_f2 = None
    nullcline_intersections = []

    if not simplex_corner_view:
        try:
            contour_f1 = plt.contour(C, S, U, levels=[0], colors='black', linewidths=2)
            contour_f2 = plt.contour(C, S, V, levels=[0], colors='blue', linewidths=2)

            # Encontrar intersecciones de las nullclinos usando los paths de los contornos
            if contour_f1 is not None and contour_f2 is not None:
                try:
                    paths1 = []
                    paths2 = []
                    if hasattr(contour_f1, 'collections'):
                        for collection1 in contour_f1.collections:
                            paths1.extend(collection1.get_paths())
                    elif hasattr(contour_f1, 'get_paths'):
                        paths1 = contour_f1.get_paths()
                    if hasattr(contour_f2, 'collections'):
                        for collection2 in contour_f2.collections:
                            paths2.extend(collection2.get_paths())
                    elif hasattr(contour_f2, 'get_paths'):
                        paths2 = contour_f2.get_paths()
                    for path1 in paths1:
                        vertices1 = path1.vertices
                        for path2 in paths2:
                            vertices2 = path2.vertices
                            for v1 in vertices1:
                                for v2 in vertices2:
                                    dist = np.sqrt((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2)
                                    if dist < 0.05:
                                        intersection = ((v1[0] + v2[0]) / 2, (v1[1] + v2[1]) / 2)
                                        is_duplicate = False
                                        for existing in nullcline_intersections:
                                            if np.sqrt(
                                                (intersection[0] - existing[0]) ** 2
                                                + (intersection[1] - existing[1]) ** 2
                                            ) < 0.1:
                                                is_duplicate = True
                                                break
                                        if not is_duplicate:
                                            nullcline_intersections.append(intersection)
                except Exception:
                    pass

            tolerance_intersection = 0.02
            mask_intersection = mask_finite & (np.abs(U) < tolerance_intersection) & (
                np.abs(V) < tolerance_intersection
            )
            intersection_points = np.column_stack((C[mask_intersection], S[mask_intersection]))

            refined_intersections = []
            if len(intersection_points) > 0:
                unique_seeds = []
                for point in intersection_points:
                    is_duplicate = False
                    for existing in unique_seeds:
                        if np.sqrt((point[0] - existing[0]) ** 2 + (point[1] - existing[1]) ** 2) < 0.1:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        unique_seeds.append(point)

                def system_eq(x):
                    c_val, s_val = x
                    try:
                        return [float(f1(c_val, s_val)), float(f2(c_val, s_val))]
                    except Exception:
                        return [np.nan, np.nan]

                for seed in unique_seeds[:5]:
                    try:
                        solution = fsolve(system_eq, [seed[0], seed[1]], xtol=1e-6, maxfev=100)
                        c_refined, s_refined = solution
                        if c_range[0] <= c_refined <= c_range[1] and s_range[0] <= s_refined <= s_range[1]:
                            f1_val = f1(c_refined, s_refined)
                            f2_val = f2(c_refined, s_refined)
                            if np.abs(f1_val) < 0.01 and np.abs(f2_val) < 0.01:
                                refined_intersections.append((c_refined, s_refined))
                    except Exception:
                        refined_intersections.append((seed[0], seed[1]))

                for point in refined_intersections:
                    is_duplicate = False
                    for existing in nullcline_intersections:
                        if np.sqrt((point[0] - existing[0]) ** 2 + (point[1] - existing[1]) ** 2) < 0.05:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        nullcline_intersections.append(point)
        except Exception:
            pass

    # Debug: imprimir información sobre intersecciones encontradas
    if len(nullcline_intersections) > 0 and not simplex_corner_view:
        print(f"    Encontradas {len(nullcline_intersections)} intersecciones de nullclinos:")
        for idx, (c_int, s_int) in enumerate(nullcline_intersections[:5]):
            print(f"      Intersección {idx+1}: c={c_int:.4f}, s={s_int:.4f}")
    
    # Marcar puntos de equilibrio explícitamente
    equilibrium_markers = []
    
    # Primero marcar las intersecciones de nullclinos encontradas
    # Mostrar TODAS las intersecciones encontradas (son las más importantes)
    if not simplex_corner_view and len(nullcline_intersections) > 0:
        # Mostrar todas las intersecciones, sin filtrar demasiado
        valid_intersections = []
        for c_int, s_int in nullcline_intersections:
            # Incluir todas las intersecciones que están razonablemente en el rango
            # Usar tolerancia muy amplia para asegurar que se muestren
            tolerance = 0.2
            if (c_range[0] - tolerance <= c_int <= c_range[1] + tolerance and 
                s_range[0] - tolerance <= s_int <= s_range[1] + tolerance):
                valid_intersections.append((c_int, s_int))
        
        # Si hay muchas intersecciones, mostrar las más importantes (hasta 4)
        if len(valid_intersections) > 4:
            # Ordenar por distancia al centro del rango y tomar las más cercanas
            c_center = (c_range[0] + c_range[1]) / 2
            s_center = (s_range[0] + s_range[1]) / 2
            valid_intersections.sort(key=lambda p: 
                np.sqrt((p[0] - c_center)**2 + (p[1] - s_center)**2))
            valid_intersections = valid_intersections[:4]
        
        print(f"    Marcando {len(valid_intersections)} intersecciones válidas en el gráfico")
        for i, (c_int, s_int) in enumerate(valid_intersections):
            # Usar las coordenadas exactas de la intersección
            c_plot = c_int
            s_plot = s_int
            
            print(f"      Marcando intersección {i+1} en ({c_plot:.4f}, {s_plot:.4f})")
            
            # Marcar el punto de intersección (más grande y visible)
            plt.plot(c_plot, s_plot, 'ro', markersize=20, markeredgecolor='darkred', 
                    markeredgewidth=4, label='Intersección nullclinos' if i == 0 else '', zorder=20, alpha=1.0)
            
            # Etiqueta con coordenadas (sin recuadro)
            plt.annotate(
                f'$c^*={c_int:.3f}$, $s^*={s_int:.3f}$',
                xy=(c_plot, s_plot),
                xytext=(15, 15),
                textcoords='offset points',
                fontsize=11,
            )
            equilibrium_markers.append((c_int, s_int))
    
    # Luego marcar los puntos del CSV si están disponibles
    # Mostrar todos los puntos que están en el rango visible
    if steady_states_points is not None and len(steady_states_points) > 0:
        # Incluir todos los puntos en rango
        valid_csv_points = []
        for point_data in steady_states_points:
            if len(point_data) == 3:
                c_eq, s_eq, i_eq = point_data
            else:
                c_eq, s_eq = point_data
                i_eq = None
            
            # Verificar que esté en el rango visible con tolerancia muy amplia
            tolerance = 0.2
            c_in_range = (c_range[0] - tolerance) <= c_eq <= (c_range[1] + tolerance)
            s_in_range = (s_range[0] - tolerance) <= s_eq <= (s_range[1] + tolerance)
            
            if c_in_range and s_in_range:
                valid_csv_points.append(point_data)
        
        # Si hay múltiples puntos válidos y ya marcamos intersecciones, 
        # solo mostrar puntos del CSV que no estén muy cerca de las intersecciones ya marcadas
        if len(valid_csv_points) > 0 and len(equilibrium_markers) > 0:
            filtered_csv_points = []
            for point_data in valid_csv_points:
                if len(point_data) == 3:
                    c_eq, s_eq, _ = point_data
                else:
                    c_eq, s_eq = point_data
                
                # Verificar si está cerca de alguna intersección ya marcada
                is_near_marked = False
                for marked in equilibrium_markers:
                    if len(marked) >= 2:
                        dist = np.sqrt((c_eq - marked[0])**2 + (s_eq - marked[1])**2)
                        if dist < 0.1:  # Muy cerca de una intersección ya marcada
                            is_near_marked = True
                            break
                
                if not is_near_marked:
                    filtered_csv_points.append(point_data)
            
            valid_csv_points = filtered_csv_points[: _max_csv_equilibrium_markers()]
        
        for i, point_data in enumerate(valid_csv_points):
            # Manejar tanto tuplas (c, s) como (c, s, i)
            if len(point_data) == 3:
                c_eq, s_eq, i_eq = point_data
            else:
                c_eq, s_eq = point_data
                i_eq = None
            
            # Verificar que el punto esté dentro del rango visible
            c_in_range = c_range[0] <= c_eq <= c_range[1]
            s_in_range = s_range[0] <= s_eq <= s_range[1]
            
            if c_in_range and s_in_range:
                # Usar las coordenadas exactas
                c_plot = c_eq
                s_plot = s_eq
                
                # Asegurar visibilidad incluso si está cerca del borde
                c_plot = max(c_range[0] - 0.05, min(c_range[1] + 0.05, c_plot))
                s_plot = max(s_range[0] - 0.05, min(s_range[1] + 0.05, s_plot))
                
                if not simplex_corner_view:
                    radius = 0.06
                    circle = Circle(
                        (c_plot, s_plot),
                        radius,
                        fill=False,
                        edgecolor="red",
                        linewidth=2.5,
                        linestyle="--",
                        alpha=0.8,
                    )
                    plt.gca().add_patch(circle)
                
                plt.plot(
                    c_plot,
                    s_plot,
                    "ro",
                    markersize=15,
                    markeredgecolor="darkred",
                    markeredgewidth=3,
                    label=("Punto de equilibrio" if i == 0 else ""),
                    zorder=15,
                    alpha=0.9,
                )
                
                if not simplex_corner_view:
                    if i_eq is not None and np.isfinite(i_eq):
                        label_text = f'$c^*={c_eq:.3f}$, $s^*={s_eq:.3f}$, $i^*={i_eq:.3f}$'
                    else:
                        label_text = f'$c^*={c_eq:.3f}$, $s^*={s_eq:.3f}$'
                    plt.annotate(
                        label_text,
                        xy=(c_plot, s_plot),
                        xytext=(12, 12),
                        textcoords="offset points",
                        fontsize=10,
                    )
                equilibrium_markers.append((c_eq, s_eq, i_eq) if i_eq is not None else (c_eq, s_eq))
            else:
                # Si está fuera del rango pero cerca, marcarlo de todos modos
                if abs(c_eq - c_range[0]) < 0.15 or abs(c_eq - c_range[1]) < 0.15 or \
                   abs(s_eq - s_range[0]) < 0.15 or abs(s_eq - s_range[1]) < 0.15:
                    c_plot = c_eq
                    s_plot = s_eq
                    plt.plot(c_plot, s_plot, 'ro', markersize=10, markeredgecolor='darkred', 
                            markeredgewidth=2, alpha=0.7, zorder=10)
                    
                    if not simplex_corner_view:
                        if i_eq is not None and np.isfinite(i_eq):
                            label_text = f'$c^*={c_eq:.3f}$, $s^*={s_eq:.3f}$, $i^*={i_eq:.3f}$'
                        else:
                            label_text = f'$c^*={c_eq:.3f}$, $s^*={s_eq:.3f}$'
                        plt.annotate(
                            label_text,
                            xy=(c_plot, s_plot),
                            xytext=(10, 10),
                            textcoords='offset points',
                            fontsize=8,
                        )
                    equilibrium_markers.append((c_eq, s_eq, i_eq) if i_eq is not None else (c_eq, s_eq))
    
    # Si no se encontraron intersecciones explícitas, usar puntos aproximados de la malla
    if not simplex_corner_view and len(nullcline_intersections) == 0 and eq_points.shape[0] > 0:
        # Agrupar puntos cercanos para evitar duplicados
        unique_eq_points = []
        for x, y in eq_points:
            is_duplicate = False
            for ux, uy in unique_eq_points:
                if np.sqrt((x - ux)**2 + (y - uy)**2) < 0.05:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_eq_points.append((x, y))
        
        # Marcar puntos aproximados encontrados en la malla
        for i, (x, y) in enumerate(unique_eq_points[:3]):  # Máximo 3 puntos
            if c_range[0] <= x <= c_range[1] and s_range[0] <= y <= s_range[1]:
                # Verificar que no esté ya marcado
                already_marked = False
                for marked in equilibrium_markers:
                    if len(marked) >= 2:
                        if np.sqrt((x - marked[0])**2 + (y - marked[1])**2) < 0.1:
                            already_marked = True
                            break
                
                if not already_marked:
                    plt.plot(x, y, 'g*', markersize=15, markeredgecolor='darkgreen', 
                            markeredgewidth=1.5, label='Equilibrio aprox.' if i == 0 else '', zorder=9)
                    plt.annotate(
                        f'$c^*={x:.3f}$, $s^*={y:.3f}$',
                        xy=(x, y),
                        xytext=(10, -15),
                        textcoords='offset points',
                        fontsize=8,
                    )
    
    # Leyenda (sin vista simplex; sin entrada "Entorno")
    if simplex_corner_view:
        legend_elements = []
    else:
        legend_elements = [
            Line2D([0], [0], color='black', lw=2, label='F1 = 0'),
            Line2D([0], [0], color='blue', lw=2, label='F2 = 0'),
        ]
    
    if equilibrium_markers and not simplex_corner_view:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor='red', markersize=10, 
                                     markeredgecolor='darkred', markeredgewidth=2,
                                     label='Punto de equilibrio'))
    
    if legend_elements:
        plt.legend(handles=legend_elements, loc='best', fontsize=10)
    
    # Configuración final
    plt.xlabel('c', fontsize=14)
    plt.ylabel('s', fontsize=14)
    if title is None:
        allee_str = 'Weak' if allee_type == 'WEAK' else 'Strong'
        plt.title(f'Phase plane - {allee_str} Allee, $\\mu = {mu_val}$', fontsize=15)
    else:
        plt.title(title, fontsize=15)
    
    plt.grid(True, alpha=0.3)
    plt.xlim(c_range)
    plt.ylim(s_range)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Guardado: {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_phase_plane_figures(
    scenarios_file: Path,
    output_dir: Path,
    steady_states_csv: Path = None,
    *,
    mark_simplex_corners: bool | None = None,
):
    """
    Genera las figuras de phase planes para FIG. 2 y FIG. 3.
    
    Args:
        scenarios_file: Ruta al archivo scenarios.json
        output_dir: Directorio base; cada figura se guarda en output_dir/<nombre_escenario>/
        steady_states_csv: Ruta al archivo CSV con estados estacionarios (opcional)
        mark_simplex_corners: Si True, ejes c,s in [-0.5, 2] y solo vértices (1,0),(0,1),(0,0);
            sin curvas F1=0/F2=0 (solo campo 2D reducido). Si None, ALLEE_PHASE_PLANE_MARK_SIMPLEX_CORNERS.
    """
    if mark_simplex_corners is None:
        mark_simplex_corners = _env_truthy("ALLEE_PHASE_PLANE_MARK_SIMPLEX_CORNERS")
    print("Generando phase planes...")
    
    # Cargar estados estacionarios si el archivo existe
    steady_states_df = None
    if steady_states_csv is not None and steady_states_csv.exists():
        try:
            steady_states_df = pd.read_csv(steady_states_csv)
            print(f"  Cargados {len(steady_states_df)} estados estacionarios desde CSV")
        except Exception as e:
            print(f"  [!] No se pudo cargar CSV de estados estacionarios: {e}")
    
    # Cargar escenarios
    with open(scenarios_file, 'r', encoding='utf-8') as f:
        scenarios_data = json.load(f)
    common_params = scenarios_data['common_params']
    scenarios = scenarios_data['scenarios']
    
    # Un plano de fase por escenario (incl. control Hill: hillY en scenarios.json suele tener USE_ADAPTIVE_CONTROL=Y).
    # El modelo 2D reducido (F1,F2) no incorpora el término Hill en la reducción; las nullclinas son las de
    # Strong/Weak sin u en F_c,F_s. Los (c*,s*,i*) del CSV marcan la proyección del equilibrio 3D (extract/Newton).
    figures_to_generate = []
    for scenario in scenarios:
        allee_label = 'Weak' if str(scenario.get('ALLEE_TYPE', 'WEAK')).upper() == 'WEAK' else 'Strong'
        mu_disp = scenario.get('mu', '?')
        sn = scenario.get('name', '')
        title = f'{allee_label} Allee, μ={mu_disp} — {sn}'
        figures_to_generate.append((scenario, title))

    if not figures_to_generate:
        print("  [!] No hay escenarios en scenarios.json; no se generan phase planes.")

    for scenario, title in figures_to_generate:
        # Usar el nombre completo del escenario para el nombre del archivo
        scenario_name = scenario.get('name', 'unknown')
        filename = f'steady_{scenario_name}.png'
        
        print(f"  Generando {filename}...")
        
        # Crear ModelParameters combinando common_params y scenario
        combined_params = {**common_params, **scenario}
        
        # Funciones auxiliares para conversión
        def get_float(key, default=0.0):
            val = combined_params.get(key)
            if val is None:
                return default
            return float(val) if isinstance(val, (int, float, str)) else default
        
        def get_bool(key, default=False):
            val = combined_params.get(key)
            if val is None:
                return default
            if isinstance(val, str):
                return val.upper() == 'Y'
            return bool(val)
        
        u_adapt = get_bool('USE_ADAPTIVE_CONTROL')
        hill_raw = combined_params.get('HILL_CONTROL')
        if hill_raw is None or (isinstance(hill_raw, str) and str(hill_raw).strip() == ''):
            control_uses_hill = u_adapt
        else:
            control_uses_hill = get_bool('HILL_CONTROL')

        params = ModelParameters(
            rc=get_float('rc'),
            rs=get_float('rs'),
            rd=get_float('rd'),
            alpha=get_float('alpha'),
            delta=get_float('delta'),
            beta=get_float('beta'),
            a=get_float('alle', get_float('a', 0.1)),
            gamma=get_float('gamma'),
            eta=get_float('eta'),
            mu=get_float('mu'),
            allee_type=combined_params.get('ALLEE_TYPE', 'WEAK').upper(),
            use_adaptive_control=u_adapt,
            control_uses_hill=control_uses_hill and u_adapt,
            ku=get_float('KU', 0.2),
            eps_u=get_float('EPS_U', 1e-3),
            u_max=get_float('U_MAX') if combined_params.get('U_MAX') else None,
        )
        
        # Construir ecuaciones según tipo de Allee
        if params.allee_type == 'STRONG':
            F1, F2 = build_reduced_model_2d_sympy_strong(params)
        else:
            from model_equations import build_reduced_model_2d_sympy
            F1, F2 = build_reduced_model_2d_sympy(params)
        
        # Convertir a funciones numéricas
        f1 = sp.lambdify((c_sym, s_sym), F1, modules='numpy')
        f2 = sp.lambdify((c_sym, s_sym), F2, modules='numpy')
        
        # Puntos de equilibrio marcados en el plano (c,s)
        steady_states_points = None
        if mark_simplex_corners:
            steady_states_points = list(simplex_corner_points_3d())
            print("    Modo simplex: ejes c,s en [-0.5, 2]; solo vertices (1,0), (0,1), (0,0)")
        elif steady_states_df is not None:
            scenario_name = scenario.get('name', '')
            
            # Primero intentar búsqueda exacta por nombre completo
            exact_match = steady_states_df[steady_states_df['scenario'] == scenario_name]
            
            if len(exact_match) == 0:
                # Si no hay coincidencia exacta, buscar por patrón más específico
                # Construir patrón: allee_type + mu (ej: "weak_mu0" o "strong_mu1")
                allee_prefix = params.allee_type.lower()
                mu_suffix = f"mu{int(params.mu)}"
                pattern = f"{allee_prefix}_{mu_suffix}"
                
                # Buscar estados estacionarios que coincidan con el patrón y mu
                matching_rows = steady_states_df[
                    (steady_states_df['scenario'].str.contains(pattern, case=False)) &
                    (steady_states_df['mu'] == float(params.mu)) &
                    (steady_states_df['allee_type'].str.upper() == params.allee_type.upper()) &
                    (~steady_states_df['scenario'].str.contains('uSi', case=False))  # Excluir control adaptativo
                ]
            else:
                matching_rows = exact_match
            
            if len(matching_rows) > 0:
                # Extraer puntos (c, s, i) del modelo completo 3D
                # El CSV tiene c_star, s_star, i_star del modelo 3D
                # Para el modelo 2D reducido, usamos c_star y s_star para la posición,
                # pero también guardamos i_star para mostrarlo en la etiqueta
                steady_states_points = []
                for _, row in matching_rows.iterrows():
                    if np.isfinite(row['c_star']) and np.isfinite(row['s_star']):
                        i_star = row.get('i_star', np.nan)
                        if np.isfinite(i_star):
                            steady_states_points.append((row['c_star'], row['s_star'], i_star))
                        else:
                            steady_states_points.append((row['c_star'], row['s_star'], None))
                
                if steady_states_points:
                    print(f"    Encontrados {len(steady_states_points)} punto(s) de equilibrio para {scenario_name}")
                else:
                    print(f"    [!] No se encontraron puntos de equilibrio válidos para {scenario_name}")
            else:
                print(f"    [!] No se encontraron estados estacionarios en CSV para {scenario_name}")
        
        # Generar gráfico (una subcarpeta por escenario bajo output_dir)
        scenario_dir = output_dir / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        output_path = scenario_dir / filename
        default_c_range = (0, 1.5)
        default_s_range = (0, 1.2)
        if mark_simplex_corners:
            c_range, s_range = SIMPLEX_PHASE_PLANE_C_RANGE, SIMPLEX_PHASE_PLANE_S_RANGE
        elif steady_states_points:
            c_range, s_range = phase_plane_ranges_from_equilibria(
                steady_states_points, default_c_range, default_s_range
            )
        else:
            c_range, s_range = default_c_range, default_s_range

        plot_nullclines_2d(
            f1,
            f2,
            c_range=c_range,
            s_range=s_range,
            mu_val=float(params.mu),
            allee_type=params.allee_type,
            title=title,
            save_path=output_path,
            steady_states_points=steady_states_points,
            simplex_corner_view=mark_simplex_corners,
        )
    
    print("OK: Phase planes generados")


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description="Planos de fase 2D desde scenarios.json")
    ap.add_argument(
        "--mark-simplex-corners",
        action="store_true",
        help="Ejes c,s en [-0.5,2]; streamlines + vertices (1,0),(0,1),(0,0); sin curvas F1=0/F2=0 (equiv. env)",
    )
    args_cli = ap.parse_args()

    # Figuras en RESULTS_DIR/<escenario>/ (mismo criterio que generate_figures_from_scenarios.py)
    scenarios_file = _ALLEE_ROOT / 'scenarios.json'
    try:
        from utils_paths import ensure_cloud_results_dir_ready, STEADY_STATES_EXTRACT_SUBDIR
        output_dir = ensure_cloud_results_dir_ready()
    except ImportError:
        STEADY_STATES_EXTRACT_SUBDIR = "steady_states_extract"
        print("Error: falta utils_paths en la raíz de Allee.")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    root_csv = output_dir / "steady_states_scenarios.csv"
    legacy_csv = output_dir / STEADY_STATES_EXTRACT_SUBDIR / "steady_states_scenarios.csv"
    local_csv = _ALLEE_ROOT / "steady_states_scenarios.csv"
    if root_csv.exists():
        steady_states_csv = root_csv
    elif legacy_csv.exists():
        steady_states_csv = legacy_csv
    elif local_csv.exists():
        steady_states_csv = local_csv
    else:
        print(
            "Error: no se encuentra steady_states_scenarios.csv en la raíz de resultados, "
            "en steady_states_extract/ ni en Allee/. Ejecuta extract (Drive montado o RESULTS_DIR)."
        )
        sys.exit(1)

    if not scenarios_file.exists():
        print(f"Error: No se encuentra {scenarios_file}")
        sys.exit(1)

    generate_phase_plane_figures(
        scenarios_file,
        output_dir,
        steady_states_csv,
        mark_simplex_corners=True if args_cli.mark_simplex_corners else None,
    )

