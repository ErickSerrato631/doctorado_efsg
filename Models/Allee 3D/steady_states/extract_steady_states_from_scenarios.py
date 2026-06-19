"""
Script para extraer estados estacionarios de los escenarios en scenarios.json
y generar tablas LaTeX formateadas para el paper y la tesis.

**Salida solo fuera de Allee/:** carpeta «Resultados Paper» resuelta por ``utils_paths``:
``RESULTS_DIR`` si está definida, o ``~/googledrive/Doctorado Erick Serrato/Resultados Paper``
si Drive está montado con rclone/FUSE (``mount_google_drive.sh``).
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_ALLEE_ROOT = Path(__file__).resolve().parent.parent
if str(_ALLEE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALLEE_ROOT))

try:
    from utils_paths import ensure_cloud_results_dir_ready
except ImportError:
    def ensure_cloud_results_dir_ready():
        raise RuntimeError("Instala utils_paths junto a este script (raíz Allee).")

# Importar desde el paquete steady_states (steady_states/steady_states.py)
try:
    from steady_states import (
        newton_root_3d,
        filter_physical_3d,
        scan_grid_3d,
        c_3d,
        s_3d,
        i_3d,
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
        build_equations_3d as ss_build_equations_3d,
    )
    import sympy as sp
except ImportError as e:
    print(f"Error importando steady_states: {e}")
    print("Asegúrate de tener Models/Allee en PYTHONPATH o ejecutar desde Allee.")
    sys.exit(1)


def scenario_uses_hill_control(scenario: dict) -> bool:
    """
    True si el escenario usa la ley de control Hill (Opción A), coherente con cancer_dynamics / steady_states.py.
    hillN en el nombre implica sin término Hill en R_i (solo comparación en el nombre del escenario).
    """
    if scenario.get("USE_ADAPTIVE_CONTROL", "N") != "Y":
        return False
    nl = scenario.get("name", "").lower()
    if "hilln" in nl:
        return False
    if "hilly" in nl:
        return True
    return scenario.get("HILL_KC") is not None or scenario.get("HILL_KI") is not None


def build_equations_3d_min_adaptive(allee_type: str = "WEAK"):
    """
    Modelo 3D con control u = min(k_u c/(i+eps), u_max) (legado; ya no coincide con Hill del PDE).
    Se usa solo si USE_ADAPTIVE_CONTROL=Y pero no es escenario Hill.
    """
    c, s, i = c_3d, s_3d, i_3d

    if allee_type == "STRONG":
        alle_term = rc * c * (1 - c) * ((c - a) / (1 - a))
    else:
        alle_term = rc * c * (c - a) * (1 - c)

    u_raw = ku * c / (i + eps_u)
    u_ctrl = sp.Min(u_raw, umax) if umax is not None else u_raw

    F_c = alle_term - c * (alpha * s**2 + beta * i**2) - mu * c * (gamma * s**2 + eta * i**2)
    F_s = rs * s * (1 - s) - gamma * c**2 * s + delta * i**2 * s - (mu * alpha * c**2 * s) / 2
    F_i = rd * i * (1 - i) + delta * i * s**2 - eta * c**2 * i - (mu * beta * c**2 * i) / 2 + u_ctrl

    return F_c, F_s, F_i


def load_scenarios(scenarios_file: Path):
    """Carga los escenarios desde scenarios.json"""
    with open(scenarios_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['common_params'], data['scenarios']


def calculate_steady_state_for_scenario(common_params: dict, scenario: dict):
    """
    Calcula el estado estacionario para un escenario específico con Newton–Raphson 3D
    (semillas en rejilla + extras, incl. cerca de c≈0, s≈1, i≈1).
    
    Returns:
        dict con los resultados del estado estacionario o None si no se encuentra
    """
    # Extraer parámetros del escenario
    mu_val = float(scenario.get('mu', common_params.get('mu', '1')))
    allee_type = scenario.get('ALLEE_TYPE', 'WEAK')
    use_control = scenario.get('USE_ADAPTIVE_CONTROL', 'N') == 'Y'
    use_hill = scenario_uses_hill_control(scenario)

    # Parámetros comunes
    a_val = float(common_params.get('a', '0.1'))

    # Parámetros específicos del escenario
    rc_val = float(scenario.get('rc', common_params.get('rc', '6.5')))
    beta_val = float(scenario.get('beta', common_params.get('beta', '3')))
    delta_val = float(scenario.get('delta', common_params.get('delta', '9')))
    eta_val = float(scenario.get('eta', common_params.get('eta', '1')))
    rd_val = float(scenario.get('rd', common_params.get('rd', '14')))

    ku_val = float(scenario.get('KU', common_params.get('KU', '0.2'))) if use_control else 0.0
    eps_val = float(scenario.get('EPS_U', common_params.get('EPS_U', '0.02'))) if use_control else 1e-3
    umax_val = float(scenario.get('U_MAX', common_params.get('U_MAX', '1.0'))) if use_control else None

    # Crear malla de semillas amplia
    c_seeds = np.linspace(0.0, 1.2, 8)
    s_seeds = np.linspace(0.0, 0.3, 6)
    i_seeds = np.linspace(0.0, 1.2, 8)
    
    seeds = [
        (c, s, i)
        for c in c_seeds
        for s in s_seeds
        for i in i_seeds
    ]
    
    # Agregar semillas específicas basadas en valores conocidos de las tablas
    scenario_name = scenario.get('name', '')
    if 'strong_mu0_uNo_bajo_umbral' in scenario_name:
        seeds.extend([(0.0, 0.255, 1.042), (0.0, 0.25, 1.04), (0.0, 0.26, 1.04), (1e-10, 0.255, 1.042)])
    elif 'strong_mu1_uNo_sobre_umbral' in scenario_name:
        seeds.extend([(1.008, 0.0, 0.068), (1.0, 0.0, 0.07), (1.01, 0.0, 0.06), (1.008, 1e-10, 0.068)])
    elif 'strong_mu1_uNo_bajo_umbral' in scenario_name:
        seeds.extend([(0.0, 0.0, 1.0), (0.0, 0.0, 1.04), (1e-10, 1e-10, 1.0)])
    elif 'weak_mu0_uNo_bajo_umbral' in scenario_name:
        seeds.extend([(0.0, 0.0, 1.0), (0.0, 0.0, 1.04), (1e-10, 1e-10, 1.0)])
    elif 'weak_mu0_uNo_sobre_umbral' in scenario_name:
        seeds.extend([(1.0, 0.0, 0.05), (1.0, 0.0, 0.06), (1.0, 1e-10, 0.05)])
    elif 'weak_mu1_uNo_bajo_umbral' in scenario_name:
        seeds.extend([(0.0, 0.0, 1.0), (0.0, 0.0, 1.04), (1e-10, 1e-10, 1.0)])
    elif 'weak_mu1_uNo_sobre_umbral' in scenario_name:
        seeds.extend([(1.01, 0.0, 0.07), (1.008, 0.0, 0.068), (1.01, 1e-10, 0.07)])

    # Régimen c≈0, s≈1, i≈1 (sanos altos, tumor ausente, inmunidad alta). La malla base
    # solo recorre s∈[0,0.3]; estas semillas permiten que Newton alcance equilibrios en esa esquina.
    # Se usan ligeramente <1 en s,i para evitar singularidades en s(1-s), i(1-i) en el borde.
    seeds.extend([
        (0.0, 1.0, 1.0),
        (1e-10, 1.0, 1.0),
        (0.0, 0.99, 1.0),
        (0.0, 1.0, 0.99),
        (1e-8, 0.995, 0.995),
        (0.0, 0.98, 1.02),
        (1e-6, 0.99, 0.99),
    ])

    # Usar scan_grid_3d para encontrar estados estacionarios
    # Nota: scan_grid_3d solo maneja Weak Allee en build_equations_3d, 
    # pero podemos modificar build_numeric_3d para aceptar allee_type
    # Por ahora, usamos el método directo pero con mejor manejo de Strong Allee
    
    # Construir ecuaciones: Hill (steady_states.py) alineado con PDE; Min adaptativo legado; o sin control
    params_dict = {
        a: a_val,
        rc: rc_val,
        rs: float(common_params.get('rs', '13.12')),
        rd: rd_val,
        alpha: float(common_params.get('alpha', '10.22')),
        delta: delta_val,
        beta: beta_val,
        gamma: float(common_params.get('gamma', '0.74')),
        eta: eta_val,
        mu: mu_val,
    }

    if use_hill:
        params_dict[kc_h] = float(scenario.get('HILL_KC', common_params.get('HILL_KC', '0.05')))
        params_dict[nc_h] = float(scenario.get('HILL_NC', common_params.get('HILL_NC', '2')))
        params_dict[ki_h] = float(scenario.get('HILL_KI', common_params.get('HILL_KI', '0.2')))
        params_dict[ni_h] = float(scenario.get('HILL_NI', common_params.get('HILL_NI', '2')))
        params_dict[umax] = float(scenario.get('U_MAX', common_params.get('U_MAX', '0.5')))
        Fc, Fs, Fi = ss_build_equations_3d(allee_type, True)
    elif use_control:
        params_dict[ku] = ku_val
        params_dict[eps_u] = eps_val
        params_dict[umax] = umax_val if umax_val is not None else sp.oo
        Fc, Fs, Fi = build_equations_3d_min_adaptive(allee_type)
    else:
        params_dict[ku] = 0.0
        params_dict[eps_u] = 1e-3
        params_dict[umax] = sp.oo
        Fc, Fs, Fi = ss_build_equations_3d(allee_type, False)

    pcur_sub = {k: v for k, v in params_dict.items() if v is not None}
    Fc_eval = Fc.subs(pcur_sub)
    Fs_eval = Fs.subs(pcur_sub)
    Fi_eval = Fi.subs(pcur_sub)
    
    # Crear función vectorial y Jacobiano
    F_vec = sp.Matrix([Fc_eval, Fs_eval, Fi_eval])
    Jsym = F_vec.jacobian([c_3d, s_3d, i_3d])
    f = sp.lambdify((c_3d, s_3d, i_3d), F_vec, modules='numpy')
    
    # Intentar encontrar raíz con cada semilla
    # Buscar múltiples equilibrios y seleccionar el correcto según el escenario
    solutions_found = []
    
    for seed in seeds:
        try:
            result = newton_root_3d(f, Jsym, seed)
            if result is not None:
                cx, sy, iz = result
                if np.isfinite(cx) and np.isfinite(sy) and np.isfinite(iz):
                    # Evitar duplicados (solutions_found es lista de dicts con c,s,i)
                    if any(
                        np.linalg.norm([cx - sol['c'], sy - sol['s'], iz - sol['i']]) < 1e-3
                        for sol in solutions_found
                    ):
                        continue
                    
                    # Calcular autovalores
                    Jnum = np.array(Jsym.subs({c_3d: cx, s_3d: sy, i_3d: iz}), dtype=float)
                    eigs = np.linalg.eigvals(Jnum)
                    max_real = max(ev.real for ev in eigs)
                    
                    # Solo considerar soluciones físicas
                    if cx >= -1e-6 and sy >= -1e-6 and iz > 0:
                        solutions_found.append({
                            'c': float(max(cx, 0.0)),
                            's': float(max(sy, 0.0)),
                            'i': float(iz),
                            'max_real': float(max_real),
                            'eigs': eigs
                        })
        except Exception as e:
            continue
    
    # Seleccionar el equilibrio correcto según el escenario
    scenario_name = scenario.get('name', '')
    best_result = None
    
    if len(solutions_found) == 0:
        return None
    
    # Criterios de selección según valores esperados de las tablas
    if 'strong_mu0_uNo_bajo_umbral' in scenario_name:
        # Esperado: c*≈0, s*=0.255, i*=1.042
        best = min(solutions_found, key=lambda x: abs(x['s'] - 0.255) + abs(x['i'] - 1.042))
    elif 'strong_mu1_uNo_sobre_umbral' in scenario_name:
        # Esperado: c*=1.008, s*≈0, i*=0.068
        best = min(solutions_found, key=lambda x: abs(x['c'] - 1.008) + abs(x['i'] - 0.068))
    elif 'weak_mu0_uNo_sobre_umbral' in scenario_name:
        # Esperado: c*≈1.0, s*≈0, i*≈0.05
        best = min(solutions_found, key=lambda x: abs(x['c'] - 1.0) + abs(x['i'] - 0.05))
    elif 'weak_mu1_uNo_sobre_umbral' in scenario_name:
        # Esperado: c*≈1.01, s*≈0, i*≈0.07
        best = min(solutions_found, key=lambda x: abs(x['c'] - 1.01) + abs(x['i'] - 0.07))
    else:
        # Para otros casos, preferir equilibrios no triviales (no el trivial 0,0,1)
        # Preferir aquellos con s* > 0.1 o c* > 0.5
        non_trivial = [s for s in solutions_found if s['s'] > 0.1 or s['c'] > 0.5]
        if non_trivial:
            best = non_trivial[0]
        else:
            # Si solo hay triviales, tomar el primero
            best = solutions_found[0]
    
    # Construir resultado final
    best_result = {
        'scenario': scenario['name'],
        'allee_type': allee_type,
        'mu': mu_val,
        'use_control': use_control,
        'rc': rc_val,
        'beta': beta_val,
        'delta': delta_val,
        'eta': eta_val,
        'rd': rd_val,
        'c_star': best['c'],
        's_star': best['s'],
        'i_star': best['i'],
        'eig1_real': float(best['eigs'][0].real),
        'eig1_imag': float(best['eigs'][0].imag),
        'eig2_real': float(best['eigs'][1].real),
        'eig2_imag': float(best['eigs'][1].imag),
        'eig3_real': float(best['eigs'][2].real),
        'eig3_imag': float(best['eigs'][2].imag),
        'max_real': best['max_real'],
        'unstable': bool(best['max_real'] > 0),
    }
    
    return best_result


def format_number_for_latex(value: float, precision: int = 4) -> str:
    """Formatea un número para LaTeX, usando notación científica si es muy pequeño"""
    if abs(value) < 1e-10:
        return "$\\approx 0$"
    elif abs(value) < 0.01:
        # Notación científica
        exp = int(np.floor(np.log10(abs(value))))
        mantissa = value / (10 ** exp)
        return f"${mantissa:.2f}\\times10^{{{exp}}}$"
    else:
        return f"${value:.{precision}f}$"


def generate_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """Genera código LaTeX para una tabla"""
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\scriptsize",
        "\\begin{tabular}{" + "c" * len(df.columns) + "}",
        "\\toprule",
        " & ".join(df.columns) + " \\\\",
        "\\midrule"
    ]
    
    for _, row in df.iterrows():
        row_str = " & ".join(str(val) for val in row.values) + " \\\\"
        lines.append(row_str)
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\end{table}"
    ])
    
    return "\n".join(lines)


def main():
    """Función principal"""
    scenarios_file = _ALLEE_ROOT / 'scenarios.json'
    
    if not scenarios_file.exists():
        print(f"Error: No se encontró {scenarios_file}")
        sys.exit(1)

    try:
        results_root = ensure_cloud_results_dir_ready()
    except RuntimeError as e:
        print(f"Error: {e}\nEste script no escribe en Allee/.")
        sys.exit(1)
    
    print("Cargando escenarios...")
    common_params, scenarios = load_scenarios(scenarios_file)
    
    print(f"Encontrados {len(scenarios)} escenarios")
    print("Calculando estados estacionarios...")
    
    results = []
    for i, scenario in enumerate(scenarios, 1):
        print(f"  [{i}/{len(scenarios)}] {scenario['name']}...", end=" ", flush=True)
        result = calculate_steady_state_for_scenario(common_params, scenario)
        if result:
            results.append(result)
            print("OK")
        else:
            print("FAILED (no encontrado)")
    
    if not results:
        print("Error: No se encontraron estados estacionarios")
        sys.exit(1)
    
    # Crear DataFrame
    df = pd.DataFrame(results)
    
    csv_agg = results_root / "steady_states_scenarios.csv"
    df.to_csv(csv_agg, index=False)
    print(f"\nCSV agregado (todos los escenarios): {csv_agg}")

    for scen in df["scenario"].unique():
        scen_dir = results_root / str(scen)
        scen_dir.mkdir(parents=True, exist_ok=True)
        sub_path = scen_dir / "steady_states_scenarios.csv"
        df[df["scenario"] == scen].to_csv(sub_path, index=False)
    print(
        f"CSV por escenario: {results_root}/<nombre_escenario>/steady_states_scenarios.csv "
        f"({df['scenario'].nunique()} carpetas)"
    )
    
    # Generar tablas LaTeX
    print("\nGenerando tablas LaTeX...")
    
    # Tabla 1: Weak Allee
    df_weak = df[df['allee_type'] == 'WEAK'].copy()
    if len(df_weak) > 0:
        df_weak_display = df_weak[['scenario', 'mu', 'use_control', 'rc', 'beta', 'delta', 'eta', 'rd', 
                                   'c_star', 's_star', 'i_star', 'max_real']].copy()
        df_weak_display['mu'] = df_weak_display['mu'].astype(int)
        df_weak_display['use_control'] = df_weak_display['use_control'].map({True: 'Sí', False: 'No'})
        
        # Formatear números pequeños
        for col in ['c_star', 's_star', 'i_star']:
            df_weak_display[col] = df_weak_display[col].apply(lambda x: format_number_for_latex(x))
        
        df_weak_display['max_real'] = df_weak_display['max_real'].apply(lambda x: f"${x:.2f}$")
        
        # Renombrar columnas
        df_weak_display.columns = ['Escenario', '$\\mu$', 'Control', '$r_c$', '$\\beta$', '$\\delta$', 
                                   '$\\eta$', '$r_d$', '$c^*$', '$s^*$', '$i^*$', 'Re $\\lambda_{\\max}$']
        
        latex_weak = generate_latex_table(df_weak_display, 
                                          "Estados estacionarios - Weak Allee", 
                                          "tab:steady_weak")
        
        tw = results_root / "table_weak_allee.tex"
        tw.write_text(latex_weak, encoding="utf-8")
        print(f"  Tabla Weak Allee: {tw}")
    
    # Tabla 2: Strong Allee
    df_strong = df[df['allee_type'] == 'STRONG'].copy()
    if len(df_strong) > 0:
        df_strong_display = df_strong[['scenario', 'mu', 'use_control', 'rc', 'beta', 'delta', 'eta', 'rd',
                                       'c_star', 's_star', 'i_star', 'max_real']].copy()
        df_strong_display['mu'] = df_strong_display['mu'].astype(int)
        df_strong_display['use_control'] = df_strong_display['use_control'].map({True: 'Sí', False: 'No'})
        
        # Formatear números pequeños
        for col in ['c_star', 's_star', 'i_star']:
            df_strong_display[col] = df_strong_display[col].apply(lambda x: format_number_for_latex(x))
        
        df_strong_display['max_real'] = df_strong_display['max_real'].apply(lambda x: f"${x:.2f}$")
        
        # Renombrar columnas
        df_strong_display.columns = ['Escenario', '$\\mu$', 'Control', '$r_c$', '$\\beta$', '$\\delta$',
                                     '$\\eta$', '$r_d$', '$c^*$', '$s^*$', '$i^*$', 'Re $\\lambda_{\\max}$']
        
        latex_strong = generate_latex_table(df_strong_display,
                                           "Estados estacionarios - Strong Allee",
                                           "tab:steady_strong")
        
        ts = results_root / "table_strong_allee.tex"
        ts.write_text(latex_strong, encoding="utf-8")
        print(f"  Tabla Strong Allee: {ts}")
    
    # Tabla 3: Comparación control adaptativo (Strong Allee)
    df_strong_control = df_strong[df_strong['mu'] == 0].copy()
    if len(df_strong_control) >= 2:
        # Agrupar por escenario base (sin _uNo/_uSi)
        control_comparison = []
        for base_name in ['strong_mu0_bajo_umbral', 'strong_mu1_bajo_umbral']:
            no_control = df_strong_control[df_strong_control['scenario'].str.contains(base_name.replace('_bajo_umbral', '_uNo_bajo_umbral'))]
            with_control = df_strong_control[df_strong_control['scenario'].str.contains(base_name.replace('_bajo_umbral', '_uSi_bajo_umbral'))]
            
            if len(no_control) > 0 and len(with_control) > 0:
                no_row = no_control.iloc[0]
                with_row = with_control.iloc[0]
                control_comparison.append({
                    'Escenario': base_name,
                    '$\\mu$': int(no_row['mu']),
                    'Sin Control $(c^*, s^*, i^*)$': f"$({format_number_for_latex(no_row['c_star'])}, {format_number_for_latex(no_row['s_star'])}, {format_number_for_latex(no_row['i_star'])})$",
                    'Con Control $(c^*, s^*, i^*)$': f"$({format_number_for_latex(with_row['c_star'])}, {format_number_for_latex(with_row['s_star'])}, {format_number_for_latex(with_row['i_star'])})$",
                    '$\\Delta$ Re $\\lambda_{\\max}$': f"${with_row['max_real'] - no_row['max_real']:.2f}$"
                })
        
        if control_comparison:
            df_control = pd.DataFrame(control_comparison)
            latex_control = generate_latex_table(df_control,
                                                 "Comparación Control Adaptativo (Strong Allee)",
                                                 "tab:control_comparison")
            
            tc = results_root / "table_control_comparison.tex"
            tc.write_text(latex_control, encoding="utf-8")
            print(f"  Tabla Comparación Control: {tc}")
    
    print("\n¡Completado!")
    print(f"\nResumen:")
    print(f"  - Escenarios procesados: {len(scenarios)}")
    print(f"  - Estados estacionarios encontrados: {len(results)}")
    print(f"  - Weak Allee: {len(df_weak)}")
    print(f"  - Strong Allee: {len(df_strong)}")


if __name__ == '__main__':
    main()

