"""
Script para generar espectros lineales (FIG. 4 y FIG. 5)
calculando λ(k) = eig(J - D·k²) para diferentes números de onda k.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp

_ALLEE_ROOT = Path(__file__).resolve().parent.parent
if str(_ALLEE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALLEE_ROOT))

# Importar funciones necesarias
try:
    from steady_states import build_equations_3d as ss_build_equations_3d, kc_h, nc_h, ki_h, ni_h
    from steady_states.extract_steady_states_from_scenarios import (
        build_equations_3d_min_adaptive,
        scenario_uses_hill_control,
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
    )
except ImportError as e:
    print(f"Error importando módulos: {e}")
    print("Asegúrate de tener Models/Allee en PYTHONPATH o ejecutar con cwd Allee.")
    sys.exit(1)


def compute_linear_spectrum(
    c_star,
    s_star,
    i_star,
    params_dict,
    allee_type="WEAK",
    D_c=0.012,
    D_s=0.022,
    D_i=0.022,
    k_max=10,
    n_points=200,
    *,
    control_mode: str = "none",
):
    """
    Calcula λ(k) = eig(J_reac - D·k²) para un rango de números de onda k.

    control_mode: ``none`` (sin término en R_i), ``hill`` (alineado con steady_states.py / PDE),
    ``min`` (legado u = min(ku c/(i+eps), u_max)).
    """
    if control_mode == "hill":
        Fc, Fs, Fi = ss_build_equations_3d(allee_type, True)
    elif control_mode == "min":
        Fc, Fs, Fi = build_equations_3d_min_adaptive(allee_type)
    else:
        Fc, Fs, Fi = ss_build_equations_3d(allee_type, False)
    
    # Crear función vectorial y Jacobiano simbólico
    F_vec = sp.Matrix([Fc, Fs, Fi])
    J_sym = F_vec.jacobian([c_3d, s_3d, i_3d])
    
    # Evaluar Jacobiano de reacción en el estado estacionario
    J_reac = np.array(J_sym.subs({
        c_3d: c_star, 
        s_3d: s_star, 
        i_3d: i_star,
        **params_dict
    }), dtype=float)
    
    # Verificar que el Jacobiano es finito
    if not np.all(np.isfinite(J_reac)):
        print(f"  ⚠️  Jacobiano no finito en ({c_star:.4f}, {s_star:.4f}, {i_star:.4f})")
        return None, None
    
    # Matriz de difusión
    D = np.diag([D_c, D_s, D_i])
    
    # Calcular espectro para diferentes k
    k_values = np.linspace(0, k_max, n_points)
    eigenvalues = []
    
    for k in k_values:
        J_k = J_reac - D * (k**2)
        try:
            eigs = np.linalg.eigvals(J_k)
            eigenvalues.append(eigs)
        except:
            eigenvalues.append([np.nan, np.nan, np.nan])
    
    eigenvalues = np.array(eigenvalues)
    
    return k_values, eigenvalues


def plot_linear_spectrum(
    k_values,
    eigenvalues,
    scenario_name,
    allee_type,
    mu_val,
    use_control=False,
    umbral_type=None,
    save_path: Path = None,
    *,
    use_hill: bool = False,
):
    """
    Grafica el espectro lineal λ(k).
    
    Args:
        k_values: Array de números de onda
        eigenvalues: Array de autovalores (n_points, 3)
        scenario_name: Nombre del escenario
        allee_type: 'WEAK' o 'STRONG'
        mu_val: Valor de μ
        use_control: Si hay control adaptativo
        umbral_type: 'bajo' o 'sobre' umbral
        save_path: Ruta donde guardar la figura
    """
    if eigenvalues is None or k_values is None:
        print(f"  ⚠️  No se pudo calcular espectro para {scenario_name}")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Graficar parte real e imaginaria de cada autovalor
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    linestyles_real = ['-', '-', '-']
    linestyles_imag = ['--', '--', '--']
    
    for i in range(3):
        # Parte real
        ax.plot(k_values, eigenvalues[:, i].real, 
               color=colors[i], linestyle=linestyles_real[i],
               linewidth=2, label=f'Re(λ_{i+1})')
        
        # Parte imaginaria (solo si no es cero)
        if np.any(np.abs(eigenvalues[:, i].imag) > 1e-10):
            ax.plot(k_values, eigenvalues[:, i].imag, 
                   color=colors[i], linestyle=linestyles_imag[i],
                   linewidth=1, alpha=0.7, label=f'Im(λ_{i+1})')
    
    # Línea de referencia en y=0
    ax.axhline(y=0, color='k', linestyle=':', alpha=0.5, linewidth=1)
    
    # Configuración del título con información completa
    allee_str = 'Weak' if allee_type == 'WEAK' else 'Strong'
    if use_hill:
        control_str = ', con control Hill'
    elif use_control:
        control_str = ', con control adaptativo (min)'
    else:
        control_str = ''
    umbral_str = f', {umbral_type} umbral' if umbral_type else ''
    title = f'Espectro lineal - {allee_str} Allee, μ={mu_val}{control_str}{umbral_str}'
    
    ax.set_xlabel('k (número de onda)', fontsize=12)
    ax.set_ylabel('λ(k)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Guardado: {save_path}")
    plt.close()


def generate_linear_spectrum_figures(scenarios_file: Path, steady_states_csv: Path, 
                                     output_dir: Path):
    """
    Genera las figuras de espectros lineales para FIG. 4 y FIG. 5.
    
    Args:
        scenarios_file: Ruta al archivo scenarios.json
        steady_states_csv: Ruta al archivo CSV con estados estacionarios
        output_dir: Directorio base; cada figura se guarda en output_dir/<nombre_escenario>/
    """
    print("Generando espectros lineales...")
    
    # Cargar estados estacionarios
    if not steady_states_csv.exists():
        print(f"  Error: No se encuentra {steady_states_csv}")
        print("  Ejecuta primero extract_steady_states_from_scenarios.py (Drive montado o RESULTS_DIR).")
        return
    
    df = pd.read_csv(steady_states_csv)
    
    # Cargar parámetros comunes de scenarios.json
    import json
    with open(scenarios_file, 'r', encoding='utf-8') as f:
        scenarios_data = json.load(f)
    common_params = scenarios_data['common_params']
    
    # Parámetros de difusión
    D_c = float(common_params.get('D_c', '0.012'))
    D_s = float(common_params.get('D_s', '0.022'))
    D_i = float(common_params.get('D_i', '0.022'))
    
    # Cargar escenarios desde JSON para obtener información de control adaptativo
    scenarios_list = scenarios_data.get('scenarios', [])
    scenario_info = {s['name']: s for s in scenarios_list}
    
    # Generar imagen para cada escenario en el CSV
    generated_count = 0
    for _, row in df.iterrows():
        scenario_name = row['scenario']
        allee_type = row['allee_type']
        mu_val = row['mu']
        use_control = row['use_control']
        
        # Obtener información adicional del JSON si está disponible
        scenario_json = scenario_info.get(scenario_name, {})
        
        # Usar el nombre completo del escenario para el nombre del archivo
        filename = f'estabilidad_lineal_{scenario_name}.png'
        
        # Determinar tipo de umbral para el título (opcional)
        if 'bajo_umbral' in scenario_name:
            umbral_type = 'bajo'
        elif 'sobre_umbral' in scenario_name:
            umbral_type = 'sobre'
        else:
            umbral_type = None
        
        print(f"  Generando {filename} desde {scenario_name}...")
        
        # Construir diccionario de parámetros simbólicos
        params_dict = {
            a: float(common_params.get('a', '0.1')),
            rc: row['rc'],
            rs: float(common_params.get('rs', '13.12')),
            rd: row['rd'],
            alpha: float(common_params.get('alpha', '10.22')),
            delta: row['delta'],
            beta: row['beta'],
            gamma: float(common_params.get('gamma', '0.74')),
            eta: row['eta'],
            mu: row['mu'],
        }

        merged_for_hill = {
            **scenario_json,
            'name': scenario_name,
            'USE_ADAPTIVE_CONTROL': 'Y' if use_control else 'N',
        }
        use_hill = scenario_uses_hill_control(merged_for_hill)
        if use_hill:
            params_dict[kc_h] = float(
                scenario_json.get('HILL_KC', common_params.get('HILL_KC', '0.05'))
            )
            params_dict[nc_h] = float(
                scenario_json.get('HILL_NC', common_params.get('HILL_NC', '2'))
            )
            params_dict[ki_h] = float(
                scenario_json.get('HILL_KI', common_params.get('HILL_KI', '0.2'))
            )
            params_dict[ni_h] = float(
                scenario_json.get('HILL_NI', common_params.get('HILL_NI', '2'))
            )
            params_dict[umax] = float(
                scenario_json.get('U_MAX', common_params.get('U_MAX', '0.5'))
            )
            control_mode = 'hill'
        elif use_control:
            ku_val = float(scenario_json.get('KU', common_params.get('KU', '0.2')))
            eps_u_val = float(scenario_json.get('EPS_U', common_params.get('EPS_U', '0.02')))
            umax_val = float(scenario_json.get('U_MAX', common_params.get('U_MAX', '1.0')))
            params_dict[ku] = ku_val
            params_dict[eps_u] = eps_u_val
            params_dict[umax] = umax_val
            control_mode = 'min'
        else:
            params_dict[ku] = 0.0
            params_dict[eps_u] = 1e-3
            params_dict[umax] = sp.oo
            control_mode = 'none'

        # Calcular espectro
        k_vals, eigs = compute_linear_spectrum(
            row['c_star'],
            row['s_star'],
            row['i_star'],
            params_dict,
            allee_type=allee_type,
            D_c=D_c,
            D_s=D_s,
            D_i=D_i,
            k_max=10,
            n_points=200,
            control_mode=control_mode,
        )
        
        if k_vals is not None and eigs is not None:
            # Graficar (una subcarpeta por escenario bajo output_dir)
            scenario_dir = output_dir / scenario_name
            scenario_dir.mkdir(parents=True, exist_ok=True)
            output_path = scenario_dir / filename
            plot_linear_spectrum(
                k_vals,
                eigs,
                scenario_name,
                allee_type,
                mu_val,
                use_control=use_control,
                umbral_type=umbral_type,
                save_path=output_path,
                use_hill=use_hill,
            )
            generated_count += 1
        else:
            print(f"  ⚠️  No se pudo calcular espectro para {scenario_name}")
    
    print(f"  Generadas {generated_count} imágenes de estabilidad lineal")
    
    print("OK: Espectros lineales generados")


if __name__ == '__main__':
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

    generate_linear_spectrum_figures(scenarios_file, steady_states_csv, output_dir)

