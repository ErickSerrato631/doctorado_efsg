"""
Script para analizar resultados termodinámicos y generar comparaciones
para integrar en el paper. Solo Allee fuerte (4 escenarios, sin sobre umbral).
Genera y actualiza imágenes en Paper copy/figures.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ruta base de resultados
BASE_DIR = Path.home() / "googledrive" / "Doctorado Erick Serrato" / "Resultados Paper"

# Directorio de salida para figuras (Paper copy)
SCRIPT_DIR = Path(__file__).parent
OUTPUT_FIGURES_DIR = SCRIPT_DIR.parent.parent / "Paper copy" / "figures"

# Solo escenarios Allee fuerte (4 escenarios, sin sobre umbral)
SCENARIOS = [
    "strong_mu0_uNo_bajo_umbral",
    "strong_mu0_uSi_bajo_umbral",
    "strong_mu1_uNo_bajo_umbral",
    "strong_mu1_uSi_bajo_umbral",
]

# Estilos para figuras (solo Strong, sin sobre umbral)
SCENARIO_STYLES = {
    "strong_mu0_uNo_bajo_umbral": {"label": r"Strong, $\mu$=0, No control", "color": "blue", "linestyle": "-"},
    "strong_mu1_uNo_bajo_umbral": {"label": r"Strong, $\mu$=1, No control", "color": "red", "linestyle": "-"},
    "strong_mu0_uSi_bajo_umbral": {"label": r"Strong, $\mu$=0, Control", "color": "green", "linestyle": "-"},
    "strong_mu1_uSi_bajo_umbral": {"label": r"Strong, $\mu$=1, Control", "color": "purple", "linestyle": "-"},
}

def load_thermodynamic_summary(scenario_name):
    """Carga el resumen termodinámico de un escenario."""
    summary_file = BASE_DIR / scenario_name / "thermodynamics" / "thermodynamic_summary.json"
    if not summary_file.exists():
        return None
    with open(summary_file, 'r') as f:
        return json.load(f)

def load_thermodynamic_time_series(scenario_name, property_name):
    """Carga series temporales de propiedades termodinámicas."""
    if property_name == "F":
        filename = "free_energy_F_t.txt"
    elif property_name == "sigma":
        filename = "entropy_production_sigma_t.txt"
    elif property_name == "mu":
        filename = "chemical_potentials_mu_t.txt"
    else:
        return None
    
    filepath = BASE_DIR / scenario_name / "thermodynamics" / filename
    if not filepath.exists():
        return None
    
    data = np.loadtxt(filepath, comments='#')
    if property_name == "F":
        return {
            'time': data[:, 0],
            'F_total': data[:, 1],
            'F_local': data[:, 2],
            'F_gradient': data[:, 3],
            'F_coupling': data[:, 4]
        }
    elif property_name == "sigma":
        return {
            'time': data[:, 0],
            'sigma': data[:, 1]
        }
    elif property_name == "mu":
        return {
            'time': data[:, 0],
            'mu_c': data[:, 1],
            'mu_s': data[:, 2],
            'mu_i': data[:, 3]
        }
    return None

def analyze_all_scenarios():
    """Analiza todos los escenarios y genera comparaciones."""
    results = {}
    
    print("Cargando resultados termodinámicos...")
    for scenario in SCENARIOS:
        summary = load_thermodynamic_summary(scenario)
        if summary:
            results[scenario] = {
                'summary': summary,
                'F_data': load_thermodynamic_time_series(scenario, "F"),
                'sigma_data': load_thermodynamic_time_series(scenario, "sigma"),
                'mu_data': load_thermodynamic_time_series(scenario, "mu")
            }
            print(f"  [OK] {scenario}")
        else:
            print(f"  [--] {scenario}: No encontrado")
    
    # Análisis comparativo
    print("\n=== Análisis Comparativo ===")
    
    # Comparar F_final entre escenarios
    print("\n1. Energía Libre Final (F_final):")
    for scenario, data in results.items():
        F_final = data['summary']['F_final']['F_total']
        print(f"   {scenario:35s}: F_final = {F_final:.6e}")
    
    # Comparar sigma_final
    print("\n2. Produccion de Entropia Final (sigma_final):")
    for scenario, data in results.items():
        sigma_final = data['summary']['sigma_final']
        print(f"   {scenario:35s}: sigma_final = {sigma_final:.6e}")
    
    # Comparar potenciales químicos finales
    print("\n3. Potenciales Quimicos Finales:")
    for scenario, data in results.items():
        mu = data['summary']['mu_final']
        print(f"   {scenario:35s}: mu_c={mu['mu_c_avg']:8.4f}, mu_s={mu['mu_s_avg']:8.4f}, mu_i={mu['mu_i_avg']:8.4f}")
    
    # Comparaciones específicas (solo Allee fuerte)
    print("\n=== Comparaciones Específicas (Allee fuerte) ===")
    
    # μ=0 vs μ=1 (strong Allee, sin control)
    strong_mu0 = results.get('strong_mu0_uNo_bajo_umbral')
    strong_mu1 = results.get('strong_mu1_uNo_bajo_umbral')
    if strong_mu0 and strong_mu1:
        print("\n4. Efecto de mu (Allee fuerte, sin control):")
        print(f"   mu=0: F_final = {strong_mu0['summary']['F_final']['F_total']:.6e}, sigma_final = {strong_mu0['summary']['sigma_final']:.6e}")
        print(f"   mu=1: F_final = {strong_mu1['summary']['F_final']['F_total']:.6e}, sigma_final = {strong_mu1['summary']['sigma_final']:.6e}")
        delta_F = strong_mu1['summary']['F_final']['F_total'] - strong_mu0['summary']['F_final']['F_total']
        print(f"   ΔF = {delta_F:.6e}")
    
    # Con vs sin control adaptativo (strong, μ=0)
    strong_no_u = results.get('strong_mu0_uNo_bajo_umbral')
    strong_with_u = results.get('strong_mu0_uSi_bajo_umbral')
    if strong_no_u and strong_with_u:
        print("\n5. Efecto del control adaptativo (Allee fuerte, mu=0):")
        print(f"   Sin control: F_final = {strong_no_u['summary']['F_final']['F_total']:.6e}, sigma_final = {strong_no_u['summary']['sigma_final']:.6e}")
        print(f"   Con control: F_final = {strong_with_u['summary']['F_final']['F_total']:.6e}, sigma_final = {strong_with_u['summary']['sigma_final']:.6e}")
        delta_F = strong_with_u['summary']['F_final']['F_total'] - strong_no_u['summary']['F_final']['F_total']
        print(f"   ΔF = {delta_F:.6e}")
    
    return results


def plot_free_energy_comparison(results):
    """Genera figura comparativa de F(t) entre escenarios (solo Allee fuerte)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for scenario, style in SCENARIO_STYLES.items():
        data = results.get(scenario, {}).get('F_data')
        if data is not None:
            ax.semilogy(data['time'], data['F_total'],
                       color=style['color'], linestyle=style['linestyle'],
                       label=style['label'], linewidth=1.5)
    ax.set_xlabel('Time $t$', fontsize=12)
    ax.set_ylabel('Free Energy $F(t)$', fontsize=12)
    ax.set_title('Free Energy Evolution: Allee Fuerte (4 escenarios)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2.0)
    plt.tight_layout()
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_FIGURES_DIR / "thermodynamic_F_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {out.name}")


def plot_entropy_production_comparison(results):
    """Genera figura comparativa de σ(t) entre escenarios (solo Allee fuerte)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for scenario, style in SCENARIO_STYLES.items():
        data = results.get(scenario, {}).get('sigma_data')
        if data is not None:
            ax.semilogy(data['time'], data['sigma'],
                       color=style['color'], linestyle=style['linestyle'],
                       label=style['label'], linewidth=1.5)
    ax.set_xlabel('Time $t$', fontsize=12)
    ax.set_ylabel('Entropy Production $\\sigma(t)$', fontsize=12)
    ax.set_title('Entropy Production: Allee Fuerte (4 escenarios)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2.0)
    plt.tight_layout()
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_FIGURES_DIR / "thermodynamic_sigma_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {out.name}")


def plot_chemical_potentials_comparison(results):
    """Genera figura comparativa de potenciales químicos (solo Allee fuerte); 2+1 filas."""
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.32, wspace=0.28)
    ax_mu_i = fig.add_subplot(gs[1, :])
    ax_mu_c = fig.add_subplot(gs[0, 0], sharex=ax_mu_i)
    ax_mu_s = fig.add_subplot(gs[0, 1], sharex=ax_mu_i)

    for scenario, style in SCENARIO_STYLES.items():
        data = results.get(scenario, {}).get('mu_data')
        if data is not None:
            ax_mu_c.plot(data['time'], data['mu_c'], color=style['color'],
                        linestyle=style['linestyle'], label=style['label'], linewidth=1.5)
            ax_mu_s.plot(data['time'], data['mu_s'], color=style['color'],
                        linestyle=style['linestyle'], label=style['label'], linewidth=1.5)
            ax_mu_i.plot(data['time'], data['mu_i'], color=style['color'],
                        linestyle=style['linestyle'], label=style['label'], linewidth=1.5)
    ax_mu_c.set_ylabel('$\\mu_c$ (Cancer)', fontsize=12)
    fig.suptitle('Chemical Potentials: Allee Fuerte (4 escenarios)', fontsize=14, y=0.98)
    ax_mu_c.legend(loc='best', fontsize=9)
    ax_mu_c.grid(True, alpha=0.3)
    ax_mu_c.axhline(y=0, color='k', linestyle=':', linewidth=0.5)
    ax_mu_c.set_xlim(0, 2.0)
    ax_mu_c.tick_params(axis='x', labelbottom=False)
    ax_mu_s.set_ylabel('$\\mu_s$ (Healthy)', fontsize=12)
    ax_mu_s.legend(loc='best', fontsize=9)
    ax_mu_s.grid(True, alpha=0.3)
    ax_mu_s.axhline(y=0, color='k', linestyle=':', linewidth=0.5)
    ax_mu_s.set_xlim(0, 2.0)
    ax_mu_s.tick_params(axis='x', labelbottom=False)
    ax_mu_i.set_xlabel('Time $t$', fontsize=12)
    ax_mu_i.set_ylabel('$\\mu_i$ (Immune)', fontsize=12)
    ax_mu_i.legend(loc='best', fontsize=9)
    ax_mu_i.grid(True, alpha=0.3)
    ax_mu_i.axhline(y=0, color='k', linestyle=':', linewidth=0.5)
    ax_mu_i.set_xlim(0, 2.0)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_FIGURES_DIR / "thermodynamic_mu_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {out.name}")


def plot_final_values_grid(results):
    """Genera grid comparativo de valores finales (solo Allee fuerte)."""
    F_final, sigma_final, mu_c_final, labels = [], [], [], []
    for scenario in SCENARIOS:
        data = results.get(scenario)
        if data and data.get('summary'):
            s = data['summary']
            F_final.append(s['F_final']['F_total'])
            sigma_final.append(s['sigma_final'])
            mu_c_final.append(s['mu_final']['mu_c_avg'])
            labels.append(scenario.replace('_', ' '))
    if not F_final:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    n = len(labels)
    axes[0].semilogy(range(n), F_final, 'o', markersize=8)
    axes[0].set_ylabel('$F_{\\text{final}}$', fontsize=12)
    axes[0].set_title('Final Free Energy (Allee fuerte)', fontsize=12)
    axes[0].set_xticks(range(n))
    axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].semilogy(range(n), sigma_final, 's', markersize=8, color='orange')
    axes[1].set_ylabel('$\\sigma_{\\text{final}}$', fontsize=12)
    axes[1].set_title('Final Entropy Production', fontsize=12)
    axes[1].set_xticks(range(n))
    axes[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(range(n), mu_c_final, '^', markersize=8, color='green')
    axes[2].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    axes[2].set_ylabel('$\\mu_{c,\\text{final}}$', fontsize=12)
    axes[2].set_title('Final Chemical Potential (Cancer)', fontsize=12)
    axes[2].set_xticks(range(n))
    axes[2].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_FIGURES_DIR / "thermodynamic_final_values_grid.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {out.name}")


def update_figures(results):
    """Genera y actualiza todas las figuras termodinámicas en Paper copy/figures."""
    print("\n=== Actualizando imágenes ===")
    print(f"  Directorio: {OUTPUT_FIGURES_DIR}")
    plot_free_energy_comparison(results)
    plot_entropy_production_comparison(results)
    plot_chemical_potentials_comparison(results)
    plot_final_values_grid(results)
    print("  [OK] Imagenes actualizadas")


if __name__ == '__main__':
    results = analyze_all_scenarios()
    if results:
        update_figures(results)
    print("\n[OK] Analisis completado")

