"""
Script para generar figuras comparativas de resultados termodinámicos
para el paper.
"""

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

# Ruta base de resultados
BASE_DIR = Path.home() / "googledrive" / "Doctorado Erick Serrato" / "Resultados Paper"
# Repo root: .../Models/Allee/termodinámica → subir 4 niveles
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "Paper" / "figures"

# Lista de escenarios
SCENARIOS = {
    "weak_mu0_uNo_bajo_umbral": {"label": "Weak, μ=0, No control", "color": "blue", "linestyle": "-"},
    "weak_mu1_uNo_bajo_umbral": {"label": "Weak, μ=1, No control", "color": "red", "linestyle": "-"},
    "strong_mu0_uNo_bajo_umbral": {"label": "Strong, μ=0, No control", "color": "blue", "linestyle": "--"},
    "strong_mu1_uNo_bajo_umbral": {"label": "Strong, μ=1, No control", "color": "red", "linestyle": "--"},
    "weak_mu0_uSi_bajo_umbral": {"label": "Weak, μ=0, Control", "color": "green", "linestyle": "-"},
    "weak_mu1_uSi_bajo_umbral": {"label": "Weak, μ=1, Control", "color": "orange", "linestyle": "-"},
}

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

def plot_free_energy_comparison():
    """Genera figura comparativa de F(t) entre escenarios."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for scenario, style in SCENARIOS.items():
        data = load_thermodynamic_time_series(scenario, "F")
        if data is not None:
            ax.semilogy(data['time'], data['F_total'], 
                       color=style['color'], linestyle=style['linestyle'],
                       label=style['label'], linewidth=1.5)
    
    ax.set_xlabel('Time $t$', fontsize=12)
    ax.set_ylabel('Free Energy $F(t)$', fontsize=12)
    ax.set_title('Free Energy Evolution: Comparison Across Scenarios', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2.0)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "thermodynamic_F_comparison.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {output_file}")
    plt.close()

def plot_entropy_production_comparison():
    """Genera figura comparativa de σ(t) entre escenarios."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for scenario, style in SCENARIOS.items():
        data = load_thermodynamic_time_series(scenario, "sigma")
        if data is not None:
            ax.semilogy(data['time'], data['sigma'], 
                       color=style['color'], linestyle=style['linestyle'],
                       label=style['label'], linewidth=1.5)
    
    ax.set_xlabel('Time $t$', fontsize=12)
    ax.set_ylabel('Entropy Production $\\sigma(t)$', fontsize=12)
    ax.set_title('Entropy Production: Comparison Across Scenarios', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2.0)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "thermodynamic_sigma_comparison.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {output_file}")
    plt.close()

def plot_chemical_potentials_comparison():
    """Genera figura comparativa de potenciales químicos (2 arriba: μ_c, μ_s; 1 abajo: μ_i)."""
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.32, wspace=0.28)
    ax_mu_i = fig.add_subplot(gs[1, :])
    ax_mu_c = fig.add_subplot(gs[0, 0], sharex=ax_mu_i)
    ax_mu_s = fig.add_subplot(gs[0, 1], sharex=ax_mu_i)
    axes = (ax_mu_c, ax_mu_s, ax_mu_i)

    for scenario, style in SCENARIOS.items():
        data = load_thermodynamic_time_series(scenario, "mu")
        if data is not None:
            ax_mu_c.plot(data['time'], data['mu_c'],
                        color=style['color'], linestyle=style['linestyle'],
                        label=style['label'], linewidth=1.5)
            ax_mu_s.plot(data['time'], data['mu_s'],
                        color=style['color'], linestyle=style['linestyle'],
                        label=style['label'], linewidth=1.5)
            ax_mu_i.plot(data['time'], data['mu_i'],
                        color=style['color'], linestyle=style['linestyle'],
                        label=style['label'], linewidth=1.5)

    ax_mu_c.set_ylabel('$\\mu_c$ (Cancer)', fontsize=12)
    fig.suptitle('Chemical Potentials: Comparison Across Scenarios', fontsize=14, y=0.98)
    ax_mu_c.legend(loc='best', fontsize=9)
    ax_mu_c.grid(True, alpha=0.3)
    ax_mu_c.set_xlim(0, 2.0)
    ax_mu_c.tick_params(axis='x', labelbottom=False)

    ax_mu_s.set_ylabel('$\\mu_s$ (Healthy)', fontsize=12)
    ax_mu_s.legend(loc='best', fontsize=9)
    ax_mu_s.grid(True, alpha=0.3)
    ax_mu_s.set_xlim(0, 2.0)
    ax_mu_s.tick_params(axis='x', labelbottom=False)

    ax_mu_i.set_xlabel('Time $t$', fontsize=12)
    ax_mu_i.set_ylabel('$\\mu_i$ (Immune)', fontsize=12)
    ax_mu_i.legend(loc='best', fontsize=9)
    ax_mu_i.grid(True, alpha=0.3)
    ax_mu_i.set_xlim(0, 2.0)

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    output_file = OUTPUT_DIR / "thermodynamic_mu_comparison.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {output_file}")
    plt.close()

def plot_final_values_grid():
    """Genera grid comparativo de valores finales."""
    # Cargar todos los resúmenes
    all_scenarios = [
        "strong_mu0_uNo_bajo_umbral",
        "strong_mu0_uSi_bajo_umbral",
        "strong_mu1_uNo_bajo_umbral",
        "strong_mu1_uNo_sobre_umbral",
        "strong_mu1_uSi_bajo_umbral",
        "weak_mu0_uNo_bajo_umbral",
        "weak_mu0_uNo_sobre_umbral",
        "weak_mu0_uSi_bajo_umbral",
        "weak_mu1_uNo_bajo_umbral",
        "weak_mu1_uNo_sobre_umbral",
        "weak_mu1_uSi_bajo_umbral",
    ]
    
    F_final = []
    sigma_final = []
    mu_c_final = []
    labels = []
    
    for scenario in all_scenarios:
        summary_file = BASE_DIR / scenario / "thermodynamics" / "thermodynamic_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                F_final.append(summary['F_final']['F_total'])
                sigma_final.append(summary['sigma_final'])
                mu_c_final.append(summary['mu_final']['mu_c_avg'])
                labels.append(scenario.replace('_', ' '))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # F_final
    axes[0].semilogy(range(len(F_final)), F_final, 'o', markersize=8)
    axes[0].set_ylabel('$F_{\\text{final}}$', fontsize=12)
    axes[0].set_title('Final Free Energy', fontsize=12)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # sigma_final
    axes[1].semilogy(range(len(sigma_final)), sigma_final, 's', markersize=8, color='orange')
    axes[1].set_ylabel('$\\sigma_{\\text{final}}$', fontsize=12)
    axes[1].set_title('Final Entropy Production', fontsize=12)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    # mu_c_final
    axes[2].plot(range(len(mu_c_final)), mu_c_final, '^', markersize=8, color='green')
    axes[2].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    axes[2].set_ylabel('$\\mu_{c,\\text{final}}$', fontsize=12)
    axes[2].set_title('Final Chemical Potential (Cancer)', fontsize=12)
    axes[2].set_xticks(range(len(labels)))
    axes[2].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "thermodynamic_final_values_grid.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {output_file}")
    plt.close()

def plot_F_grid_mu_comparison():
    """Genera grid 2x2 comparando μ=0 vs μ=1 para F(t) (sin control)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    scenarios_grid = [
        ("weak_mu0_uNo_bajo_umbral", "weak_mu1_uNo_bajo_umbral", "Weak Allee"),
        ("strong_mu0_uNo_bajo_umbral", "strong_mu1_uNo_bajo_umbral", "Strong Allee")
    ]
    
    for row, (scenario_mu0, scenario_mu1, title) in enumerate(scenarios_grid):
        data_mu0 = load_thermodynamic_time_series(scenario_mu0, "F")
        data_mu1 = load_thermodynamic_time_series(scenario_mu1, "F")
        
        # Panel izquierdo: μ=0
        ax = axes[row, 0]
        if data_mu0 is not None:
            ax.semilogy(data_mu0['time'], data_mu0['F_total'], 'b-', linewidth=2, label='$F_{\\text{total}}$')
            ax.semilogy(data_mu0['time'], data_mu0['F_local'], 'r--', linewidth=1.5, alpha=0.7, label='$F_{\\text{local}}$')
        ax.set_ylabel('$F(t)$', fontsize=11)
        ax.set_title(f'{title}, $\\mu=0$', fontsize=11)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 2.0)
        
        # Panel derecho: μ=1
        ax = axes[row, 1]
        if data_mu1 is not None:
            ax.semilogy(data_mu1['time'], data_mu1['F_total'], 'r-', linewidth=2, label='$F_{\\text{total}}$')
            ax.semilogy(data_mu1['time'], data_mu1['F_local'], 'r--', linewidth=1.5, alpha=0.7, label='$F_{\\text{local}}$')
        ax.set_ylabel('$F(t)$', fontsize=11)
        ax.set_title(f'{title}, $\\mu=1$', fontsize=11)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 2.0)
    
    axes[1, 0].set_xlabel('Time $t$', fontsize=11)
    axes[1, 1].set_xlabel('Time $t$', fontsize=11)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "thermo_grid_mu_F.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {output_file}")
    plt.close()

def plot_sigma_grid_u_comparison():
    """Genera grid 2x2 comparando con/sin control para σ(t)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    scenarios_grid = [
        ("weak_mu0_uNo_bajo_umbral", "weak_mu0_uSi_bajo_umbral", "Weak Allee"),
        ("strong_mu0_uNo_bajo_umbral", "strong_mu0_uSi_bajo_umbral", "Strong Allee")
    ]
    
    for row, (scenario_no_u, scenario_with_u, title) in enumerate(scenarios_grid):
        data_no_u = load_thermodynamic_time_series(scenario_no_u, "sigma")
        data_with_u = load_thermodynamic_time_series(scenario_with_u, "sigma")
        
        # Panel izquierdo: sin control
        ax = axes[row, 0]
        if data_no_u is not None:
            ax.semilogy(data_no_u['time'], data_no_u['sigma'], 'b-', linewidth=2, label='No control')
        ax.set_ylabel('$\\sigma(t)$', fontsize=11)
        ax.set_title(f'{title}, $u=0$', fontsize=11)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 2.0)
        
        # Panel derecho: con control
        ax = axes[row, 1]
        if data_with_u is not None:
            ax.semilogy(data_with_u['time'], data_with_u['sigma'], 'g-', linewidth=2, label='With control')
        ax.set_ylabel('$\\sigma(t)$', fontsize=11)
        ax.set_title(f'{title}, $u>0$', fontsize=11)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 2.0)
    
    axes[1, 0].set_xlabel('Time $t$', fontsize=11)
    axes[1, 1].set_xlabel('Time $t$', fontsize=11)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "thermo_grid_u_sigma.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {output_file}")
    plt.close()

def plot_mu_grid_comparison():
    """Genera grid 2x2 comparando potenciales químicos entre escenarios clave."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Escenarios clave para comparar
    scenarios = {
        "weak_mu0_uNo_bajo_umbral": {"label": "Weak, μ=0, No control", "color": "blue"},
        "weak_mu0_uSi_bajo_umbral": {"label": "Weak, μ=0, Control", "color": "green"},
        "strong_mu0_uNo_bajo_umbral": {"label": "Strong, μ=0, No control", "color": "blue", "linestyle": "--"},
        "strong_mu0_uSi_bajo_umbral": {"label": "Strong, μ=0, Control", "color": "green", "linestyle": "--"},
    }
    
    # Panel 1: μ_c
    ax = axes[0, 0]
    for scenario, style in scenarios.items():
        data = load_thermodynamic_time_series(scenario, "mu")
        if data is not None:
            linestyle = style.get("linestyle", "-")
            ax.plot(data['time'], data['mu_c'], color=style['color'], 
                   linestyle=linestyle, linewidth=1.5, label=style['label'])
    ax.set_ylabel('$\\mu_c$ (Cancer)', fontsize=11)
    ax.set_title('Chemical Potential: Cancer', fontsize=11)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle=':', linewidth=0.5)
    ax.set_xlim(0, 2.0)
    
    # Panel 2: μ_s
    ax = axes[0, 1]
    for scenario, style in scenarios.items():
        data = load_thermodynamic_time_series(scenario, "mu")
        if data is not None:
            linestyle = style.get("linestyle", "-")
            ax.plot(data['time'], data['mu_s'], color=style['color'], 
                   linestyle=linestyle, linewidth=1.5, label=style['label'])
    ax.set_ylabel('$\\mu_s$ (Healthy)', fontsize=11)
    ax.set_title('Chemical Potential: Healthy', fontsize=11)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle=':', linewidth=0.5)
    ax.set_xlim(0, 2.0)
    
    # Panel 3: μ_i
    ax = axes[1, 0]
    for scenario, style in scenarios.items():
        data = load_thermodynamic_time_series(scenario, "mu")
        if data is not None:
            linestyle = style.get("linestyle", "-")
            ax.plot(data['time'], data['mu_i'], color=style['color'], 
                   linestyle=linestyle, linewidth=1.5, label=style['label'])
    ax.set_xlabel('Time $t$', fontsize=11)
    ax.set_ylabel('$\\mu_i$ (Immune)', fontsize=11)
    ax.set_title('Chemical Potential: Immune', fontsize=11)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle=':', linewidth=0.5)
    ax.set_xlim(0, 2.0)
    
    # Panel 4: Comparación final de μ_i (enfatizando efecto del control)
    ax = axes[1, 1]
    control_scenarios = {
        "weak_mu0_uNo_bajo_umbral": {"label": "Weak, No control", "color": "blue"},
        "weak_mu0_uSi_bajo_umbral": {"label": "Weak, Control", "color": "green"},
    }
    for scenario, style in control_scenarios.items():
        data = load_thermodynamic_time_series(scenario, "mu")
        if data is not None:
            ax.plot(data['time'], data['mu_i'], color=style['color'], 
                   linewidth=2, label=style['label'])
    ax.set_xlabel('Time $t$', fontsize=11)
    ax.set_ylabel('$\\mu_i$ (Immune)', fontsize=11)
    ax.set_title('Control Effect on $\\mu_i$', fontsize=11)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle=':', linewidth=0.5)
    ax.set_xlim(0, 2.0)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "thermo_grid_mu_potentials.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {output_file}")
    plt.close()

if __name__ == '__main__':
    print("Generando figuras comparativas termodinámicas...")
    plot_free_energy_comparison()
    plot_entropy_production_comparison()
    plot_chemical_potentials_comparison()
    plot_final_values_grid()
    print("\nGenerando grids comparativos...")
    plot_F_grid_mu_comparison()
    plot_sigma_grid_u_comparison()
    plot_mu_grid_comparison()
    print("\n✓ Todas las figuras generadas")

