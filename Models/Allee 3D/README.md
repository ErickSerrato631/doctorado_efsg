# Modelo Allee — scripts y estado operativo

## Cálculo de propiedades termodinámicas (`termodinámica/calculate_thermodynamic_properties.py`)

Los scripts de post-proceso termodinámico están en la carpeta **`termodinámica/`**. Ejecútalos con `cd` en la raíz `Allee` y `python termodinámica/<script>.py`.

Migrado desde `ESTADO_TERMODINAMICOS.md` (raíz del repo, eliminado tras consolidación). Actualiza esta sección cuando completes corridas o cambie `scenarios.json`.

### Escenarios totales (11)

Orden de procesamiento según `scenarios.json`:

1. `strong_mu0_uNo_bajo_umbral`
2. `strong_mu0_uSi_bajo_umbral`
3. `strong_mu1_uNo_sobre_umbral`
4. `strong_mu1_uNo_bajo_umbral`
5. `strong_mu1_uSi_bajo_umbral`
6. `weak_mu0_uNo_bajo_umbral`
7. `weak_mu0_uNo_sobre_umbral`
8. `weak_mu0_uSi_bajo_umbral` — **último estado registrado: corrida interrumpida** (~4.5 % en t = 0.089 de 2001 pasos hasta T = 2.0)
9. `weak_mu1_uNo_bajo_umbral`
10. `weak_mu1_uNo_sobre_umbral`
11. `weak_mu1_uSi_bajo_umbral`

### Estado snapshot (última anotación migrada)

- **Incompleto:** `weak_mu0_uSi_bajo_umbral` (#8)
- **Pendientes de ejecutar (si el snapshot sigue vigente):** `weak_mu1_uNo_bajo_umbral`, `weak_mu1_uNo_sobre_umbral`, `weak_mu1_uSi_bajo_umbral`
- **Probablemente completos (#1–#7):** verificar en el directorio de resultados que exista `[escenario]/thermodynamics/` con todos los archivos listados abajo.

### Comandos (WSL + conda)

Usar el entorno **`fenicsx-env`**, no `base`. Creación y paquetes: **[../environment.yml](../environment.yml)** en `Models/` (FEniCSx + stack científico unificado). Ajusta rutas si tu instalación difiere.

**Opción 1 — solo el interrumpido**

```bash
wsl bash -c "cd /mnt/c/Users/Erick\ Serrato/Documents/Doctorado/doctorado_efsg-main/Models/Allee && source ~/miniconda3/etc/profile.d/conda.sh && conda activate fenicsx-env && export RESULTS_DIR=\$HOME/googledrive/Doctorado\ Erick\ Serrato/Resultados\ Paper && python termodinámica/calculate_thermodynamic_properties.py --scenario weak_mu0_uSi_bajo_umbral"
```

**Opción 2 — los cuatro pendientes (según snapshot)**

```bash
wsl bash -c "cd /mnt/c/Users/Erick\ Serrato/Documents/Doctorado/doctorado_efsg-main/Models/Allee && source ~/miniconda3/etc/profile.d/conda.sh && conda activate fenicsx-env && export RESULTS_DIR=\$HOME/googledrive/Doctorado\ Erick\ Serrato/Resultados\ Paper && python termodinámica/calculate_thermodynamic_properties.py --scenarios weak_mu0_uSi_bajo_umbral weak_mu1_uNo_bajo_umbral weak_mu1_uNo_sobre_umbral weak_mu1_uSi_bajo_umbral"
```

**Opción 3 — script auxiliar**

```bash
wsl bash /mnt/c/Users/Erick\ Serrato/Documents/Doctorado/doctorado_efsg-main/Models/Allee/termodinámica/run_thermodynamic_missing.sh
```

**Opción 4 — todos (reprocesa también completos)**

```bash
wsl bash -c "cd /mnt/c/Users/Erick\ Serrato/Documents/Doctorado/doctorado_efsg-main/Models/Allee && source ~/miniconda3/etc/profile.d/conda.sh && conda activate fenicsx-env && python termodinámica/calculate_thermodynamic_properties.py --all"
```

### Archivos esperados por escenario

En `[scenario_dir]/thermodynamics/`:

- `free_energy_F_t.txt`
- `entropy_production_sigma_t.txt`
- `chemical_potentials_mu_t.txt`
- `thermodynamic_summary.json`
- `thermodynamic_properties_combined.png`

### Scripts relacionados (en `termodinámica/`)

- `termodinámica/run_thermodynamic_missing.sh`
- `termodinámica/analyze_thermodynamic_results.py`
- `termodinámica/generate_thermodynamic_figures.py`
- `copy_thermodynamic_figures.sh` (si existe en tu árbol; no está versionado aquí)

### Termodinámica fuera del equilibrio (`nonequilibrium_termodynamics/`)

- `nonequilibrium_termodynamics/reciprocity_jacobian_analysis.py` — jacobiano \(A=\partial R_a/\partial\phi_b\), descomposición \(S+N\), normas de no reciprocidad.
- `nonequilibrium_termodynamics/fenics_nonequilibrium.py` — densidad \(\sigma\) y total \(\Sigma\) en FEniCSx (UFL), convenciones `field` / `reaction_derivative` / `positive_proxy`.
- `nonequilibrium_termodynamics/visualize_fluxes_and_entropy_density.py` — figuras desde `matrices/*.txt`: \(\|\nabla\phi\|\), \(\|\mathbf{J}_a\|\), \(\sigma^+\), quiver de \(\mathbf{J}_c\); por defecto `scenarios_v1.json` y `get_scenario_dir` (p. ej. Drive si `RESULTS_DIR` apunta allí). Salida en `<escenario>/nonequilibrium_plots/`.

Ejemplo:

```bash
python nonequilibrium_termodynamics/visualize_fluxes_and_entropy_density.py --scenario strong_mu0_uNo_bajo_umbral --time 0.5 1.0 --sigma-csv
```

Documentación: `Biblioteca/markdowns/termodinamica_fuera_equilibrio_allee.md` y `nonequilibrium_termodynamics/MARCO_FISICO_MATEMATICO.md` (marco teórico y analogías TdC).

Contexto teórico del funcional y energía libre: `Biblioteca/markdowns/contexto_fisica.md`.
