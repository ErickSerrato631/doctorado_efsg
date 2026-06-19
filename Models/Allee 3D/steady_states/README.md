# Estados estacionarios (`steady_states/`)

Este directorio agrupa el **análisis de equilibrios** del modelo de reacción–difusión de dinámica tumoral (variables `c`, `s`, `i`) y utilidades para escenarios, figuras y tablas. Lo que sigue resume los **conceptos lógico-matemáticos** que subyacen al código.

---

## 1. Definición de estado estacionario (espacialmente homogéneo)

En el subsistema de **reacción** (sin difusión ni tiempo), un estado estacionario \((c^*, s^*, i^*)\) es un punto donde el campo vectorial se anula:

\[
F_c(c^*, s^*, i^*) = 0,\quad
F_s(c^*, s^*, i^*) = 0,\quad
F_i(c^*, s^*, i^*) = 0.
\]

Numéricamente, el pipeline principal en `steady_states.py` trata esto como un **problema de raíces** de un sistema no lineal \(\mathbf{F}(\mathbf{x}) = \mathbf{0}\) con \(\mathbf{x} = (c,s,i)\) (**solo 3D**).

Una **reducción 2D** en el plano \((c,s)\) sigue disponible en **`model_equations.build_reduced_model_2d_sympy`** para otras herramientas (p. ej. nullclines en `generate_phase_planes.py`); no forma parte del barrido de equilibrios en `steady_states.py`.

---

## 2. Sistema 3D (`build_equations_3d`, `scan_grid_3d`)

Se resuelven simultáneamente las tres componentes. Entran términos de **Allee** en \(F_c\), acoplamientos competitivos y, según la variante, **control terapéutico** en \(F_i\).

---

## 3. Efecto Allee débil y fuerte

- **Débil:** crecimiento per cápita del tipo \(r_c\, c\,(c-a)\,(1-c)\) (forma polinomial clásica).
- **Fuerte:** forma que anula el crecimiento en \(c \to 0^+\) de manera distinta, usando \(r_c\, c\,(1-c)\,\frac{c-a}{1-a}\).

La elección cambia la forma de \(F_c\) y por tanto el conjunto de equilibrios y su estabilidad. En `extract_steady_states_from_scenarios.py` el tipo se toma del escenario (`ALLEE_TYPE`). El barrido genérico en `steady_states.py` para 3D con control Hill asume la forma **débil** en `build_equations_3d` de ese mismo archivo (coherente con el análisis “story control” consolidado allí).

---

## 4. Control en la ecuación de \(i\): formulaciones en el código

Es importante no mezclar interpretaciones entre archivos:

| Ubicación | Forma del control |
|-----------|-------------------|
| `steady_states.py` → `build_equations_3d(..., use_hill=True)` | **Hill:** \(u = u_{\max}\, H_{\mathrm{act}}(c)\, H_{\mathrm{inh}}(i)\) con saturaciones tipo Michaelis–Menten generalizadas (`kc_h`, `nc_h`, `ki_h`, `ni_h`). |
| `extract_steady_states_from_scenarios.py` | Si `scenario_uses_hill_control` es verdadero → **misma Hill** que arriba (`ss_build_equations_3d`). Si `USE_ADAPTIVE_CONTROL=Y` pero **no** es escenario Hill (p. ej. nombre con `hillN`, o sin parámetros Hill) → **`build_equations_3d_min_adaptive`:** \(u = \min\!\big(k_u c/(i+\varepsilon_u),\, u_{\max}\big)\). Sin control adaptativo → `build_equations_3d(..., use_hill=False)`. |

Todas las variantes son **Lipschitz** (o se regularizan con `eps_u` / `Max(i,0)`) para Jacobianos y Newton. Si comparas equilibrios, verifica qué rama aplica tu escenario (Hill vs min) además de los parámetros.

---

## 5. Método numérico: Newton–Raphson y Jacobianos

- **`newton_root_3d`:** evalúa \(\mathbf{F}\) con `lambdify` y el **Jacobiano simbólico** sustituido en el punto actual.

Guía detallada (matemática + ejemplo reproducible): **`proceso_estados_estacionarios.pdf`** / **`proceso_estados_estacionarios.tex`** en esta carpeta; copia en markdown bajo **`Biblioteca/markdowns/codigos/steady_states_newton_3d_guia.md`** (repo doctorado, relativo a la raíz del proyecto).

Criterios de rechazo habituales: matriz mal condicionada, no finitud, fallo de `solve`, falta de convergencia en el máximo de iteraciones.

**Implicación matemática:** los sistemas pueden tener **varios equilibrios**; Newton converge a uno dependiendo de la **semilla** \((c_0,s_0,i_0)\). Por eso el código usa **rejillas de semillas** (productos de `linspace`) y deduplicación por distancia euclidiana entre soluciones encontradas.

---

## 6. Estabilidad lineal local (ODE)

En un equilibrio \(\mathbf{x}^*\), la linealización \(\dot{\mathbf{x}} \approx J(\mathbf{x}^*)\,(\mathbf{x}-\mathbf{x}^*)\) con \(J = \partial \mathbf{F}/\partial \mathbf{x}\) determina la **estabilidad local**:

- Si **toda** la parte real de los autovalores de \(J\) es \(< 0\), el equilibrio es **localmente asintóticamente estable** (como punto fijo del flujo ODE).
- Si **algún** autovalor tiene parte real \(> 0\), es **inestable** (el código marca `unstable`).

Los DataFrames guardan partes reales e imaginarias de los autovalores para análisis posterior.

---

## 7. “Combinaciones” de parámetros: producto cartesiano

`scan_grid_3d` recorre listas de valores \((r_c, \beta, \delta, \eta, r_d, \ldots)\) con `itertools.product`. Eso es un **diseño factorial completo** sobre esas mallas discretas: cada tupla de parámetros define un campo \(\mathbf{F}\) distinto y se vuelve a ejecutar la búsqueda de raíces con las mismas semillas.

Complejidad: crece multiplicativamente con el tamaño de cada lista; es exploración sistemática, no optimización bayesiana ni muestreo aleatorio.

---

## 8. Filtros “físicos” (`filter_physical_3d`)

Tras el barrido, se descartan equilibrios que caen fuera de rangos **ad hoc** (positividad de \(c^*, s^*\), rango de \(i^*\), cota superior en \(\max_j \mathrm{Re}(\lambda_j)\), etc.). Son **criterios de plausibilidad biológica / numérica**, no deducciones del modelo: sirven para reducir soluciones espurias o explosivas del espectro.

---

## 9. Espectro con difusión (inestabilidad de Turing linealizada)

En `generate_linear_spectra.py` se usa la matriz efectiva

\[
J(k) = J_{\mathrm{reac}} - \mathrm{diag}(D_c, D_s, D_i)\, k^2
\]

y sus autovalores \(\lambda(k)\) para modos planos de número de onda \(k\). La idea es la **dispersión lineal** del modelo reacción–difusión alrededor de un equilibrio homogéneo: buscar si algún modo con \(k>0\) se vuelve inestable aunque \(k=0\) sea estable (esqueleto del análisis de Turing).

---

## 10. Flujo de escenarios y derivados

1. **`steady_states.py`**: por defecto (**`control-3d`**) encadena **WEAK** y **STRONG** sobre la **misma rejilla factorial** (`control_3d_parameter_mesh`), cada uno con **μ∈{0,1}** y **con/sin Hill** (salvo **`--no-sweep-weak`**, que acorta ambos tramos). Un solo **`steady_states_full_run.json`** bajo **`Resultados Paper/estados_estacionarios/`** (o **`--local-only`**). El bloque JSON `strong_corner` incluye `all`, `steady_states_filtered` y `near_corner_only`. **`--mode corner-strong`** ejecuta solo el tramo STRONG en rejilla (μ×Hill completos). **`--generate-scenarios`** solo desde filas WEAK con Hill.
2. **`extract_steady_states_from_scenarios.py`**: lee `scenarios.json`, recalcula equilibrios por escenario (Newton 3D + selección cuando hay varias raíces), exporta `steady_states_scenarios.csv` y tablas LaTeX.
3. **`generate_phase_planes.py`**: nullclines y retratos de fase en 2D coherente con `ModelParameters` / `model_equations`.
4. **`generate_figures_from_scenarios.py`**: orquesta fase y espectros si existen los CSV necesarios.
5. **`generate_linear_spectra.py`**: curvas \(\mathrm{Re}\,\lambda(k)\) por escenario.

---

## 11. Ejecución práctica

Usa el mismo entorno conda **`fenicsx-env`** que las simulaciones FEniCSx (definición en **[../../environment.yml](../../environment.yml)** respecto a esta carpeta, es decir `Models/environment.yml`). Ahí deben estar `sympy`, `numpy`, `pandas`, `matplotlib`, `scipy` (conda-forge), no un `venv` aparte.

Desde el directorio **`Allee`**, típicamente:

```bash
python steady_states/steady_states.py --help
# Pipeline completo: WEAK (mu en {0,1} x Hill on/off) + STRONG (igual) → .../steady_states_full_run.json
# (monta Drive con mount_google_drive.sh o define RESULTS_DIR; en Windows sin Drive: --local-only)
python steady_states/steady_states.py --umax 0.5
python steady_states/steady_states.py --umax 0.5 --local-only
# WEAK más rápido: un solo --mu y siempre Hill (--no-sweep-weak)
python steady_states/steady_states.py --no-sweep-weak --mu 1 --umax 0.5 --local-only
# Opcional: fusionar scenarios.json desde el df WEAK filtrado
python steady_states/steady_states.py --generate-scenarios
# Solo STRONG en la misma rejilla (weak_grid vacío)
python steady_states/steady_states.py --mode corner-strong --umax 0.5
# Salidas clásicas repartidas (CSV + catalog + corner JSON en Allee/)
python steady_states/steady_states.py --legacy-split-outputs
python steady_states/extract_steady_states_from_scenarios.py
# Nullclinas / planos de fase (PNG bajo RESULTS_DIR o Drive; requiere scenarios.json + steady_states_scenarios.csv)
python steady_states/generate_phase_planes.py
python steady_states/generate_phase_planes.py --mark-simplex-corners
# Tabla I paper (4 escenarios Strong, desde steady_states en scenarios.json)
python steady_states/generate_table_control_strong_paper.py
python steady_states/generate_table_control_strong_paper.py --also-paper-copy
# Equiv.: ALLEE_PHASE_PLANE_MARK_SIMPLEX_CORNERS=1  (ejes [-0.5,2]; streamlines + vértices; sin nullclinas F1=0/F2=0)
python steady_states/generate_figures_from_scenarios.py
# Residuo 3D en los vértices (1,0,0), (0,1,0), (0,0,1)
python steady_states/verify_equilibrium_point.py --simplex-corners --allee STRONG --mu 1
```

Asegúrate de que `Allee` esté en `PYTHONPATH` si invocas módulos como paquete (`python -m` desde la raíz adecuada).

---

## Referencias cruzadas en el repositorio

- Ecuaciones de reacción y control en **`model_equations.py`**.
- Parámetros y `scenarios.json` en **`model_parameters.py`**.
- La reducción 2D simbólica en `model_equations.build_reduced_model_2d_sympy` (uso en fase/nullclines, no en el catálogo de equilibrios de `steady_states.py`).
