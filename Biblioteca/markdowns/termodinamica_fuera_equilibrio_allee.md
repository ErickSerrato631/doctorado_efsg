# Termodinámica fuera del equilibrio y análogo en el modelo Allee (c, s, i)

Este documento alinea el marco de **transporte irreversible** y **dinámica no integrable** con el código del proyecto (`model_equations.py`, simulación FEniCSx, análisis de estados estacionarios). Complementa y matiza el enfoque variacional descrito en [`contexto_fisica.md`](contexto_fisica.md).

---

## 1. Forma local de conservación

Cada densidad de población \(\phi_a \in \{c, s, i\}\) satisface

\[
\partial_t \phi_a = -\nabla \cdot \mathbf{J}_a + R_a(\phi),
\]

con **corriente difusiva** (ley de Fick)

\[
\mathbf{J}_a = -D_a \nabla \phi_a
\]

y **fuente reaccional** \(R_a = f_a(c,s,i)\) (crecimiento tipo Allee, competencia, términos en \(\mu\), control inmunológico, etc.). En el repositorio esto está explícito en [`Models/Allee/model_equations.py`](../Models/Allee/model_equations.py).

Interpretación: **transporte** (redistribución espacial) separado de **reacción** (creación/destrucción neta local).

---

## 2. ¿Cuándo existe una energía libre \(\mathcal{F}\) global?

Una dinámica **gradiente** en espacio de campos tiene la forma

\[
\partial_t \phi_a = -\sum_b M_{ab}\,\frac{\delta \mathcal{F}}{\delta \phi_b},
\]

con movilidades \(M_{ab}\) (a menudo diagonales) y un único funcional \(\mathcal{F}[\phi]\). Entonces los potenciales químicos habituales son \(\mu_a = \delta \mathcal{F}/\delta \phi_a\) (Cahn–Hilliard, modelos A/B de Hohenberg–Halperin en su forma puramente variacional).

En el **subsistema espacialmente homogéneo** (sin difusión), \(\dot{\phi}_a = f_a(\phi)\). Si existiera un potencial \(V(\phi)\) con \(f_a = -\partial V/\partial \phi_a\), la matriz jacobiana

\[
A_{ab} = \frac{\partial f_a}{\partial \phi_b}
\]

sería **simétrica** (condición tipo Helmholtz en dimensión finita: el campo vectorial \(f\) sería conservativo).

Si \(A \neq A^\top\), **no** existe \(V\) tal que \(f = -\nabla V\). En ese sentido la parte reaccional es **no integrable**: no puede generarse toda la dinámica local como gradiente de una sola energía libre escalar \(V(c,s,i)\).

La **difusión** no arregla por sí sola esta conclusión: añade términos \(\nabla^2 \phi_a\) pero la no reciprocidad de las **interacciones locales** permanece en \(A\).

---

## 3. Matriz de interacción y descomposición recíproca / no recíproca

Definición operativa (evaluada en un punto \((c,s,i)\), o campo a campo en cada \((\mathbf{x},t)\)):

\[
A_{ab} = \frac{\partial R_a}{\partial \phi_b}.
\]

Descomposición estándar:

\[
S = \frac{A + A^\top}{2}, \qquad N = \frac{A - A^\top}{2}.
\]

- \(S\): parte **simétrica** (“reciproca efectiva” o acoplamientos que podrían integrarse en un potencial cuadrático local alrededor del punto).
- \(N\): parte **antisimétrica** pura: **no reciprocidad** activa (rotación en el espacio de fases del subsistema \(3\)D).

Métricas útiles para comparar escenarios (por ejemplo \(\mu = 0\) vs \(\mu > 0\), control on/off):

- \(\|N\|_F\) (norma de Frobenius),
- \(\|N\|_2\) (espectral, vía valores singulares),
- cociente \(\|N\|_F / \|A\|_F\).

El código [`Models/Allee/nonequilibrium_termodynamics/reciprocity_jacobian_analysis.py`](../Models/Allee/nonequilibrium_termodynamics/reciprocity_jacobian_analysis.py) calcula \(A\), \(S\), \(N\) y estas normas a partir de `build_reaction_equations_sympy`.

---

## 4. Potencial químico efectivo sin \(\mathcal{F}\) único

En termodinámica clásica del transporte se usa \(\mathbf{J} \propto -\nabla \mu\). Si **no** hay un \(\mathcal{F}\) global que genere todos los términos reaccionales, \(\mu_a\) **no** es únicamente \(\delta \mathcal{F}/\delta \phi_a\).

Convenciones operativas (documentar siempre cuál se usa en figuras y tablas):

| Convención | Definición local | Uso |
|------------|------------------|-----|
| **Campo** | \(\mu_a \equiv \phi_a\) (con \(M_a = D_a\)) | Flujo \(\mathbf{J}_a = -D_a \nabla \phi_a = -D_a \nabla \mu_a\); analiza disipación asociada al gradiente de densidad. |
| **Derivada de reacción** | \(\mu_a \equiv \partial R_a / \partial \phi_a\) | “Fuerza” local que responde a pequeñas variaciones de la especie \(a\) en el punto; útil cerca de equilibrios homogéneos. |
| **Variacional (proxy)** | \(\mu_a\) como \(-\delta \mathcal{F}/\delta \phi_a\) de un funcional **elegido** (p. ej. el de `contexto_fisica.md`) | Solo consistente si los \(R_a\) coinciden con las variaciones de ese \(\mathcal{F}\); si \(N \neq 0\), es una **aproximación narrativa**, no una demostración de integrabilidad. |

Lo que suele enfatizar la dirección asesoría (“potencial químico como gradiente que impulsa la corriente”) es: fijar **constitutivamente** \(\mathbf{J}_a = -D_a \nabla \phi_a\) y elegir \(\mu_a\) para estudiar \(\mathbf{J}_a \cdot \nabla \mu_a\), sin afirmar que \(\mu\) proviene de una sola \(\mathcal{F}\) válida para todo el vector \((R_c, R_s, R_i)\).

---

## 5. Producción de entropía (observable fenomenológico)

En el formalismo de fuerzas y flujos, un escalar tipo

\[
\sigma_{\mathrm{tr}}(\mathbf{x},t) = \sum_a \mathbf{J}_a \cdot \nabla \mu_a
\]

mide un **producto flujo × fuerza** asociado al transporte difusivo bajo la convención elegida para \(\mu_a\).

- Con \(\mathbf{J}_a = -D_a \nabla \phi_a\) y \(\mu_a = \phi_a\):

\[
\sigma_{\mathrm{tr}} = -\sum_a D_a \|\nabla \phi_a\|^2 \leq 0.
\]

A menudo se reporta el **proxy positivo** \(\sigma^+ = \sum_a D_a \|\nabla \phi_a\|^2\), que coincide con la forma usada en [`calculate_thermodynamic_properties.py`](../Models/Allee/termodynamics/calculate_thermodynamic_properties.py) al combinar \(D_a (\nabla \mu_a)^2\) con potenciales \(\mu\) espacialmente suaves.

**Advertencia:** \(\sigma\) o \(\sigma^+\) construidos solo con campos y difusión **no** incluyen por separado la producción entrópica de las **reacciones químicas locales** (\(R_a\)). En un artículo conviene separar explícitamente:

1. disipación / mezcla por **gradientes** (transporte);
2. **fuentes** \(R_a\) y no reciprocidad (\(N \neq 0\)).

Así se evita confundir un escalar gráficado con la entropía microscópica completa del sistema biológico.

---

## 6. Modelo C de Hohenberg–Halperin: analogía estructural

El **modelo C** clásico acopla un campo **conservado** y uno **no conservado** mediante **un mismo** \(\mathcal{F}\). Fenomenológicamente, tu sistema tiene:

- campos con difusión \(c, s, i\);
- acoplamientos no lineales y, según parámetros, **no reciprocidad** en \(A\).

Si \(A\) no es simétrica (o los \(R_a\) no coinciden con las variaciones de un único \(\mathcal{F}\)), la analogía con Model C es **tipológica** (qué tipo de campos y términos aparecen), no un **certificado variacional** de que existe \(F[c,s,i]\) que genere exactamente las ecuaciones implementadas.

---

## 7. Mapa símbolo físico ↔ proyecto

| Símbolo / concepto | En el proyecto |
|--------------------|----------------|
| \(\phi = (c,s,i)\) | Campos tumor, sanos, inmune |
| \(\mathbf{J}_a\) | \(-D_a \nabla \phi_a\) (FEniCSx: gradientes de `Function`) |
| \(R_c, R_s, R_i\) | `build_reaction_equations_sympy` / `build_reaction_terms_ufl` |
| \(A, S, N\) | Jacobiano simbólico de \((R_c,R_s,R_i)^\top\); script `reciprocity_jacobian_analysis.py` |
| Estados de referencia | Raíces / catálogo de steady states (`steady_states/`) |
| \(\sigma\), \(\Sigma\) en FEniCS | Módulo `fenics_nonequilibrium.py` (formas UFL + ensamblado) |
| Figuras desde matrices `.txt` | `visualize_fluxes_and_entropy_density.py` (\(\|\nabla\phi\|\), \(\|\mathbf{J}\|\), \(\sigma^+\), quiver) |
| Postproceso en malla regular | [`calculate_thermodynamic_properties.py`](../Models/Allee/termodynamics/calculate_thermodynamic_properties.py): en **cada tiempo** con matrices, escribe `thermodynamics/entropy_production_sigma_t.txt` (total \(\int\sum D_a|\nabla\mu_a|^2\)), `entropy_production_by_field_t.txt` (total y **por campo** para \(\mu\) y para disipación \(D_a|\nabla\phi_a|^2\)), y `chemical_potentials_mu_t.txt` (promedios espaciales de \(\mu\)). Por defecto usa `scenarios_v1.json`. |
| \(R_a\) como UFL escalar | `build_reaction_rates_ufl` en [`model_equations.py`](../Models/Allee/model_equations.py) (para `ufl.diff` y FEniCSx) |

---

## 8. Referencias orientativas

- P. Hohenberg, A. P. Halperin, *Theory of dynamic critical phenomena*, Rev. Mod. Phys. (modelos A–F; Model C como referencia de acoplamiento conservado / no conservado).
- S. R. de Groot, P. Mazur, *Non-Equilibrium Thermodynamics* (fuerzas, flujos, relaciones constitutivas).
- I. Prigogine, marco de entropía y sistemas abiertos (nivel fenomenológico).
- S. Ramaswamy, M. C. Marchetti et al., revisiones sobre **materia activa** e hidrodinámica sin funcional de energía libre único para todo el conjunto de ecuaciones.

---

## 9. Archivos relacionados

- [`Models/Allee/nonequilibrium_termodynamics/MARCO_FISICO_MATEMATICO.md`](../Models/Allee/nonequilibrium_termodynamics/MARCO_FISICO_MATEMATICO.md) — marco físico-matemático completo, analogías con TdC (temperatura, \(\mu\), flujos, límites del \(\sigma\) implementado).
- [`contexto_fisica.md`](contexto_fisica.md) — enfoque variacional; leer junto con la sección de límites añadida allí.
- [`Models/Allee/nonequilibrium_termodynamics/reciprocity_jacobian_analysis.py`](../Models/Allee/nonequilibrium_termodynamics/reciprocity_jacobian_analysis.py)
- [`Models/Allee/nonequilibrium_termodynamics/fenics_nonequilibrium.py`](../Models/Allee/nonequilibrium_termodynamics/fenics_nonequilibrium.py)
- [`Models/Allee/nonequilibrium_termodynamics/visualize_fluxes_and_entropy_density.py`](../Models/Allee/nonequilibrium_termodynamics/visualize_fluxes_and_entropy_density.py) — PNG: `grad_mag_t_*.png`, `J_mag_t_*.png`, `sigma_plus_t_*.png`, `quiver_Jc_t_*.png`; opcional `sigma_plus_integral_vs_time.csv`. Entrada: `scenarios_v1.json` + carpeta del escenario (`RESULTS_DIR`/Drive) con `matrices/`.
