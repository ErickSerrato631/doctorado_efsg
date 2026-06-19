# Marco físico-matemático: termodinámica fuera del equilibrio en el modelo Allee (c, s, i)

Este documento describe **qué se implementa** en esta carpeta y en [`termodynamics/calculate_thermodynamic_properties.py`](../termodynamics/calculate_thermodynamic_properties.py), cómo se relaciona con la **termodinámica clásica del no equilibrio (TdC)** y **qué analogías son útiles** (y cuáles no aplican literalmente).

---

## 1. Ecuaciones del modelo (continuo)

Se consideran tres campos adimensionales en un dominio espacial \(\Omega \subset \mathbb{R}^2\):

- \(c(\mathbf{x},t)\): densidad relativa de células tumorales  
- \(s(\mathbf{x},t)\): densidad relativa de tejido sano  
- \(i(\mathbf{x},t)\): actividad / densidad relativa del sistema inmune  

La evolución tiene la forma **reacción–difusión**:

\[
\partial_t \phi_a = D_a \nabla^2 \phi_a + R_a(c,s,i), \qquad \phi_a \in \{c,s,i\},
\]

con coeficientes de difusión \(D_c, D_s, D_i > 0\) y términos de reacción \(R_a\) que incluyen crecimiento tipo Allee, competencia cruzada, parámetro \(\mu\) del modelo y, si aplica, control inmunológico (p. ej. término en \(R_i\)). Las expresiones exactas están en [`model_equations.py`](../model_equations.py).

### 1.1 Forma “transporte + fuente”

Equivalente a la forma conservativa usada en TdC:

\[
\partial_t \phi_a = -\nabla \cdot \mathbf{J}_a + R_a,
\qquad
\mathbf{J}_a = -D_a \nabla \phi_a .
\]

- \(\mathbf{J}_a\): **corriente difusiva** (ley de Fick).  
- \(R_a\): **fuente** (reacción / nacimiento–muerte local, no conservativa en general).

---

## 2. Analogía con termodinámica del no equilibrio (TdC)

En el formalismo de **flujos** \(J_\alpha\) y **fuerzas termodinámicas** \(X_\alpha\) (Onsager / de Groot–Mazur), la **producción de entropía** local suele escribirse como una suma de productos bilineales:

\[
\sigma_S = \sum_\alpha J_\alpha X_\alpha \geq 0,
\]

en condiciones de validad del marco (sistemas continuos, relaciones constitutivas lineales o fenomenológicas, etc.).

En un fluido con difusión de especies y **temperatura uniforme** \(T\), un término típico es el de **difusión química**:

\[
\sigma_S = -\frac{1}{T}\sum_a \mathbf{J}_a \cdot \nabla \mu_a + \cdots
\]

con \(\mathbf{J}_a\) flujo de masa de la especie \(a\) y \(\mu_a\) potencial químico.

### 2.1 ¿Dónde está la “temperatura” en tu modelo?

**No hay una variable explícita \(T(\mathbf{x},t)\)** en las ecuaciones implementadas: el modelo es **isotermo efectivo** a nivel fenomenológico. Las densidades \(c,s,i\) no son energía ni temperatura; son **fracciones o densidades normalizadas** de poblaciones.

| Cantidad en TdC continua | Analogía en este proyecto | Comentario |
|--------------------------|---------------------------|------------|
| Temperatura \(T\) | **No modelada**; puede pensarse **constante** y **absorbida en la escala** de tiempos y coeficientes | Los \(D_a\) y las tasas \(r_c,r_s,r_d,\ldots\) ya incorporan “efectos biológicos” sin separar \(1/T\). |
| \(1/T\) como factor en \(\sigma\) | **Omisión explícita**: los escalares que graficas son **proporcionales** a una disipación “a temperatura efectiva constante” | Si se desea, \(\sigma_{\mathrm{TdC}} = \sigma_{\mathrm{código}}/T_{\mathrm{eff}}\) con una \(T_{\mathrm{eff}}\>0\) arbitraria solo cambia unidades, no la forma temporal ni espacial relativa. |
| Potencial químico \(\mu_a\) | **Tres usos distintos en el código** (ver §3) | Ninguno coincide automáticamente con el \(\mu\) de un gas ideal; son **potenciales efectivos** del continuo. |
| Flujo difusivo \(\mathbf{J}_a\) | \(\mathbf{J}_a = -D_a \nabla \phi_a\) | Directo. |
| Densidad de entropía | **No calculada** | Solo **producción / disipación proxy** (§4). |

**Conclusión:** la comparación útil con la TdC es **estructural** (flujo generado por gradientes, suma de términos de disipación por especie), **no** una identificación literal \(c \equiv T\) o \(i \equiv 1/T\). Si en un texto quieres un análogo verbal de “intensidad térmica”, lo honesto es decir que **no hay campo térmico** y que el sistema es **químico–ecológico** con difusión.

---

## 3. Potenciales químicos \(\mu_c, \mu_s, \mu_i\) en el código

### 3.1 Convención “variacional” (`calculate_chemical_potentials`)

En [`calculate_thermodynamic_properties.py`](../termodynamics/calculate_thermodynamic_properties.py), \(\mu_c,\mu_s,\mu_i\) se construyen como **derivadas funcionales locales** del mismo funcional de energía libre efectiva \(F[c,s,i]\) que usa ese módulo (densidad local + términos de acoplamiento coherentes con el paper de contexto físico). Simbólicamente:

\[
\mu_a(\mathbf{x},t) \sim \frac{\delta F}{\delta \phi_a}\Big|_{\text{densidad local}} .
\]

**Advertencia conceptual:** si el subsistema reaccional **no** es integrable como gradiente de un único potencial en \((c,s,i)\) (no reciprocidad en el jacobiano de \(R_a\)), esta \(\mu_a\) es un **objeto de trabajo** útil para cuantificar **gradientes espaciales** y un término \(\sum_a D_a|\nabla\mu_a|^2\), pero **no** garantiza que toda la dinámica global sea relajación de ese mismo \(F\). Ver [`Biblioteca/markdowns/termodinamica_fuera_equilibrio_allee.md`](../../../Biblioteca/markdowns/termodinamica_fuera_equilibrio_allee.md).

### 3.2 Convención “campo igual a actividad” (`fenics_nonequilibrium.py`, modo `field`)

\(\mu_a \equiv \phi_a\): entonces \(\mathbf{J}_a = -D_a\nabla\mu_a\) y el producto \(\mathbf{J}_a\cdot\nabla\mu_a = -D_a|\nabla\phi_a|^2 \leq 0\). El **proxy positivo** \(\sigma^+ = \sum_a D_a|\nabla\phi_a|^2\) es el que se alinea con la **disipación por mezcla** espacial.

### 3.3 Convención “derivada de la reacción” (`reaction_derivative`)

\(\mu_a = \partial R_a / \partial \phi_a\) evaluado en los campos (UFL). Describe una **sensibilidad local del kinetismo**; sus gradientes espaciales aparecen en \(\sigma\) solo si se elige esa convención en el postproceso FEniCS.

---

## 4. Producción de entropía y disipación implementadas

### 4.1 Densidad local (sobre la malla guardada)

Con diferencias finitas y paso espacial \(\Delta x\) (derivado de `space_size` y el tamaño de la matriz):

1. **Término “\(\mu\)” (módulo termodinámico)**  
   \[
   \sigma_\mu(\mathbf{x},t) = D_c|\nabla\mu_c|^2 + D_s|\nabla\mu_s|^2 + D_i|\nabla\mu_i|^2 .
   \]  
   Integral: \(\Sigma_\mu(t) = \int_\Omega \sigma_\mu\, dA \approx \sum_{ij} \sigma_\mu \,\Delta x^2\).

2. **Término “disipación difusiva por especie” (TdC, \(\mathbf{J}=-D\nabla\phi\))**  
   \[
   \sigma_{\mathrm{diss},a} = D_a|\nabla\phi_a|^2,\qquad
   \sigma_{\mathrm{diss,tot}} = \sum_a \sigma_{\mathrm{diss},a}.
   \]  
   Integrales por campo: \(\Sigma_{\mathrm{diss},c}(t)\), etc.  
   Guardado: `thermodynamics/entropy_production_by_field_t.txt`.

3. **Coherencia con figuras espaciales**  
   [`visualize_fluxes_and_entropy_density.py`](visualize_fluxes_and_entropy_density.py) grafica \(\sigma_{\mathrm{diss,tot}}\) local como `sigma_plus_t_*.png`.

### 4.2 Qué **no** incluye este \(\sigma\)

- **Reacciones químicas locales** \(R_a\) contribuyen a producción entrópica en un tratamiento completo; aquí **no** se descompone \(\sigma_{\mathrm{reacción}}\) separadamente.  
- **Intercambios con el exterior** (sistema abierto biológico) no están en el funcional continuo simplificado.  
- Por tanto, \(\Sigma_\mu\) y \(\Sigma_{\mathrm{diss}}\) son **observables fenomenológicos del PDE**, no la entropía del tejido medida en laboratorio.

---

## 5. No reciprocidad (subsistema reaccional homogéneo)

El jacobiano

\[
A_{ab} = \frac{\partial R_a}{\partial \phi_b}
\]

se descompone en \(S = (A+A^\top)/2\) y \(N = (A-A^\top)/2\). La norma de \(N\) cuantifica **cuánta parte del acoplamiento linealizado no puede venir de un único potencial escalar** en \((c,s,i)\).

Herramienta: [`reciprocity_jacobian_analysis.py`](reciprocity_jacobian_analysis.py).

---

## 6. Mapa de scripts y flujo de trabajo

| Objetivo | Script / módulo |
|----------|-----------------|
| Serie temporal **completa** de \(F\), \(\mu\) promedio, \(\Sigma_\mu\), \(\Sigma_{\mathrm{diss}}\) y **por campo** | [`termodynamics/calculate_thermodynamic_properties.py`](../termodynamics/calculate_thermodynamic_properties.py) (por defecto `scenarios_v1.json`) |
| Mapas \(\|\nabla\phi\|\), \(\|\mathbf{J}\|\), \(\sigma_{\mathrm{diss,tot}}\), quiver de \(\mathbf{J}_c\) | [`visualize_fluxes_and_entropy_density.py`](visualize_fluxes_and_entropy_density.py) |
| \(A=S+N\) y métricas de no reciprocidad en un punto \((c,s,i)\) | [`reciprocity_jacobian_analysis.py`](reciprocity_jacobian_analysis.py) |
| \(\sigma\) y proyección en malla dentro de FEniCSx | [`fenics_nonequilibrium.py`](fenics_nonequilibrium.py) (import, no CLI) |

Orden típico: simulación (`run_scenarios` / `cancer_dynamics`) → postproceso termodinámico → visualizaciones puntuales / reciprocidad según necesidad.

**Interpretación** de `grad_mag`, `J_mag`, `sigma_plus` y quiver (marco físico-matemático y termodinámico en §1 del documento; tumor / inmunidad; materia activa): [`INTERPRETACION_FIGURAS_FLUJOS_MATERIA_ACTIVA.md`](INTERPRETACION_FIGURAS_FLUJOS_MATERIA_ACTIVA.md).

**Pipeline completo** (incluye `steady_states`, qué ejecuta `--all` en `calculate_thermodynamic_properties.py`, tabla por etapa con entradas/salidas y distinción equilibrio vs fuera de equilibrio): ver [`PIPELINE_EJECUCION_Y_FISICA.md`](../PIPELINE_EJECUCION_Y_FISICA.md) en la raíz de `Allee`.

---

## 7. Redacción sugerida para tesis o artículo

1. Declarar **sistema isotermo efectivo** sin campo \(T(\mathbf{x},t)\).  
2. Definir \(\mathbf{J}_a = -D_a\nabla\phi_a\) y la **disipación difusiva** \(\sum_a D_a|\nabla\phi_a|^2\) como **proxy** de tasa de disipación asociada al transporte.  
3. Si se usa \(\mu_a\) del funcional \(F\), explicitar que es un **potencial efectivo** del modelo Landau–Ginzburg adoptado y que la **no reciprocidad** en \(R_a\) limita la interpretación global como relajación de un único \(F\).  
4. Separar siempre **contribución difusiva** (implementada) de **contribución reaccional** (no desglosada en \(\sigma\) aquí).

---

## 8. Referencias conceptuales

- de Groot & Mazur, *Non-Equilibrium Thermodynamics* (flujos, fuerzas, producción de entropía).  
- Prigogine (sistemas abiertos y producción de entropía, nivel fenomenológico).  
- Documentación alineada en el repo: [`termodinamica_fuera_equilibrio_allee.md`](../../../Biblioteca/markdowns/termodinamica_fuera_equilibrio_allee.md), [`contexto_fisica.md`](../../../Biblioteca/markdowns/contexto_fisica.md).

---

*Última actualización: alineado con la implementación en `calculate_thermodynamic_properties.py` (desglose por campo y `scenarios_v1` por defecto) y scripts en esta carpeta.*
