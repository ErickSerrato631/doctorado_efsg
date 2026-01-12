# Resumen de estados estacionarios y candidatos CI (Weak y Strong)

## Candidatos escogidos para simulaciones (a=0.1)
- 20 puntos inestables con $i^*>0$ (5 bajo umbral $c^*\to 0$ y 5 sobre umbral $c^*>0.9$) para cada modelo y $\mu\in\{0,1\}$.
- Parám. fijos no barridos: $r_s=13.12$, $\alpha=10.22$, $\gamma=0.74$, $a=0.1$; $\mu$ según el grupo.
- Archivos: `filesystem/mu0/Weak_Allee/steady_states_candidates_story_under.csv`, `..._over.csv` y análogos para Strong y para $\mu=1`.

### Weak Allee
- $\mu=0$ (bajo umbral): 5 puntos con $c^*\sim 10^{-18}$, $s^*\sim 2\\times10^{-17}$, $i^*\approx1.042$, Re $\lambda_{\max}\approx16.12$ (silla).
- $\mu=1$ (sobre umbral): 5 puntos con $c^*\in[1.0066,1.0088]$, $s^*\sim10^{-21}$–$10^{-25}$, $i^*\approx0.05$–0.067, Re $\lambda_{\max}\approx4.63$–4.64 (silla).

### Strong Allee
- $\mu=0$ (bajo umbral): 5 puntos con $c^*\sim -10^{-23}$–$10^{-22}$, $s^*\approx0.2553$, $i^*\approx1.042$, Re $\lambda_{\max}\approx4.92$ (silla).
- $\mu=1$ (sobre umbral): 5 puntos con $c^*\in[1.0082,1.0100]$, $s^*\sim10^{-20}$–$10^{-21}$, $i^*\approx0.068$–0.107, Re $\lambda_{\max}\approx6.27$–6.73 (silla).

Tablas completas en `Tesis/main.tex` (sección “Candidatos usados como CI inestables…”).

## Datos generados (Weak, corrida previa)
- Tabla completa de soluciones: `steady_states_scan.csv` (raíz del repo). 628 soluciones con `i*>0`, todas marcadas inestables (Re λ>0).
- Figuras de síntesis: `steady_states_summary.png` (raíz del repo), generada en `steady_states.ipynb`.
- Figuras/barras adicionales: `steady_states_bars_story.png` (raíz), con Re(λ) promedio y fracción inestable por parámetro.
- Candidatos filtrados: `steady_states_candidates_story.csv` (raíz), después del post-procesado.
- Notebook: `new_model_10112024/Weak Allee/steady_states.ipynb` contiene:
  - Barrido de parámetros y construcción del CSV.
  - Visualización 2D (c–s coloreado por `i*`) y curva 3D (c, s, i).
  - Celda de análisis estadístico y correlaciones.
  - Celda de selección de candidatos “moderadamente inestables”.

## Hallazgos numéricos
- Todas las soluciones encontradas en el barrido están inestables (Re λ>0). No apareció un estado estable dentro del rango muestreado.
- Rango (s*>0, c*>0 filtro físico):
  - `c*` ≈ [~0, 1.18], mediana ~1.01
  - `s*` ≈ [~0, 0.58], mediana ~0 (muchos valores pequeños)
  - `i*` ≈ [0.0075, 1.0], mediana ~0.37
  - Re(λ) máx típica de orden 40–60; hay casos >180 sin filtro.
- Correlaciones (filtro s*>0, c*>0):
  - `i*` vs `c*`: −0.95 (fuerte anticorrelación)
  - `i*` vs `s*`: −0.48 (moderada)
  - Parámetros vs `i*`: correlaciones cercanas a 0 en el rango muestreado (paso discreto).

### Criterios aplicados en las selecciones
- Barrido completo: se aceptan raíces con `i*>0` (para evitar puntos inmunológicamente no físicos); no se filtró por s*>0 en el CSV, pero sí en los análisis.
- Filtros “físicos” en análisis: s*>0 y c*>0 (densidades no negativas); a veces i*>0.05 para descartar casi-ceros numéricos.
- Candidatos moderadamente inestables (celda de selección, criterios relajados):
  - s*>0, c*>0
  - 0.05 < i* < 1.05
  - 0 < max_real < 70 (inestabilidad moderada, no explosiva)
  - c* < 2.5, s* < 1.2
  - Ordenados por menor max_real y luego por c*
### Ecuaciones reducidas empleadas
Con la hipótesis quasi-estática de \(i\) (∂i/∂t = 0, i>0) se despeja
\[
i(c,s) = 1 + \frac{\delta s^2 - c^2(\eta + \beta \mu/2)}{r_d}.
\]
Sustituyendo en las ecuaciones de \(c\) y \(s\) y fijando ∂c/∂t=∂s/∂t=0 se obtienen las nullclines:
\[
F1(c,s)=\frac14 c\Big[-4(1-c)(a-c)r_c - 4s^2\alpha - \frac{\beta}{r_d^2}(2r_d+2s^2\delta-c^2(2\eta+\beta\mu))^2 -4\mu\big(s^2\gamma + \frac{\eta}{4r_d^2}(2r_d+2s^2\delta-c^2(2\eta+\beta\mu))^2\big)\Big],
\]
\[
F2(c,s)=\frac14 s\Big[-4r_s(1-s) -4c^2\gamma -2c^2\alpha\mu + \frac{\delta}{r_d^2}(2r_d+2s^2\delta-c^2(2\eta+\beta\mu))^2\Big].
\]
Los puntos de equilibrio del sistema reducido satisfacen \(F1=0, F2=0\).

### Sobre λ (autovalores)
En este análisis, λ se refiere a los autovalores del Jacobiano 2×2 del sistema reducido \((F1,F2)\) evaluado en cada punto de equilibrio \((c^*,s^*)\):
- Re λ < 0 ⇒ estabilidad lineal local.
- Re λ > 0 ⇒ inestabilidad (silla/foco inestable).
Los λ aquí no son longitudes de onda; para longitudes de onda espaciales habría que analizar el problema PDE completo (modos de Fourier y patrones de Turing). Si necesitas longitudes de onda, debemos estudiar el espectro de perturbaciones espaciales en la PDE (linealización con términos de difusión) o usar los espectros ya calculados en otros notebooks.

### Candidatos “moderadamente inestables”
Criterios en notebook (relajados para obtener puntos): s*>0, c*>0, 0.05<i*<1.05, 0<Re λ<70, c*<2.5, s*<1.2.
- Se listan las primeras 10 filas ordenadas por menor Re λ y menor c* en la tabla de la celda correspondiente.
- Los más inestables dentro del filtro moderado suelen concentrarse en `rc≈6.5`, `delta≈7.0`, `eta≈3.0`, `rd≈10.9–12.5`, con `beta` en {5.0, 7.6, 9.0}, y (c*, s*) ~ (1.1–1.17, 0.52–0.58), `i*` ~ 0.4–0.6, Re λ máx ~60.

## Discusión física/biológica
- El barrido muestra que, con los rangos discretos usados, la dinámica estacionaria en el plano reducido (c, s) no produce atractores estables cuando se exige `i*>0`; todos los puntos son saddles/focos inestables (Re λ>0).
- `i*` decrece al aumentar `c*` (competencia fuerte cáncer–inmune) y también disminuye con `s*` alto, señal de que la supresión cuadrática domina sobre la activación.
- Los puntos con mayor inestabilidad combinan:
  - `delta` alto (interacción s–i) que sostiene `i*` moderado,
  - `eta` bajo (supresión inmune menor),
  - `rc` alto, que empuja el crecimiento tumoral y favorece Re λ positivo.
- Para obtener estados estacionarios físicamente “aceptables” (i*>0) y no explosivamente inestables, se requiere balancear interacción inmune (`delta`), supresión (`eta`, `beta`) y tasa inmune `rd`. El barrido sugiere que reducir `beta`/`eta` o aumentar `rd` facilita `i*>0`, pero no garantiza estabilidad lineal en el rango explorado.

## Recomendaciones próximas
1. Ampliar/afinar el rango de parámetros continuo (no solo tres valores discretos) y probar búsqueda con `scipy.optimize.root` y semillas adaptativas para detectar posibles atractores estables.
2. Restringir el dominio de semillas a s≥0 y c≥0 pero incluir i*>0 explícito para evitar raíces con s muy pequeñas → ajustar malla a [0,1] si se busca interpretación estricta de densidades.
3. Si el interés es excitar dinámica espacial partiendo de un equilibrio inestable, usar los candidatos moderados (Re λ ~ 40–60) como CI en `cancer_dynamics.ipynb`.
4. Si se necesita un estado estable, explorar reducción de `rc`, aumento de `rd`, y disminución conjunta de `beta` y `eta`, o elevar `delta`; añadir un barrido más fino centrado en esas direcciones.

## Cómo reproducir
1. Ejecutar las celdas de barrido en `steady_states.ipynb` para generar `steady_states_scan.csv`.
2. Ejecutar las celdas de visualización y análisis para generar `steady_states_summary.png` y la tabla de candidatos.
3. Usar las coordenadas seleccionadas como condiciones iniciales en `cancer_dynamics.ipynb` si se desea estudiar la evolución PDE desde un equilibrio inestable.

## Procedimiento matemático usado en `steady_states_story.ipynb` (Weak y Strong)

1. **Reducción cuasi-estática de \(i\).** Se impone \(\partial i/\partial t = 0\), \(i>0\), y se despeja
   \[
   i(c,s) = 1 + \frac{\delta s^2 - c^2(\eta + \beta \mu/2)}{r_d}.
   \]
   El sistema PDE 3D se reduce al plano \((c,s)\) con dos nullclines:
   - Weak Allee: \(F1, F2\) como en las ecuaciones de la sección anterior (con el término logístico débil en \(c\)).
   - Strong Allee: mismo esquema pero con la forma fuerte \(rc\,c(1-c)\frac{c-a}{1-a}\) en \(F1\); el notebook implementa las fórmulas explícitas para ambos casos.
   Los equilibrios reducidos satisfacen \(F1(c,s)=0,\,F2(c,s)=0\) y luego \(i^*(c,s)\).

2. **Newton–Raphson en rejilla discreta.** Para cada punto del hiper-parámetro:
   - Weak (μ=1): \(rc\in\{5.0, 5.84, 6.5\}\), \(\beta\in\{5.0, 7.6, 9.0\}\), \(\delta\in\{3.5, 5.4, 7.0\}\), \(\eta\in\{3.0, 5.08, 7.0\}\), \(rd\in\{9.0, 10.92, 12.5\}\); semillas 5×5 en \([0.2,3.0]^2\).
   - Strong (μ=1): \(rc\in\{4.5, 5.0, 5.5, 6.5\}\), \(\beta\in\{3,5,7\}\), \(\delta\in\{5,7,9\}\), \(\eta\in\{1,3,5\}\), \(rd\in\{8,10,12\}\); semillas 5×5 en \([0.2,2.5]^2\).
   Newton se detiene si \(\|Δ\|<10^{-8}\), condición de número \(<10^{12}\) y descarta raíces no finitas o duplicadas (distancia <\(10^{-3}\)).

3. **Filtrado físico y cálculo de eigenvalores.** Se rechaza \(i^*\le 0\). Se arma el Jacobiano \(J(c^*,s^*)\) del sistema reducido y se calculan \(\lambda_{1,2}\); se guarda \(\mathrm{Re}\,\lambda_{\max}\). Filtros moderados: \(s^*>0\), \(c^*>0\), \(0.05<i^*<1.05\), \(0<\mathrm{Re}\,\lambda_{\max}<70\), \(c^*<2.5\), \(s^*<1.2\) (Weak) o \(s^*<1.0\) (Strong), y se ordena por menor \(\mathrm{Re}\,\lambda_{\max}\).

4. **Clasificación de inestabilidad.** Para las tablas de top 10:
   - Weak: umbrales en \(\mathrm{Re}\,\lambda_{\max}\) (Menos ≤38.54, Intermedia ≤39.60, Más >39.60); se añade \(\mathrm{Re}\,\lambda_{\max}\) normalizado en \([0,1]\).
   - Strong: umbrales (Menos ≤8.72, Intermedia ≤9.35, Más >9.35); se marca \(s^*\approx 0\) en la corrida actual.

5. **Persistencia de resultados.** Los notebooks escriben sin fecha en `filesystem/mu1/Weak_Allee/` y `filesystem/mu1/Strong_Allee/`:
   - `steady_states_scan_story.csv` y `steady_states_candidates_story.csv` (Weak).
   - `steady_states_scan_story_strong.csv` y `steady_states_candidates_story_strong.csv` (Strong).
   - Figuras asociadas (`*_summary*.png`, `*_bars*.png`, `*_heatmaps*.png`) según el caso.
   Las carpetas `filesystem/mu0/...` quedan reservadas para futuras corridas con \(\mu=0\).

## Nota sobre longitudes de onda y patrones espaciales
Para conectar con escalas espaciales/termodinámica:
1. Linealizar la PDE completa incluyendo difusión y evaluar el símbolo en Fourier: λ(k) = eigenvals(J_reac − D·k²), donde J_reac es el Jacobiano de reacción y D la matriz de difusiones.
2. Modos inestables (Re λ(k)>0) dan número de onda k*; la longitud de onda característica es \( \ell = 2\pi / k^* \).
3. Comparar k* con los espectros ya generados en notebooks de correlación/FFT para validar coherencia entre análisis ODE reducido y patrones PDE.

## Discusión de resultados generados (última corrida)
- Barrido (`steady_states_scan.csv`): 628 raíces con `i*>0`; todas inestables (Re λ>0).
- Filtro moderado (`steady_states_candidates_story.csv`): 622 puntos con s*>0, c*>0, 0.05<i*<1.05, 0<Re λ<70, c*<2.5, s*<1.2. Los más inestables: `rc≈6.5`, `delta≈7.0`, `eta≈3.0`, `rd≈10.9–12.5`, `beta` en {5.0, 7.6, 9.0}; (c*, s*) ~ (1.1–1.17, 0.52–0.58), `i*` ~ 0.4–0.6, Re λ máx ~60.
- Estadísticos (filtro s*>0,c*>0): mediana `c*≈1.01`, `s*≈0`, `i*≈0.37`; Re λ mediana ~40.7.
- Correlaciones (filtro s*>0,c*>0): i* vs c* fuerte negativa (≈−0.95); i* vs s* moderada (≈−0.48); parámetros vs i* ~0 en el rango discreto.
- Figuras: `steady_states_summary.png` (scatter c-s color i*, hist i*, Re λ vs c*, s*); `steady_states_bars_story.png` (Re λ promedio y fracción inestable por parámetro; fracción inestable ≈1 en el barrido actual).