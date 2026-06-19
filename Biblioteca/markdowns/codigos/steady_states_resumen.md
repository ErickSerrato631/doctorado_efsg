# Resumen de estados estacionarios y candidatos CI (Weak y Strong)

**Análisis 3D directo (sin reducción cuasi-estática de \(i\)):** guía paso a paso con SymPy, `lambdify`, Jacobiano simbólico y Newton–Raphson en [`steady_states_newton_3d_guia.md`](steady_states_newton_3d_guia.md) (y PDF `Models/Allee/steady_states/proceso_estados_estacionarios.pdf`). El cuerpo de *este* documento sigue centrado en la **reducción 2D**, tablas de los 10 escenarios y procedimiento con nullclines.

## Datos actuales: 10 escenarios de scenarios.json

Los 10 escenarios definidos en `Models/Allee/scenarios.json` combinan:
- Efecto Allee: Weak (6 escenarios) y Strong (5 escenarios)
- Control inmunológico: $\mu \in \{0, 1\}$
- Control adaptativo (Hill): con/sin $u = u_{\max}\,H_{\mathrm{act}}(c;K_c,n_c)\,H_{\mathrm{inh}}(i;K_i,n_i)$, con defaults $u_{\max}=1$, $K_c=0.05$, $n_c=2$, $K_i=0.2$, $n_i=2$

### Parámetros comunes
- $r_c = 6.5$, $r_s = 13.12$, $a = 0.1$
- $\alpha = 10.22$, $\gamma = 0.74$, $\delta = 9$, $\eta = 1$
- Parámetros variables por escenario: $\beta \in \{3, 5\}$, $r_d \in \{10, 14\}$

### Resultados principales
- **Todos los equilibrios tienen $i^* > 0$ y son inestables** (Re $\lambda_{\max} > 0$)
- **Weak Allee** (6 escenarios):
  - Estados bajo umbral ($c^* \approx 0$): Re $\lambda_{\max} \approx 4.9$–22.1
  - Estados sobre umbral ($c^* \approx 1.0$): Re $\lambda_{\max} \approx 6.7$–16.1
  - Con control adaptativo: Re $\lambda_{\max} \approx 22.1$
- **Strong Allee** (5 escenarios):
  - Estados bajo umbral: Re $\lambda_{\max} \approx 4.9$–22.1 ($s^* \approx 0$–0.255)
  - Estado sobre umbral ($c^* \approx 1.008$): Re $\lambda_{\max} \approx 6.7$
  - Con control adaptativo: Re $\lambda_{\max} \approx 22.1$

Ver tablas completas en:
- `Tesis/chapters/04_analisis.tex` (Tablas 4.1 y 4.2)
- `Paper/sections/03_meanfield.tex` y `05_control.tex`

## Ecuaciones reducidas empleadas

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

## Sobre λ (autovalores)

En este análisis, λ se refiere a los autovalores del Jacobiano 2×2 del sistema reducido \((F1,F2)\) evaluado en cada punto de equilibrio \((c^*,s^*)\):
- Re λ < 0 ⇒ estabilidad lineal local.
- Re λ > 0 ⇒ inestabilidad (silla/foco inestable).
Los λ aquí no son longitudes de onda; para longitudes de onda espaciales habría que analizar el problema PDE completo (modos de Fourier y patrones de Turing). Si necesitas longitudes de onda, debemos estudiar el espectro de perturbaciones espaciales en la PDE (linealización con términos de difusión) o usar los espectros ya calculados en otros notebooks.

## Procedimiento matemático usado

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

## Nota sobre longitudes de onda y patrones espaciales

Para conectar con escalas espaciales/termodinámica:
1. Linealizar la PDE completa incluyendo difusión y evaluar el símbolo en Fourier: λ(k) = eigenvals(J_reac − D·k²), donde J_reac es el Jacobiano de reacción y D la matriz de difusiones.
2. Modos inestables (Re λ(k)>0) dan número de onda k*; la longitud de onda característica es \( \ell = 2\pi / k^* \).
3. Comparar k* con los espectros ya generados en notebooks de correlación/FFT para validar coherencia entre análisis ODE reducido y patrones PDE.

## Interpretación para la tesis

Contenido alineado con `Tesis/chapters/04_analisis.tex` y `07_discusion.tex`; mapa general en `Biblioteca/markdowns/indice_redaccion_tesis_paper.md`.

- **Separatrices inestables, no atractores**: en los 10 escenarios tabulados, todos los puntos con \(i^*>0\) cumplen Re \(\lambda_{\max}>0\). La lectura física es que los equilibrios homogéneos relevantes organizan el flujo como **umbrales inestables** (fronteras entre rutas hacia proliferación o hacia confinamiento cercano a casi-extinción), no como puntos de convergencia asintótica estable.
- **Robustez del hallazgo**: las tablas con \((c^*, s^*, i^*)\) y Re \(\lambda_{\max}\) dan evidencia sistemática; la inestabilidad lineal local es coherente en weak/strong Allee, con y sin control adaptativo (el control \(u\) puede **aumentar** Re \(\lambda_{\max}\), p. ej. \(\approx 22.1\) en varios casos con \(u\) activo).
- **Weak vs strong**: strong Allee refuerza la “dureza” del umbral y, en conjunto con la dinámica espacial, se interpreta como un **tope geométrico** a la expansión de dominios; weak permite patrones más amplios y sensibles a \(\mu\) y \(u\) (detalle en Cap. 5–7 de la tesis).
- **Mean-field vs PDE**: estos equilibrios y \(\lambda\) provienen del **sistema reducido** (nullclinas en \((c,s)\) con \(i\) cuasi-estática). Las afirmaciones sobre patrones y longitudes de correlación \(\xi(t)\) requieren el análisis **PDE** y figuras de correlación; no deben confundirse con estabilidad asintótica del sistema completo en dominio finito.
