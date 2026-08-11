# Índice de redacción: tesis, paper y biblioteca

> **Ubicación canónica** (2026-03-22). Sustituye al archivo raíz `ANALISIS_PAPER_Y_ABSTRACT.md`, eliminado tras consolidación. Aquí están el **mapa de contenidos**, el **resumen ejecutivo** para el paper y **tres borradores de abstract en inglés** para revista. La versión larga original del análisis sigue disponible en el historial de Git si la necesitas.

## Correcciones del asesor (junio 2026) — estado

| ID | Tema | Paper | Tesis |
|----|------|-------|-------|
| C1 | ODE/PDE unificados (forma cuadrática) | `Paper/sections/02_model.tex` | `Tesis/chapters/03_modelo.tex` |
| C2 | Weak Allee eliminado del artículo (solo strong) | `Paper/sections/02_model.tex` | Intro: nota de alcance del artículo derivado |
| C3 | Disclaimer μ_a≡φ_a | `Paper/sections/08_thermodynamics.tex` | `Tesis/chapters/07_discusion.tex` § termodinámica |
| R1–R4 | Bibliografía ampliada | `Paper/bibliography.bib` | `Tesis/references.bib` |
| N1 | Convergencia malla/Δt (texto + tabla pendiente) | `04_spatial.tex`, `06_discussion.tex` | `appendices/B_numerico.tex` |
| N2 | nb=1 + R² intra-serie | `04_spatial.tex`, `06_discussion.tex` | `07_discusion.tex`, Cap. 5 |
| N3 | Tolerancias SNES/Picard | `04_spatial.tex` § Numerical implementation | `appendices/B_numerico.tex` |
| P1 | Escala física ilustrativa (100 µm) | `04_spatial.tex` | `05_dinamica_espacial.tex` |
| P2 | Raíces i*<0 en cuerpo | `04_spatial.tex` | `04_analisis.tex` |
| P3 | Alcance strong Allee | `02_model.tex`, `06_discussion.tex` | Resumen tesis mantiene weak+strong |
| P4 | Notación Hill consolidada | Tab. I + Tab. III | `06_control.tex` |

**Pendiente antes de envío final a revista (requiere simulaciones):** barridos N1 (malla/$\Delta t$) y realizaciones $nb\geq 3$ (N2). En esta revisión se documenta la discretización baseline (Tab. discretización) y la limitación $nb=1$ sin nuevas corridas.

## Dónde quedó cada bloque

| Tema (antiguo § del análisis) | Destino principal |
|-------------------------------|-------------------|
| Resumen / Abstract balanceado | `Tesis/main.tex` (capítulos sin numerar *Resumen* y *Abstract*) |
| Objetivos y aportes a la frontera | `Tesis/chapters/01_introduccion.tex` |
| Lectura de tablas: separatrices inestables | `Tesis/chapters/04_analisis.tex` (tras Tablas weak/strong) |
| Mean-field vs PDE; coste \(C_u=\iint u\,dx\,dy\,dt\) | `Tesis/chapters/06_control.tex` |
| Grado de evidencia; literatura sin modelo sin Allee; limitaciones | `Tesis/chapters/07_discusion.tex` (`\ref{sec:grado_evidencia}`) |
| Trabajo futuro ampliado | `Tesis/chapters/08_conclusiones.tex` |
| Interpretación equilibrios / \(\lambda_{\max}\) | `Biblioteca/markdowns/codigos/steady_states_resumen.md` → *Interpretación para la tesis* |
| Pipeline Newton 3D, SymPy, Jacobiano, ejemplo numérico | `Biblioteca/markdowns/codigos/steady_states_newton_3d_guia.md` (+ PDF `Models/Allee/steady_states/proceso_estados_estacionarios.pdf`) |
| Aportes físicos | `Biblioteca/markdowns/contexto_fisica.md` → *Aportes a la frontera del conocimiento (física)* |
| Aportes biológicos | `Biblioteca/markdowns/contexto_inmunoterapia_cancer.md` → *Aportes a la frontera…* |
| Limitaciones de estudio + código | `Biblioteca/markdowns/codigos/cancer_dynamics_control_issues.md` → §7 |

Referencias cruzadas en LaTeX: `\label{chap:espacial}` (Cap. 5), `\label{chap:discusion}` (Cap. 7), `\label{chap:conclusiones}` (Cap. 8).

### Otros documentos (raíz del repo, ya eliminados)

| Antes | Ahora |
|-------|--------|
| `ESTADO_TERMODINAMICOS.md` | `Models/Allee/README.md` |
| `FORMULARIO_COMPLETADO.md` | `Tesis/avance_institucional.md` |

---

## Resumen ejecutivo (para el paper)

1. Diez escenarios: todos los equilibrios homogéneos tabulados con \(i^*>0\) son linealmente inestables → umbrales como **separatrices**, no atractores estables.
2. En PDE: patrones mesoscópicos, coarsening \(\xi(t)\sim e^{\alpha}t^{1/2}\), \(\mu\) como **selector morfológico**, \(u\) como **intervención geométrica** (\(c_c\), \(c_i\)).
3. **Pendientes explícitos**: medias espaciales largas para el “colapso” con \(u\); métricas \(\iint u\,dx\,dt\) vs beneficio; sensibilidad; corrida control sin Allee; validación experimental.

---

## Borradores de abstract (inglés) — respaldo para revista

### Versión 1 (énfasis resultados)

We investigate a spatial cancer--immune model combining weak/strong Allee effects, nonreciprocal interactions, and immunological control. Our analysis of 10 systematic scenarios reveals that all steady states are unstable (Re λ_max > 0), establishing the Allee threshold as an unstable separator organizing extinction vs proliferation routes rather than a stable attractor. In reaction--diffusion dynamics, we observe rapid self-organization into mesoscopic domains with subdiffusive coarsening ξ(t) ~ e^α t^(1/2), where the prefactor α is modulated by Allee hardness and control protocols. The variational parameter μ acts as a morphological selector that attenuates short-wavelength modes and reduces spatial fragmentation without stabilizing the system, as quantified by correlation-length grids across weak/strong Allee and under/over-threshold regimes. The adaptive control field u(x,t) functions as a geometric intervention that reshapes tumor coherence (measured by c_c correlation length) and cancer--immune co-localization (measured by c_i correlation length), effectively steering trajectories toward near-extinction by locally pushing tumor density below the Allee threshold. Strong Allee imposes a geometric cap on domain sizes, while weak Allee permits broader patterns sensitive to control modulation. Our results establish a threshold-driven organization framework where control protocols select spatial scales and coupling structures rather than creating stable coexistence, providing quantitative metrics for comparing intervention strategies.

### Versión 2 (énfasis frontera)

We present a nonreciprocal cancer--immune model with Allee effects and immunological control, revealing a threshold-driven organization where unstable separators, not stable attractors, govern the dynamics. Analysis of 10 systematic scenarios (weak/strong Allee × μ∈{0,1} × adaptive control) shows all steady states are unstable (Re λ_max = 4.9-22.1), establishing the Allee threshold as an energy-like barrier organizing transient routes. In spatial dynamics, we quantify emergent length scales via correlation functions, finding subdiffusive coarsening ξ(t) ~ e^α t^(1/2) consistent with diffusion-limited domain formation. The variational parameter μ reorganizes spatial spectra as a morphological selector (attenuating high-frequency modes without stabilization), while adaptive control u(x,t) acts as a geometric intervention reshaping tumor coherence and cancer--immune coupling. Strong Allee enforces a geometric cap on domain expansion, contrasting with weak Allee's broader, control-sensitive patterns. Our framework provides quantitative mesoscopic metrics (correlation-length grids) to compare intervention strategies, separating morphological selection (μ) from geometric control (u) effects. These results advance understanding of threshold-mediated pattern formation in cancer--immune competition and establish a bridge between mechanistic control rules and emergent spatial organization.

### Versión 3 (balanceada) — alineada con `Tesis/main.tex` Abstract

We study a spatial cancer--immune model with weak/strong Allee effects, nonreciprocal interactions, and immunological control. Systematic analysis of 10 scenarios reveals that all steady states are unstable (Re λ_max > 0), establishing Allee thresholds as unstable separators organizing extinction vs proliferation routes rather than stable attractors. In reaction--diffusion dynamics, rapid self-organization into mesoscopic domains exhibits subdiffusive coarsening ξ(t) ~ e^α t^(1/2), with the prefactor modulated by Allee hardness and control. The variational parameter μ acts as a morphological selector, attenuating short-wavelength structure and reducing fragmentation without stabilization, as quantified by correlation-length grids. Adaptive control u(x,t) functions as a geometric intervention reshaping tumor coherence (c_c correlations) and cancer--immune co-localization (c_i correlations), steering trajectories toward near-extinction by locally pushing tumor density below the Allee threshold. Strong Allee imposes a geometric cap on domain sizes, while weak Allee permits broader, control-sensitive patterns. Our results establish a threshold-driven framework where control selects spatial scales and coupling structures, providing quantitative mesoscopic metrics to compare intervention strategies and advancing understanding of pattern formation in cancer--immune competition.
