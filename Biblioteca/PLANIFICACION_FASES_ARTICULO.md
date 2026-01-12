# Fases para el Artículo de Investigación



**Título del artículo:** "Dynamics of coupled biological populations: pattern formation in reaction-diffusion systems with Allee effect and immune interactions"

**Revistas objetivo:** ¿En donde se planea publicar?

---

## FASE 1: Definición del Problema de Investigación Físico

### ¿Qué problema físico quiero resolver?

#### Objetivos:
- [ ] Documentar claramente el problema físico-matemático
- [ ] Identificar las limitaciones de modelos previos en física
- [ ] Establecer la necesidad de un nuevo enfoque desde perspectiva de física estadística y sistemas dinámicos


**1.1 Problema Físico-Biológico Fundamental:**
- **Sistemas de reacción-difusión de poblaciones biológicas acopladas**: Estudio de sistemas de campos escalares que representan poblaciones biológicas (células cancerosas, células sanas, sistema inmune) que exhiben dinámicas espaciotemporales complejas, formación de patrones y transiciones de fase.

- **Efecto Allee como no-linealidad crítica en poblaciones biológicas**: 
  - Modela umbral crítico en sistemas cooperativos biológicos
  - Genera múltiples estados estacionarios y bifurcaciones
  - Conecta con teoría de transiciones de fase en sistemas extendidos
  - Relacionado con fenómenos de extinción y persistencia poblacional en física estadística
  - Relevante para entender dinámicas de poblaciones pequeñas

- **Interacciones inmunes y control espaciotemporal**: 
  - Estudio de sistemas bajo control externo (parámetro u) que modela intervención inmunológica
  - Control puede ser constante (u = constante), temporal (u(t)) o espaciotemporal (u(x,y,t))
  - Análisis de respuesta del sistema a perturbaciones inmunes mediante inmunoterapia
  - Conexión con teoría de control en sistemas dinámicos biológicos
  - Modelado de interacciones complejas entre poblaciones biológicas
  - Escenarios de inmunoterapia: pasiva (u=0), constante, adaptativa espacial, adaptativa temporal
- **Parámetro μ de encendido/apagado modelo energía libre**:
  - μ = 0: Aproximación biológica directa (modelado fenomenológico)
  - μ = 1 (o > 0): Modelo derivado desde energía libre tipo C de Halperin-Hohenberg (enfoque físico de transiciones de fase)
  - Permite comparar ambos enfoques y sus implicaciones físicas

**1.2 Problema Matemático-Físico:**
- Resolver sistemas de ecuaciones diferenciales parciales no lineales acopladas en 2D que modelan poblaciones biológicas
- Analizar formación de patrones espaciotemporales: patrones de Turing, ondas viajeras, caos espacial
- Estudiar el efecto de parámetros de control (u) que modelan interacciones inmunes en la evolución del sistema y transiciones de fase
  - Control constante: u = constante (inmunoterapia sistémica uniforme)
  - Control temporal: u(t) (protocolos de tratamiento con dosis variables)
  - Control espaciotemporal: u(x,y,t) (inmunoterapia adaptativa y localizada)
- Análisis de estabilidad lineal y no lineal de estados estacionarios de poblaciones biológicas
- Caracterización de escalas espaciales y temporales características en sistemas biológicos
- Conexión con modelo de energía libre tipo C de Halperin-Hohenberg mediante parámetro μ (encendido/apagado)
  - μ = 0: Aproximación biológica directa
  - μ = 1 (o > 0): Modelo derivado desde energía libre

**1.3 Problema Computacional-Físico:**
- Implementación eficiente usando método de elementos finitos (formulación débil) con FEniCS para sistemas biológicos
- Análisis de grandes volúmenes de datos espaciotemporales usando herramientas de física estadística aplicadas a poblaciones biológicas
- Cuantificación de formación de patrones mediante análisis espectral y funciones de correlación
- Cálculo de longitudes de correlación y escalas características en sistemas de poblaciones acopladas
- Implementación de control adaptativo espaciotemporal u(x,y,t) que responde al estado local del sistema
- Comparación entre formulación desde energía libre (modelo C de Halperin-Hohenberg) y aproximación biológica directa mediante parámetro μ
  - μ = 0: Aproximación biológica directa (modelado fenomenológico)
  - μ = 1 (o > 0): Modelo derivado desde energía libre tipo C de Halperin-Hohenberg

- ✅ Documentación en `Biblioteca/markdowns/contexto_inmunoterapia_cancer.md`
- ✅ Documentación en `Biblioteca/markdowns/contexto_fisica.md`
- ✅ Referencias bibliográficas en carpeta `Biblioteca/`
- ✅ **Notas personales:** `Biblioteca/Mis notas/Tesis_Phd_EFSG.pdf` (tesis completa con resultados y análisis)
- ✅ **Notas personales:** `Biblioteca/Mis notas/Notas_para_tesis.pdf` (notas de trabajo previo)
- ✅ **Poster:** `Biblioteca/Mis notas/poster_CNF_2024.pdf` (resultados presentados en conferencia)

#### Referencias Bibliográficas Específicas para esta Fase:
- **`The_role_of_mathematical_modelling_in_understandin.pdf`**: Papel del modelado matemático en sistemas biológicos
- **`thomas2005.pdf`**, **`thomas2006.pdf`**: Modelos de dinámicas poblacionales y sistemas de reacción-difusión
- **`logistic_generaliced.pdf`**: Modelos logísticos generalizados aplicados a poblaciones biológicas
- **`AlleeEffect_2008.01692.pdf`**: Fundamentos teóricos del efecto Allee en sistemas poblacionales
- **`Wang_allee_extintion_2019.pdf`**: Extinción y persistencia poblacional con efecto Allee
- **`control_theory_inmune.pdf`**: Teoría de control aplicada a sistemas inmunológicos
- **`Cornel_2020_mhc.pdf`**: Complejo mayor de histocompatibilidad y respuestas inmunes

#### Entregables:
- [ ] Sección de Introducción del artículo enfocada en física-biológica
- [ ] Revisión crítica de literatura de física aplicada a biología (reacción-difusión en poblaciones, sistemas de Turing biológicos, física estadística de poblaciones)
- [ ] Justificación del modelo desde perspectiva de física teórica aplicada a poblaciones biológicas
- [ ] Conexión con modelos clásicos: Fisher-Kolmogorov (poblaciones), sistemas de Turing (formación de patrones biológicos), modelos de interacciones poblacionales
- [ ] Citas apropiadas a referencias bibliográficas relevantes

---

## FASE 2: Estado del Arte en Física y Obstáculos Iniciales

### ¿Qué se había hecho antes en física y cuáles fueron los obstáculos iniciales?

#### Objetivos:
- [ ] Revisar literatura de física sobre sistemas de reacción-difusión
- [ ] Revisar literatura sobre efecto Allee en sistemas físicos y biológicos
- [ ] Revisar literatura sobre control en sistemas dinámicos no lineales
- [ ] Documentar los obstáculos técnicos y metodológicos desde perspectiva física
- [ ] Identificar las contribuciones previas en física teórica/computacional

**2.1 Trabajos Previos en Física sobre Sistemas de Reacción-Difusión Biológicos:**
- [ ] Revisar literatura sobre modelos Fisher-Kolmogorov aplicados a poblaciones biológicas
- [ ] Revisar: **`thomas2005.pdf`**, **`thomas2006.pdf`** - Modelos de dinámicas poblacionales y sistemas de reacción-difusión
- [ ] Revisar: **`The_role_of_mathematical_modelling_in_understandin.pdf`** - Papel del modelado matemático en sistemas biológicos
- [ ] Revisar: **`logistic_generaliced.pdf`** - Modelos logísticos generalizados aplicados a poblaciones
- [ ] Revisar literatura sobre sistemas de Turing y formación de patrones en sistemas biológicos
- [ ] Revisar literatura sobre dinámicas de poblaciones acopladas
- [ ] Revisar literatura sobre ondas viajeras en sistemas biológicos extendidos
- [ ] Revisar literatura sobre caos espacial en sistemas de poblaciones biológicas
- [ ] Revisar: **`10.3389@fphy.2020.00377.pdf`** - Física aplicada a sistemas biológicos (si aplica)

**2.2 Trabajos Previos sobre Efecto Allee en Poblaciones Biológicas:**
- [ ] Revisar: **`AlleeEffect_2008.01692.pdf`** - Fundamentos teóricos del efecto Allee (perspectiva física-biológica)
- [ ] Revisar: **`Wang_allee_extintion_2019.pdf`** - Extinción y persistencia poblacional con efecto Allee
- [ ] Revisar: **`Kaitlyn_e_cancer_allee_2019.pdf`** - Aplicaciones del efecto Allee en cáncer
- [ ] Revisar: **`Marcello_Allee_cancer_terapy_2020.pdf`** - Terapia y efecto Allee en cáncer
- [ ] Revisar: **`Philip_g_autocrine_allee_efect2022.pdf`** - Efectos autocrinos y efecto Allee
- [ ] Revisar literatura sobre umbrales críticos en sistemas cooperativos biológicos
- [ ] Revisar literatura sobre extinción y persistencia poblacional en física estadística

**2.3 Trabajos Previos sobre Interacciones Inmunes y Control en Sistemas Biológicos:**
- [ ] Revisar: **`control_theory_inmune.pdf`** - Teoría de control aplicada a sistemas inmunológicos (control en sistemas biológicos dinámicos)
- [ ] Revisar: **`Cornel_2020_mhc.pdf`** - Complejo mayor de histocompatibilidad y respuestas inmunes
- [ ] Revisar: **`Cordula_r_allee_virus_2022.pdf`** - Interacciones virus-cáncer-inmunidad con efecto Allee
- [ ] Revisar literatura sobre control de sistemas no lineales biológicos
- [ ] Revisar literatura sobre respuesta de poblaciones biológicas a perturbaciones inmunes

**2.4 Trabajos Previos sobre Análisis Espectral y Correlaciones:**
- [ ] Revisar: **`Nullclines.pdf`** - Análisis de nullclines y estados estacionarios
- [ ] Revisar literatura sobre funciones de correlación en física estadística
- [ ] Revisar literatura sobre análisis espectral en sistemas espaciales extendidos
- [ ] Revisar literatura sobre longitudes de correlación y escalas características

**2.4 Obstáculos Iniciales Enfrentados:**

*Obstáculos Técnicos:*
- [ ] Documentar dificultades en la implementación numérica (convergencia de solvers)
- [ ] Documentar desafíos en el manejo de condiciones iniciales
- [ ] Documentar problemas de estabilidad numérica con diferentes parámetros

*Obstáculos Metodológicos-Físicos:*
- [ ] Documentar la selección de parámetros físicamente relevantes (números adimensionales)
- [ ] Documentar la adimensionalización del sistema
- [ ] Documentar la elección entre Weak vs Strong Allee desde perspectiva de física (bifurcaciones)
- [ ] Documentar el análisis de escalas características (tiempo de difusión vs tiempo de reacción)
- [ ] Documentar el cálculo del número de Damköhler y su significado físico

*Obstáculos Computacionales:*
- [ ] Documentar tiempos de ejecución y optimizaciones realizadas
- [ ] Documentar manejo de memoria para simulaciones largas
- [ ] Documentar estrategias de paralelización (si aplica)

- ✅ Archivos PDF en `Biblioteca/` (13 referencias principales)
- ✅ Código de implementación en `new_model_10112024/`
- ✅ Documentación técnica en `Biblioteca/markdowns/contexto_tecnico_software.md`
- ✅ Documento de referencias: `REFERENCIAS_BIBLIOGRAFICAS_POR_PREGUNTA.md`
- ✅ **Notas personales:** `Biblioteca/Mis notas/Tesis_Phd_EFSG.pdf` (revisar sección de estado del arte y obstáculos)
- ✅ **Notas personales:** `Biblioteca/Mis notas/Notas_para_tesis.pdf` (obstáculos y soluciones documentados)

#### Entregables:
- [ ] Sección de "Estado del Arte" enfocada en física teórica/computacional
- [ ] Tabla comparativa de modelos previos vs modelo propuesto (desde perspectiva física)
- [ ] Discusión de limitaciones de trabajos previos en física con citas específicas
- [ ] Documentación de obstáculos y soluciones desde perspectiva de física computacional
- [ ] Conexión con literatura clásica de física (Fisher, Turing, Ginzburg-Landau)
- [ ] Citas apropiadas a todas las referencias relevantes de la biblioteca

---

## FASE 3: Originalidad y Relevancia en Física

### ¿Qué es lo original y relevante de mi trabajo desde perspectiva física?

#### Objetivos:
- [ ] Identificar las contribuciones novedosas desde perspectiva de física teórica
- [ ] Comparar con trabajos previos en física para destacar diferencias
- [ ] Establecer la relevancia para física estadística y sistemas dinámicos
- [ ] Justificar la importancia del enfoque metodológico en física computacional


**3.1 Contribuciones Originales en Física:**

*Contribución 1: Sistema de Reacción-Difusión de Poblaciones Biológicas Acopladas*
- [ ] Sistema de tres poblaciones biológicas acopladas (células cancerosas, células sanas, sistema inmune) con efecto Allee
- [ ] Comparar con sistemas de dos poblaciones (predator-prey, activator-inhibitor)
- [ ] Documentar nuevas dinámicas emergentes y formación de patrones en sistemas de tres poblaciones
- [ ] Análisis de acoplamiento no lineal entre poblaciones biológicas

*Contribución 2: Interacciones Inmunes y Control Espaciotemporal en Poblaciones Biológicas*
- [ ] Implementación de control externo (parámetro u) que modela interacciones inmunes en sistema de reacción-difusión
  - Control constante: u = constante (inmunoterapia sistémica)
  - Control temporal: u(t) (protocolos de tratamiento)
  - Control espaciotemporal adaptativo: u(x,y,t) = f(c(x,y,t), i(x,y,t)) (inmunoterapia de precisión)
- [ ] Análisis de respuesta de poblaciones biológicas a control inmunológico: transiciones de fase, cambios en estabilidad
- [ ] Estudio de eficacia del control desde perspectiva de física estadística aplicada a poblaciones
- [ ] Conexión con teoría de control en sistemas dinámicos biológicos no lineales
- [ ] Escenarios de inmunoterapia: cómo diferentes estrategias de control u afectan la dinámica del cáncer

*Contribución 3: Análisis Espectral y Funciones de Correlación*
- [ ] Análisis de Fourier para identificar escalas espaciales características (números de onda k)
- [ ] Cálculo de longitudes de correlación ξ usando funciones de correlación
- [ ] Análisis de correlaciones cruzadas entre campos acoplados
- [ ] Caracterización de escalas características y su evolución temporal
- [ ] Conexión con física estadística y teoría de campos

*Contribución 4: Método Numérico y Física Computacional*
- [ ] Implementación con elementos finitos usando formulación débil (FEniCS)
- [ ] Manejo robusto de sistemas no lineales acoplados
- [ ] Validación numérica: conservación, estabilidad, convergencia
- [ ] Análisis de escalas numéricas vs escalas físicas
- [ ] Conexión con modelo de energía libre tipo C de Halperin-Hohenberg mediante parámetro μ (encendido/apagado)
  - μ = 0: Aproximación biológica directa (modelado fenomenológico)
  - μ = 1 (o > 0): Modelo derivado desde energía libre tipo C de Halperin-Hohenberg (enfoque físico de transiciones de fase)
- [ ] Comparación entre ambos enfoques y sus implicaciones físicas

**3.2 Relevancia para Física Teórica Aplicada a Biología:**
- [ ] Contribución a teoría de sistemas de reacción-difusión no lineales en poblaciones biológicas
- [ ] Análisis de bifurcaciones y transiciones de fase en sistemas biológicos
- [ ] Estudio de formación de patrones espaciales emergentes (Turing, ondas, caos) en poblaciones biológicas
- [ ] Conexión con física estadística de sistemas biológicos fuera de equilibrio

**3.3 Relevancia para Física Computacional Aplicada a Biología:**
- [ ] Metodología numérica aplicable a otros sistemas de reacción-difusión biológicos
- [ ] Herramientas de análisis (espectral, correlaciones) reutilizables para poblaciones biológicas
- [ ] Validación de métodos numéricos para sistemas de poblaciones biológicas no lineales acoplados

- ✅ Implementación completa en `new_model_10112024/`
- ✅ Dos variantes del modelo (Weak y Strong Allee)
- ✅ Sistema de control inmunológico implementado
- ✅ Referencias para comparación: `thomas2005.pdf`, `thomas2006.pdf` (sistemas de dos poblaciones)
- ✅ Referencias sobre efecto Allee: `AlleeEffect_2008.01692.pdf`, `Kaitlyn_e_cancer_allee_2019.pdf`, `Marcello_Allee_cancer_terapy_2020.pdf`
- ✅ Referencias sobre control: `control_theory_inmune.pdf`, `Cordula_r_allee_virus_2022.pdf`
- ✅ **Notas personales:** `Biblioteca/Mis notas/Tesis_Phd_EFSG.pdf` (revisar sección de contribuciones y originalidad)
- ✅ **Poster:** `Biblioteca/Mis notas/poster_CNF_2024.pdf` (contribuciones destacadas en presentación)

#### Entregables:
- [ ] Sección de "Contribuciones" enfocada en física
- [ ] Tabla comparativa con trabajos previos en física (usar referencias de biblioteca)
- [ ] Justificación de la relevancia para física teórica/computacional
- [ ] Conexión con problemas abiertos en física de sistemas no lineales
- [ ] Comparación explícita con trabajos previos citando referencias específicas

---

## FASE 4: Resultados Cuantitativos en Física

### ¿Cuáles son los resultados cuantitativos físicos obtenidos?

#### Objetivos:
- [ ] Inventariar todos los resultados generados
- [ ] Organizar resultados por tipo y categoría
- [ ] Documentar métricas cuantitativas clave
- [ ] Crear un catálogo completo de resultados

#### 4a) Inventario de Resultados del Universo Completo

**Tipos de Resultados Generados:**

**A. Campos Espaciales Temporales (Campos Escalares):**
- [ ] Matrices de campo c(x,y,t): `matrix_c_{time}_nb_{block}.txt`
- [ ] Matrices de campo s(x,y,t): `matrix_s_{time}_nb_{block}.txt`
- [ ] Matrices de campo i(x,y,t): `matrix_i_{time}_nb_{block}.txt`
- [ ] Visualizaciones: `fields_block_{block}_step_{time}.png`
- **Métricas cuantitativas físicas:**
  - Evolución temporal de valores promedio: ⟨c(t)⟩, ⟨s(t)⟩, ⟨i(t)⟩
  - Evolución temporal de valores extremos: c_max(t), c_min(t)
  - Distribución espacial: varianza espacial σ²(t)
  - Tasas de cambio: ∂⟨c⟩/∂t, ∂⟨s⟩/∂t, ∂⟨i⟩/∂t
  - Números adimensionales: número de Damköhler Da = τ_D/τ_R

**B. Análisis Espectral (Transformada de Fourier):**
- [ ] Espectros de potencia 2D: P(k_x, k_y, t) = |FFT[φ(x,y,t)]|²
- [ ] Archivos: `power_spectrum_{field}_{time}.png`
- **Métricas cuantitativas físicas:**
  - Números de onda dominantes: k_dominante (escala espacial característica)
  - Longitud característica: λ = 2π/k_dominante
  - Distribución de energía espectral: E(k) = ∫ P(k,θ) dθ
  - Evolución temporal de escalas características: k_dominante(t)
  - Comparación entre campos: relación entre escalas de c, s, i
  - Análisis de modos inestables (si aplica para patrones de Turing)

**C. Funciones de Correlación Espacial (Física Estadística):**
- [ ] Correlaciones cruzadas: C_{ab}(r,t) = ⟨a(x)·b(x+r)⟩
  - Archivos: `correlacion_{field1}_{field2}_{time}.txt`
  - Correlación cruzada c-s: C_{cs}(r,t)
  - Correlación cruzada c-i: C_{ci}(r,t)
  - Correlación cruzada s-i: C_{si}(r,t)
- [ ] Autocorrelaciones: C_{aa}(r,t) = ⟨a(x)·a(x+r)⟩
  - Autocorrelación de campo c: C_{cc}(r,t)
  - Autocorrelación de campo s: C_{ss}(r,t)
  - Autocorrelación de campo i: C_{ii}(r,t)
- [ ] Longitudes de correlación: ξ (distancia donde C(ξ) = C(0)/e)
  - Archivos: `corr_length_real_inverse_nb_{block}_{field1}_{field2}.txt`
- **Métricas cuantitativas físicas:**
  - Longitud de correlación ξ(t) para cada par de campos
  - Evolución temporal de longitudes de correlación: ξ(t)
  - Exponentes de decaimiento: C(r) ~ r^(-α) o C(r) ~ exp(-r/ξ)
  - Comparación entre autocorrelaciones y correlaciones cruzadas
  - Escala de coherencia espacial del sistema

**D. Análisis de Estados Estacionarios y Estabilidad:**
- [ ] Análisis de nullclines: F₁(c,s) = 0, F₂(c,s) = 0
- [ ] Puntos de equilibrio (estados estacionarios): (c*, s*, i*)
- [ ] Análisis de estabilidad lineal: autovalores λ de matriz jacobiana J
- [ ] Análisis de bifurcaciones: cambios cualitativos al variar parámetros
- **Métricas cuantitativas físicas:**
  - Valores de puntos fijos: (c*, s*, i*) para cada escenario
  - Autovalores λ₁, λ₂, λ₃ y su parte real/imaginaria
  - Tipo de estabilidad: nodo estable/inestable, foco, silla
  - Diagrama de fases: cuencas de atracción
  - Comparación entre Weak y Strong Allee (efecto en bifurcaciones)
  - Comparación entre μ=0 y μ>0 (efecto del control en estabilidad)
  - Parámetros críticos donde ocurren bifurcaciones

**E. Comparaciones entre Escenarios (Análisis Paramétrico):**
- [ ] Comparación Weak Allee vs Strong Allee
- [ ] Comparación u=0 vs u>0 (sistema sin control vs con control externo)
- [ ] Comparación diferentes tipos de control:
  - Control constante: u = constante
  - Control temporal: u(t) (protocolos de tratamiento)
  - Control espaciotemporal adaptativo: u(x,y,t) = f(c,i)
- [ ] Comparación modelo energía libre (μ=1 o >0) vs aproximación biológica (μ=0)
- [ ] Comparación diferentes valores de parámetros (análisis de sensibilidad)
- **Métricas cuantitativas físicas:**
  - Diferencias en valores finales de campos: Δc_final, Δs_final, Δi_final
  - Diferencias en longitudes de correlación: Δξ
  - Diferencias en escalas espaciales características: Δk_dominante
  - Eficacia del control: reducción relativa de campo c para diferentes estrategias u
  - Cambios en estabilidad: transiciones de fase inducidas por control
  - Comparación entre eficacia de control constante vs adaptativo
  - Números adimensionales comparativos: Da, números de Reynolds (si aplica)

**F. Análisis de Formas Funcionales y Escalado:**
- [ ] Archivos: `resultados_regresion.json`
- [ ] Ajustes de funciones a correlaciones: C(r) ~ f(r)
- [ ] Análisis de escalado: comportamiento asintótico
- **Métricas cuantitativas físicas:**
  - Parámetros de ajuste: exponentes críticos, longitudes características
  - Coeficientes de determinación (R²) para validar ajustes
  - Formas funcionales: exponencial C(r) ~ exp(-r/ξ), potencia C(r) ~ r^(-α)
  - Exponentes críticos α (si hay comportamiento de potencia)
  - Validación de leyes de escalado universales

#### 4b) Documento de Inventario de Resultados

**Estructura del Inventario:**

```markdown
# Inventario de Resultados Cuantitativos

## 1. Resultados por Tipo de Análisis

### 1.1 Dinámicas Temporales
- Total de simulaciones ejecutadas: [contar]
- Rango de tiempos simulados: [t_min, t_max]
- Pasos temporales: [dt]
- Bloques de simulación: [nb]

### 1.2 Campos Espaciales
- Resolución espacial: [nodes_in_xaxis × nodes_in_yaxis]
- Tamaño del dominio: [space_size × space_size]
- Archivos de matrices generados: [contar]
- Visualizaciones generadas: [contar]

### 1.3 Análisis Espectral
- Espectros de potencia calculados: [contar]
- Escalas espaciales identificadas: [listar]
- Archivos de análisis espectral: [contar]

### 1.4 Correlaciones
- Correlaciones cruzadas calculadas: [contar]
- Longitudes de correlación calculadas: [contar]
- Archivos de correlación: [contar]

### 1.5 Estados Estacionarios
- Puntos de equilibrio identificados: [listar]
- Análisis de estabilidad realizados: [contar]

## 2. Resultados por Escenario

### 2.1 Weak Allee, μ=0
- [Listar resultados específicos]

### 2.2 Weak Allee, μ>0
- [Listar resultados específicos]

### 2.3 Strong Allee, μ=0
- [Listar resultados específicos]

### 2.4 Strong Allee, μ>0
- [Listar resultados específicos]

## 3. Métricas Cuantitativas Clave

### 3.1 Densidades
- Densidad máxima de cáncer alcanzada: [valor]
- Densidad mínima de cáncer alcanzada: [valor]
- Densidad promedio final: [valor]

### 3.2 Longitudes de Correlación
- Longitud de correlación c-s: [valor ± desviación]
- Longitud de correlación c-i: [valor ± desviación]
- Longitud de correlación s-i: [valor ± desviación]

### 3.3 Escalas Espaciales
- Escala dominante en espectro de cáncer: [valor]
- Escala dominante en espectro de células sanas: [valor]
- Escala dominante en espectro de sistema inmune: [valor]

### 3.4 Eficacia del Control
- Reducción de densidad tumoral con μ>0: [porcentaje]
- Cambio en longitud de correlación con control: [valor]
```

- ✅ Notebooks que generan resultados en `new_model_10112024/`
- ✅ Sistema Django para gestionar experimentos y resultados
- ✅ Archivos de resultados guardados (según rutas en notebooks)
- ✅ Referencias para validación: `AlleeEffect_2008.01692.pdf`, `Wang_allee_extintion_2019.pdf`, `Kaitlyn_e_cancer_allee_2019.pdf`
- ✅ Referencias para comparación: `Marcello_Allee_cancer_terapy_2020.pdf`, `control_theory_inmune.pdf`
- ✅ Referencias para métodos: `Nullclines.pdf`, `thomas2005.pdf`, `thomas2006.pdf`
- ✅ **Notas personales - Resultados:** `Biblioteca/Mis notas/Tesis_Phd_EFSG.pdf` (resultados cuantitativos ya documentados)
- ✅ **Notas personales - Correlaciones:** `Biblioteca/Mis notas/Correlations.pdf` (análisis de correlaciones ya realizado)
- ✅ **Poster:** `Biblioteca/Mis notas/poster_CNF_2024.pdf` (resultados clave presentados)
- ✅ **Notas de trabajo:** `Biblioteca/Mis notas/Notas_para_tesis.pdf` (resultados y métricas documentadas)

#### Entregables:
- [ ] Documento completo: `INVENTARIO_RESULTADOS.md`
- [ ] Tablas resumen de resultados cuantitativos
- [ ] Gráficos y figuras organizadas por categoría
- [ ] Base de datos de resultados (posiblemente usando Django)
- [ ] Comparación de resultados con trabajos previos (usar referencias de biblioteca)
- [ ] Validación de métodos numéricos comparando con literatura

---

## FASE 5: Interpretación Física de Resultados

### ¿Qué significan esos resultados físicamente?

#### Objetivos:
- [ ] Interpretar resultados en contexto de física teórica
- [ ] Conectar resultados con teoría de sistemas dinámicos y física estadística
- [ ] Explicar las implicaciones físicas de los hallazgos
- [ ] Validar o refutar hipótesis físicas iniciales
- [ ] Conectar con fenómenos físicos conocidos (patrones de Turing, ondas, etc.)


**5.1 Interpretación Física de Dinámicas Temporales:**
- [ ] ¿Qué regímenes dinámicos se observan? (transitorio, estacionario, oscilatorio, caótico)
- [ ] ¿Hay oscilaciones? ¿Cuál es su frecuencia característica?
- [ ] ¿Cómo afecta el control (μ) a las dinámicas? ¿Hay transiciones de fase?
- [ ] ¿Qué diferencias físicas hay entre Weak y Strong Allee? (bifurcaciones diferentes)
- [ ] Análisis de escalas temporales: τ_D vs τ_R, número de Damköhler

**5.2 Interpretación Física de Patrones Espaciales:**
- [ ] ¿Qué estructuras espaciales emergen? (patrones de Turing, ondas viajeras, frentes, espirales)
- [ ] ¿Hay formación de patrones de Turing? ¿Cuáles son los modos inestables?
- [ ] ¿Cómo se relacionan las escalas espaciales con parámetros físicos? (D, r, números adimensionales)
- [ ] ¿Qué tipo de ondas se observan? (velocidad, perfil, estabilidad)
- [ ] ¿Hay caos espacial? ¿Cómo se caracteriza?

**5.3 Interpretación Física de Correlaciones:**
- [ ] ¿Qué significan las correlaciones positivas/negativas entre campos?
- [ ] ¿Cómo evolucionan las correlaciones en el tiempo? (dinámica de correlaciones)
- [ ] ¿Qué implican las longitudes de correlación ξ? (escala de coherencia espacial)
- [ ] ¿Cómo se relacionan con la organización espacial del sistema?
- [ ] ¿Hay comportamiento crítico? (exponentes críticos, escalado)

**5.4 Interpretación Física de Estados Estacionarios:**
- [ ] ¿Qué estados de equilibrio son físicamente accesibles?
- [ ] ¿Qué condiciones llevan a extinción de campos?
- [ ] ¿Qué condiciones llevan a coexistencia de campos?
- [ ] ¿Qué condiciones llevan a crecimiento ilimitado?
- [ ] Análisis de estabilidad: tipos de puntos fijos (nodos, focos, sillas)

**5.5 Interpretación Física del Control:**
- [ ] ¿Es efectivo el control (u>0)? ¿Cómo se cuantifica?
- [ ] Comparación de eficacia entre diferentes estrategias de control:
  - Control constante u = constante vs control adaptativo u(x,y,t)
  - Control temporal u(t) vs control espaciotemporal u(x,y,t)
- [ ] ¿Bajo qué condiciones físicas es más efectivo cada tipo de control?
- [ ] ¿Qué parámetros optimizan la respuesta del sistema para inmunoterapia?
- [ ] ¿Hay efectos secundarios en otros campos (células sanas, sistema inmune)?
- [ ] ¿El control induce transiciones de fase?
- [ ] ¿Cómo afecta la inmunoterapia adaptativa u(x,y,t) a la dinámica espacial del cáncer?
- [ ] Comparación física entre modelo energía libre (μ=1 o >0) y aproximación biológica (μ=0)

**5.6 Comparación Física Weak vs Strong Allee:**
- [ ] ¿Qué diferencias cualitativas y cuantitativas se observan?
- [ ] ¿Cómo afectan las bifurcaciones? (diferentes diagramas de fase)
- [ ] ¿Qué implicaciones tienen para física estadística? (umbral crítico diferente)

- ✅ Resultados de simulaciones
- ✅ Documentación biológica en `Biblioteca/markdowns/`
- ✅ Análisis implementados en notebooks
- ✅ Referencias para interpretación: `AlleeEffect_2008.01692.pdf`, `Wang_allee_extintion_2019.pdf`, `Kaitlyn_e_cancer_allee_2019.pdf`
- ✅ Referencias para interpretación inmunológica: `control_theory_inmune.pdf`, `Cornel_2020_mhc.pdf`, `Cordula_r_allee_virus_2022.pdf`
- ✅ Referencias para interpretación de patrones: `The_role_of_mathematical_modelling_in_understandin.pdf`, `thomas2005.pdf`, `thomas2006.pdf`
- ✅ Referencias para interpretación de estados: `Nullclines.pdf`
- ✅ **Notas personales - Interpretaciones:** `Biblioteca/Mis notas/Tesis_Phd_EFSG.pdf` (interpretaciones ya realizadas de resultados)
- ✅ **Notas personales - Correlaciones:** `Biblioteca/Mis notas/Correlations.pdf` (interpretación física de correlaciones)
- ✅ **Notas de trabajo:** `Biblioteca/Mis notas/Notas_para_tesis.pdf` (interpretaciones y análisis previos)

#### Entregables:
- [ ] Sección de "Resultados" con interpretación física
- [ ] Sección de "Discusión" conectando resultados con teoría física
- [ ] Figuras con interpretación física (patrones, escalas, correlaciones)
- [ ] Tablas resumen con métricas físicas (ξ, k, λ, Da, etc.)
- [ ] Comparación con predicciones teóricas usando referencias de biblioteca
- [ ] Interpretación de resultados comparando con trabajos previos citados

---

## FASE 6: Impacto e Implicaciones en Física

### ¿Qué impacto o implicaciones tiene mi investigación para física?

#### Objetivos:
- [ ] Establecer el impacto en física teórica/computacional
- [ ] Identificar direcciones futuras de investigación en física
- [ ] Discutir limitaciones y extensiones posibles desde perspectiva física
- [ ] Conectar con problemas abiertos en física de sistemas no lineales


**6.1 Impacto en Física de Sistemas de Reacción-Difusión Biológicos:**
- [ ] Contribución a teoría de sistemas de reacción-difusión no lineales de poblaciones biológicas acopladas
- [ ] Nuevos insights sobre formación de patrones en sistemas de tres poblaciones biológicas
- [ ] Metodología numérica aplicable a otros sistemas biológicos
- [ ] Herramientas de análisis (espectral, correlaciones) reutilizables en física aplicada a biología

**6.2 Impacto en Física Estadística de Poblaciones Biológicas:**
- [ ] Comprensión de correlaciones en sistemas de poblaciones biológicas fuera de equilibrio
- [ ] Análisis de escalas características y comportamiento crítico en poblaciones acopladas
- [ ] Conexión entre escalas espaciales y temporales en sistemas biológicos extendidos
- [ ] Estudio de transiciones de fase en sistemas de poblaciones biológicas no lineales

**6.3 Impacto en Teoría de Sistemas Dinámicos Biológicos:**
- [ ] Análisis de bifurcaciones en sistemas de reacción-difusión de poblaciones biológicas
- [ ] Estudio de control inmunológico en sistemas no lineales biológicos
- [ ] Comprensión de efectos de umbral crítico (Allee) en dinámicas poblacionales
- [ ] Análisis de estabilidad y cuencas de atracción en poblaciones biológicas acopladas

**6.4 Limitaciones del Trabajo Actual (Perspectiva Física):**
- [ ] Modelo 2D (extensión a 3D para mayor realismo físico)
- [ ] Parámetros constantes (heterogeneidad espacial no incluida en algunos parámetros)
- [ ] Sin efectos estocásticos (ruido, fluctuaciones)
- [ ] Simplificaciones en acoplamientos no lineales
- [ ] Dominio finito (efectos de frontera)
- [ ] Control u implementado como función determinista (extensión a control estocástico posible)
- [ ] Comparación limitada entre modelo energía libre (μ>0) y aproximación biológica (μ=0) (más análisis necesarios)

**6.5 Direcciones Futuras en Física-Biología:**
- [ ] Extensión a 3D (sistemas espaciales completos de poblaciones biológicas)
- [ ] Incorporación de ruido estocástico (ecuaciones de Langevin para poblaciones)
- [ ] Análisis de efectos de frontera y condiciones de contorno en sistemas biológicos
- [ ] Estudio de comportamiento crítico y exponentes críticos en poblaciones
- [ ] Optimización de control inmunológico usando teoría de control óptimo
  - Optimización de estrategias u(t) y u(x,y,t) para maximizar eficacia terapéutica
  - Minimización de efectos secundarios mediante control adaptativo
- [ ] Análisis de caos espacial y transición a turbulencia en poblaciones biológicas
- [ ] Validación con datos experimentales de poblaciones celulares
- [ ] Estudio comparativo más profundo entre modelo energía libre (μ>0, Halperin-Hohenberg) y aproximación biológica (μ=0)
- [ ] Desarrollo de estrategias de control u basadas en aprendizaje automático
- [ ] Análisis de resistencia a inmunoterapia mediante evolución de parámetros

**6.6 Aplicaciones en Otros Sistemas Biológicos:**
- [ ] Sistemas ecológicos (dinámicas poblacionales de especies)
- [ ] Sistemas de crecimiento celular (patrones de desarrollo)
- [ ] Sistemas de interacciones predador-presa con efecto Allee
- [ ] Sistemas de respuesta inmune en diferentes contextos biológicos

- ✅ Trabajo completo implementado
- ✅ Sistema extensible (Django, notebooks modulares)
- ✅ Documentación de limitaciones en markdowns
- ✅ Referencias para impacto en cáncer: `Kaitlyn_e_cancer_allee_2019.pdf`, `Marcello_Allee_cancer_terapy_2020.pdf`, `Philip_g_autocrine_allee_efect2022.pdf`
- ✅ Referencias para impacto en inmunoterapia: `control_theory_inmune.pdf`, `Cornel_2020_mhc.pdf`, `Cordula_r_allee_virus_2022.pdf`
- ✅ Referencias para impacto general: `The_role_of_mathematical_modelling_in_understandin.pdf`, `10.3389@fphy.2020.00377.pdf`
- ✅ Referencias para direcciones futuras: `thomas2005.pdf`, `thomas2006.pdf`, `logistic_generaliced.pdf`
- ✅ **Notas personales - Impacto:** `Biblioteca/Mis notas/Tesis_Phd_EFSG.pdf` (discusión de impacto e implicaciones)
- ✅ **Poster:** `Biblioteca/Mis notas/poster_CNF_2024.pdf` (impacto presentado en conferencia)
- ✅ **Notas de trabajo:** `Biblioteca/Mis notas/Notas_para_tesis.pdf` (direcciones futuras documentadas)

#### Entregables:
- [ ] Sección de "Conclusiones" enfocada en contribuciones físicas
- [ ] Sección de "Trabajo Futuro" con direcciones en física (conectando con referencias)
- [ ] Sección de "Limitaciones" desde perspectiva física
- [ ] Conexión con problemas abiertos en física teórica/computacional
- [ ] Discusión de impacto citando trabajos relevantes de la biblioteca
- [ ] Direcciones futuras basadas en limitaciones de trabajos previos citados

---

## PLAN DE EJECUCIÓN POR FASES

### Cronograma Sugerido:

**FASE 1 (Semana 1-2):** Definición del Problema
- Revisar y consolidar documentación existente
- Escribir introducción del artículo
- Identificar gaps en literatura

**FASE 2 (Semana 3-4):** Estado del Arte
- Revisar PDFs de biblioteca
- Crear tabla comparativa
- Documentar obstáculos y soluciones

**FASE 3 (Semana 5):** Originalidad
- Identificar contribuciones únicas
- Comparar con trabajos previos
- Escribir sección de contribuciones

**FASE 4 (Semana 6-8):** Resultados Cuantitativos
- Ejecutar inventario completo de resultados
- **Revisar:** `Biblioteca/Mis notas/Tesis_Phd_EFSG.pdf` para extraer resultados ya documentados
- **Revisar:** `Biblioteca/Mis notas/Correlations.pdf` para resultados de correlaciones
- **Revisar:** `Biblioteca/Mis notas/poster_CNF_2024.pdf` para resultados clave presentados
- Crear documento de inventario
- Organizar figuras y tablas
- Generar métricas cuantitativas clave

**FASE 5 (Semana 9-10):** Interpretación
- Analizar cada tipo de resultado
- **Revisar:** `Biblioteca/Mis notas/Tesis_Phd_EFSG.pdf` para interpretaciones ya realizadas
- **Revisar:** `Biblioteca/Mis notas/Correlations.pdf` para interpretación de correlaciones
- **Revisar:** `Biblioteca/Mis notas/Notas_para_tesis.pdf` para análisis previos
- Conectar con problema de investigación
- Escribir sección de resultados y discusión

**FASE 6 (Semana 11-12):** Impacto
- Escribir conclusiones
- Identificar trabajo futuro
- Discutir implicaciones

### Priorización:

**ALTA PRIORIDAD (Hacer primero):**
1. FASE 4: Inventario de resultados (necesario para todo lo demás)
2. FASE 1: Definición del problema (estructura del artículo)
3. FASE 5: Interpretación (contenido principal)

**MEDIA PRIORIDAD:**
4. FASE 3: Originalidad (importante para publicación)
5. FASE 6: Impacto (cierra el artículo)

**BAJA PRIORIDAD (puede hacerse en paralelo):**
6. FASE 2: Estado del arte (revisión bibliográfica continua)

---

## CHECKLIST GENERAL

### Documentación:
- [ ] Artículo completo estructurado
- [ ] Figuras y tablas preparadas
- [ ] Referencias bibliográficas completas (13 PDFs de biblioteca + literatura adicional)
- [ ] Todas las referencias de `Biblioteca/` citadas apropiadamente en el texto
- [ ] Documento `REFERENCIAS_BIBLIOGRAFICAS_POR_PREGUNTA.md` consultado y usado
- [ ] Apéndices con detalles técnicos (si necesario)

### Validación:
- [ ] Resultados verificados y reproducibles
- [ ] Código documentado y disponible
- [ ] Parámetros documentados
- [ ] Comparaciones con literatura

### Comunicación:
- [ ] Abstract claro y conciso
- [ ] Introducción que motive el problema
- [ ] Resultados presentados de forma clara
- [ ] Conclusiones que resuman contribuciones

---

## NOTAS ADICIONALES

- Este plan es flexible y puede ajustarse según necesidades
- Las fases pueden solaparse (ej: revisión bibliográfica continua)
- Priorizar completar FASE 4 (inventario) para tener datos concretos
- Usar el sistema Django para organizar y gestionar resultados
- Considerar crear scripts automatizados para generar inventarios
- **IMPORTANTE**: Revisar y citar todos los PDFs de `Biblioteca/` según corresponda a cada fase
- Consultar `REFERENCIAS_BIBLIOGRAFICAS_POR_PREGUNTA.md` para mapeo de referencias

### Referencias Bibliográficas Disponibles (13 PDFs):
1. `10.3389@fphy.2020.00377.pdf` - Física aplicada a sistemas biológicos
2. `AlleeEffect_2008.01692.pdf` - Fundamentos del efecto Allee
3. `control_theory_inmune.pdf` - Control inmunológico
4. `Cordula_r_allee_virus_2022.pdf` - Interacciones virus-cáncer-inmunidad
5. `Cornel_2020_mhc.pdf` - Complejo mayor de histocompatibilidad
6. `Kaitlyn_e_cancer_allee_2019.pdf` - Efecto Allee en cáncer
7. `logistic_generaliced.pdf` - Modelos logísticos generalizados
8. `Marcello_Allee_cancer_terapy_2020.pdf` - Terapia y efecto Allee
9. `Nullclines.pdf` - Análisis de nullclines
10. `Philip_g_autocrine_allee_efect2022.pdf` - Efecto Allee autocrino
11. `The_role_of_mathematical_modelling_in_understandin.pdf` - Modelado matemático
12. `thomas2005.pdf`, `thomas2006.pdf` - Dinámicas poblacionales
13. `Wang_allee_extintion_2019.pdf` - Extinción y efecto Allee

---

**Última actualización:** [Fecha]
**Estado:** Planificación inicial con referencias bibliográficas asociadas
**Próximos pasos:** Comenzar con FASE 4 (Inventario de Resultados) y revisar referencias bibliográficas

