# Contexto de Física

## Resumen
Este documento describe el contexto físico y matemático del modelo de dinámicas de cáncer con efectos Allee e interacciones inmunológicas.

## Modelo Físico-Matemático

### Sistema de Ecuaciones de Reacción-Difusión

El modelo consiste en un sistema de tres ecuaciones diferenciales parciales acopladas que describen la evolución espaciotemporal de tres poblaciones:

1. **Cáncer (c)**: Densidad de células cancerosas
2. **Células Sanas (s)**: Células del tejido sano circundante
3. **Sistema Inmune (i)**: Células del sistema inmunológico

### Formulación Matemática

#### Ecuación para el Cáncer (c)

**Caso μ = 0 (Sin control inmunológico adicional)**:
```
∂c/∂t = D_c ∇²c + rc·c·(c - alle)·(1 - c) - c·(α·s² + β·i²)
```

**Caso μ ≠ 0 (Con control inmunológico)**:
```
∂c/∂t = D_c ∇²c + rc·c·(c - alle)·(1 - c) - c·(α·s² + β·i²) - μ·c·(γ·s² + η·i²)
```

**Interpretación física**:
- `D_c`: Coeficiente de difusión del cáncer (movimiento espacial)
- `rc`: Tasa de crecimiento intrínseca
- `alle`: Parámetro de efecto Allee (umbral crítico)
- `α`, `β`: Coeficientes de interacción con células sanas e inmunes
- `γ`, `η`: Coeficientes adicionales de supresión inmunológica cuando μ ≠ 0

#### Ecuación para Células Sanas (s)

**Caso μ = 0**:
```
∂s/∂t = D_s ∇²s + rs·s·(1 - s) - γ·c²·s + δ·i²·s
```

**Caso μ ≠ 0**:
```
∂s/∂t = D_s ∇²s + rs·s·(1 - s) - γ·c²·s + δ·i²·s - (s·c²·α·μ)/2
```

**Interpretación física**:
- `D_s`: Coeficiente de difusión/regeneración de células sanas
- `rs`: Tasa de crecimiento logístico de células sanas
- `γ`: Supresión de células sanas por cáncer (invasión tumoral)
- `δ`: Activación/regeneración de células sanas por sistema inmune

#### Ecuación para Sistema Inmune (i)

**Caso μ = 0**:
```
∂i/∂t = D_i ∇²i + rd·i·(1 - i) + δ·i·s² - c²·i·η
```

**Caso μ ≠ 0**:
```
∂i/∂t = D_i ∇²i + rd·i·(1 - i) + δ·i·s² - c²·i·η - (i·c²·β·μ)/2
```

**Interpretación física**:
- `D_i`: Coeficiente de difusión del sistema inmune
- `rd`: Tasa de crecimiento logístico
- `δ`: Interacción con células sanas
- `η`: Supresión por cáncer

## Efecto Allee

### Definición Física

El efecto Allee es un fenómeno poblacional donde la tasa de crecimiento per cápita aumenta con la densidad poblacional hasta cierto umbral crítico. En el contexto del cáncer:

- **Allee Débil (Weak Allee)**: El crecimiento puede ocurrir desde densidades muy bajas, pero es más lento
- **Allee Fuerte (Strong Allee)**: Existe un umbral mínimo (`alle`) por debajo del cual la población no puede crecer

### Formulación Matemática

**Weak Allee**:
```
rc·c·(c - alle)·(1 - c)
```

**Strong Allee**:
```
rc·c·(1 - c)·((c - alle)/(1 - alle))
```

### Significado Biológico

- **`alle < 0`**: Efecto Allee débil
- **`0 < alle < 1`**: Efecto Allee fuerte con umbral crítico
- El efecto Allee modela la necesidad de cooperación entre células cancerosas para sobrevivir y proliferar

## Fenómenos Físicos Involucrados

### Difusión

La difusión modela el movimiento espacial de las poblaciones:

```
D·∇²φ = D·(∂²φ/∂x² + ∂²φ/∂y²)
```

- **Difusión homogénea**: Coeficientes constantes en el espacio
- **Patrones espaciales**: La difusión puede generar estructuras espaciales complejas

### Reacción-Difusión

La combinación de términos de reacción (local) y difusión (no local) genera:

- **Ondas viajeras**: Propagación de frentes
- **Patrones estacionarios**: Estructuras espaciales que no cambian en el tiempo
- **Turbulencia espacial**: Comportamiento caótico espacial

### No Linealidades

Las interacciones no lineales (`c²`, `s²`, `i²`, `c·s²`, etc.) generan:

- **Múltiples estados estacionarios**: Bifurcaciones
- **Comportamiento complejo**: Caos, oscilaciones
- **Dependencia de condiciones iniciales**: Histéresis

## Análisis Espectral

### Espectro de Potencia

El análisis de Fourier permite identificar escalas espaciales características:

```
P(k) = |FFT[φ(x,y)]|²
```

- **k**: Número de onda (frecuencia espacial)
- **P(k)**: Potencia en cada escala espacial
- **Análisis**: Identificación de patrones periódicos y escalas dominantes

### Funciones de Correlación

#### Correlación Cruzada

```
C_{ab}(r) = ⟨a(x)·b(x+r)⟩
```

Mide la correlación espacial entre dos campos diferentes.

#### Autocorrelación

```
C_{aa}(r) = ⟨a(x)·a(x+r)⟩
```

Mide la estructura espacial de un solo campo.

#### Longitud de Correlación

La longitud de correlación `ξ` se define como la distancia a la cual la correlación cae a `1/e` de su valor máximo:

```
C(ξ) = C(0)/e
```

**Interpretación física**:
- Mide el tamaño característico de las estructuras espaciales
- Indica la escala de coherencia espacial
- Puede cambiar con el tiempo (dinámica de correlaciones)

## Estados Estacionarios

### Análisis de Nullclines

Las nullclines son curvas donde la derivada temporal es cero:

```
F₁(c, s) = 0  (nullcline de c)
F₂(c, s) = 0  (nullcline de s)
```

**Puntos de equilibrio**: Intersecciones de nullclines

### Nullclines reducidas (Weak Allee, notebook `steady_states.ipynb`)

El notebook `new_model_10112024/Weak Allee/steady_states.ipynb` analiza las nullclines en el plano \((c,s)\) suponiendo que la población inmune \(i\) está en equilibrio cuasi-estático (\(\partial i/\partial t = 0\)) y \(i>0\). De la ecuación de \(i\) se obtiene:
```
i*(c,s) = 1 + (δ·s² - c²·(η + β·μ/2)) / rd
```
Sustituyendo \(i(c,s)\) en las ecuaciones de \(c\) y \(s\) y fijando \(\partial c/\partial t = \partial s/\partial t = 0\), se obtienen las nullclines:
```
F1(c,s) = (1/4)·c·[ -4·(1 - c)·(a - c)·rc
                 - 4·s²·α
                 - β·(2·rd + 2·s²·δ - c²·(2·η + β·μ))² / rd²
                 - 4·μ·( s²·γ
                        + η·(2·rd + 2·s²·δ - c²·(2·η + β·μ))² / (4·rd²) ) ]

F2(c,s) = (1/4)·s·[ -4·rs·(1 - s)
                 - 4·c²·γ
                 - 2·c²·α·μ
                 + δ·(2·rd + 2·s²·δ - c²·(2·η + β·μ))² / rd² ]
```
Las nullclines son \(F1(c,s)=0\) y \(F2(c,s)=0\); sus intersecciones \((c^\*, s^\*)\) son candidatos a estados estacionarios del sistema reducido.

### Parámetros usados en el notebook
```
rc=0.01, rs=0.013, rd=0.011,
α=0.81, δ=0.82, β=1.0, a=0.7,
γ=0.74, η=5.08, μ=1
```

### Procedimiento numérico (Newton–Raphson)

1. Se evalúan \(F1\) y \(F2\) numéricamente con `lambdify`.
2. Se construye una malla \((c,s) ∈ [0,1]²\) para visualizar nullclines y localizar aproximaciones iniciales.
3. Se aplica Newton–Raphson en \((c,s)\) resolviendo el sistema \(F1=0, F2=0\):
```
[J]·Δ = (F1, F2)ᵀ ,   (c, s) ← (c, s) - Δ
```
4. Se filtra si la matriz Jacobiana es mal condicionada (`cond(J) > 1e12`) para evitar divergencias numéricas.

Puntos de partida utilizados en el notebook:
- Semilla 1: \(c₀ = 1.4103, s₀ = 1.0414\)
- Semilla 2: \(c₁ = 2.7,   s₁ = 1.5\)

El notebook grafica el campo vectorial \((F1, F2)\), colorea por cercanía a los puntos de equilibrio detectados en la malla y muestra las nullclines \(F1=0\) (negro) y \(F2=0\) (azul).

### Estabilidad

El análisis de estabilidad se realiza mediante:

1. **Análisis lineal**: Jacobiano del sistema en el punto de equilibrio
2. **Autovalores**: Determinan la estabilidad local
3. **Bifurcaciones**: Cambios cualitativos en el comportamiento

### Puntos de Equilibrio Conocidos

Para el modelo con μ = 0:
- `P₁ = (1.4103, 1.0414, 0.6110)`
- `P₂ = (1.0826, 0.4885, 0.1649)` (cuando μ = 1)

## Parámetros Físicos

### Coeficientes de Difusión

- **`D_c`**: Movilidad de células cancerosas
- **`D_s`**: Movilidad/regeneración de células sanas
- **`D_i`**: Movilidad de células inmunes

**Rango típico**: `10⁻³` a `10⁻¹` (unidades adimensionales)

### Tasas de Crecimiento

- **`rc`**: Tasa de crecimiento del cáncer (incluye efecto Allee)
- **`rs`**: Tasa de crecimiento de células sanas
- **`rd`**: Tasa de crecimiento del sistema inmune

**Rango típico**: `10⁻³` a `10⁻¹`

### Parámetros de Interacción

- **`α`**: Supresión del cáncer por células sanas
- **`β`**: Supresión del cáncer por sistema inmune
- **`γ`**: Supresión de células sanas por cáncer (invasión tumoral)
- **`δ`**: Interacción entre sistema inmune y células sanas
- **`η`**: Supresión del sistema inmune por cáncer

**Rango típico**: `0.1` a `10`

### Parámetro de Control

- **`μ`**: Intensidad del control inmunológico adicional
  - `μ = 0`: Sin control adicional
  - `μ > 0`: Control inmunológico activo

## Escalas Características

### Escala Temporal

- **Tiempo de difusión**: `τ_D = L²/D` (donde L es la longitud característica)
- **Tiempo de reacción**: `τ_R = 1/r` (donde r es la tasa de reacción)
- **Número de Damköhler**: `Da = τ_D/τ_R` determina si domina difusión o reacción

### Escala Espacial

- **Longitud de difusión**: `l_D = √(D/r)`
- **Longitud de correlación**: `ξ` (calculada numéricamente)
- **Tamaño del dominio**: `space_size` (parámetro de simulación)

## Fenómenos Emergentes

### Patrones Espaciales

El sistema puede exhibir:

1. **Patrones de Turing**: Estructuras periódicas espaciales
2. **Frentes de propagación**: Ondas viajeras
3. **Vórtices y espirales**: Estructuras rotatorias
4. **Caos espacial**: Comportamiento irregular

### Dinámicas Temporales

- **Oscilaciones**: Comportamiento periódico en el tiempo
- **Caos temporal**: Comportamiento aperiódico
- **Transiciones de fase**: Cambios abruptos en el comportamiento

## Métodos Numéricos

### Formulación Débil

Las ecuaciones se resuelven mediante el método de elementos finitos usando formulación débil:

```
∫ φ·(∂u/∂t) dx + ∫ D·∇u·∇φ dx = ∫ f(u)·φ dx
```

donde `φ` es una función de prueba.

### Discretización Temporal

- **Esquema implícito**: Estabilidad numérica mejorada
- **Paso temporal adaptativo**: Ajuste según la dinámica

### Discretización Espacial

- **Elementos finitos**: Polinomios de grado 1 (`P1`)
- **Malla rectangular**: Dominio cuadrado discretizado
- **Resolución**: Controlada por `nodes_in_xaxis` y `nodes_in_yaxis`

## Referencias Físicas

### Ecuaciones de Reacción-Difusión

- Modelo de Fisher-Kolmogorov
- Ecuaciones de Ginzburg-Landau
- Sistemas de Turing

### Efecto Allee

- Modelos poblacionales con umbral crítico
- Dinámicas de metapoblaciones
- Extinción y persistencia poblacional

### Análisis Espectral

- Teoría de funciones de correlación en física estadística
- Análisis de Fourier en sistemas espaciales extendidos
- Escalas características en sistemas complejos

