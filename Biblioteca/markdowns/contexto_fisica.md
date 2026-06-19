# Contexto de Física

## Resumen
Este documento describe el contexto físico y matemático del modelo de dinámicas de cáncer con efectos Allee e interacciones inmunológicas.

### Marco variacional y límites (lectura conjunta recomendada)

Gran parte de lo que sigue interpreta la dinámica mediante un **funcional de energía libre** \(\mathcal{F}[c,s,i]\) y variaciones \(-\delta\mathcal{F}/\delta\phi\), en línea con modelos tipo Landau–Ginzburg y con la **analogía fenomenológica** con el modelo C de Hohenberg–Halperin. Esa narrativa es útil cuando los términos reaccionales pueden alinearse con un único potencial.

Sin embargo, si el vector de reacciones \((R_c,R_s,R_i)\) tiene jacobiano **no simétrico** respecto de \((c,s,i)\), la parte **local** \(\dot{\phi}_a = R_a\) **no** es integrable como gradiente de un potencial escalar \(V(c,s,i)\): no existe una energía libre local única cuyo gradiente reproduzca simultáneamente todos los flujos reaccionales. En ese régimen conviene complementar (o reemplazar) el discurso variacional por **transporte irreversible**: corrientes \(\mathbf{J}_a = -D_a\nabla\phi_a\), descomposición \(A=S+N\) del jacobiano, y observables tipo producción de entropía **definidos con convención explícita** (sin identificarlos con la entropía microscópica completa del tejido).

Documento técnico dedicado, tabla de convenciones para \(\mu_a\), mapa al código y referencias: **[`termodinamica_fuera_equilibrio_allee.md`](termodinamica_fuera_equilibrio_allee.md)**. Herramientas: `Models/Allee/nonequilibrium_termodynamics/reciprocity_jacobian_analysis.py` y `nonequilibrium_termodynamics/fenics_nonequilibrium.py`.

## Modelo Físico-Matemático

### Origen Variacional: Funcional de Energía Libre

Las ecuaciones dinámicas del modelo se obtienen a partir de un **funcional de energía libre** (free energy functional), siguiendo el formalismo de física estadística y teoría de campos. Esta estructura variacional conecta el modelo con sistemas fuera del equilibrio y permite interpretar las dinámicas en términos de minimización de energía libre.

**Formulación general**:
Las ecuaciones de evolución temporal se derivan de:
```
∂φ/∂t = -δF[φ]/δφ + términos de ruido
```

donde `F[φ]` es el funcional de energía libre que depende de los campos `φ = (c, s, i)` y sus gradientes espaciales.

### No Reciprocidad y el Parámetro μ

El parámetro **μ** actúa como un **interruptor variacional** (variational switch) que modifica la estructura del funcional de energía libre:

- **μ = 0**: Sistema con estructura variacional estándar (puede ser recíproco o no recíproco según otros parámetros)
- **μ = 1**: Sistema con estructura variacional modificada que introduce **no reciprocidad** explícita

**Interpretación física de la no reciprocidad**:
Cuando μ = 0 o μ = 1, las interacciones entre las poblaciones no son simétricas:
- La fuerza y forma funcional de la supresión del cáncer por el sistema inmune **difiere** de la supresión del sistema inmune por el cáncer
- Las interacciones entre células sanas y cáncer también pueden ser asimétricas
- Esta asimetría rompe la reciprocidad típica de modelos Lotka-Volterra y genera dinámicas más ricas

**Efecto en la estructura variacional**:
- **μ = 0**: Los términos de interacción tienen una estructura que puede derivarse de un funcional de energía libre estándar
- **μ = 1**: Se introducen términos adicionales que modifican el funcional de energía libre, cambiando la estructura variacional y generando no reciprocidad explícita

Esta diferencia estructural se manifiesta en:
- Cambios en la morfología de patrones espaciales (μ = 1 produce patrones menos fragmentados)
- Modificación de las escalas características (longitudes de correlación)
- Reorganización del espectro espacial (atenuación de modos de alta frecuencia cuando μ = 1)

### Modelo C de Halperin-Hohenberg (μ = 1)

Cuando **μ = 1**, el sistema puede integrarse mediante el **modelo C de Halperin-Hohenberg**, que describe la dinámica de campos conservados y no conservados acoplados. Este modelo es fundamental en física estadística para sistemas fuera del equilibrio con acoplamiento entre parámetros de orden y campos conservados.

#### Funcional de Energía Libre para μ = 1

El funcional de energía libre tiene la estructura del modelo C:

```
F[c, s, i] = ∫ [f_local(c, s, i) + f_gradient(c, s, i) + f_coupling(c, s, i)] dx
```

**Forma explícita del funcional**:

```
F[c, s, i] = ∫ {
    // Términos locales (potenciales)
    -∫[0→c] r_c·c'·f_Allee(c') dc' 
    + (r_s/2)·s²·(1 - s/2)
    + (r_i/2)·i²·(1 - i/2)
    
    // Términos de gradiente (rigidez)
    + (D_c/2)·(∇c)²
    + (D_s/2)·(∇s)²
    + (D_i/2)·(∇i)²
    
    // Términos de acoplamiento base
    + (α/2)·c·s²
    + (β/2)·c·i²
    + (γ/2)·c²·s
    + (η/2)·c²·i
    + (δ/2)·s²·i²
    
    // Términos adicionales de no reciprocidad (cuando μ = 1)
    + (α/4)·μ·c²·s²
    + (β/4)·μ·c²·i²
    + (γ/2)·μ·c·s²
    + (η/2)·μ·c·i²
} dx
```

#### Derivación de las Ecuaciones Dinámicas

Las ecuaciones dinámicas se obtienen mediante variación del funcional:

```
∂c/∂t = -δF/δc + D_c·∇²c
∂s/∂t = -δF/δs + D_s·∇²s  
∂i/∂t = -δF/δi + D_i·∇²i
```

**Variación con respecto a c** (cuando μ = 1):
```
-δF/δc = r_c·c·f_Allee(c) - (α·s² + β·i²) - μ·(γ·s² + η·i²) - (α·μ/2)·c·s² - (β·μ/2)·c·i²
```

**Variación con respecto a s** (cuando μ = 1):
```
-δF/δs = r_s·s·(1 - s) - γ·c² - (α·μ/2)·c²·s + δ·i²·s
```

**Variación con respecto a i** (cuando μ = 1):
```
-δF/δi = r_i·i·(1 - i) - η·c² - (β·μ/2)·c²·i + δ·s²·i
```

#### Características del Modelo C cuando μ = 1

1. **Campos acoplados**: 
   - `c` (cáncer): Campo no conservado (parámetro de orden)
   - `s` (células sanas): Campo con dinámica propia
   - `i` (sistema inmune): Campo con dinámica propia

2. **Acoplamiento no recíproco**: 
   - Los términos con μ = 1 rompen la simetría de las interacciones
   - La fuerza de interacción de `c` sobre `s` e `i` difiere de la fuerza inversa
   - Esto genera dinámicas fuera del equilibrio con características específicas

3. **Estructura variacional**: 
   - Permite derivar las ecuaciones dinámicas desde el funcional
   - Facilita el análisis de estabilidad y bifurcaciones
   - Conecta con la teoría de dinámica crítica

4. **Diferencias con μ = 0**:
   - **μ = 0**: El funcional tiene una estructura más simple, sin los términos de acoplamiento adicionales que generan no reciprocidad explícita
   - **μ = 1**: El funcional incluye los términos de acoplamiento que generan la no reciprocidad explícita y permiten la integración mediante el modelo C

#### Interpretación Física

El modelo C de Halperin-Hohenberg describe sistemas donde:
- Un campo no conservado (cáncer `c`) evoluciona según dinámica tipo modelo A (relajación hacia el mínimo de energía libre)
- Campos adicionales (`s`, `i`) están acoplados al campo principal
- El acoplamiento genera dinámicas cooperativas y no recíprocas
- La estructura variacional permite predecir comportamientos colectivos y patrones espaciales

Esta formulación es particularmente útil para:
- Analizar la formación de patrones espaciales
- Estudiar transiciones de fase en sistemas biológicos
- Entender la organización espontánea de dominios mesoscópicos
- Predecir escalas características de correlación

### Propiedades Termodinámicas del Funcional de Energía Libre

Tener acceso al funcional de energía libre `F[c,s,i]` proporciona acceso a una amplia gama de propiedades termodinámicas que permiten caracterizar el sistema desde una perspectiva de física estadística y termodinámica fuera del equilibrio.

#### Tipos de Energías Libres

**1. Energía Libre de Helmholtz (F)**
En el modelo, `F[c,s,i]` es una energía libre de Helmholtz funcional:
- **Definición**: `F = U - TS` (energía interna menos entropía por temperatura)
- **En el modelo**: Representa el potencial termodinámico del sistema
- **Minimización**: El sistema evoluciona hacia estados que minimizan F
- **Interpretación**: Equilibrio cuando `δF/δφ = 0` para cada campo

**2. Energía Libre de Gibbs (G)**
Para sistemas con presión o campos externos:
- **Relación**: `G = F + pV` o `G = F + ∫ campos_externos`
- **En el modelo**: El control adaptativo `u(x,t)` actúa como campo externo que modifica G
- **Aplicación**: Estudiar el efecto de terapias externas sobre el equilibrio

**3. Potencial Gran Canónico (Ω)**
Para sistemas abiertos con intercambio de partículas:
- **Relevancia**: El modelo describe poblaciones que crecen/disminuyen (sistema abierto)
- **Aplicación**: Analizar flujos de materia (células) entre el tumor y el entorno

#### Propiedades Termodinámicas Fundamentales

**1. Entropía (S)**
```
S = -∂F/∂T  (a temperatura constante)
```

**En el modelo**:
- **Entropía configuracional**: Mide el desorden espacial de las poblaciones
- **Entropía de mezcla**: Relacionada con la heterogeneidad espacial
- **Producción de entropía**: `σ = dS/dt` (tasa de generación de entropía fuera del equilibrio)

**Cálculo práctico**:
```
S[c,s,i] ≈ -∫ [c·ln(c) + s·ln(s) + i·ln(i)] dx
```

**2. Energía Interna (U)**
```
U = F + TS
```

**Componentes en el modelo**:
- **Energía de interacción**: Términos de acoplamiento `α·c·s²`, `β·c·i²`, etc.
- **Energía de gradiente**: Términos de difusión `(D/2)·(∇φ)²`
- **Energía potencial**: Términos de crecimiento logístico y Allee

**3. Potencial Químico (μ)**
```
μ_i = ∂F/∂n_i  (derivada con respecto al número de partículas)
```

**En el modelo**:
- **Potencial químico del cáncer**: `μ_c = δF/δc`
- **Potencial químico de células sanas**: `μ_s = δF/δs`
- **Potencial químico del sistema inmune**: `μ_i = δF/δi`

**Interpretación biológica**:
- Mide la "tendencia" de cada población a crecer o disminuir
- Equilibrio cuando `μ_c = μ_s = μ_i` (en sistemas cerrados)
- Fuera del equilibrio, los potenciales químicos determinan los flujos

**4. Capacidad Calorífica (C)**
```
C = -T·(∂²F/∂T²)
```

**En el modelo**:
- **Capacidad de respuesta**: Mide la sensibilidad del sistema a cambios en parámetros
- **Análogo**: `C_param = -∂²F/∂param²` (respuesta a cambios en `α`, `β`, etc.)

**5. Susceptibilidad (χ)**
```
χ = -∂²F/∂h²  (donde h es un campo externo)
```

**En el modelo**:
- **Susceptibilidad al control**: `χ_u = -∂²F/∂u²`
- Mide qué tan sensible es el sistema al control inmunológico
- Valores altos indican que pequeños cambios en `u` producen grandes efectos

**6. Función de Correlación y Longitud de Correlación**
```
G(r) = ⟨φ(x)·φ(x+r)⟩ - ⟨φ(x)⟩²
ξ = distancia donde G(ξ) = G(0)/e
```

**Propiedades termodinámicas**:
- **Longitud de correlación**: `ξ ~ (∂²F/∂φ²)^(-1/2)`
- **Exponentes críticos**: Relacionados con la divergencia de `ξ` cerca de transiciones de fase

**7. Presión (P)**
```
P = -∂F/∂V  (en sistemas con volumen)
```

**Análogo en el modelo**:
- **Presión poblacional**: Mide la "fuerza" que ejerce cada población
- **Presión de interacción**: Relacionada con los términos de acoplamiento

#### Termodinámica Fuera del Equilibrio

**1. Producción de Entropía**
```
σ = dS/dt = ∫ J·X dx
```

donde:
- **J**: Flujos (corrientes de células, difusión)
- **X**: Fuerzas termodinámicas (gradientes de potencial químico)

**En el modelo**:
```
σ = ∫ [D_c·(∇μ_c)² + D_s·(∇μ_s)² + D_i·(∇μ_i)²] dx
```

**2. Principio de Mínima Producción de Entropía**
- Estados estacionarios fuera del equilibrio minimizan la producción de entropía
- El sistema busca estados con `σ` mínima compatible con las restricciones

**3. Relaciones de Onsager**
Para sistemas lineales fuera del equilibrio:
```
J_i = Σ_j L_ij·X_j
```

donde `L_ij` son coeficientes de acoplamiento.

**En el modelo**: Los términos de acoplamiento `α`, `β`, `γ`, `δ`, `η` pueden interpretarse como coeficientes de Onsager.

**4. Fluctuaciones y Teorema de Fluctuación-Disipación**
```
⟨δφ(x)·δφ(x')⟩ = k_B·T·(δ²F/δφ²)^(-1)
```

- Relaciona las fluctuaciones con la respuesta del sistema
- Permite calcular varianzas y correlaciones desde el funcional

#### Propiedades Específicas del Modelo

**1. Transiciones de Fase**
**Puntos críticos**: Donde `∂²F/∂φ² = 0`
- **Transición extinción-proliferación**: Relacionada con el umbral de Allee
- **Transición orden-desorden**: Relacionada con la formación de patrones espaciales

**2. Diagramas de Fase**
- **Líneas de coexistencia**: Donde `μ_c = μ_s = μ_i`
- **Puntos triples**: Intersección de múltiples fases
- **Regiones metaestables**: Mínimos locales de F

**3. Histeresis**
- **Ciclos de histéresis**: Relacionados con la existencia de múltiples mínimos de F
- **Área del ciclo**: Relacionada con la disipación de energía

**4. Energía de Interfaz**
```
γ_interfaz = ∫ [F(φ_interface) - F(φ_bulk)] dx
```

- Mide la energía necesaria para crear interfaces entre dominios
- Relacionada con la tensión superficial entre regiones de alta/baja densidad tumoral

**5. Trabajo y Calor**
```
dF = dW - dQ
```

- **Trabajo (dW)**: Cambios en F debido a trabajo externo (control `u`)
- **Calor (dQ)**: Disipación de energía (producción de entropía)

#### Aplicaciones Prácticas

**1. Optimización de Terapias**
- **Minimizar F**: Encontrar estrategias de control que minimicen la energía libre del tumor
- **Maximizar producción de entropía**: Identificar condiciones que favorezcan la extinción

**2. Predicción de Comportamiento**
- **Análisis de estabilidad**: `∂²F/∂φ² > 0` indica estabilidad
- **Bifurcaciones**: Donde `∂²F/∂φ² = 0`

**3. Caracterización de Patrones**
- **Energía de formación de patrones**: Diferencia entre F(pattern) y F(homogéneo)
- **Escalas características**: Relacionadas con la estructura de F

**4. Comparación de Escenarios**
- **Diferencia de energía libre**: `ΔF = F(μ=1) - F(μ=0)`
- Mide el efecto del control inmunológico sobre la termodinámica del sistema

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

### Parámetro de Control y No Reciprocidad

- **`μ`**: Interruptor variacional que modifica la estructura del funcional de energía libre y define la no reciprocidad
  - **`μ = 0`**: Estructura variacional estándar (sin términos adicionales de no reciprocidad)
  - **`μ = 1`**: Estructura variacional modificada que introduce **no reciprocidad explícita**
    - Los términos adicionales modifican el funcional de energía libre
    - Genera interacciones asimétricas entre poblaciones
    - Actúa como selector morfológico que reorganiza el espectro espacial
    - Produce patrones menos fragmentados con interfaces más suaves
    - Atenúa modos de alta frecuencia sin estabilizar el sistema
  - **`μ > 0` (general)**: Control inmunológico activo con estructura variacional modificada

**Nota importante**: Los valores μ = 0 y μ = 1 son los casos principales que definen la no reciprocidad en el modelo. El parámetro μ no estabiliza las dinámicas, sino que funciona como un **selector morfológico** que cambia la geometría de las trayectorias transitorias y la escala característica de los dominios espaciales.

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

### Formulación Débil y Funcional de Energía Libre

Las ecuaciones se resuelven mediante el método de elementos finitos usando formulación débil (weak form), que está directamente relacionada con el funcional de energía libre:

```
∫ φ·(∂u/∂t) dx + ∫ D·∇u·∇φ dx = ∫ f(u)·φ dx
```

donde `φ` es una función de prueba (test function).

**Conexión con el funcional de energía libre**:
La formulación débil puede interpretarse como la condición de que la variación del funcional de energía libre con respecto a los campos sea cero:
```
δF/δu = 0  (en el sentido débil)
```

Esta estructura variacional es fundamental para entender cómo μ = 0 y μ = 1 modifican las ecuaciones: cuando μ cambia, se modifica el funcional de energía libre subyacente, lo que a su vez cambia las ecuaciones dinámicas derivadas de él.

### Discretización Temporal

- **Esquema implícito**: Estabilidad numérica mejorada
- **Paso temporal adaptativo**: Ajuste según la dinámica

### Discretización Espacial

- **Elementos finitos**: Polinomios de grado 1 (`P1`)
- **Malla rectangular**: Dominio cuadrado discretizado
- **Resolución**: Controlada por `nodes_in_xaxis` y `nodes_in_yaxis`

## Aportes a la frontera del conocimiento (física)

Síntesis para redacción de tesis/paper; ver `Biblioteca/markdowns/indice_redaccion_tesis_paper.md` (mapa y abstracts).

1. **Modelo C (Halperin–Hohenberg) y marco variacional**: integrar cáncer–inmune–tejido sano con un funcional de energía libre \(F[c,s,i]\) conecta el modelo con sistemas fuera del equilibrio y con la formación de patrones tipo modelo C cuando \(\mu=1\).
2. **Coarsening subdifusivo**: las longitudes de correlación \(\xi(t)\) en simulaciones 2D muestran crecimiento compatible con \(\xi(t)\sim e^{\alpha} t^{1/2}\) en el horizonte simulado, interpretable como coarsening limitado por difusión/competencia; el prefactor \(\alpha\) depende de Allee, \(\mu\) y protocolo de control.
3. **No reciprocidad y funcional**: la asimetría cáncer\(\leftrightarrow\)inmune (y resto de acoplamientos) modifica la estructura efectiva del funcional y el espectro lineal; \(\mu\) actúa como **selector morfológico** (atenuación de modos cortos, menos fragmentación espacial) **sin** estabilizar los equilibrios homogéneos en los barridos actuales (Re \(\lambda_{\max}>0\)).
4. **Métricas mesoscópicas**: rejillas de \(\xi(t)\) para autocorrelaciones (\(c_c\), \(s_s\), \(i_i\)) y cruzadas (\(c_i\), etc.) ofrecen un marco reproducible para comparar protocolos y separar el papel de \(\mu\) del de \(u(x,t)\).

## Pipeline termodinámico (energía libre, producción de entropía)

El script `Models/Allee/calculate_thermodynamic_properties.py` genera series \(F(t)\), \(\sigma(t)\), potenciales y figuras por escenario. El **orden de escenarios**, **comandos WSL/conda** y el **estado de corridas** (completos / interrumpidos / pendientes) viven en **`Models/Allee/README.md`**; actualízalo cuando reanudes o termines procesamiento.

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

