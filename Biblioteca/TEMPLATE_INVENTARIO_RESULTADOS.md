# Template: Inventario de Resultados Cuantitativos en Física-Biología
## Dynamics of coupled biological populations: pattern formation in reaction-diffusion systems with Allee effect and immune interactions

## Instrucciones de Uso

Este template debe completarse con los resultados físico-biológicos obtenidos de las simulaciones de poblaciones biológicas acopladas. Para cada sección, documentar:
- Valores numéricos específicos con unidades físicas (adimensionales)
- Rangos de valores observados en poblaciones biológicas
- Tendencias y patrones físicos en formación de patrones
- Comparaciones entre escenarios (Weak/Strong Allee, con/sin control inmunológico)
- Números adimensionales relevantes
- Interpretación biológica de resultados físicos

**Fuentes de información adicionales:**
- ✅ `Biblioteca/Mis notas/Tesis_Phd_EFSG.pdf` - Revisar para resultados ya documentados
- ✅ `Biblioteca/Mis notas/Correlations.pdf` - Revisar para análisis de correlaciones
- ✅ `Biblioteca/Mis notas/poster_CNF_2024.pdf` - Revisar para resultados clave presentados
- ✅ `Biblioteca/Mis notas/Notas_para_tesis.pdf` - Revisar para métricas y resultados previos

---

## 1. CONFIGURACIÓN DE SIMULACIONES

### 1.1 Parámetros Físicos del Modelo (Adimensionales)

**Interpretación biológica:** Los campos c, s, i representan densidades de poblaciones biológicas (células cancerosas, células sanas, sistema inmune)

#### Weak Allee
- **alle**: [valor] (parámetro de umbral crítico - efecto Allee débil en poblaciones)
- **rc**: [valor] (tasa de crecimiento población c, unidades: 1/t)
- **rs**: [valor] (tasa de crecimiento población s)
- **rd**: [valor] (tasa de crecimiento población i)
- **alpha**: [valor] (coeficiente de interacción c-s)
- **beta**: [valor] (coeficiente de interacción c-i - interacción inmune)
- **gamma**: [valor] (coeficiente de interacción s-c)
- **delta**: [valor] (coeficiente de interacción s-i - interacción inmune)
- **eta**: [valor] (coeficiente de interacción i-c - supresión inmune)
- **u**: [valor(es) probados] (parámetro de control inmunológico)
  - u = constante: [valor] (control constante)
  - u(t): [función temporal] (control temporal)
  - u(x,y,t): [función espaciotemporal] (control adaptativo)
- **μ (mu)**: [0 o >0] (parámetro de encendido/apagado modelo energía libre)
  - μ = 0: Aproximación biológica directa (modelado fenomenológico)
  - μ = 1 (o > 0): Modelo derivado desde energía libre tipo C de Halperin-Hohenberg

#### Strong Allee
- **alle**: [valor] (umbral crítico positivo - efecto Allee fuerte en poblaciones)
- [mismos parámetros que arriba, incluyendo u y parámetro_modelo]

### 1.2 Parámetros Numéricos y Escalas Físicas
- **T**: [tiempo total de simulación] (unidades: tiempo adimensional)
- **dt**: [paso temporal] (debe ser << tiempo característico del sistema)
- **nb**: [número de bloques] (réplicas estadísticas)
- **nodes_in_xaxis**: [resolución espacial X] (puntos de malla)
- **nodes_in_yaxis**: [resolución espacial Y] (puntos de malla)
- **space_size**: [tamaño del dominio] (longitud característica L)

### 1.3 Escalas Características Físicas
- **Tiempo de difusión**: τ_D = L²/D_c = [calcular]
- **Tiempo de reacción**: τ_R = 1/rc = [calcular]
- **Número de Damköhler**: Da = τ_D/τ_R = [calcular]
- **Longitud de difusión**: l_D = √(D_c/rc) = [calcular]

### 1.4 Condiciones Iniciales
- Tipo: [Aleatoria / Puntual / Otra]
- Valores iniciales (densidades poblacionales):
  - c₀: [rango o valor] (densidad inicial población c - células cancerosas)
  - s₀: [rango o valor] (densidad inicial población s - células sanas)
  - i₀: [rango o valor] (densidad inicial población i - sistema inmune)

### 1.5 Configuración del Modelo
- **Parámetro μ (encendido/apagado modelo energía libre)**: [0 o >0]
  - μ = 0: Aproximación biológica directa (modelado fenomenológico)
  - μ = 1 (o > 0): Modelo derivado desde energía libre tipo C de Halperin-Hohenberg (enfoque físico de transiciones de fase)
- **Tipo de control inmunológico u**:
  - [ ] Constante: u = [valor]
  - [ ] Temporal: u(t) = [función]
  - [ ] Espaciotemporal adaptativo: u(x,y,t) = [función, ej: 0.2·c/(i+ε)]

---

## 2. RESULTADOS DE DINÁMICAS TEMPORALES (Poblaciones Biológicas)

### 2.1 Evolución Temporal de Valores Promedio de Poblaciones

#### Weak Allee, u=0 (sin control)
| Tiempo | ⟨c(t)⟩ | ⟨s(t)⟩ | ⟨i(t)⟩ | σ_c(t) | σ_s(t) | σ_i(t) | ∂⟨c⟩/∂t | ∂⟨s⟩/∂t | ∂⟨i⟩/∂t |
|--------|--------|--------|--------|--------|--------|--------|---------|---------|---------|
| 0.0    |        |        |        |        |        |        |         |         |         |
| ...    |        |        |        |        |        |        |         |         |         |

**Nota:** 
- ⟨·⟩ denota promedio espacial sobre el dominio
- σ denota desviación estándar espacial (heterogeneidad espacial)
- Interpretación biológica: valores promedio representan densidades poblacionales globales

#### Weak Allee, u=constante (control constante)
[Tabla similar, incluir valor de u]

#### Weak Allee, u(t) (control temporal)
[Tabla similar, incluir función u(t) usada]

#### Weak Allee, u(x,y,t) (control adaptativo espaciotemporal)
[Tabla similar, incluir función u(x,y,t) usada]

#### Strong Allee, u=0 (sin control)
[Tabla similar]

#### Strong Allee, u=constante (control constante)
[Tabla similar]

#### Strong Allee, u(t) (control temporal)
[Tabla similar]

#### Strong Allee, u(x,y,t) (control adaptativo espaciotemporal)
[Tabla similar]

### 2.2 Valores Extremos de Poblaciones

#### Valor Máximo de la Población c (células cancerosas)
- Weak Allee, u=0: c_max = [valor] en t=[tiempo]
- Weak Allee, u=constante: c_max = [valor] en t=[tiempo] (u=[valor])
- Weak Allee, u(t): c_max = [valor] en t=[tiempo] (control temporal)
- Weak Allee, u(x,y,t): c_max = [valor] en t=[tiempo] (control adaptativo)
- Strong Allee, u=0: c_max = [valor] en t=[tiempo]
- Strong Allee, u=constante: c_max = [valor] en t=[tiempo] (u=[valor])
- Strong Allee, u(t): c_max = [valor] en t=[tiempo] (control temporal)
- Strong Allee, u(x,y,t): c_max = [valor] en t=[tiempo] (control adaptativo)

**Interpretación biológica:** Densidad máxima alcanzada por población cancerosa. Comparar eficacia de diferentes estrategias de control u.

#### Valor Mínimo de la Población c
[Similar para c_min - puede indicar extinción local o global]

#### Valores Finales (t=T) - Estados Estacionarios de Poblaciones
- Weak Allee, u=0: c_final = [valor], s_final = [valor], i_final = [valor]
- Weak Allee, u=constante: [similar] (u=[valor], efecto del control constante)
- Weak Allee, u(t): [similar] (control temporal, función u(t) usada)
- Weak Allee, u(x,y,t): [similar] (control adaptativo, función u(x,y,t) usada)
- Strong Allee, u=0: [similar]
- Strong Allee, u=constante: [similar] (u=[valor])
- Strong Allee, u(t): [similar] (control temporal)
- Strong Allee, u(x,y,t): [similar] (control adaptativo)

**Comparación modelo energía libre vs biológico:**
- Modelo energía libre (μ=1 o >0): c_final = [valor], s_final = [valor], i_final = [valor]
- Aproximación biológica (μ=0): c_final = [valor], s_final = [valor], i_final = [valor]
- Diferencias: Δc = [valor], Δs = [valor], Δi = [valor]

**Interpretación biológica:** Estados de equilibrio poblacional (coexistencia, extinción, etc.). Comparar eficacia de diferentes estrategias de control u y diferencias entre modelos.

### 2.3 Tasas de Cambio y Escalas Temporales

#### Tasa de Crecimiento Inicial (∂⟨c⟩/∂t en t≈0)
- Weak Allee, u=0: [valor] (unidades: 1/t)
- Weak Allee, u=constante: [valor] (u=[valor])
- Weak Allee, u(t): [valor] (control temporal)
- Weak Allee, u(x,y,t): [valor] (control adaptativo)
- Strong Allee, u=0: [valor]
- Strong Allee, u=constante: [valor] (u=[valor])
- Strong Allee, u(t): [valor] (control temporal)
- Strong Allee, u(x,y,t): [valor] (control adaptativo)

#### Tiempo Característico de Evolución
- Tiempo para alcanzar 50% del valor final: τ_50% = [valor]
- Tiempo para alcanzar 90% del valor final: τ_90% = [valor]

#### Comparación con Escalas Teóricas
- τ_D (tiempo de difusión) = [valor]
- τ_R (tiempo de reacción) = [valor]
- Da (número de Damköhler) = [valor]
- ¿Domina difusión o reacción? [Análisis]

---

## 3. RESULTADOS DE ANÁLISIS ESPECTRAL (Transformada de Fourier)

### 3.1 Escalas Espaciales Dominantes (Números de Onda)

#### Población c(x,y,t) - Células Cancerosas
| Escenario | Tipo Control u | k_dominante | λ = 2π/k | l_D = √(D/rc) | Energía E(k) | Tipo de Patrón | Interpretación Biológica |
|-----------|----------------|-------------|----------|---------------|--------------|----------------|-------------------------|
| Weak, u=0 | Sin control    |             |          |               |              |                |                         |
| Weak, u=cte| Constante      |             |          |               |              |                |                         |
| Weak, u(t)| Temporal       |             |          |               |              |                |                         |
| Weak, u(x,y,t)| Adaptativo |             |          |               |              |                |                         |
| Strong, u=0| Sin control   |             |          |               |              |                |                         |
| Strong, u=cte| Constante    |             |          |               |              |                |                         |
| Strong, u(t)| Temporal      |             |          |               |              |                |                         |
| Strong, u(x,y,t)| Adaptativo|             |          |               |              |                |                         |

**Nota:** 
- k_dominante: número de onda con máxima energía en espectro P(k)
- λ: longitud de onda característica del patrón espacial
- l_D: longitud de difusión teórica
- Comparar λ con l_D para validar física del sistema
- Interpretación biológica: λ representa escala espacial característica de estructuras poblacionales

#### Campo de Células Sanas (s)
[Tabla similar]

#### Campo de Sistema Inmune (i)
[Tabla similar]

### 3.2 Evolución Temporal de Escalas Espaciales

#### Escala Dominante vs Tiempo
| Tiempo | k_dominante(c) | k_dominante(s) | k_dominante(i) | λ_c | λ_s | λ_i |
|--------|-----------------|----------------|----------------|-----|-----|-----|
| 0.0    |                |                |                |     |     |     |
| ...    |                |                |                |     |     |     |

**Análisis:**
- ¿Hay modos inestables? (k donde P(k) crece exponencialmente)
- ¿Se forman patrones de Turing? (modos con k específico)
- ¿Cómo evolucionan las escalas? (constante, creciente, decreciente)

### 3.3 Comparación entre Campos y Acoplamiento

#### Relación entre Escalas
- ¿Las escalas de c, s, i son similares o diferentes?
- Razón λ_c/λ_s = [valor]
- Razón λ_c/λ_i = [valor]
- ¿Hay acoplamiento entre escalas? (modos correlacionados)
- [Documentar observaciones físicas]

---

## 4. RESULTADOS DE FUNCIONES DE CORRELACIÓN ESPACIAL (Física Estadística)

### 4.1 Longitudes de Correlación ξ

#### Correlación Cruzada C_{cs}(r,t) = ⟨c(x)·s(x+r)⟩ (Cáncer-Células Sanas)
| Escenario | Tipo Control u | ξ_{cs}(t=0) | ξ_{cs}(t=T/2) | ξ_{cs}(t=T) | Tendencia | Comparación con l_D | Interpretación Biológica |
|----------|----------------|-------------|---------------|-------------|-----------|---------------------|---------------------------|
| Weak, u=0| Sin control    |             |               |             |           |                     |                           |
| Weak, u=cte| Constante   |             |               |             |           |                     |                           |
| Weak, u(t)| Temporal      |             |               |             |           |                     |                           |
| Weak, u(x,y,t)| Adaptativo|             |               |             |           |                     |                           |
| Strong, u=0| Sin control  |             |               |             |           |                     |                           |
| Strong, u=cte| Constante  |             |               |             |           |                     |                           |
| Strong, u(t)| Temporal     |             |               |             |           |                     |                           |
| Strong, u(x,y,t)| Adaptativo|             |               |             |           |                     |                           |

**Nota:** 
- ξ se define como C(ξ) = C(0)/e, donde e ≈ 2.718
- Interpretación biológica: ξ_{cs} representa la escala espacial de correlación entre poblaciones cancerosas y sanas

#### Correlación Cáncer-Sistema Inmune (c-i)
[Tabla similar]

#### Correlación Células Sanas-Sistema Inmune (s-i)
[Tabla similar]

#### Autocorrelación Cáncer (c-c)
[Tabla similar]

#### Autocorrelación Células Sanas (s-s)
[Tabla similar]

#### Autocorrelación Sistema Inmune (i-i)
[Tabla similar]

### 4.2 Forma Funcional de Correlaciones y Escalado

#### Tipo de Decaimiento Asintótico
- ¿Exponencial? C(r) ~ exp(-r/ξ) para r >> ξ
- ¿Potencia? C(r) ~ r^(-α) para r >> ξ (comportamiento crítico)
- ¿Oscilatorio? C(r) ~ exp(-r/ξ) cos(kr + φ)
- ¿Otro? [Especificar]

#### Parámetros de Ajuste y Exponentes Críticos
| Correlación | Escenario | Tipo Control u | Tipo | ξ | α (exponente) | R² | Comportamiento |
|-------------|-----------|----------------|------|---|---------------|----|----------------|
| C_{cs}      | Weak, u=0 | Sin control    | Exp  |   |               |    |                |
| C_{cs}      | Weak, u=cte| Constante     | Exp  |   |               |    |                |
| C_{cs}      | Weak, u(t)| Temporal       | Exp  |   |               |    |                |
| C_{cs}      | Weak, u(x,y,t)| Adaptativo| Exp  |   |               |    |                |
| ...         | ...       |                |      |   |               |    |                |

**Análisis Físico:**
- ¿Hay comportamiento crítico? (exponente α)
- ¿Las correlaciones siguen leyes de escalado universales?
- Comparar con predicciones teóricas (si aplica)

### 4.3 Análisis de Correlaciones Cruzadas

#### Signo y Magnitud de Correlación en r=0
- C_{cs}(0): [valor] → [Positiva/Negativa/Alternante] 
  - Interpretación físico-biológica: [poblaciones correlacionadas/anticorrelacionadas espacialmente]
- C_{ci}(0): [valor] → [Positiva/Negativa/Alternante]
  - Interpretación físico-biológica: [correlación entre cáncer y sistema inmune - interacción inmune]
- C_{si}(0): [valor] → [Positiva/Negativa/Alternante]
  - Interpretación físico-biológica: [correlación entre células sanas y sistema inmune]

#### Evolución Temporal de Correlaciones
- ¿Cómo cambia C(0) con el tiempo?
- ¿Hay transiciones en el signo de correlación?
- ¿Las correlaciones se fortalecen o debilitan con el tiempo?

---

## 5. RESULTADOS DE ESTADOS ESTACIONARIOS Y ESTABILIDAD

### 5.1 Puntos Fijos (Estados Estacionarios)

#### Weak Allee, u=0 (Sin Control Inmunológico)
| Punto | c* | s* | i* | Tipo de Estabilidad | λ₁ | λ₂ | λ₃ | Re(λ) | Im(λ) | Estado Biológico |
|-------|----|----|----|---------------------|----|----|----|-------|-------|------------------|
| P₁    |    |    |    | Nodo estable        |    |    |    | < 0   | 0     | Coexistencia     |
| P₂    |    |    |    | Foco inestable      |    |    |    | > 0   | ≠ 0   | Oscilaciones     |
| ...   |    |    |    |                     |    |    |    |       |       |                  |

**Tipos de Estabilidad:**
- Nodo estable: todos Re(λ) < 0, Im(λ) = 0 → Estado de equilibrio poblacional estable
- Foco estable: todos Re(λ) < 0, Im(λ) ≠ 0 → Oscilaciones amortiguadas

**Comparación Modelo Energía Libre vs Biológico:**
- Estados estacionarios modelo energía libre (μ=1 o >0): [listar puntos]
- Estados estacionarios aproximación biológica (μ=0): [listar puntos]
- Diferencias en estabilidad: [análisis]
- Silla: algunos Re(λ) > 0, algunos < 0 → Estado inestable
- Nodo inestable: todos Re(λ) > 0 → Crecimiento exponencial

#### Weak Allee, u=constante (Control Constante)
[Tabla similar, incluir valor de u]

#### Weak Allee, u(t) (Control Temporal)
[Tabla similar, incluir función u(t) usada]

#### Weak Allee, u(x,y,t) (Control Adaptativo)
[Tabla similar, incluir función u(x,y,t) usada]

#### Strong Allee, u=0 (Sin Control)
[Tabla similar]

#### Strong Allee, u=constante (Control Constante)
[Tabla similar]

#### Strong Allee, u(t) (Control Temporal)
[Tabla similar]

#### Strong Allee, u(x,y,t) (Control Adaptativo)
[Tabla similar]

### 5.2 Análisis de Bifurcaciones

#### Bifurcaciones al Variar Parámetros
- ¿Hay cambios de estabilidad al variar u (control inmunológico)?
  - Control constante: Valor crítico u_c = [valor] donde ocurre bifurcación
  - Control temporal: Función u(t) crítica = [especificar]
  - Control adaptativo: Función u(x,y,t) crítica = [especificar]
  - Tipo de bifurcación: [silla-nodo, Hopf, pitchfork, etc.]
- ¿Hay cambios de estabilidad al variar μ (parámetro encendido/apagado modelo energía libre)?
  - Comparación entre μ=0 (aproximación biológica) y μ>0 (modelo energía libre): [análisis]
  - Valor crítico μ_c donde cambia el comportamiento: [valor]
- ¿Hay cambios de estabilidad al variar alle?
  - Valor crítico alle_c = [valor]
  - Tipo de bifurcación: [especificar]
- Diagrama de bifurcación: [describir o referenciar figura]

### 5.3 Atractores y Cuencas de Atracción

#### Convergencia de Simulaciones
- ¿A qué punto fijo convergen las simulaciones?
- ¿Depende de condiciones iniciales? (múltiples atractores)
- Tiempo de convergencia: τ_convergencia = [valor]
- Comparación con predicción teórica: [coincide/no coincide]

#### Análisis de Cuencas de Atracción
- Tamaño relativo de cuencas: [si aplica]
- Dependencia de condiciones iniciales: [documentar]

---

## 6. COMPARACIÓN ENTRE ESCENARIOS (Análisis Paramétrico)

### 6.1 Efecto del Control Externo (u)

#### Reducción Relativa del Campo c
| Escenario Base | Escenario Control | Tipo Control u | Δc_rel = (c_u=0 - c_u>0)/c_u=0 | τ_reducción |
|----------------|-------------------|----------------|--------------------------------|-------------|
| Weak, u=0      | Weak, u=constante | Constante      |                                |             |
| Weak, u=0      | Weak, u(t)        | Temporal        |                                |             |
| Weak, u=0      | Weak, u(x,y,t)    | Adaptativo      |                                |             |
| Strong, u=0    | Strong, u=constante | Constante    |                                |             |
| Strong, u=0    | Strong, u(t)      | Temporal        |                                |             |
| Strong, u=0    | Strong, u(x,y,t)  | Adaptativo      |                                |             |

#### Comparación Eficacia entre Tipos de Control
| Tipo Control | Weak Allee - Reducción | Strong Allee - Reducción | Ventajas | Desventajas |
|--------------|------------------------|---------------------------|----------|-------------|
| u=constante  | [porcentaje]          | [porcentaje]              |          |             |
| u(t)         | [porcentaje]          | [porcentaje]              |          |             |
| u(x,y,t)     | [porcentaje]          | [porcentaje]              |          |             |

#### Cambios en Estabilidad
- ¿El control induce transiciones de fase?
- ¿Cambian los puntos fijos?
- ¿Cambian las longitudes de correlación?

#### Cambio en Longitudes de Correlación
[Tabla similar]

#### Cambio en Escalas Espaciales
[Tabla similar]

### 6.2 Efecto del Tipo de Allee (Bifurcaciones)

#### Comparación Weak vs Strong Allee
| Métrica Física | Weak Allee | Strong Allee | Diferencia | Interpretación |
|----------------|------------|--------------|------------|----------------|
| c* (punto fijo) |            |              |            |                |
| ξ_{cs} (longitud correlación) | |              |            |                |
| k_dominante (escala espacial) | |              |            |                |
| Tipo de estabilidad |        |              |            |                |
| Número de puntos fijos |      |              |            |                |

**Análisis Físico:**
- ¿Cómo afecta el umbral crítico (alle) a las bifurcaciones?
- ¿Qué diferencias cualitativas se observan?

### 6.3 Eficacia del Control y Transiciones de Fase

#### Condiciones para Extinción de Población c (Eliminación Tumoral)
- ¿Bajo qué condiciones físicas se logra c → 0? (extinción poblacional)
- ¿Qué valores de u (control inmunológico) son efectivos? (umbral u_c para cada tipo de control)
  - Control constante: u_c = [valor]
  - Control temporal: función u(t) crítica = [especificar]
  - Control adaptativo: función u(x,y,t) crítica = [especificar]
- ¿Depende del tipo de Allee? (Weak vs Strong)
- ¿Hay transición de fase de coexistencia a extinción?
- Comparación eficacia: ¿qué tipo de control u es más efectivo para eliminación tumoral?
- Parámetros críticos: [documentar]
- Comparación modelo energía libre vs biológico:
  - Condiciones extinción en modelo energía libre (μ=1 o >0): [documentar]
  - Condiciones extinción en aproximación biológica (μ=0): [documentar]
  - Diferencias: [análisis]
- Interpretación biológica: condiciones para eliminación tumoral mediante diferentes estrategias de inmunoterapia

#### Análisis de Respuesta del Sistema
- Sensibilidad a cambios en u: ∂c*/∂u = [valor] (para cada tipo de control)
- Eficiencia del control: [métrica a definir]

---

## 7. PATRONES ESPACIALES EMERGENTES

### 7.1 Tipos de Patrones Observados
- [ ] Frentes de propagación (ondas viajeras unidireccionales)
- [ ] Ondas viajeras (periódicas en espacio y tiempo)
- [ ] Patrones de Turing (estructuras periódicas estacionarias)
- [ ] Espirales/vórtices (estructuras rotatorias)
- [ ] Caos espacial (comportamiento irregular)
- [ ] Otros: [especificar]

### 7.2 Caracterización Física de Patrones

#### Para cada tipo de patrón observado:
- Escenario donde aparece: [Weak/Strong, u=0/u=constante/u(t)/u(x,y,t)]
- Tiempo de aparición: t_aparición = [valor]
- Escala espacial característica: λ = [valor] o k = [valor]
- Velocidad de propagación (si aplica): v = [valor]
- Estabilidad temporal: [Estable/Transitorio]
- Modos inestables (números de onda): k_inestable = [valor]
- Descripción física: [texto]
- Comparación con teoría: [coincide con predicción teórica?]

---

## 8. RESULTADOS DE REGRESIÓN Y AJUSTES

### 8.1 Ajustes a Correlaciones

#### Archivo: `resultados_regresion.json`
[Extraer y documentar resultados principales]

### 8.2 Parámetros de Regresión

| Archivo | Correlación | m (pendiente) | b (intercepto) | R² | Función |
|---------|-------------|---------------|----------------|----|---------|
| ...     | ...         |               |                |    |         |

---

## 9. ARCHIVOS GENERADOS

### 9.1 Matrices de Campos
- Total de archivos: [número]
- Formato: `matrix_{field}_{time}_nb_{block}.txt`
- Tamaño promedio: [MB]
- Ubicación: [ruta]

### 9.2 Correlaciones
- Total de archivos: [número]
- Formato: `correlacion_{field1}_{field2}_{time}.txt`
- Ubicación: [ruta]

### 9.3 Longitudes de Correlación
- Total de archivos: [número]
- Formato: `corr_length_real_inverse_nb_{block}_{field1}_{field2}.txt`
- Ubicación: [ruta]

### 9.4 Visualizaciones
- Total de imágenes: [número]
- Formatos: PNG
- Resolución: [dpi]
- Ubicación: [ruta]

### 9.5 Resultados de Análisis
- Archivos JSON: [listar]
- Archivos de regresión: [listar]
- Otros: [listar]

---

## 10. MÉTRICAS FÍSICAS RESUMEN

### 10.1 Métricas Clave para el Artículo en Física

#### Métrica 1: Eficacia del Control Inmunológico
- Definición: Reducción relativa de la población c con control inmunológico (u>0)
- Fórmula: η_control = (c_u=0 - c_u>0)/c_u=0 (para cada tipo de control u)
- Comparación entre tipos de control: η_constante vs η_temporal vs η_adaptativo
- Valor: [X] (adimensional, 0-1)
- Interpretación físico-biológica: Eficacia del control inmunológico en reducir densidad tumoral

#### Métrica 2: Longitud de Correlación Característica (Cáncer-Sistema Inmune)
- Definición: Longitud promedio de correlación cruzada entre poblaciones c-i
- Fórmula: ξ_{ci} = [valor donde C_{ci}(ξ) = C_{ci}(0)/e]
- Valor: [X] unidades espaciales adimensionales
- Comparación con l_D: ξ_{ci}/l_D = [valor]
- Interpretación físico-biológica: Escala espacial de interacción entre cáncer y sistema inmune

#### Métrica 3: Escala Espacial Dominante
- Definición: Longitud de onda del modo con máxima energía espectral
- Fórmula: λ_dominante = 2π/k_dominante
- Valor: [X] unidades espaciales adimensionales
- Comparación con l_D: λ_dominante/l_D = [valor]
- Interpretación física: Escala característica de estructuras espaciales

#### Métrica 4: Número de Damköhler
- Definición: Razón entre tiempo de difusión y tiempo de reacción
- Fórmula: Da = τ_D/τ_R = (L²/D)/(1/rc)
- Valor: [X] (adimensional)
- Interpretación física: 
  - Da >> 1: domina difusión
  - Da << 1: domina reacción
  - Da ~ 1: competencia entre difusión y reacción

#### Métrica 5: Exponente Crítico de Correlación
- Definición: Exponente en decaimiento de potencia C(r) ~ r^(-α)
- Valor: α = [valor]
- Interpretación física: Indica comportamiento crítico si α < dimensión espacial

[Agregar más métricas físicas según necesidad]

---

## 11. OBSERVACIONES Y NOTAS

### 11.1 Resultados Inesperados
- [Documentar cualquier resultado sorprendente o no esperado]

### 11.2 Limitaciones Observadas
- [Documentar limitaciones numéricas, computacionales, etc.]

### 11.3 Validaciones Realizadas
- [Documentar comparaciones con literatura, verificaciones numéricas, etc.]

---

## 12. FIGURAS Y TABLAS PARA EL ARTÍCULO

### 12.1 Figuras Principales Propuestas

#### Figura 1: Evolución Temporal de Densidades
- Contenido: [Gráfico de ⟨c⟩, ⟨s⟩, ⟨i⟩ vs tiempo]
- Escenarios: [Todos o selección]
- Ubicación de datos: [archivos]

#### Figura 2: Campos Espaciales en Diferentes Tiempos
- Contenido: [Mapas de color de c, s, i]
- Tiempos seleccionados: [t1, t2, t3, ...]
- Ubicación: [archivos PNG]

#### Figura 3: Espectros de Potencia
- Contenido: [Espectros 2D de Fourier]
- Campos: [c, s, i]
- Ubicación: [archivos PNG]

#### Figura 4: Correlaciones Espaciales
- Contenido: [Gráficos de C(r) vs r]
- Correlaciones: [c-s, c-i, s-i]
- Ubicación: [archivos de correlación]

#### Figura 5: Longitudes de Correlación vs Tiempo
- Contenido: [Gráfico de ξ(t)]
- Correlaciones: [seleccionar]
- Ubicación: [archivos de longitud]

#### Figura 6: Estados Estacionarios y Nullclines
- Contenido: [Diagrama de fase con nullclines]
- Escenarios: [seleccionar]
- Ubicación: [resultados de steady_states.ipynb]

#### Figura 7: Comparación Weak vs Strong Allee
- Contenido: [Comparación lado a lado]
- Métricas: [seleccionar]
- Ubicación: [varios archivos]

#### Figura 8: Efecto del Control Inmunológico
- Contenido: [Comparación u=0 vs u=constante vs u(t) vs u(x,y,t)]
- Comparación modelo energía libre (μ>0) vs biológico (μ=0): [si aplica]
- Métricas: [seleccionar]
- Ubicación: [varios archivos]

### 12.2 Tablas Principales Propuestas

#### Tabla 1: Parámetros del Modelo
- Contenido: [Todos los parámetros usados]

#### Tabla 2: Puntos de Equilibrio
- Contenido: [Valores de puntos de equilibrio]

#### Tabla 3: Longitudes de Correlación
- Contenido: [Valores de ξ para diferentes correlaciones]

#### Tabla 4: Escalas Espaciales Dominantes
- Contenido: [Valores de k_dominante]

#### Tabla 5: Comparación entre Escenarios
- Contenido: [Métricas clave comparadas]

---

## INSTRUCCIONES PARA COMPLETAR

1. **Ejecutar scripts de análisis** para extraer métricas automáticamente cuando sea posible
2. **Revisar archivos generados** manualmente para validar resultados
3. **Completar tablas** con valores numéricos específicos
4. **Documentar observaciones** cualitativas importantes
5. **Identificar figuras clave** para el artículo
6. **Validar consistencia** entre diferentes tipos de análisis

---

**Fecha de inicio:** [fecha]
**Fecha de última actualización:** [fecha]
**Estado:** [En progreso / Completo]
**Responsable:** [nombre]

**Notas importantes:**
- Revisar `Biblioteca/Mis notas/Tesis_Phd_EFSG.pdf` para extraer resultados cuantitativos ya documentados
- Revisar `Biblioteca/Mis notas/Correlations.pdf` para resultados de correlaciones ya calculados
- Revisar `Biblioteca/Mis notas/poster_CNF_2024.pdf` para identificar resultados clave ya presentados
- Usar `Biblioteca/Mis notas/Notas_para_tesis.pdf` como referencia para métricas y análisis previos

