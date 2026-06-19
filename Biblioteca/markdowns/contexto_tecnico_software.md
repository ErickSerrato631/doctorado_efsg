# Contexto Técnico de Software Python

## Resumen
Este documento describe el contexto técnico y las herramientas de software utilizadas en el proyecto de modelado matemático de dinámicas de cáncer con efectos Allee e inmunoterapia.

## Stack Tecnológico Principal

### Bibliotecas Core

#### FEniCS (DOLFIN)
- **Propósito**: Resolución de ecuaciones diferenciales parciales (PDEs) mediante el método de elementos finitos
- **Uso en el proyecto**:
  - Resolución de sistemas de reacción-difusión espaciotemporales
  - Implementación de formulaciones débiles (weak forms) para las ecuaciones del modelo
  - Manejo de mallas 2D (`RectangleMesh`) y espacios de funciones (`FunctionSpace`)
  - Solución de sistemas no lineales mediante `NonlinearVariationalSolver`
  
**Ejemplo de uso**:
```python
from fenics import *
V = FunctionSpace(mesh, 'P', 1)
c = Function(V)
phi_c = TestFunction(V)
F_c = ((c - c_n) / dt) * phi_c * dx + D_c * dot(grad(c), grad(phi_c)) * dx
```

#### NumPy
- **Propósito**: Computación numérica eficiente con arrays multidimensionales
- **Uso en el proyecto**:
  - Conversión de campos FEniCS a arrays NumPy para análisis posterior
  - Operaciones de álgebra lineal y transformadas de Fourier (`np.fft.fft2`, `np.fft.fftshift`)
  - Manejo de datos espaciales y temporales
  - Cálculo de correlaciones y espectros de potencia

#### Matplotlib
- **Propósito**: Visualización científica y generación de gráficos
- **Uso en el proyecto**:
  - Visualización de campos espaciales (`plot()`)
  - Gráficos de correlaciones 2D y 3D
  - Mapas de calor con colormaps personalizados (`seismic`, `viridis`, `magma`, `gray`)
  - Visualización de nullclines y campos vectoriales
  - Generación de figuras con múltiples subplots

#### SciPy
- **Propósito**: Funciones científicas adicionales
- **Uso en el proyecto**:
  - Interpolación (`scipy.interpolate`)
  - Procesamiento de imágenes (`scipy.ndimage.zoom`)
  - Operaciones de señal y análisis espectral

### Herramientas de Desarrollo

#### Jupyter Notebooks
- **Propósito**: Entorno interactivo para desarrollo y análisis
- **Estructura del proyecto**:
  - `cancer_dynamics.ipynb`: Simulación principal de dinámicas
  - `cancer_dynamics_control.ipynb`: Variante Weak Allee con control inmunológico adaptativo (μ opcional)
  - `correlation_fourier.ipynb`: Análisis espectral y correlaciones
  - `correlation_real.ipynb`: Correlaciones en espacio real
  - `steady_states.ipynb`: Análisis de estados estacionarios
  - `main.ipynb`: Orquestación de ejecución de múltiples notebooks

#### Python-dotenv
- **Propósito**: Gestión de variables de entorno desde archivos `.env`
- **Uso**: Configuración de parámetros del modelo (coeficientes de difusión, tasas de crecimiento, parámetros de Allee, etc.)

## Arquitectura del Código

### Estructura de Simulación

#### 1. Inicialización
- Carga de parámetros desde variables de entorno
- Creación de malla y espacio de funciones
- Definición de condiciones iniciales (aleatorias o puntuales)

#### 2. Resolución Temporal
- Implementación de esquemas temporales implícitos
- Solución secuencial de sistemas no lineales acoplados
- Actualización de campos en cada paso temporal

#### 3. Análisis Post-procesamiento
- Conversión de campos FEniCS a arrays NumPy
- Cálculo de espectros de potencia mediante FFT
- Análisis de correlaciones espaciales
- Cálculo de longitudes de correlación

### Componentes Principales

#### Notebook `cancer_dynamics_control.ipynb`
- **Entorno**: Kernel `fenicsproject`; si no está activo, FEniCS no se carga.
- **Parámetros**: Se leen desde `.env` (`D_c`, `D_s`, `D_i`, `rc`, `rs`, `rd`, `alpha`, `beta`, `gamma`, `delta`, `eta`, `mu`, `alle`, `T`, `dt`, `nb`, `nodes_in_xaxis`, `nodes_in_yaxis`, `space_size`, `SAVE_IMAGES`).
- **Condiciones iniciales**: `RandomExpression` para `c`, `s`, `i` (valores en rangos); existe `CancerInitialCondition` para focos puntuales.
- **Control adaptativo (Hill)**: `u = u_max * (c^nc/(Kc^nc + c^nc)) * (Ki^ni/(Ki^ni + i^ni))` (con `i=max(i,0)`), aplicado en la ecuación de `i`; se reevalúa en cada paso temporal.
- **Ecuaciones**:
  - `μ = 0`: términos base con efecto Allee en `c` y acoplamientos cuadráticos; control adaptativo suma `+ u*phi_i*dx`.
  - `μ > 0`: añade términos proporcionales a `μ` que refuerzan supresión del cáncer y modulan `s` e `i`.
- **Solver**: `NonlinearVariationalSolver` (SNES) con prueba de múltiples `linear_solver` (`cg`, `gmres`, `bicgstab`) y precondicionadores (`ilu`, `amg`, `icc`); tolerancias relajadas.
- **Bucle temporal**: `nb` bloques; reinicia campos por bloque; avanza `t` en pasos `dt` hasta `T`; guarda matrices (`field_to_numpy_array`) y grafica `c`, `s`, `i`, `u` en el primer bloque.

#### Solvers No Lineales
```python
def NonlinearSolver(F, field):
    J = derivative(F, field)
    problem = NonlinearVariationalProblem(F, field, bcs=[], J=J)
    solver = NonlinearVariationalSolver(problem)
    # Configuración de parámetros del solver
    prm = solver.parameters["snes_solver"]
    prm["method"] = "vinewtonrsls"
    prm["linear_solver"] = "mumps"
    # ...
```

**Características**:
- Métodos iterativos con tolerancias configurables
- Fallback automático entre diferentes solvers lineales
- Manejo robusto de convergencia

#### Condiciones Iniciales
- `RandomExpression`: Condiciones iniciales aleatorias con rangos definidos
- `CancerInitialCondition`: Condiciones iniciales con concentraciones puntuales en regiones específicas

#### Conversión de Campos
- `field_to_numpy_array()`: Convierte campos FEniCS a matrices NumPy para análisis
- Manejo de valores NaN e Inf
- Muestreo espacial configurable

### Análisis Espectral

#### Transformadas de Fourier
- Cálculo de espectros de potencia 2D
- Análisis de correlaciones cruzadas en espacio de Fourier
- Transformadas inversas para obtener correlaciones en espacio real

#### Funciones de Correlación
- Correlaciones cruzadas entre campos (c-s, c-i, s-i)
- Autocorrelaciones para cada campo
- Cálculo de longitudes de correlación mediante promedios radiales

### Visualización

#### Campos Espaciales
- Visualización simultánea de múltiples campos (cáncer, células supresoras, sistema inmune)
- Mapas de color personalizados según el tipo de campo
- Barras de color ajustadas dinámicamente

#### Análisis Temporal
- Gráficos 3D de correlaciones en función del tiempo
- Contornos de longitud de correlación
- Visualización de nullclines y campos vectoriales

## Configuración y Parámetros

### Variables de Entorno (.env)
- `D_c`, `D_s`, `D_i`: Coeficientes de difusión
- `rc`, `rs`, `rd`: Tasas de crecimiento
- `alpha`, `beta`, `gamma`, `delta`, `eta`: Parámetros de interacción
- `alle`: Parámetro de efecto Allee
- `mu`: Parámetro de control inmunológico
- `T`, `dt`: Tiempo total y paso temporal
- `nodes_in_xaxis`, `nodes_in_yaxis`: Resolución espacial
- `space_size`: Tamaño del dominio espacial

## Optimizaciones y Mejores Prácticas

### Rendimiento
- Uso de solvers optimizados (MUMPS, SuperLU)
- Paralelización implícita en FEniCS
- Manejo eficiente de memoria para simulaciones largas

### Robustez
- Manejo de errores en evaluación de campos
- Validación de valores NaN/Inf
- Fallback entre diferentes métodos de solución

### Reproducibilidad
- Configuración mediante archivos `.env`
- Guardado de resultados en formato texto
- Versionado de notebooks y parámetros

## Dependencias Principales

```python
# Core
fenics >= 2019.1.0
numpy >= 1.19.0
matplotlib >= 3.3.0
scipy >= 1.5.0

# Utilidades
python-dotenv >= 0.19.0
jupyter >= 1.0.0
sympy >= 1.7.0  # Para análisis simbólico de estados estacionarios
```

## Flujo de Trabajo Típico

1. **Configuración**: Definir parámetros en `.env`
2. **Simulación**: Ejecutar `cancer_dynamics.ipynb` para generar datos temporales
3. **Análisis Espectral**: Ejecutar `correlation_fourier.ipynb` para análisis de Fourier
4. **Análisis de Correlaciones**: Ejecutar `correlation_real.ipynb` para correlaciones espaciales
5. **Análisis de Estados Estacionarios**: Ejecutar `steady_states.ipynb` para análisis de equilibrio
6. **Visualización**: Generar figuras y videos con `create_videos.ipynb`

## Notas Técnicas

- **Versión de Python**: 3.9.7
- **Kernel**: IPython (Jupyter)
- **Sistema de archivos**: Guardado de matrices en formato texto para análisis posterior
- **Manejo de memoria**: Cuidado especial con simulaciones de múltiples bloques (nb > 1)

## Referencias Técnicas

- Documentación FEniCS: https://fenicsproject.org/
- NumPy User Guide: https://numpy.org/doc/stable/
- Matplotlib Documentation: https://matplotlib.org/
- SciPy Reference Guide: https://docs.scipy.org/doc/scipy/

