# `cancer_dynamics_control.ipynb`

## Propósito
Simulación 2D de un sistema reacción-difusión con efecto Allee débil/fuerte para tres poblaciones (cáncer `c`, sanas `s`, sistema inmune `i`) y un control inmunológico adaptativo opcional.

## Entorno y dependencias
- Kernel: `fenicsproject` (si no está activo, falla `from fenics import *`).
- Principales: `fenics`, `dolfin`, `numpy`, `matplotlib`, `python-dotenv`, `random`, `math`.

## Parámetros (desde `.env`)
`D_c`, `D_s`, `D_i`, `rc`, `rs`, `rd`, `alpha`, `beta`, `gamma`, `delta`, `eta`, `mu`, `alle`, `T`, `dt`, `nb`, `nodes_in_xaxis`, `nodes_in_yaxis`, `space_size`, `SAVE_IMAGES`.

## Estructura principal
- **Condiciones iniciales**: `RandomExpression` (rangos configurados); existe `CancerInitialCondition` para focos puntuales (no se usa en el flujo actual).
- **Control adaptativo (Hill)**: `u = u_max * (c^nc/(Kc^nc + c^nc)) * (Ki^ni/(Ki^ni + i^ni))` (con `i=max(i,0)`), se reevalúa cada paso; suma `+ u * phi_i * dx` en la ecuación de `i`.
- **Formulación débil (casos)**  
  - `mu = 0`: términos base con efecto Allee en `c` (`rc * c * (c - alle) * (1 - c)`) y acoplamientos `-c*(alpha*s**2 + beta*i**2)`, etc.  
  - `mu > 0`: añade términos proporcionales a `mu` que refuerzan supresión tumoral y modulan `s` e `i`.
- **Solver**: `NonlinearVariationalSolver` (SNES) con fallback entre `linear_solver` (`cg`, `gmres`, `bicgstab`) y precondicionadores (`ilu`, `amg`, `icc`); tolerancias relajadas; no se usan BC explícitas.
- **Malla y espacio**: `RectangleMesh(Point(0,0), Point(space_size, space_size), nodes_in_xaxis, nodes_in_yaxis, "right/left")`; espacio `P1`.

## Flujo de ejecución
1) Cargar parámetros `.env` y banderas.  
2) Para cada bloque `block` en `1..nb`:
   - Crear espacio `V`, inicializar `c, s, i` y copias previas `c_n, s_n, i_n`.  
   - Construir formas `F_c`, `F_s`, `F_i` y solvers.  
   - Bucle de tiempo `t` de `0` a `< T` con paso `dt`:
     - Actualizar control adaptativo con campos actuales.  
     - Resolver `c, s, i`; asignar a `*_n`.  
     - Guardar matrices en texto (`matrix_{field}_{t}_nb_{block}.txt`); evitar NaN/Inf.  
     - Graficar `c, s, i, u` solo en el bloque 1 si `SAVE_IMAGES` activa.

## Entradas y salidas
- **Entrada**: archivo `.env` con parámetros; no requiere datos externos.  
- **Salida**: matrices de los campos por paso temporal; figuras PNG si `SAVE_IMAGES='Y'`.

## Notas de uso
- Seleccionar kernel: `Kernel -> Change Kernel -> fenicsproject (Python 3.10.19)` y reiniciar.  
- Si aparece `ModuleNotFoundError: fenics`, reabrir con el kernel correcto.  
- El control adaptativo depende de `i`; valores muy bajos de `i` intensifican `u`, considera estabilidad numérica.
