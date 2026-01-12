# Observaciones sobre `cancer_dynamics_control.ipynb`

## 1) Salidas no se guardan
- En `field_to_numpy_array` las llamadas a `np.savetxt` están comentadas.  
- Efecto: no se generan los archivos `matrix_*.txt` aunque se impriman los nombres.  
- Fix: descomentar `np.savetxt` (o mover a una ruta válida) para persistir los campos.

## 2) Control adaptativo fuera del Jacobiano
- `AdaptiveControl` usa `UserExpression` con `self.c(x)` e `self.i(x)`.  
- Al ensamblar `F_i`, el término `u` se trata como fuente externa; el Jacobiano de SNES no incluye la dependencia de `c`/`i`.  
- Riesgo: peor convergencia o precisión.  
- Fix: definir `u = 0.2 * c / (i + 0.01)` directamente en la forma variacional para que el derivativo lo capte.

## 3) Acoplamiento débil entre ecuaciones
- Cada campo (`c`, `s`, `i`) se resuelve con su propio `NonlinearVariationalSolver` una vez por paso de tiempo.  
- No hay iteración interna (Picard/Newton) sobre el sistema acoplado.  
- Riesgo: con términos cuadráticos y control, una sola pasada tipo Gauss-Seidel puede generar deriva o inestabilidad.  
- Fix: resolver monolíticamente o añadir iteraciones internas hasta convergencia por paso.

## 4) Control puede explotar con `i` pequeño
- `u = 0.2 * c / (i + 0.01)` puede ser grande si `i` inicia bajo.  
- Riesgo: dificultar la convergencia o generar valores extremos.  
- Mitigación: acotar `u` (p.ej. `min(u, u_max)`) o usar un piso mayor para `i` inicial.

## 5) Entorno
- Error típico: `ModuleNotFoundError: fenics` cuando el kernel no es `fenicsproject`.  
- Mitigación: seleccionar kernel `fenicsproject` y reiniciar antes de ejecutar.

## 6) Resolución de arrays para `correlation_fourier`
- Actualmente se muestrea con `sample_rate = 0.1` → con `space_size=4` se obtienen matrices `41x41` y `dx = sample_rate`.  
- Para mayor densidad: usar `sample_rate` menor (ej. `0.02` → `201x201`; `0.01` → `401x401`).  
- Mantener coherencia de paso espacial: usar el mismo `sample_rate` como `dx` en `correlation_fourier.ipynb` (o derivarlo de la misma variable de entorno).  
- Asegurar rutas: `correlation_fourier` lee `matrix_{field}_{t:.3f}_nb_{block}.txt` en el cwd (`os.chdir(nueva_ruta)`); guardar los `.txt` en el mismo directorio o ajustar `nueva_ruta`.  
- Activar guardado: descomentar `np.savetxt` en `field_to_numpy_array` y, si quieres, definir `sample_rate` en `.env` para controlar la resolución de salida.
