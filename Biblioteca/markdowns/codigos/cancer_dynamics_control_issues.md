# Observaciones sobre `cancer_dynamics_control.ipynb`

## 1) Salidas no se guardan
- En `field_to_numpy_array` las llamadas a `np.savetxt` están comentadas.  
- Efecto: no se generan los archivos `matrix_*.txt` aunque se impriman los nombres.  
- Fix: descomentar `np.savetxt` (o mover a una ruta válida) para persistir los campos.

## 2) Control adaptativo fuera del Jacobiano
- `AdaptiveControl` usa `UserExpression` con `self.c(x)` e `self.i(x)`.  
- Al ensamblar `F_i`, el término `u` se trata como fuente externa; el Jacobiano de SNES no incluye la dependencia de `c`/`i`.  
- Riesgo: peor convergencia o precisión.  
- Fix: definir el control Hill directamente en la forma variacional para que el derivativo lo capte:
  `u = u_max * (c^nc/(Kc^nc + c^nc)) * (Ki^ni/(Ki^ni + i^ni))` (con `i = max(i,0)`).

## 3) Acoplamiento débil entre ecuaciones
- Cada campo (`c`, `s`, `i`) se resuelve con su propio `NonlinearVariationalSolver` una vez por paso de tiempo.  
- No hay iteración interna (Picard/Newton) sobre el sistema acoplado.  
- Riesgo: con términos cuadráticos y control, una sola pasada tipo Gauss-Seidel puede generar deriva o inestabilidad.  
- Fix: resolver monolíticamente o añadir iteraciones internas hasta convergencia por paso.

## 4) Control puede explotar con `i` pequeño
- El control Hill es acotado en `[0, u_max]`, evitando explosiones cuando `i` es bajo.  
- Riesgo: dificultar la convergencia o generar valores extremos.  
- Mitigación adicional: usar damping de Newton (`relaxation_parameter`) y/o incrementar `max_it` si hay rigidez.

## 5) Entorno
- Error típico: `ModuleNotFoundError: fenics` cuando el kernel no es `fenicsproject`.  
- Mitigación: seleccionar kernel `fenicsproject` y reiniciar antes de ejecutar.

## 6) Resolución de arrays para `correlation_fourier`
- Actualmente se muestrea con `sample_rate = 0.1` → con `space_size=4` se obtienen matrices `41x41` y `dx = sample_rate`.  
- Para mayor densidad: usar `sample_rate` menor (ej. `0.02` → `201x201`; `0.01` → `401x401`).  
- Mantener coherencia de paso espacial: usar el mismo `sample_rate` como `dx` en `correlation_fourier.ipynb` (o derivarlo de la misma variable de entorno).  
- Asegurar rutas: `correlation_fourier` lee `matrix_{field}_{t:.3f}_nb_{block}.txt` en el cwd (`os.chdir(nueva_ruta)`); guardar los `.txt` en el mismo directorio o ajustar `nueva_ruta`.  
- Activar guardado: descomentar `np.savetxt` en `field_to_numpy_array` y, si quieres, definir `sample_rate` en `.env` para controlar la resolución de salida.

## 7) Limitaciones del estudio (PDE y protocolo numérico)

Alineado con `Biblioteca/markdowns/indice_redaccion_tesis_paper.md` y con `Tesis/chapters/06_control.tex`, `07_discusion.tex`.

- **Horizonte temporal corto** (\(T\) del orden 2 en muchas corridas reportadas): los resultados espacio-temporales son **transitorios**; no se pretende caracterización asintótica \(t\to\infty\).
- **Una realización por escenario** (`nb=1` típico): no hay estadística de ensamble sobre condiciones iniciales; la variabilidad estocástica no está cuantificada.
- **Rigidez numérica con control adaptativo**: tolerancias del solver y la forma en que \(u\) entra en la forma débil (véase §2 arriba) pueden exigir **co-diseño** de paso temporal, malla y regularización del control para horizontes largos.
- **Equilibrios homogéneos**: en los barridos tabulados no se reportan equilibrios estables con \(i^*>0\); las conclusiones sobre “umbrales” se formulan en régimen de **inestabilidad lineal** y organización transitoria.
- **Evidencia moderada/débil explícita**: (i) el colapso hacia \(c\approx 0\), \(i\approx 1\) con \(u\) está documentado en **mean-field**; conviene respaldarlo con medias espaciales \(\langle c\rangle(t)\), \(\langle i\rangle(t)\) en PDE largas; (ii) tablas cualitativas coste–beneficio deben reforzarse con \(\iint u\,dx\,dt\), reducción de pico y carga integrada; (iii) no hay comparación directa con un modelo **sin** Allee ni validación contra datos experimentales en este repositorio.
