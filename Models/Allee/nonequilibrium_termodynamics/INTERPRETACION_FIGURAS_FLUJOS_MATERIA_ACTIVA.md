# Interpretación de las figuras de flujos, gradientes y \(\sigma^+\)

Este documento describe **qué información aportan** los PNG generados por [`visualize_fluxes_and_entropy_density.py`](visualize_fluxes_and_entropy_density.py) y **cómo leerlos**, primero en términos del continuo reacción–difusión y luego en el **contexto biológico** (tumor, tejido sano, inmunidad) y en una **mirada inspirada en materia activa**.

**Salida típica** (por escenario, carpeta `nonequilibrium_plots/`):

| Archivo | Contenido |
|---------|-----------|
| `grad_mag_t_<tiempo>.png` | Tres paneles: \(\|\nabla c\|\), \(\|\nabla s\|\), \(\|\nabla i\|\) |
| `J_mag_t_<tiempo>.png` | Tres paneles: \(\|\mathbf{J}_c\|\), \(\|\mathbf{J}_s\|\), \(\|\mathbf{J}_i\|\) con \(\mathbf{J}_a = -D_a\nabla\phi_a\) |
| `sigma_plus_t_<tiempo>.png` | Densidad \(\sigma^+ = D_c\|\nabla c\|^2 + D_s\|\nabla s\|^2 + D_i\|\nabla i\|^2\) |
| `quiver_Jc_t_<tiempo>.png` | Campo vectorial \(\mathbf{J}_c\) sobre el mapa de \(c\) (si no usas `--no-quiver`) |

**Variables del modelo** (alineado con [`MARCO_FISICO_MATEMATICO.md`](MARCO_FISICO_MATEMATICO.md)):

- \(c(\mathbf{x},t)\): densidad relativa de **células tumorales**  
- \(s(\mathbf{x},t)\): densidad relativa de **tejido sano**  
- \(i(\mathbf{x},t)\): **actividad / densidad relativa del sistema inmune**

Opción útil: `--sigma-csv` escribe `sigma_plus_integral_vs_time.csv` (integral espacial de \(\sigma^+\) frente al tiempo).

---

## 1. Contexto físico-matemático y termodinámico

Esta sección resume el **marco continuo** y la **analogía con la termodinámica clásica del no equilibrio (TdC)** tal como están documentados en el MARCO; las figuras del script son **casos particulares** de esas cantidades espaciales.

### 1.1 Ecuaciones en el continuo (reacción–difusión)

En \(\Omega \subset \mathbb{R}^2\) los campos \(\phi_a \in \{c,s,i\}\) obedecen

\[
\partial_t \phi_a = D_a \nabla^2 \phi_a + R_a(c,s,i),
\qquad D_a > 0 .
\]

Forma equivalente **transporte + fuente** (balance local tipo TdC):

\[
\partial_t \phi_a = -\nabla\cdot \mathbf{J}_a + R_a,
\qquad
\mathbf{J}_a = -D_a \nabla \phi_a .
\]

- \(\mathbf{J}_a\): **flujo difusivo** (Fick lineal isotrópico).  
- \(R_a\): **fuente** reactiva (no conservativa en general): crecimiento, muerte, competencia, inmunidad, Allee, etc.

Las **matrices** guardadas por la simulación son muestreos de \(\phi_a(\mathbf{x},t)\) en una malla; el script de visualización **reconstruye gradientes** por diferencias finitas y el paso \(\Delta x\) a partir de `space_size` y el tamaño de la matriz (igual criterio que en [`calculate_thermodynamic_properties.py`](../termodynamics/calculate_thermodynamic_properties.py)).

### 1.2 Termodinámica del no equilibrio: flujos y fuerzas

En el formalismo de **de Groot–Mazur** / Onsager, la **densidad de producción de entropía** local suele escribirse como suma de productos **flujo × fuerza termodinámica**:

\[
\sigma_S = \sum_\alpha J_\alpha X_\alpha ,
\]

bajo las hipótesis del marco (relaciones constitutivas, localidad, etc.), con \(\sigma_S \geq 0\) en situaciones estándar.

Para **difusión de especies** a **temperatura uniforme** \(T\), un término clásico es

\[
\sigma_S = -\frac{1}{T}\sum_a \mathbf{J}_a \cdot \nabla \mu_a + \cdots
\]

donde \(\mathbf{J}_a\) es el flujo de la especie \(a\) y \(\mu_a\) su **potencial químico**. Con la ley de Fick **\(\mathbf{J}_a = -D_a \nabla \mu_a\)** en el caso ideal donde el potencial coincide con el “motor” del flujo, se obtiene

\[
-\mathbf{J}_a \cdot \nabla \mu_a = D_a \|\nabla \mu_a\|^2 \geq 0,
\]

de modo que ese bloque contribuye **positivamente** a \(\sigma_S\) (al dividir por \(T\)).

### 1.3 Convención en `visualize_fluxes_and_entropy_density.py`: \(\mu_a \equiv \phi_a\)

En este script **no** se cargan los \(\mu_a\) del funcional \(F\) del módulo termodinámico. Se trabaja con **los propios campos** \(\phi_a \in \{c,s,i\}\) como variables que generan el flujo:

\[
\mathbf{J}_a = -D_a \nabla \phi_a .
\]

Entonces el escalar que se grafica como **`sigma_plus_t_*.png`** es la **disipación difusiva total por especie** en la notación del MARCO §4.1:

\[
\sigma_{\mathrm{diss},a} = D_a \|\nabla \phi_a\|^2,
\qquad
\sigma^+ = \sigma_{\mathrm{diss,tot}} = \sum_a \sigma_{\mathrm{diss},a}
= D_c\|\nabla c\|^2 + D_s\|\nabla s\|^2 + D_i\|\nabla i\|^2 .
\]

Es el mismo objeto que el MARCO llama \(\sigma_{\mathrm{diss,tot}}\) y que en [`calculate_thermodynamic_properties.py`](../termodynamics/calculate_thermodynamic_properties.py) se integra y guarda en series temporales (`entropy_production_by_field_t.txt`, etc.). **Coherencia:** las figuras espaciales `sigma_plus_t_*` son el **mapa local** de esa misma cantidad.

A **temperatura efectiva constante** \(T_{\mathrm{eff}}\) (no modelada en el PDE), podrías escribir, por analogía dimensional,

\[
\sigma_S \supset \frac{1}{T_{\mathrm{eff}}} \sum_a D_a \|\nabla \phi_a\|^2
\]

para enfatizar el vínculo con un **término de producción entrópica por difusión**; **\(T_{\mathrm{eff}}\)** solo **rescala** la magnitud, **no** cambia patrones espaciales ni la forma de la evolución temporal relativa de \(\sigma^+\).

### 1.4 Distinción importante: \(\sigma_\mu\) frente a \(\sigma_{\mathrm{diss,tot}}\)

En [`calculate_thermodynamic_properties.py`](../termodynamics/calculate_thermodynamic_properties.py) también se calcula, con potenciales \(\mu_a\) definidos a partir de un funcional de **energía libre efectiva** \(F[c,s,i]\),

\[
\sigma_\mu = D_c\|\nabla \mu_c\|^2 + D_s\|\nabla \mu_s\|^2 + D_i\|\nabla \mu_i\|^2 .
\]

En general **\(\sigma_\mu \neq \sigma_{\mathrm{diss,tot}}\)** salvo que la relación entre \((\mu_c,\mu_s,\mu_i)\) y \((c,s,i)\) sea tal que los gradientes coincidan up to scale. Para **interpretar las figuras de este documento**, lo relevante es **\(\sigma^+ = \sigma_{\mathrm{diss,tot}}\)** en la convención \(\mathbf{J}_a = -D_a\nabla\phi_a\). Las series \(\Sigma_\mu\) del otro módulo responden a otra pregunta: **gradientes del potencial variacional** asociado a \(F\).

### 1.5 Isotermia efectiva: ¿dónde está la temperatura?

**No hay** \(T(\mathbf{x},t)\) en las ecuaciones implementadas: el modelo es **isotermo efectivo**. Los campos \(c,s,i\) son **densidades poblacionales** (o actividades normalizadas), no energía interna.

| Cantidad en TdC continua | En este proyecto |
|--------------------------|------------------|
| \(T\) | No explícita; puede absorbse en \(D_a\) y tasas |
| \(\mathbf{J}_a\) | \(\mathbf{J}_a = -D_a\nabla\phi_a\) (idéntico al formalismo) |
| \(\mu_a\) (TdC estricta) | Aquí, en `visualize_*`, se usa **\(\phi_a\)** como variable del flujo; en el otro módulo, \(\mu_a \sim \delta F/\delta\phi_a\) |
| Producción entrópica completa | **No** implementada: solo **parte difusiva** \(\sigma_{\mathrm{diss,tot}}\) en mapas `sigma_plus_*` |

La comparación útil con la TdC es **estructural** (balance conservativo + flujos + disipación por gradientes), no una medición literal de entropía del tejido.

### 1.6 Qué **no** incluye \(\sigma^+\) (límites del proxy)

Tal como en el MARCO §4.2:

- **Reacciones** \(R_a\) contribuyen a la producción entrópica en un tratamiento completo del no equilibrio químico; aquí **no** se grafica una densidad separada \(\sigma_{\mathrm{reacción}}(\mathbf{x},t)\).  
- **Intercambios con el exterior** (sistema biológico abierto) no están desglosados.  
- Por tanto, \(\sigma^+\) y su integral son **observables fenomenológicos del PDE** sobre la trayectoria simulada: miden **coste estructural de mezcla difusiva** en el continuo, no “entropía medida en laboratorio”.

### 1.7 Energía libre \(F\), potenciales \(\mu_a\) y no reciprocidad (síntesis)

El postproceso termodinámico introduce \(F[c,s,i]\) y \(\mu_a \sim \delta F/\delta\phi_a\) para cuantificar **gradientes de un potencial efectivo**. Sin embargo, si el jacobiano de las tasas,

\[
A_{ab} = \frac{\partial R_a}{\partial \phi_b},
\]

tiene parte **antisimétrica** \(N = (A-A^\top)/2\) no trivial, el subsistema reaccional **no** es integrable en general como gradiente de un único potencial escalar en \((c,s,i)\). Entonces **la dinámica global no tiene por qué** ser relajación pura de ese \(F\), aunque \(F\) y \(\mu_a\) sigan siendo **herramientas** útiles. Herramienta numérica: [`reciprocity_jacobian_analysis.py`](reciprocity_jacobian_analysis.py).

**Para las figuras `grad_mag` / `J_mag` / `sigma_plus`:** esto no cambia la definición de \(\mathbf{J}_a\) ni de \(\sigma_{\mathrm{diss,tot}}\); sí **acota el lenguaje** cuando conectes esas figuras con “descenso de energía libre” o “equilibrio químico local”.

### 1.8 Resumen: encaje formal de cada figura

| Figura | Objeto matemático | Papel en el marco TdC (analogía) |
|--------|-------------------|----------------------------------|
| `grad_mag_t_*` | \(\|\nabla \phi_a\|\) | Magnitud de la “pendiente” espacial que, con \(D_a\), fija la intensidad de flujo y de \(\sigma_{\mathrm{diss},a}\) |
| `J_mag_t_*` | \(\|\mathbf{J}_a\| = D_a\|\nabla\phi_a\|\) | **Flujo difusivo** (par conjugado en productos tipo \(\mathbf{J}_a\cdot\nabla(\cdot)\)) |
| `sigma_plus_t_*` | \(\sigma_{\mathrm{diss,tot}} = \sum_a D_a\|\nabla\phi_a\|^2\) | **Proxy** de contribución **difusiva** a disipación / producción entrópica (sin reacción ni \(1/T\) explícitos) |
| `quiver_Jc_t_*` | \(\mathbf{J}_c\) sobre \(c\) | Misma corriente que en la tabla anterior, con **dirección** en el plano |

Detalle y demás matices: [`MARCO_FISICO_MATEMATICO.md`](MARCO_FISICO_MATEMATICO.md) §§1–5 y §7 (redacción sugerida).

---

## 2. `grad_mag_t_*.png` — Magnitudes de gradiente \(\|\nabla c\|\), \(\|\nabla s\|\), \(\|\nabla i\|\)

### Qué representan

En cada panel se muestra **cuán abrupto es el cambio espacial** del campo correspondiente: cuanto mayor sea \(\|\nabla\phi_a\|\), más fuerte es el contraste entre regiones vecinas en la malla.

### Cómo leer la figura

- **Valores altos** (colores claros en la escala `viridis`): **frentes**, **bordes** de manchas o **interfaces** donde la densidad pasa de un nivel a otro en poca distancia.  
- **Valores bajos** (azul oscuro): regiones **casi uniformes** (poco perfil espacial).  
- Los **tres paneles comparten** el mismo `vmax` cuando el código usa `share_vmax=True`, lo que permite **comparar en un mismo instante** qué compartimento es más “heterogéneo espacialmente” en gradiente puro (antes de ponderar por \(D_a\)).

### Uso interpretativo

Sirve para **localizar patrones** (invasión, huecos, mezclas) y para ver **dónde** el continuo “dobla” más fuerte, independientemente del transporte efectivo.

---

## 3. `J_mag_t_*.png` — Magnitudes de flujo difusivo \(\|\mathbf{J}_a\|\)

### Qué representan

\(\mathbf{J}_a = -D_a\nabla\phi_a\) es la **corriente difusiva** (Fick). Su magnitud es \(D_a\|\nabla\phi_a\|\): combina **pendiente espacial** y **movilidad** \(D_a\).

### Cómo leer la figura

- Un **gradiente grande** con **\(D_a\) pequeño** puede dar un flujo modesto.  
- Un **gradiente moderado** con **\(D_a\) grande** puede dar flujos intensos.  
- Comparar **este panel** con el de `grad_mag` del mismo tiempo aclara si lo que ves es “mucho contraste” o “mucho transporte”.

### Uso interpretativo

Describe **hacia dónde y con qué intensidad** el modelo redistribuye masa (o densidad efectiva) **por difusión** en cada compartimento.

---

## 4. `sigma_plus_t_*.png` — Densidad \(\sigma^+\)

### Qué representa

\[
\sigma^+ = D_c\|\nabla c\|^2 + D_s\|\nabla s\|^2 + D_i\|\nabla i\|^2
= \sigma_{\mathrm{diss,tot}}
\]

(densidad local sobre la malla; véase §1.3–1.6).

### Cómo leer la figura

- **Máximos locales**: zonas donde la **reorganización espacial por difusión** es más intensa en este proxy — suele coincidir con **frentes** y **bordes** donde varios campos cambian.  
- **Regiones oscuras**: configuraciones más **homogéneas** o con gradientes débiles.  
- El script imprime **`integral_sigma_plus`** por tiempo: integral discreta \(\sum \sigma^+ \Delta x \Delta y\). Comparar esa integral **a lo largo de \(t\)** indica si el sistema mantiene mucha estructura de mezcla difusiva o tiende a **relajar** perfiles.

### Uso interpretativo

Útil para **un mapa único** que agrega los tres compartimentos y para **series temporales** con `--sigma-csv`, alineadas con \(\Sigma_{\mathrm{diss}}\) del módulo termodinámico.

---

## 5. `quiver_Jc_t_*.png` — Quiver de \(\mathbf{J}_c\) sobre \(c\)

### Qué representa

Flechas (submuestreadas con `--quiver-skip`) con la **dirección** de \(\mathbf{J}_c = -D_c\nabla c\) y longitud proporcional a la magnitud local, superpuestas al mapa de **densidad tumoral** \(c\).

### Cómo leerla

- **Dirección**: hacia dónde la difusión tiende a **redistribuir** \(c\) (ley de Fick lineal).  
- **Longitud / leyenda `quiverkey`**: dónde el transporte de \(c\) es más intenso.

### Uso interpretativo

Complementa `J_mag` con **orientación** en el caso del tumor; es la figura más directa para narrativas tipo “flujo de células tumorales” en el lenguaje del continuo.

---

## 6. Lectura temporal (varios `t`)

1. **Mismo \(t\), distintos tipos de figura**: localizas **interfaces** (`grad_mag`), **transporte** (`J_mag`), **disipación difusiva** (`sigma_plus`), y opcionalmente **dirección del flujo tumoral** (quiver).  
2. **Mismo tipo de figura, distintos \(t\)**: observas **migración de frentes**, **fragmentación** o **homogeneización**, y la evolución de la **integral** de \(\sigma^+\).  
3. Coherencia: si a un tiempo dado hay fuertes \(\|\nabla c\|\) pero poco \(\|\mathbf{J}_c\|\), revisa **\(D_c\)** en el JSON de escenarios.

---

## 7. Contexto de **materia activa** aplicado a tumor, tejido sano e inmunidad

En física, **materia activa** suele referirse a muchos agentes que **consumen energía** y generan **fases fuera de equilibrio**, **patrones** y **frentes** que no existirían solo con equilibrio térmico pasivo. Tu implementación es un **continuo reacción–difusión** (no un enjambre explícito de partículas autopropulsadas), pero varias ideas del lenguaje de materia activa **iluminan la lectura** de las figuras.

### 7.1 Tres “especies” acopladas fuera de equilibrio

- **Tumor (\(c\))**: crecimiento, Allee y competencia con \(s\) introducen **fuentes locales** \(R_c\) que mantienen el sistema alejado de un equilibrio trivial: es el compartimento que suele mostrar **invasión**, **núcleos** o **colas de frente**.  
- **Tejido sano (\(s\))**: actúa como reservorio modificado por **competencia** y **dinámica de reacción**; sus gradientes reflejan **límites** entre tejido viable y regiones dominadas por \(c\) o por fuertes perfiles de \(i\).  
- **Inmunidad (\(i\))**: en el modelo, \(i\) acopla las tasas (p. ej. control tipo Hill en \(R_i\)). Interpretado en clave activa, el **sistema inmune** es el que introduce **retroalimentación no trivial** y puede generar **estructuras espaciales** (zonas de alta actividad inmune frente a nichos tumorales).

Las imágenes **no** distinguen “actividad” microscópica de cada célula; muestran **campos efectivos** cuya evolución ya incorpora esos procesos activos en \(R_a\).

### 7.2 Frentes e interfaces como “defectos” o estructuras dissipativas

En muchos sistemas activos, los **frentes** son donde se concentra el **intercambio** y la **disipación** efectiva. En tus mapas:

- **`grad_mag`** marca **dónde** está el “borde” entre fases o densidades distintas (análogo a interfaces en medios activos).  
- **`sigma_plus`** suele **iluminar esos bordes**: son los lugares donde la **difusión** mezcla y suaviza contrastes, contribuyendo a \(\sigma_{\mathrm{diss,tot}}\).

Así puedes narrar: *los hotspots de \(\sigma^+\) coinciden con frentes tumor–sano o con regiones de fuerte reorganización del campo inmune*.

### 7.3 Flujos \(\mathbf{J}_a\) y “corrientes” efectivas

En materia activa a menudo se habla de **corrientes** y **polarización**. Aquí la **única corriente espacial explícita** en las figuras es la **difusiva** \(\mathbf{J}_a\). Las **reacciones** no aparecen como vectores en el plano, pero **moldean** los gradientes que luego ves. Por tanto:

- **`J_mag` y el quiver de \(\mathbf{J}_c\)** describen **redistribución espacial** en el sentido de Fick.  
- La **“actividad”** (crecimiento, muerte, ataque inmune) aparece en **cómo cambian** esos campos en el tiempo, no como otra flecha en el mismo gráfico.

Esa distinción es útil en redacción científica: **transporte difusivo** vs **dinámica reaccional local**.

### 7.4 Producción de entropía / disipación (lenguaje cuidadoso)

En sistemas activos reales, la entropía y la energía libre requieren **identificar** baños y potencias suministradas. En este modelo:

- \(\sigma^+\) es la **disipación difusiva** \(\sigma_{\mathrm{diss,tot}}\), no la producción entrópica completa (§1.6).  
- Para **tesis o artículo**, conviene declarar **sistema isotermo efectivo**, flujos \(\mathbf{J}_a = -D_a\nabla\phi_a\) y \(\sigma_{\mathrm{diss,tot}}\) como **proxy** de la parte **difusiva**; separar siempre **reacción** \(R_a\) y citar **no reciprocidad** si usas \(\mu_a\) de \(F\) (MARCO §7 y [`MARCO_FISICO_MATEMATICO.md`](MARCO_FISICO_MATEMATICO.md)).

### 7.5 Preguntas concretas que puedes responder con las figuras

- ¿El tumor avanza con un **frente delgado** (altos \(\|\nabla c\|\) localizados) o con una **cola ancha**?  
- ¿El tejido sano muestra **mesas** casi uniformes o **gradientes** sostenidos por el acoplamiento con \(c\) e \(i\)?  
- ¿La inmunidad \(i\) forma **manchas** o **coronas** alrededor del tumor (visibles en \(\|\nabla i\|\) y en \(\sigma^+\))?  
- ¿La **integral** de \(\sigma^+\) **crece** en fases de invasión rápida y **decae** si el perfil se homogeneiza?

---

## 8. Referencias cruzadas en el repositorio

- Versión formal en LaTeX (mismo contenido, ecuaciones numeradas, PDF vía `pdflatex`): [`interpretacion_flujos_sigma_figuras.tex`](interpretacion_flujos_sigma_figuras.tex)  
- Marco termodinámico completo (\(F\), \(\mu\), \(\sigma_\mu\), reciprocidad, redacción sugerida): [`MARCO_FISICO_MATEMATICO.md`](MARCO_FISICO_MATEMATICO.md)  
- Orden de ejecución del pipeline: [`PIPELINE_EJECUCION_Y_FISICA.md`](../PIPELINE_EJECUCION_Y_FISICA.md)  
- Implementación: [`visualize_fluxes_and_entropy_density.py`](visualize_fluxes_and_entropy_density.py)  
- Termodinámica en malla y series \(\Sigma_\mu\), \(\Sigma_{\mathrm{diss}}\): [`calculate_thermodynamic_properties.py`](../termodynamics/calculate_thermodynamic_properties.py)  

---

*Documento orientado a interpretación de figuras; las fórmulas de \(\mathbf{J}_a\) y \(\sigma^+\) coinciden con [`visualize_fluxes_and_entropy_density.py`](visualize_fluxes_and_entropy_density.py) y con el MARCO §4.1.*
