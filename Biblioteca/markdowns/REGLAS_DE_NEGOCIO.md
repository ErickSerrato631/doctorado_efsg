# Reglas de Negocio - Sistema de Simulaciones de Cáncer

## Resumen
Este documento describe las reglas de negocio que conectan los diferentes notebooks de simulación y gestionan el flujo de trabajo entre ellos.

## Arquitectura del Sistema

### Componentes Principales

1. **Experimentos (Experiments)**: Representan una configuración completa de simulación
2. **Parámetros (SimulationParameters)**: Valores numéricos y configuración para las simulaciones
3. **Flujos de Trabajo (NotebookWorkflow)**: Define el orden de ejecución de notebooks
4. **Ejecuciones (SimulationRun)**: Registro de cada ejecución de un experimento
5. **Resultados (SimulationResult)**: Archivos y datos generados por cada notebook
6. **Reglas de Negocio (BusinessRule)**: Lógica que conecta notebooks y automatiza decisiones

## Flujo de Trabajo Estándar

### Para Weak Allee
```
1. cancer_dynamics.ipynb      → Simulación de dinámicas
2. correlation_fourier.ipynb   → Análisis espectral
3. correlation_real.ipynb      → Correlaciones espaciales
4. steady_states.ipynb         → Análisis de estados estacionarios
```

### Para Strong Allee
```
1. cancer_dynamics.ipynb      → Simulación de dinámicas
2. correlation_fourier.ipynb   → Análisis espectral
3. correlation_real.ipynb      → Correlaciones espaciales
4. steady_states.ipynb         → Análisis de estados estacionarios
```

## Reglas de Negocio Implementadas

### 1. Regla de Ejecución Condicional de Análisis

**Nombre**: `Ejecutar análisis de correlación solo si dinámicas completaron`

**Condición**:
- Tipo: `notebook_success`
- Notebook fuente: `cancer_dynamics.ipynb`
- Condición: El notebook debe ejecutarse exitosamente

**Acción**:
- Tipo: `execute_notebook`
- Notebook objetivo: `correlation_fourier.ipynb`
- Descripción: Solo ejecuta el análisis de correlación si la simulación de dinámicas fue exitosa

**Configuración JSON**:
```json
{
  "condition_type": "notebook_success",
  "condition_config": {},
  "action_type": "execute_notebook",
  "action_config": {
    "notebook_name": "correlation_fourier.ipynb",
    "result_type": "correlation_analysis"
  },
  "source_notebook": "cancer_dynamics.ipynb",
  "target_notebook": "correlation_fourier.ipynb"
}
```

### 2. Regla de Umbral de Parámetros

**Nombre**: `Ejecutar análisis avanzado si mu > 0`

**Condición**:
- Tipo: `parameter_threshold`
- Parámetro: `mu`
- Operador: `>`
- Umbral: `0`

**Acción**:
- Tipo: `execute_notebook`
- Notebook objetivo: `cancer_dynamics_control.ipynb` (solo para Weak Allee)
- Descripción: Si hay control inmunológico (μ > 0), ejecuta la versión con control

**Configuración JSON**:
```json
{
  "condition_type": "parameter_threshold",
  "condition_config": {
    "parameter": "mu",
    "operator": ">",
    "threshold": 0
  },
  "action_type": "execute_notebook",
  "action_config": {
    "notebook_name": "cancer_dynamics_control.ipynb",
    "result_type": "control_simulation"
  }
}
```

### 3. Regla de Generación de Resultados

**Nombre**: `Crear resultado después de análisis de correlación`

**Condición**:
- Tipo: `notebook_success`
- Notebook fuente: `correlation_real.ipynb`

**Acción**:
- Tipo: `create_result`
- Descripción: Registra automáticamente los resultados de correlación

**Configuración JSON**:
```json
{
  "condition_type": "notebook_success",
  "condition_config": {},
  "action_type": "create_result",
  "action_config": {
    "result_type": "correlation_length",
    "file_path": "corr_length_real_inverse_nb_*_*.txt",
    "metadata": {
      "analysis_type": "spatial_correlation"
    }
  },
  "source_notebook": "correlation_real.ipynb"
}
```

### 4. Regla de Modificación de Parámetros

**Nombre**: `Ajustar paso temporal según resolución espacial`

**Condición**:
- Tipo: `parameter_threshold`
- Parámetro: `nodes_in_xaxis`
- Operador: `>`
- Umbral: `100`

**Acción**:
- Tipo: `modify_parameters`
- Descripción: Reduce el paso temporal si la resolución espacial es alta para mantener estabilidad

**Configuración JSON**:
```json
{
  "condition_type": "parameter_threshold",
  "condition_config": {
    "parameter": "nodes_in_xaxis",
    "operator": ">",
    "threshold": 100
  },
  "action_type": "modify_parameters",
  "action_config": {
    "parameter": "dt",
    "value": 0.05
  }
}
```

## Tipos de Condiciones Soportadas

### 1. `notebook_success`
Evalúa si un notebook se ejecutó exitosamente.

**Configuración**:
```json
{
  "condition_type": "notebook_success",
  "condition_config": {}
}
```

### 2. `parameter_threshold`
Compara un parámetro con un umbral.

**Configuración**:
```json
{
  "condition_type": "parameter_threshold",
  "condition_config": {
    "parameter": "mu",
    "operator": ">",  // >, <, >=, <=, ==
    "threshold": 0.5
  }
}
```

### 3. `always`
Siempre se cumple (útil para acciones que siempre deben ejecutarse).

**Configuración**:
```json
{
  "condition_type": "always",
  "condition_config": {}
}
```

## Tipos de Acciones Soportadas

### 1. `execute_notebook`
Ejecuta un notebook específico.

**Configuración**:
```json
{
  "action_type": "execute_notebook",
  "action_config": {
    "notebook_name": "correlation_fourier.ipynb",
    "result_type": "spectral_analysis"
  }
}
```

### 2. `modify_parameters`
Modifica un parámetro del experimento.

**Configuración**:
```json
{
  "action_type": "modify_parameters",
  "action_config": {
    "parameter": "dt",
    "value": 0.05
  }
}
```

### 3. `create_result`
Crea un registro de resultado manualmente.

**Configuración**:
```json
{
  "action_type": "create_result",
  "action_config": {
    "result_type": "custom_analysis",
    "file_path": "/path/to/result.txt",
    "metadata": {
      "custom_field": "value"
    }
  }
}
```

## Ejemplos de Uso

### Ejemplo 1: Flujo Completo con Reglas

```python
from simulations.models import Experiment, SimulationParameters, BusinessRule
from simulations.services import SimulationService

# Crear experimento
experiment = Experiment.objects.create(
    name="Test Weak Allee",
    allee_type="weak"
)

# Crear parámetros
params = SimulationParameters.objects.create(
    experiment=experiment,
    mu=1.0,  # Con control inmunológico
    alle=0.7,
    T=100.0,
    dt=0.1
)

# Crear regla: si mu > 0, ejecutar versión con control
rule = BusinessRule.objects.create(
    name="Control inmunológico activo",
    description="Ejecuta versión con control si mu > 0",
    condition_type="parameter_threshold",
    condition_config={
        "parameter": "mu",
        "operator": ">",
        "threshold": 0
    },
    action_type="execute_notebook",
    action_config={
        "notebook_name": "cancer_dynamics_control.ipynb"
    },
    source_notebook="cancer_dynamics.ipynb",
    target_notebook="cancer_dynamics_control.ipynb",
    is_active=True
)

# Ejecutar experimento
service = SimulationService()
simulation_run = service.run_experiment(experiment)
```

### Ejemplo 2: Regla de Validación

```python
# Regla que valida que los resultados de correlación existen antes de continuar
validation_rule = BusinessRule.objects.create(
    name="Validar correlaciones antes de análisis avanzado",
    description="Solo ejecuta análisis avanzado si hay resultados de correlación",
    condition_type="notebook_success",
    condition_config={},
    action_type="execute_notebook",
    action_config={
        "notebook_name": "correlation_comparison.ipynb"
    },
    source_notebook="correlation_real.ipynb",
    target_notebook="correlation_comparison.ipynb",
    is_active=True
)
```

## Integración con Notebooks

Los notebooks deben seguir estas convenciones para integrarse correctamente:

1. **Lectura de parámetros**: Los notebooks leen parámetros desde variables de entorno (archivo `.env`)
2. **Guardado de resultados**: Los resultados se guardan con nombres consistentes:
   - Matrices: `matrix_{field}_{time}_nb_{block}.txt`
   - Correlaciones: `corr_length_real_inverse_nb_{block}_{field1}_{field2}.txt`
3. **Manejo de errores**: Los notebooks deben manejar errores apropiadamente
4. **Logging**: Usar logging para facilitar el debugging

## API REST

### Endpoints Disponibles

- `GET /api/experiments/` - Listar experimentos
- `POST /api/experiments/` - Crear experimento
- `POST /api/experiments/{id}/run/` - Ejecutar experimento
- `GET /api/experiments/{id}/runs/` - Ver ejecuciones de un experimento
- `GET /api/runs/{id}/results/` - Ver resultados de una ejecución
- `GET /api/rules/` - Listar reglas de negocio
- `POST /api/rules/` - Crear regla de negocio
- `GET /api/workflows/` - Listar flujos de trabajo

## Próximos Pasos

1. **Crear reglas adicionales** según necesidades específicas
2. **Configurar workflows personalizados** para diferentes tipos de análisis
3. **Implementar notificaciones** cuando se completen simulaciones
4. **Agregar validaciones** de parámetros antes de ejecutar
5. **Implementar scheduling** para ejecuciones programadas






