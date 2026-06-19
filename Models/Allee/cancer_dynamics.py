"""
cancer_dynamics.py

Aplica mejoras de los issues 3 (acoplamiento con iteraciones internas) y 6 (muestreo denso configurable para guardado) 
al caso sin control cancer_dynamics.ipynb.
"""

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx import log
import ufl
from mpi4py import MPI
import numpy as np
import os
from dotenv import load_dotenv
import gc
import sys
import time
import tempfile
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import pyvista
from scipy.interpolate import griddata
import json
from datetime import datetime

# Cargar variables de entorno
load_dotenv()
log.set_log_level(log.LogLevel.WARNING)

# MPI communicator
comm = MPI.COMM_WORLD

# ----------------------------------------------------------------------------
# Configurar directorio temporal (evita errores de SciPy/QHull si /tmp es RO)
# ----------------------------------------------------------------------------
def _pick_writable_tmp_dir(candidates):
    for d in candidates:
        try:
            os.makedirs(d, exist_ok=True)
            test_path = os.path.join(d, ".tmp_write_test")
            with open(test_path, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(test_path)
            return d
        except Exception:
            continue
    return None

_tmp_candidates = [
    os.getenv("TMPDIR"),
    "/var/tmp/allee_tmp",
    "/tmp/allee_tmp",
    os.path.join(os.getcwd(), "tmp"),
]
_tmp_candidates = [d for d in _tmp_candidates if d]
_tmp_dir = _pick_writable_tmp_dir(_tmp_candidates)
if _tmp_dir:
    os.environ["TMPDIR"] = _tmp_dir
    os.environ["TEMP"] = _tmp_dir
    os.environ["TMP"] = _tmp_dir
    tempfile.tempdir = _tmp_dir
    if comm.rank == 0:
        print(f"[Temp] Usando TMPDIR={_tmp_dir}")
else:
    if comm.rank == 0:
        print("⚠ [Temp] No se pudo configurar TMPDIR writable; SciPy puede fallar si /tmp no es escribible.")

# Parámetros del modelo (sin control)
D_c = float(os.getenv('D_c'))
D_s = float(os.getenv('D_s'))
D_i = float(os.getenv('D_i'))
rc = float(os.getenv('rc'))
rs = float(os.getenv('rs'))
rd = float(os.getenv('rd'))
alpha = float(os.getenv('alpha'))
delta = float(os.getenv('delta'))
beta = float(os.getenv('beta'))
alle = float(os.getenv('alle'))
gamma = float(os.getenv('gamma'))
eta = float(os.getenv('eta'))
mu = float(os.getenv('mu'))

# Tipo de efecto Allee
allee_type = os.getenv('ALLEE_TYPE', 'WEAK').upper()
if allee_type not in ['WEAK', 'STRONG']:
    allee_type = 'WEAK'  # Por defecto Weak Allee
    print(f"Advertencia: ALLEE_TYPE inválido, usando WEAK por defecto")

print(f"Tipo de efecto Allee: {allee_type}")

# Control adaptativo u (independiente de mu)
use_adaptive_control = os.getenv('USE_ADAPTIVE_CONTROL', 'N').upper() == 'Y'

if use_adaptive_control:
    u_max = float(os.getenv('U_MAX', '1.0'))
    # Control Hill (Opción A): u = u_max * H_act(c;Kc,nc) * H_inh(i;Ki,ni)
    hill_kc = float(os.getenv("HILL_KC", "0.05"))
    hill_nc = float(os.getenv("HILL_NC", "2"))
    hill_ki = float(os.getenv("HILL_KI", "0.2"))
    hill_ni = float(os.getenv("HILL_NI", "2"))
    print(
        "Control adaptativo (Hill): ACTIVADO "
        f"(u_max={u_max}, Kc={hill_kc}, nc={hill_nc}, Ki={hill_ki}, ni={hill_ni})"
    )
else:
    print("Control adaptativo: DESACTIVADO")

# Mallado / tiempo
nodes_in_xaxis = int(os.getenv('nodes_in_xaxis'))
nodes_in_yaxis = int(os.getenv('nodes_in_yaxis'))
space_size = float(os.getenv('space_size'))
T = float(os.getenv('T'))
dt = float(os.getenv('dt'))
nb = int(os.getenv('nb'))

# Muestreo denso para guardado (issue 6)
sample_rate = float(os.getenv('sample_rate', 0.02))  # menor => matrices más densas
save_images = os.getenv('SAVE_IMAGES', 'N')

# Iteraciones internas (issue 3)
inner_max_iter = int(os.getenv('inner_max_iter', 3))
inner_tol = float(os.getenv('inner_tol', 1e-3))

# Parámetros del solver Newton (dolfinx usa max_it, no max_iter)
newton_max_it_c = int(os.getenv("NEWTON_MAX_IT_C", "100"))
newton_max_it_s = int(os.getenv("NEWTON_MAX_IT_S", "100"))
newton_max_it_i = int(os.getenv("NEWTON_MAX_IT_I", "150"))

newton_relax_c = float(os.getenv("NEWTON_RELAX_C", "1.0"))
newton_relax_s = float(os.getenv("NEWTON_RELAX_S", "1.0"))
# Default más conservador para i cuando hay control (suele ser el más rígido)
_relax_i_env = os.getenv("NEWTON_RELAX_I", "").strip()
if _relax_i_env:
    newton_relax_i = float(_relax_i_env)
else:
    newton_relax_i = 0.7 if use_adaptive_control else 1.0

# Condiciones iniciales desde .env
c_init_min = float(os.getenv('C_INIT_MIN', '0.01'))
c_init_max = float(os.getenv('C_INIT_MAX', '0.02'))
s_init_min = float(os.getenv('S_INIT_MIN', '0.01'))
s_init_max = float(os.getenv('S_INIT_MAX', '0.02'))
i_init_min = float(os.getenv('I_INIT_MIN', '0.9'))
i_init_max = float(os.getenv('I_INIT_MAX', '1.0'))

# Validar que min < max para cada campo
assert c_init_min < c_init_max, "C_INIT_MIN debe ser menor que C_INIT_MAX"
assert s_init_min < s_init_max, "S_INIT_MIN debe ser menor que S_INIT_MAX"
assert i_init_min < i_init_max, "I_INIT_MIN debe ser menor que I_INIT_MAX"

# Perfil espacial de condiciones iniciales.
# UNIFORM conserva el comportamiento histórico. CIRCLE permite sembrar un disco
# aleatorio con rangos distintos dentro/fuera para c, s, i.
init_profile_raw = os.getenv('INIT_PROFILE', 'UNIFORM').strip().upper()
_init_profile_aliases = {
    'UNIFORM': 'UNIFORM',
    'RANDOM': 'UNIFORM',
    'ALEATORIO': 'UNIFORM',
    'CIRCLE': 'CIRCLE',
    'CIRCULAR': 'CIRCLE',
    'DISK': 'CIRCLE',
    'DISC': 'CIRCLE',
}
init_profile = _init_profile_aliases.get(init_profile_raw, 'UNIFORM')
if init_profile_raw and init_profile_raw not in _init_profile_aliases:
    print(f"Advertencia: INIT_PROFILE inválido ({init_profile_raw}); usando UNIFORM")

init_circle_center_x = float(os.getenv('INIT_CIRCLE_CENTER_X', str(space_size / 2.0)))
init_circle_center_y = float(os.getenv('INIT_CIRCLE_CENTER_Y', str(space_size / 2.0)))
init_circle_radius = float(os.getenv('INIT_CIRCLE_RADIUS', str(space_size / 6.0)))
init_circle_edge_width = float(os.getenv('INIT_CIRCLE_EDGE_WIDTH', '0.0'))


def _env_float_or_default(name, default):
    value = os.getenv(name, '').strip()
    return float(value) if value else float(default)


def _read_init_range(prefix, region, default_min, default_max):
    """Lee rangos de CI por región: C_INIT_INSIDE_MIN/MAX, C_INIT_OUTSIDE_MIN/MAX, etc."""
    region = region.upper()
    lo = _env_float_or_default(f'{prefix}_INIT_{region}_MIN', default_min)
    hi = _env_float_or_default(f'{prefix}_INIT_{region}_MAX', default_max)
    if lo >= hi:
        raise ValueError(
            f"{prefix}_INIT_{region}_MIN debe ser menor que {prefix}_INIT_{region}_MAX "
            f"(recibido {lo} >= {hi})"
        )
    return lo, hi


c_init_inside_min, c_init_inside_max = _read_init_range('C', 'INSIDE', c_init_min, c_init_max)
c_init_outside_min, c_init_outside_max = _read_init_range('C', 'OUTSIDE', c_init_min, c_init_max)
s_init_inside_min, s_init_inside_max = _read_init_range('S', 'INSIDE', s_init_min, s_init_max)
s_init_outside_min, s_init_outside_max = _read_init_range('S', 'OUTSIDE', s_init_min, s_init_max)
i_init_inside_min, i_init_inside_max = _read_init_range('I', 'INSIDE', i_init_min, i_init_max)
i_init_outside_min, i_init_outside_max = _read_init_range('I', 'OUTSIDE', i_init_min, i_init_max)

if init_profile == 'CIRCLE':
    if init_circle_radius <= 0.0:
        raise ValueError("INIT_CIRCLE_RADIUS debe ser positivo cuando INIT_PROFILE=CIRCLE")
    if init_circle_edge_width < 0.0:
        raise ValueError("INIT_CIRCLE_EDGE_WIDTH debe ser >= 0")
    print(
        "Condiciones iniciales: CIRCLE "
        f"(center=({init_circle_center_x:.3g}, {init_circle_center_y:.3g}), "
        f"radius={init_circle_radius:.3g}, edge_width={init_circle_edge_width:.3g})"
    )
    print(
        "  Rangos dentro/fuera: "
        f"C=[{c_init_inside_min:g},{c_init_inside_max:g}] / [{c_init_outside_min:g},{c_init_outside_max:g}], "
        f"S=[{s_init_inside_min:g},{s_init_inside_max:g}] / [{s_init_outside_min:g},{s_init_outside_max:g}], "
        f"I=[{i_init_inside_min:g},{i_init_inside_max:g}] / [{i_init_outside_min:g},{i_init_outside_max:g}]"
    )
else:
    print("Condiciones iniciales: UNIFORM")

# Semilla aleatoria (opcional)
random_seed_str = os.getenv('RANDOM_SEED', '')
if random_seed_str.strip():
    random_seed = int(random_seed_str)
    np.random.seed(random_seed)
    print(f"Semilla aleatoria configurada: {random_seed}")

print(f"sample_rate={sample_rate}, inner_iter={inner_max_iter}, inner_tol={inner_tol}")

# ----------------------------------------------------------------------------
# Proyección física a [0, 1] (regularización por capacidad de carga)
# ----------------------------------------------------------------------------
# Las ecs. de la tesis (ecs. 58–60, cap. 3) incluyen los términos de cooperación
# +δ i² s en R_s y +δ s² i en R_i con δ > 0. Cuando c → 0, esos términos pueden
# crecer más rápido que el freno logístico r·φ·(1−φ) y empujar las soluciones
# fuera del cono físico [0,1]³ (k_c = k_s = k_i = 1). Esta proyección recorta
# c, s, i a [0,1] tras cada paso temporal convergido para mantener el resultado
# dentro del rango físico de las capacidades de carga del modelo. Es una
# regularización por barrera; debe reportarse el % de nodos saturados como
# diagnóstico (ver clip_log_interval).
physical_clipping = os.getenv('PHYSICAL_CLIPPING', 'Y').upper() == 'Y'
clip_log_interval = int(os.getenv('CLIP_LOG_INTERVAL', '100'))
if physical_clipping:
    print(
        "Proyección física [0,1] en c, s, i: ACTIVADA "
        f"(log cada {clip_log_interval} pasos)"
    )
else:
    print("Proyección física [0,1] en c, s, i: DESACTIVADA")


def _clip_to_unit_box(c_fn, s_fn, i_fn):
    """Recorta in-place los arrays de c, s, i al intervalo [0, 1].

    Devuelve (n_clip_c, n_clip_s, n_clip_i, n_total): número de nodos
    modificados por cada campo y el total local, para diagnóstico.
    """
    arr_c = c_fn.x.array
    arr_s = s_fn.x.array
    arr_i = i_fn.x.array
    n_c = int(np.count_nonzero((arr_c < 0.0) | (arr_c > 1.0)))
    n_s = int(np.count_nonzero((arr_s < 0.0) | (arr_s > 1.0)))
    n_i = int(np.count_nonzero((arr_i < 0.0) | (arr_i > 1.0)))
    np.clip(arr_c, 0.0, 1.0, out=arr_c)
    np.clip(arr_s, 0.0, 1.0, out=arr_s)
    np.clip(arr_i, 0.0, 1.0, out=arr_i)
    return n_c, n_s, n_i, int(arr_c.size)

# Parámetros de gestión de memoria
monitor_memory = os.getenv('MONITOR_MEMORY', 'Y').upper() == 'Y'
memory_cleanup_interval = int(os.getenv('MEMORY_CLEANUP_INTERVAL', '100'))
memory_warning_threshold_mb = float(os.getenv('MEMORY_WARNING_THRESHOLD_MB', '0'))  # 0 = desactivado
memory_warning_threshold_pct = float(os.getenv('MEMORY_WARNING_THRESHOLD_PCT', '80'))  # Porcentaje de RAM
solver_recreate_interval = int(os.getenv('SOLVER_RECREATE_INTERVAL', '500'))  # Recrear solvers cada N pasos

# Función para monitorear memoria
def get_memory_usage():
    """Retorna el uso de memoria en MB. Usa psutil si está disponible, sino tracemalloc."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / (1024 * 1024)  # Convertir a MB
    except ImportError:
        try:
            import tracemalloc
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            current, peak = tracemalloc.get_traced_memory()
            return current / (1024 * 1024)  # Convertir a MB
        except:
            return 0.0

def get_memory_percentage():
    """Retorna el porcentaje de RAM usado. Requiere psutil."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        mem_percent = process.memory_percent()
        return mem_percent
    except ImportError:
        return 0.0

def check_memory_warning():
    """Verifica si el consumo de memoria excede los umbrales configurados."""
    warnings = []
    if not monitor_memory:
        return warnings
    
    mem_mb = get_memory_usage()
    mem_pct = get_memory_percentage()
    
    if memory_warning_threshold_mb > 0 and mem_mb > memory_warning_threshold_mb:
        warnings.append(f"Memoria alta: {mem_mb:.1f} MB (umbral: {memory_warning_threshold_mb} MB)")
    
    if mem_pct > 0 and mem_pct > memory_warning_threshold_pct:
        warnings.append(f"Memoria alta: {mem_pct:.1f}% de RAM (umbral: {memory_warning_threshold_pct}%)")
    
    return warnings

if monitor_memory:
    print(f"Monitoreo de memoria: ACTIVADO (limpieza cada {memory_cleanup_interval} pasos)")
    print(f"Recreación de solvers: cada {solver_recreate_interval} pasos")
else:
    print("Monitoreo de memoria: DESACTIVADO")

# Parámetros de checkpoint/restart
enable_checkpoint = os.getenv('ENABLE_CHECKPOINT', 'Y').upper() == 'Y'
checkpoint_interval = int(os.getenv('CHECKPOINT_INTERVAL', '500'))
checkpoint_memory_threshold_pct = float(os.getenv('CHECKPOINT_MEMORY_THRESHOLD_PCT', '80'))
checkpoint_restart_threshold_pct = float(os.getenv('CHECKPOINT_RESTART_THRESHOLD_PCT', '85'))
# Mínimo de pasos entre checkpoints “por memoria” (> umbral %) para no saturar I/O (antes: cada paso).
checkpoint_memory_min_interval = int(os.getenv('CHECKPOINT_MEMORY_MIN_INTERVAL', '25'))
checkpoint_max_step_str = os.getenv('CHECKPOINT_MAX_STEP', '')
checkpoint_max_step = int(checkpoint_max_step_str) if checkpoint_max_step_str.strip() else None
if checkpoint_max_step is not None:
    print(f"  [Checkpoint] CHECKPOINT_MAX_STEP configurado: {checkpoint_max_step}")

# Perfil opcional de memoria Python (tracemalloc) en el bucle temporal
memory_profile_interval = int(os.getenv('MEMORY_PROFILE_INTERVAL', '0'))

if enable_checkpoint:
    checkpoint_info = (
        f"Sistema de checkpoint: ACTIVADO (intervalo: {checkpoint_interval} pasos, "
        f"umbral memoria: {checkpoint_memory_threshold_pct}%, umbral reinicio: {checkpoint_restart_threshold_pct}%, "
        f"min_interval_memoria: {checkpoint_memory_min_interval})"
    )
    if checkpoint_max_step is not None:
        checkpoint_info += f"\n  Checkpoint máximo permitido: paso {checkpoint_max_step} (evitará checkpoints posteriores)"
    print(checkpoint_info)
else:
    print("Sistema de checkpoint: DESACTIVADO")

# ============================================================================
# Diagnóstico de fallos/no-convergencia
# ============================================================================
enable_diagnostics = os.getenv('ENABLE_DIAGNOSTICS', 'Y').upper() == 'Y'
diagnostic_sample_rate = float(os.getenv('DIAGNOSTIC_SAMPLE_RATE', '0.05'))
diagnostic_on_warning = os.getenv('DIAGNOSTIC_ON_WARNING', 'Y').upper() == 'Y'
diagnostic_warning_min_step_gap = int(os.getenv('DIAGNOSTIC_WARNING_MIN_STEP_GAP', '10'))

if memory_profile_interval > 0 and comm.rank == 0:
    print(
        f"Perfil de memoria Python: ACTIVADO (cada {memory_profile_interval} pasos; "
        "RSS + tracemalloc top). Desactivar con MEMORY_PROFILE_INTERVAL=0"
    )

if enable_diagnostics:
    print(
        "Diagnóstico: ACTIVADO "
        f"(sample_rate={diagnostic_sample_rate}, "
        f"on_warning={diagnostic_on_warning}, "
        f"min_step_gap={diagnostic_warning_min_step_gap})"
    )
else:
    print("Diagnóstico: DESACTIVADO")

# Función para limpiar cachés de PETSc
def cleanup_petsc_caches():
    """Intenta limpiar cachés de PETSc si está disponible"""
    try:
        from petsc4py import PETSc
        PETSc.garbage_cleanup()
        return True
    except ImportError:
        return False
    except Exception:
        return False


# Reintentos ante fallos E/S (p. ej. Google Drive / FUSE: errno 5 EIO)
io_write_max_attempts = max(1, int(os.getenv("IO_WRITE_MAX_ATTEMPTS", "3")))
io_write_retry_delay_sec = max(0.0, float(os.getenv("IO_WRITE_RETRY_DELAY_SEC", "3")))


def _io_write_with_retries(operation, description: str):
    """Ejecuta ``operation()`` reintentando solo ante ``OSError``."""
    last_exc = None
    for attempt in range(1, io_write_max_attempts + 1):
        try:
            operation()
            return
        except OSError as e:
            last_exc = e
            if attempt >= io_write_max_attempts:
                break
            if comm.rank == 0:
                print(
                    f"  ⚠ {description}: fallo E/S (intento {attempt}/{io_write_max_attempts}): {e}\n"
                    f"     Reintento en {io_write_retry_delay_sec:.1f}s..."
                )
            time.sleep(io_write_retry_delay_sec)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("_io_write_with_retries: estado inesperado")


if comm.rank == 0 and io_write_max_attempts > 1:
    print(
        f"Reintentos E/S: hasta {io_write_max_attempts} intentos, "
        f"pausa {io_write_retry_delay_sec:.1f}s (IO_WRITE_MAX_ATTEMPTS / IO_WRITE_RETRY_DELAY_SEC)"
    )

# Funciones de checkpoint/restart
def save_checkpoint(c, s, i, c_n, s_n, i_n, t, step, block):
    """Guarda el estado actual de la simulación en un archivo checkpoint."""
    checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Guardar checkpoint latest (siempre)
    checkpoint_path_latest = os.path.join(checkpoint_dir, 'checkpoint_latest.npz')
    
    # Guardar checkpoint numerado (para poder seleccionar uno anterior)
    checkpoint_path_numbered = os.path.join(checkpoint_dir, f'checkpoint_step_{step:06d}.npz')
    
    checkpoint_data = {
        'c_array': c.x.array.copy(),
        's_array': s.x.array.copy(),
        'i_array': i.x.array.copy(),
        'c_n_array': c_n.x.array.copy(),
        's_n_array': s_n.x.array.copy(),
        'i_n_array': i_n.x.array.copy(),
        't': t,
        'step': step,
        'block': block
    }

    def _write_checkpoints():
        np.savez_compressed(checkpoint_path_latest, **checkpoint_data)
        np.savez_compressed(checkpoint_path_numbered, **checkpoint_data)

    _io_write_with_retries(_write_checkpoints, f"checkpoint step={step} t={t:.4f}")

    return checkpoint_path_latest

def load_checkpoint(max_step=None):
    """
    Carga el estado de la simulación desde un checkpoint si existe.
    
    Args:
        max_step: Si se especifica, busca el checkpoint más reciente antes de este paso.
                  Útil para evitar checkpoints problemáticos.
    """
    checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')
    import glob
    
    # Primero verificar el checkpoint latest si existe
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.npz')
    latest_step = None
    
    if os.path.exists(latest_path):
        try:
            latest_data = np.load(latest_path)
            latest_step = int(latest_data['step'])
            if max_step is not None:
                print(f"  [Checkpoint] Latest encontrado: paso {latest_step}, max_step permitido: {max_step}")
        except Exception as e:
            if max_step is not None:
                print(f"  [Checkpoint] Error al leer latest: {e}")
            pass
    
    # Si max_step está especificado y el latest es >= max_step, buscar uno anterior
    if max_step is not None and latest_step is not None and latest_step >= max_step:
        print(f"  [Checkpoint] Latest ({latest_step}) >= max_step ({max_step}), buscando anterior...")
        # Buscar el checkpoint más reciente antes de max_step
        checkpoint_pattern = os.path.join(checkpoint_dir, 'checkpoint_step_*.npz')
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        best_checkpoint = None
        best_step = -1
        
        for checkpoint_file in checkpoint_files:
            try:
                # Extraer el paso del nombre del archivo
                filename = os.path.basename(checkpoint_file)
                step_str = filename.replace('checkpoint_step_', '').replace('.npz', '')
                step = int(step_str)
                
                # Si el paso es menor que max_step y es el más reciente hasta ahora
                if step < max_step and step > best_step:
                    best_step = step
                    best_checkpoint = checkpoint_file
            except (ValueError, IndexError):
                continue
        
        if best_checkpoint:
            checkpoint_path = best_checkpoint
            print(f"  ⚠ Checkpoint latest es del paso {latest_step} (problemático)")
            print(f"  ✓ Usando checkpoint anterior: paso {best_step} (evitando paso {max_step})")
        else:
            # No se encontró checkpoint anterior, no cargar ninguno (empezar desde cero)
            print(f"  ⚠ Checkpoint latest es del paso {latest_step} (problemático)")
            print(f"  ⚠ No se encontró checkpoint anterior al paso {max_step}")
            print(f"  → Empezando desde cero (t=0)")
            return None
    elif max_step is not None:
        # Buscar el checkpoint más reciente antes de max_step (aunque latest sea válido)
        checkpoint_pattern = os.path.join(checkpoint_dir, 'checkpoint_step_*.npz')
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        best_checkpoint = None
        best_step = -1
        
        for checkpoint_file in checkpoint_files:
            try:
                filename = os.path.basename(checkpoint_file)
                step_str = filename.replace('checkpoint_step_', '').replace('.npz', '')
                step = int(step_str)
                
                if step < max_step and step > best_step:
                    best_step = step
                    best_checkpoint = checkpoint_file
            except (ValueError, IndexError):
                continue
        
        if best_checkpoint:
            checkpoint_path = best_checkpoint
            print(f"  ✓ Usando checkpoint anterior: paso {best_step} (evitando paso {max_step})")
        else:
            # No se encontró checkpoint anterior, empezar desde cero
            print(f"  ⚠ No se encontró checkpoint anterior al paso {max_step}")
            print(f"  → Empezando desde cero (t=0)")
            return None
    else:
        # Comportamiento normal: usar latest solo si no hay restricción de max_step
        # IMPORTANTE: Si max_step está definido, solo usar latest si latest_step < max_step
        if max_step is not None and latest_step is not None and latest_step >= max_step:
            # Latest es problemático, no usar
            print(f"  ⚠ Latest ({latest_step}) >= max_step ({max_step}), rechazando latest")
            return None
        checkpoint_path = latest_path
    
    if not os.path.exists(checkpoint_path):
        if max_step is not None:
            print(f"  [Checkpoint] No existe checkpoint en {checkpoint_path}")
        return None
    
    try:
        checkpoint_data = np.load(checkpoint_path)
        loaded_step = int(checkpoint_data['step'])
        
        # Verificación CRÍTICA: si el checkpoint cargado es >= max_step, rechazarlo SIEMPRE
        if max_step is not None:
            print(f"  [Checkpoint] Verificando checkpoint cargado: paso {loaded_step}, max_step: {max_step}")
            if loaded_step >= max_step:
                print(f"  ⚠ Checkpoint cargado es del paso {loaded_step} (>= {max_step}, RECHAZADO)")
                print(f"  → Empezando desde cero (t=0)")
                return None
            else:
                print(f"  ✓ Checkpoint válido: paso {loaded_step} < {max_step}")
        
        return {
            'c_array': checkpoint_data['c_array'],
            's_array': checkpoint_data['s_array'],
            'i_array': checkpoint_data['i_array'],
            'c_n_array': checkpoint_data['c_n_array'],
            's_n_array': checkpoint_data['s_n_array'],
            'i_n_array': checkpoint_data['i_n_array'],
            't': float(checkpoint_data['t']),
            'step': int(checkpoint_data['step']),
            'block': int(checkpoint_data['block'])
        }
    except Exception as e:
        print(f"⚠ Error al cargar checkpoint: {e}")
        return None

# Verificar instalación de FEniCSx
try:
    import dolfinx
    print(f"FEniCSx version: {dolfinx.__version__}")
except ImportError:
    print("FEniCSx no está instalado. Usa el kernel 'Python (FEniCSx)'")

# Ruta de salida - usar directorio actual cuando se ejecuta desde run_all_scenarios.py
nueva_ruta = os.getcwd()
os.chdir(nueva_ruta)
print(f"Salida a: {os.getcwd()}")

# Crear subcarpetas: matrices e imágenes en el directorio actual (junto con las matrices)
matrices_dir = os.path.join(nueva_ruta, 'matrices')
images_dir = os.path.join(nueva_ruta, 'images')
diagnostics_dir = os.path.join(nueva_ruta, 'diagnostics')
os.makedirs(matrices_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)
os.makedirs(diagnostics_dir, exist_ok=True)
print(f"✓ Matrices se guardarán en: {matrices_dir}")
print(f"✓ Imágenes se guardarán en: {images_dir}")
print(f"✓ Diagnósticos se guardarán en: {diagnostics_dir}")

# ============================================================================
# Funciones auxiliares
# ============================================================================

def _gather_coords_values_for_rank0(fenics_field):
    """
    Retorna (coords, values) en rank 0. En otros ranks retorna (None, None).
    Nota: diseñado para ser robusto si se ejecuta en MPI; en serial no hace nada especial.
    """
    domain = fenics_field.function_space.mesh
    coords_local = domain.geometry.x[:, :2]
    values_local = fenics_field.x.array

    if comm.size == 1:
        return coords_local, values_local

    coords_list = comm.gather(coords_local, root=0)
    values_list = comm.gather(values_local, root=0)
    if comm.rank != 0:
        return None, None

    coords = np.vstack([c for c in coords_list if c is not None and len(c) > 0])
    values = np.concatenate([v for v in values_list if v is not None and len(v) > 0])
    return coords, values


def sample_field_on_grid(fenics_field, space_size, sample_rate):
    """
    Muestrea un campo FEniCSx en una grilla regular usando interpolación `griddata`.
    Retorna (grid, x, y) en rank 0; en otros ranks retorna (None, None, None).
    """
    coords, values = _gather_coords_values_for_rank0(fenics_field)
    if comm.rank != 0:
        return None, None, None

    sample_points = np.linspace(0, space_size, int(space_size / sample_rate) + 1)
    X, Y = np.meshgrid(sample_points, sample_points)
    grid = griddata(coords, values, (X, Y), method='linear', fill_value=0.0)
    grid = np.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)
    del X, Y
    return grid, sample_points, sample_points


def compute_field_stats(grid):
    """Estadísticas robustas de un grid 2D (numpy)."""
    if grid is None:
        return {}
    flat = grid.ravel()
    return {
        'min': float(np.min(flat)),
        'max': float(np.max(flat)),
        'mean': float(np.mean(flat)),
        'std': float(np.std(flat)),
        'p01': float(np.percentile(flat, 1)),
        'p05': float(np.percentile(flat, 5)),
        'p50': float(np.percentile(flat, 50)),
        'p95': float(np.percentile(flat, 95)),
        'p99': float(np.percentile(flat, 99)),
    }


def compute_gradient_stats(grid, h):
    """
    Estadísticas de gradiente (magnitud) sobre una grilla regular.
    """
    if grid is None:
        return {}
    try:
        gx, gy = np.gradient(grid, h, h)
        gmag = np.sqrt(gx * gx + gy * gy)
        flat = gmag.ravel()
        return {
            'max_grad': float(np.max(flat)),
            'mean_grad': float(np.mean(flat)),
            'p95_grad': float(np.percentile(flat, 95)),
            'p99_grad': float(np.percentile(flat, 99)),
        }
    except Exception as e:
        return {'error': str(e)}


def compute_u_grid_and_stats(c_grid, i_grid):
    """
    Calcula u tipo Hill (Opción A) sobre la grilla muestreada:
      u = u_max * H_act(c;Kc,nc) * H_inh(i;Ki,ni)
    """
    if (c_grid is None) or (i_grid is None) or (not use_adaptive_control):
        return None, {}

    # Proteger contra i negativa (overshoot numérico)
    i_pos = np.maximum(i_grid, 0.0)
    # Hill activation in c
    c_pow = np.power(c_grid, hill_nc)
    kc_pow = float(hill_kc) ** float(hill_nc)
    h_act = c_pow / (kc_pow + c_pow + 1e-16)
    # Hill inhibition in i
    i_pow = np.power(i_pos, hill_ni)
    ki_pow = float(hill_ki) ** float(hill_ni)
    h_inh = ki_pow / (ki_pow + i_pow + 1e-16)
    u_grid = float(u_max) * h_act * h_inh
    u_grid = np.nan_to_num(u_grid, nan=0.0, posinf=float(u_max), neginf=0.0)
    # Asegurar rango [0, u_max]
    u_grid = np.clip(u_grid, 0.0, float(u_max))

    flat = u_grid.ravel()
    saturated = float(np.mean(flat >= (0.99 * u_max))) if u_max > 0 else 0.0
    return u_grid, {
        'min': float(np.min(flat)),
        'max': float(np.max(flat)),
        'mean': float(np.mean(flat)),
        'std': float(np.std(flat)),
        'saturation_ratio': saturated
    }


def save_diagnostic_snapshot(c, s, i, t, step, block, reason, extra=None):
    """
    Guarda snapshot diagnóstico (NPZ + JSON) en diagnostics/ del directorio del escenario.
    Solo rank 0 escribe archivos.
    """
    if not enable_diagnostics:
        return None

    # Preparar muestreo a grilla regular (rank 0)
    c_grid, x, y = sample_field_on_grid(c, space_size, diagnostic_sample_rate)
    s_grid, _, _ = sample_field_on_grid(s, space_size, diagnostic_sample_rate)
    i_grid, _, _ = sample_field_on_grid(i, space_size, diagnostic_sample_rate)

    if comm.rank != 0:
        return None

    # Espaciado real de grilla
    h = float(x[1] - x[0]) if (x is not None and len(x) > 1) else float(diagnostic_sample_rate)

    u_grid, u_stats = compute_u_grid_and_stats(c_grid, i_grid)

    payload = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'reason': reason,
        't': float(t),
        'step': int(step),
        'block': int(block),
        'dt': float(dt),
        'inner_max_iter': int(inner_max_iter),
        'inner_tol': float(inner_tol),
        'diagnostic_sample_rate': float(diagnostic_sample_rate),
        'use_adaptive_control': bool(use_adaptive_control),
        'U_MAX': float(u_max) if use_adaptive_control else None,
        'HILL_KC': float(hill_kc) if use_adaptive_control else None,
        'HILL_NC': float(hill_nc) if use_adaptive_control else None,
        'HILL_KI': float(hill_ki) if use_adaptive_control else None,
        'HILL_NI': float(hill_ni) if use_adaptive_control else None,
        'c_stats': compute_field_stats(c_grid),
        's_stats': compute_field_stats(s_grid),
        'i_stats': compute_field_stats(i_grid),
        'u_stats': u_stats,
        'grad_c': compute_gradient_stats(c_grid, h),
        'grad_s': compute_gradient_stats(s_grid, h),
        'grad_i': compute_gradient_stats(i_grid, h),
        'extra': extra or {}
    }

    # Nombres de archivo
    safe_t = f"{t:.6f}".replace('.', 'p')
    base = f"diagnostic_step_{int(step):06d}_t_{safe_t}_b_{int(block):02d}"
    out_npz = os.path.join(diagnostics_dir, f"{base}.npz")
    out_json = os.path.join(diagnostics_dir, f"{base}.json")

    # Guardar NPZ (grids + algunos scalars)
    np.savez_compressed(
        out_npz,
        c_grid=c_grid,
        s_grid=s_grid,
        i_grid=i_grid,
        u_grid=u_grid if u_grid is not None else np.array([], dtype=float),
        x=x,
        y=y,
        t=float(t),
        step=int(step),
        block=int(block),
        reason=str(reason),
    )

    # Guardar JSON legible
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"  [Diagnóstico] Snapshot guardado: {out_json}")
    return out_json


def _log_memory_profile_step(step: int, t_phys: float) -> None:
    """Registro opcional RSS + tracemalloc para localizar crecimiento en el bucle temporal."""
    if memory_profile_interval <= 0 or comm.rank != 0:
        return
    if step <= 0 or step % memory_profile_interval != 0:
        return
    rss = get_memory_usage()
    msg = f"  [MemProfile] paso={step} t={t_phys:.4f} RSS_MB={rss:.1f}"
    try:
        import tracemalloc
        if not tracemalloc.is_tracing():
            tracemalloc.start(25)
        cur, peak = tracemalloc.get_traced_memory()
        msg += f" | tracemalloc_cur_MB={cur / 1024 / 1024:.2f} peak_MB={peak / 1024 / 1024:.2f}"
        snap = tracemalloc.take_snapshot()
        for i, st in enumerate(snap.statistics("lineno")[:5], 1):
            loc = st.trace[0] if st.trace else ""
            msg += f"\n      #{i} {st.count} blks {st.size / 1024:.0f} KB {loc}"
    except Exception as ex:
        msg += f" | ({type(ex).__name__}: {ex})"
    print(msg)


def create_space_function(space_size, nx, ny):
    domain = mesh.create_rectangle(comm, [np.array([0.0, 0.0]), np.array([space_size, space_size])], [nx, ny], cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("CG", 1))  # Usar functionspace (minúscula), no FunctionSpace
    return domain, V

def field_to_numpy_array(fenics_field, space_size, step, field_name, block, sample_rate=0.02):
    sample_points = np.linspace(0, space_size, int(space_size / sample_rate) + 1)
    field_array = np.empty((len(sample_points), len(sample_points)), dtype=float)
    
    # Obtener coordenadas y valores del campo directamente desde la malla
    domain = fenics_field.function_space.mesh
    coords = domain.geometry.x[:, :2]  # Solo coordenadas x, y
    values = fenics_field.x.array
    
    # Interpolar valores en los puntos de muestreo
    X, Y = np.meshgrid(sample_points, sample_points)
    field_array = griddata(coords, values, (X, Y), method='linear', fill_value=0.0)
    
    # Limpiar arrays temporales de interpolación
    del X, Y
    
    field_array = np.nan_to_num(field_array, nan=0.0, posinf=0.0, neginf=0.0)
    # Guardar en subcarpeta matrices/ del directorio actual
    matrices_dir = os.path.join(os.getcwd(), 'matrices')
    os.makedirs(matrices_dir, exist_ok=True)
    filename = f"matrix_{field_name}_{step}_nb_{block}.txt"
    filepath = os.path.join(matrices_dir, filename)
    tmp_path = filepath + ".part"

    def _write_matrix_file():
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        np.savetxt(tmp_path, field_array, delimiter="\t", fmt="%.8e")
        os.replace(tmp_path, filepath)

    _io_write_with_retries(
        _write_matrix_file,
        f"matrix {field_name} t={step} nb={block}",
    )
    
    # Limpiar array después de guardar
    del field_array
    
    return filepath

# Colores de campos en plot_fields: menor densidad = negro; ver run_scenarios / images/
_FIELD_CMAP_C = LinearSegmentedColormap.from_list(
    "allee_c", [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0)], N=256
)  # c: negro → amarillo
_FIELD_CMAP_S = LinearSegmentedColormap.from_list(
    "allee_s", [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)], N=256
)  # s: negro → blanco
_FIELD_CMAP_I = LinearSegmentedColormap.from_list(
    "allee_i", [(0.0, 0.0, 0.0), (0.66, 0.84, 1.0)], N=256
)  # i: negro → azul claro
_FIELD_CMAP_U = LinearSegmentedColormap.from_list(
    "allee_u", [(0.0, 0.0, 0.0), (0.56, 0.93, 0.56)], N=256
)  # Hill u: negro → verde claro (~#90EE90)


def plot_fields(field_c, field_s, field_i, block, t, u_expr=None, V=None):
    # Obtener la malla y coordenadas directamente
    mesh_c = field_c.function_space.mesh
    
    # En DOLFINx 0.10.0, acceder directamente a las coordenadas
    coords_c = mesh_c.geometry.x[:, :2]  # Solo coordenadas x, y
    values_c = field_c.x.array
    
    mesh_s = field_s.function_space.mesh
    coords_s = mesh_s.geometry.x[:, :2]
    values_s = field_s.x.array
    
    mesh_i = field_i.function_space.mesh
    coords_i = mesh_i.geometry.x[:, :2]
    values_i = field_i.x.array
    
    # Determinar número de subplots
    if u_expr is not None and V is not None:
        n_subplots = 4
        figsize = (20, 5)
    else:
        n_subplots = 3
        figsize = (16, 5)
    
    plt.figure(figsize=figsize)

    # Crear malla regular para interpolación
    x = np.linspace(0, space_size, 100)
    y = np.linspace(0, space_size, 100)
    X, Y = np.meshgrid(x, y)

    # Cáncer
    plt.subplot(1, n_subplots, 1)
    Z_c = griddata(coords_c, values_c, (X, Y), method='linear', fill_value=0.0)
    p1 = plt.contourf(X, Y, Z_c, levels=20, cmap=_FIELD_CMAP_C)
    plt.colorbar(p1)
    plt.title(f'c at t = {t:.3f}')

    # Sanas
    plt.subplot(1, n_subplots, 2)
    Z_s = griddata(coords_s, values_s, (X, Y), method='linear', fill_value=0.0)
    p2 = plt.contourf(X, Y, Z_s, levels=20, cmap=_FIELD_CMAP_S)
    plt.colorbar(p2)
    plt.title(f's at t = {t:.3f}')

    # Inmune
    plt.subplot(1, n_subplots, 3)
    Z_i = griddata(coords_i, values_i, (X, Y), method='linear', fill_value=0.0)
    p3 = plt.contourf(X, Y, Z_i, levels=20, cmap=_FIELD_CMAP_I)
    plt.colorbar(p3)
    plt.title(f'i at t = {t:.3f}')

    # Control adaptativo u (si está activado)
    if u_expr is not None and V is not None:
        plt.subplot(1, n_subplots, 4)
        # Calcular u (Hill) directamente desde los valores de c e i en nodos
        coords_u = mesh_c.geometry.x[:, :2]
        values_c_at_u = field_c.x.array
        values_i_at_u = field_i.x.array
        i_pos = np.maximum(values_i_at_u, 0.0)
        # Hill activation in c
        c_pow = np.power(values_c_at_u, hill_nc)
        kc_pow = float(hill_kc) ** float(hill_nc)
        h_act = c_pow / (kc_pow + c_pow + 1e-16)
        # Hill inhibition in i
        i_pow = np.power(i_pos, hill_ni)
        ki_pow = float(hill_ki) ** float(hill_ni)
        h_inh = ki_pow / (ki_pow + i_pow + 1e-16)
        values_u = float(u_max) * h_act * h_inh
        values_u = np.clip(values_u, 0.0, float(u_max))
        Z_u = griddata(coords_u, values_u, (X, Y), method='linear', fill_value=0.0)
        p4 = plt.contourf(X, Y, Z_u, levels=20, cmap=_FIELD_CMAP_U)
        plt.colorbar(p4)
        plt.title(f'u at t = {t:.3f}')

    plt.tight_layout(pad=4)
    # Guardar en subcarpeta images/ del directorio actual (junto con las matrices)
    if save_images == 'Y':
        try:
            images_dir = os.path.join(os.getcwd(), 'images')
            os.makedirs(images_dir, exist_ok=True)
            image_path = os.path.join(images_dir, f'fields_block_{block}_step_{t:.3f}.png')
            plt.savefig(image_path, dpi=200)
        except Exception as save_error:
            # Si falla el guardado de imagen, continuar sin detener la simulación
            print(f"⚠ ADVERTENCIA: Error al guardar imagen en t={t:.4f}: {save_error}")
            print(f"  Continuando con la simulación...")
    try:
        plt.close()
    except:
        pass  # Ignorar errores al cerrar matplotlib
    
    # Limpiar arrays temporales de interpolación usados en plot_fields
    try:
        del X, Y, Z_c, Z_s, Z_i
        if u_expr is not None and V is not None:
            del Z_u, values_u, coords_u, values_c_at_u, values_i_at_u
    except NameError:
        pass  # Variables ya eliminadas o no existen


def _diagnostic_rate_limit_should_save(last_step_saved, current_step, min_gap):
    """Rate-limit simple por step para evitar miles de snapshots."""
    if last_step_saved is None:
        return True
    return (current_step - last_step_saved) >= max(1, int(min_gap))

# ============================================================================
# Setup del problema sin control
# ============================================================================

def _configure_nlp_snes(nlp, rtol, atol, max_it, relaxation):
    """SNES en dolfinx ≥0.10 (NonlinearProblem): tolerancias y amortiguación (equiv. relaxation_parameter)."""
    from petsc4py import PETSc

    snes = nlp.solver
    snes.setTolerances(rtol=rtol, atol=atol, max_it=max_it)
    # petsc4py a veces no expone SNESLineSearch.setDamping; usar opciones PETSc (portable).
    opts = PETSc.Options()
    opts.prefixPush(snes.getOptionsPrefix())
    opts["snes_linesearch_damping"] = str(float(relaxation))
    opts.prefixPop()
    snes.setFromOptions()
    # setFromOptions puede tocar tolerancias; fijarlas de nuevo por código.
    snes.setTolerances(rtol=rtol, atol=atol, max_it=max_it)


def _random_values_in_range(lo, hi, n):
    return float(lo) + (float(hi) - float(lo)) * np.random.rand(int(n))


def _circle_weights_for_dofs(V, n_values):
    """
    Peso espacial del disco para cada DOF: 1 dentro, 0 fuera.

    Si INIT_CIRCLE_EDGE_WIDTH > 0, el borde se suaviza con una transición logística,
    útil para evitar discontinuidades iniciales demasiado rígidas.
    """
    coords = V.tabulate_dof_coordinates()[:, :2]
    if len(coords) != int(n_values):
        coords = V.mesh.geometry.x[:, :2]
    if len(coords) != int(n_values):
        raise ValueError(
            "No se pudieron empatar coordenadas de DOFs con el tamaño del campo "
            f"({len(coords)} coords vs {n_values} valores)"
        )

    dx0 = coords[:, 0] - init_circle_center_x
    dy0 = coords[:, 1] - init_circle_center_y
    radius = np.sqrt(dx0 * dx0 + dy0 * dy0)
    if init_circle_edge_width > 0.0:
        z = (radius - init_circle_radius) / init_circle_edge_width
        z = np.clip(z, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(z))
    return (radius <= init_circle_radius).astype(float)


def _assign_circle_random_field(field, V, inside_range, outside_range):
    n = len(field.x.array)
    weights = _circle_weights_for_dofs(V, n)
    inside = _random_values_in_range(inside_range[0], inside_range[1], n)
    outside = _random_values_in_range(outside_range[0], outside_range[1], n)
    field.x.array[:] = weights * inside + (1.0 - weights) * outside


def assign_initial_conditions(c_n, s_n, i_n, V):
    """Asigna CI aleatorias uniformes o espacialmente enfocadas según INIT_PROFILE."""
    if init_profile == 'CIRCLE':
        _assign_circle_random_field(
            c_n, V,
            (c_init_inside_min, c_init_inside_max),
            (c_init_outside_min, c_init_outside_max),
        )
        _assign_circle_random_field(
            s_n, V,
            (s_init_inside_min, s_init_inside_max),
            (s_init_outside_min, s_init_outside_max),
        )
        _assign_circle_random_field(
            i_n, V,
            (i_init_inside_min, i_init_inside_max),
            (i_init_outside_min, i_init_outside_max),
        )
    else:
        c_n.x.array[:] = _random_values_in_range(c_init_min, c_init_max, len(c_n.x.array))
        s_n.x.array[:] = _random_values_in_range(s_init_min, s_init_max, len(s_n.x.array))
        i_n.x.array[:] = _random_values_in_range(i_init_min, i_init_max, len(i_n.x.array))


def solve_dynamics():
    domain, V = create_space_function(space_size, nodes_in_xaxis, nodes_in_yaxis)
    c = fem.Function(V)
    s = fem.Function(V)
    i = fem.Function(V)

    phi_c = ufl.TestFunction(V)
    phi_s = ufl.TestFunction(V)
    phi_i = ufl.TestFunction(V)

    # Condiciones iniciales aleatorias suaves
    c_n = fem.Function(V)
    s_n = fem.Function(V)
    i_n = fem.Function(V)

    assign_initial_conditions(c_n, s_n, i_n, V)

    # Formas variacionales
    dx = ufl.Measure("dx", domain=domain)
    
    # Control adaptativo u (si está activado)
    if use_adaptive_control:
        # Control Hill: u = u_max * H_act(c;Kc,nc) * H_inh(i;Ki,ni), con i_pos=max(i,0).
        # El +1e-16 evita NaN cuando (c, i) tocan exactamente 0 al inicio del transitorio.
        i_pos = ufl.conditional(i > 0.0, i, 0.0)
        c_pow = c ** hill_nc
        kc_pow = hill_kc ** hill_nc
        h_act = c_pow / (kc_pow + c_pow + 1e-16)
        i_pow = i_pos ** hill_ni
        ki_pow = hill_ki ** hill_ni
        h_inh = ki_pow / (ki_pow + i_pow + 1e-16)
        u_expr = u_max * h_act * h_inh
    else:
        u_expr = None
    
    # Construir ecuaciones según mu y u
    # Ecuación para c - término del efecto Allee según tipo
    if allee_type == 'STRONG':
        # Strong Allee: rc * c * (1 - c) * ((c - alle) / (1 - alle))
        allee_term = rc * c * (1 - c) * ((c - alle) / (1 - alle)) * phi_c * dx
    else:
        # Weak Allee: rc * c * (c - alle) * (1 - c)
        allee_term = rc * c * (c - alle) * (1 - c) * phi_c * dx
    
    F_c_base = ((c - c_n) / dt) * phi_c * dx + D_c * ufl.dot(ufl.grad(c), ufl.grad(phi_c)) * dx - \
               allee_term + c * (alpha * s**2 + beta * i**2) * phi_c * dx
    
    if mu > 0:
        F_c = F_c_base + mu * c * (gamma * s**2 + eta * i**2) * phi_c * dx
    else:
        F_c = F_c_base
    
    # Ecuación para s
    F_s_base = ((s - s_n) / dt) * phi_s * dx + D_s * ufl.dot(ufl.grad(s), ufl.grad(phi_s)) * dx - \
               rs * s * (1 - s) * phi_s * dx + gamma * c**2 * s * phi_s * dx - delta * i**2 * s * phi_s * dx
    
    if mu > 0:
        F_s = F_s_base + (s * c**2 * alpha * mu / 2) * phi_s * dx
    else:
        F_s = F_s_base
    
    # Ecuación para i
    F_i_base = ((i - i_n) / dt) * phi_i * dx + D_i * ufl.dot(ufl.grad(i), ufl.grad(phi_i)) * dx - \
               rd * i * (1 - i) * phi_i * dx - delta * i * s**2 * phi_i * dx + c**2 * i * eta * phi_i * dx
    
    if mu > 0:
        F_i_base = F_i_base + (i * c**2 * beta * mu / 2) * phi_i * dx
    
    # Agregar control adaptativo u si está activado
    if use_adaptive_control:
        F_i = F_i_base - u_expr * phi_i * dx
    else:
        F_i = F_i_base

    # NonlinearProblem (dolfinx ≥0.10) = SNES; no usar nls.petsc.NewtonSolver (API antigua NewtonSolverNonlinearProblem).
    nlp_c = NonlinearProblem(F_c, c, petsc_options_prefix="nl_c_")
    nlp_s = NonlinearProblem(F_s, s, petsc_options_prefix="nl_s_")
    nlp_i = NonlinearProblem(F_i, i, petsc_options_prefix="nl_i_")
    _configure_nlp_snes(nlp_c, 1e-6, 1e-8, newton_max_it_c, newton_relax_c)
    _configure_nlp_snes(nlp_s, 1e-6, 1e-8, newton_max_it_s, newton_relax_s)
    _configure_nlp_snes(nlp_i, 1e-5, 1e-7, newton_max_it_i, newton_relax_i)

    def solve_c():
        nlp_c.solve()

    def solve_s():
        nlp_s.solve()

    def solve_i():
        nlp_i.solve()

    return solve_c, solve_s, solve_i, c, s, i, c_n, s_n, i_n, V, u_expr, domain, nlp_c, nlp_s, nlp_i

def recreate_solvers_preserving_state(c, s, i, c_n, s_n, i_n, V, domain, u_expr_old=None):
    """
    Recrea los solvers preservando el estado actual de los campos.
    Esto ayuda a limpiar cachés internos de PETSc/FEniCSx.
    """
    # Preservar valores actuales
    c_vals = c.x.array.copy()
    s_vals = s.x.array.copy()
    i_vals = i.x.array.copy()
    c_n_vals = c_n.x.array.copy()
    s_n_vals = s_n.x.array.copy()
    i_n_vals = i_n.x.array.copy()
    
    # Recrear problemas no lineales (esto limpia cachés de PETSc)
    dx = ufl.Measure("dx", domain=domain)
    
    # Control adaptativo u (si estaba activado)
    if use_adaptive_control:
        i_pos = ufl.conditional(i > 0.0, i, 0.0)
        c_pow = c ** hill_nc
        kc_pow = hill_kc ** hill_nc
        h_act = c_pow / (kc_pow + c_pow + 1e-16)
        i_pow = i_pos ** hill_ni
        ki_pow = hill_ki ** hill_ni
        h_inh = ki_pow / (ki_pow + i_pow + 1e-16)
        u_expr = u_max * h_act * h_inh
    else:
        u_expr = None
    
    # Reconstruir formas variacionales
    phi_c = ufl.TestFunction(V)
    phi_s = ufl.TestFunction(V)
    phi_i = ufl.TestFunction(V)
    
    if allee_type == 'STRONG':
        allee_term = rc * c * (1 - c) * ((c - alle) / (1 - alle)) * phi_c * dx
    else:
        allee_term = rc * c * (c - alle) * (1 - c) * phi_c * dx
    
    F_c_base = ((c - c_n) / dt) * phi_c * dx + D_c * ufl.dot(ufl.grad(c), ufl.grad(phi_c)) * dx - \
               allee_term + c * (alpha * s**2 + beta * i**2) * phi_c * dx
    
    if mu > 0:
        F_c = F_c_base + mu * c * (gamma * s**2 + eta * i**2) * phi_c * dx
    else:
        F_c = F_c_base
    
    F_s_base = ((s - s_n) / dt) * phi_s * dx + D_s * ufl.dot(ufl.grad(s), ufl.grad(phi_s)) * dx - \
               rs * s * (1 - s) * phi_s * dx + gamma * c**2 * s * phi_s * dx - delta * i**2 * s * phi_s * dx
    
    if mu > 0:
        F_s = F_s_base + (s * c**2 * alpha * mu / 2) * phi_s * dx
    else:
        F_s = F_s_base
    
    F_i_base = ((i - i_n) / dt) * phi_i * dx + D_i * ufl.dot(ufl.grad(i), ufl.grad(phi_i)) * dx - \
               rd * i * (1 - i) * phi_i * dx - delta * i * s**2 * phi_i * dx + c**2 * i * eta * phi_i * dx
    
    if mu > 0:
        F_i_base = F_i_base + (i * c**2 * beta * mu / 2) * phi_i * dx
    
    if use_adaptive_control:
        F_i = F_i_base - u_expr * phi_i * dx
    else:
        F_i = F_i_base
    
    # Recrear problemas no lineales (esto limpia cachés)
    nlp_c = NonlinearProblem(F_c, c, petsc_options_prefix="nl_c_")
    nlp_s = NonlinearProblem(F_s, s, petsc_options_prefix="nl_s_")
    nlp_i = NonlinearProblem(F_i, i, petsc_options_prefix="nl_i_")
    _configure_nlp_snes(nlp_c, 1e-6, 1e-8, newton_max_it_c, newton_relax_c)
    _configure_nlp_snes(nlp_s, 1e-6, 1e-8, newton_max_it_s, newton_relax_s)
    _configure_nlp_snes(nlp_i, 1e-5, 1e-7, newton_max_it_i, newton_relax_i)

    # Restaurar valores
    c.x.array[:] = c_vals
    s.x.array[:] = s_vals
    i.x.array[:] = i_vals
    c_n.x.array[:] = c_n_vals
    s_n.x.array[:] = s_n_vals
    i_n.x.array[:] = i_n_vals

    # Mantener el invariante físico [0,1] al recrear solvers
    if physical_clipping:
        _clip_to_unit_box(c, s, i)
        _clip_to_unit_box(c_n, s_n, i_n)

    def solve_c():
        nlp_c.solve()

    def solve_s():
        nlp_s.solve()

    def solve_i():
        nlp_i.solve()

    return solve_c, solve_s, solve_i, u_expr, nlp_c, nlp_s, nlp_i

# ============================================================================
# Bucle principal con iteraciones internas (issue 3) y guardado denso (issue 6)
# ============================================================================
# Flujo resumido:
# 1. Carga parámetros y `sample_rate` / iteraciones internas desde `.env`.
# 2. Malla `P1` en `[0, space_size]^2`.
# 3. Resuelve el sistema sin control con iteraciones internas de Picard por paso (`inner_max_iter`, `inner_tol`).
# 4. Guarda `matrix_{c,s,i}_t_nb_block.txt` con paso espacial `sample_rate` (útil para análisis en Fourier/correlaciones).

if __name__ == "__main__":
    for block in range(1, nb + 1):
        t = 0.0
        solver_c, solver_s, solver_i, c, s, i, c_n, s_n, i_n, V, u_expr, domain, solver_c_newton, solver_s_newton, solver_i_newton = solve_dynamics()
        step = 0

        # Rate-limit de diagnósticos en warnings (por bloque)
        last_warning_diag_step = None
        last_exception_diag_step = None
        diag_saved_steps = set()
        
        # Intentar cargar checkpoint si está habilitado
        checkpoint_loaded = False
        if enable_checkpoint:
            # Cargar checkpoint evitando el paso problemático si se especifica
            # Si checkpoint_max_step está definido, busca checkpoint anterior a ese paso
            if checkpoint_max_step is not None:
                print(f"  [Checkpoint] Buscando checkpoint anterior al paso {checkpoint_max_step}")
            checkpoint_data = load_checkpoint(max_step=checkpoint_max_step)
            if checkpoint_data is not None:
                # Verificar que el checkpoint corresponde al bloque actual
                if checkpoint_data['block'] == block:
                    # Restaurar valores de campos
                    c.x.array[:] = checkpoint_data['c_array']
                    s.x.array[:] = checkpoint_data['s_array']
                    i.x.array[:] = checkpoint_data['i_array']
                    c_n.x.array[:] = checkpoint_data['c_n_array']
                    s_n.x.array[:] = checkpoint_data['s_n_array']
                    i_n.x.array[:] = checkpoint_data['i_n_array']

                    # Reaplicar la proyección a [0, 1] al cargar checkpoint:
                    # garantiza el invariante físico aún si el checkpoint fue
                    # producido por una corrida previa sin clipping.
                    if physical_clipping:
                        _clip_to_unit_box(c, s, i)
                        _clip_to_unit_box(c_n, s_n, i_n)

                    # Restaurar tiempo y paso
                    t = checkpoint_data['t']
                    step = checkpoint_data['step']
                    checkpoint_loaded = True
                    
                    print(f"\n{'='*60}")
                    print(f"✓ Checkpoint cargado para bloque {block}")
                    print(f"  Continuando desde paso {step}, t={t:.4f}")
                    print(f"{'='*60}\n")

        last_memory_checkpoint_step = step

        # Buffers de Picard: una sola vez por bloque (evita fugas por crear fem.Function cada paso)
        c_prev = fem.Function(c.function_space)
        s_prev = fem.Function(s.function_space)
        i_prev = fem.Function(i.function_space)

        # Calcular número total de pasos estimados
        total_steps_estimate = int(T / dt) + 1
        start_time = time.time()
        
        if not checkpoint_loaded:
            print(f"\n{'='*60}")
            print(f"Bloque {block}/{nb} - Iniciando simulación")
            print(f"Tiempo total: T={T}, dt={dt}")
            print(f"Pasos estimados: ~{total_steps_estimate}")
            print(f"{'='*60}\n")

        while t < T + 1e-12:
            # Iteraciones internas de Picard (reutiliza c_prev/s_prev/i_prev)
            c_prev.x.array[:] = c.x.array[:]
            s_prev.x.array[:] = s.x.array[:]
            i_prev.x.array[:] = i.x.array[:]
            warn_not_converged = True
            convergence_failed_count = 0
            max_failed_iterations = 10  # Límite de iteraciones fallidas consecutivas
            diff = float('inf')  # Inicializar diff para evitar NameError
            
            for k in range(inner_max_iter):
                try:
                    solver_c()  # Cambiado de solver_c.solve()
                    solver_s()  # Cambiado de solver_s.solve()
                    solver_i()  # Cambiado de solver_i.solve()
                except Exception as solver_error:
                    convergence_failed_count += 1
                    # Guardar diagnóstico (una vez por step) cuando ocurre una excepción del solver
                    if enable_diagnostics and _diagnostic_rate_limit_should_save(
                        last_exception_diag_step, step, 1
                    ):
                        try:
                            extra = {
                                'iteration_k': int(k),
                                'inner_max_iter': int(inner_max_iter),
                                'convergence_failed_count': int(convergence_failed_count),
                                'error_type': type(solver_error).__name__,
                                'error_message': str(solver_error),
                            }
                            save_diagnostic_snapshot(
                                c, s, i, t, step, block,
                                reason="solver_exception",
                                extra=extra
                            )
                            last_exception_diag_step = step
                            diag_saved_steps.add(step)
                        except Exception as diag_err:
                            if comm.rank == 0:
                                print(f"⚠ [Diagnóstico] Error al guardar snapshot por excepción: {diag_err}")

                    if convergence_failed_count >= max_failed_iterations:
                        # Snapshot final antes de abortar
                        if enable_diagnostics:
                            try:
                                extra = {
                                    'iteration_k': int(k),
                                    'inner_max_iter': int(inner_max_iter),
                                    'convergence_failed_count': int(convergence_failed_count),
                                    'final_failure': True,
                                    'error_type': type(solver_error).__name__,
                                    'error_message': str(solver_error),
                                }
                                save_diagnostic_snapshot(
                                    c, s, i, t, step, block,
                                    reason="solver_failed_abort",
                                    extra=extra
                                )
                            except Exception as diag_err:
                                if comm.rank == 0:
                                    print(f"⚠ [Diagnóstico] Error al guardar snapshot final: {diag_err}")

                        error_msg = f"ERROR: El solver falló {convergence_failed_count} veces consecutivas en t={t:.4f}, paso {step}"
                        print(f"\n✗ {error_msg}")
                        print(f"  Último error: {solver_error}")
                        raise RuntimeError(error_msg) from solver_error
                    else:
                        warning_msg = f"ADVERTENCIA: El solver falló en iteración {k+1} de {inner_max_iter} (t={t:.4f}): {solver_error}"
                        print(f"⚠ {warning_msg}")
                        # Continuar con la siguiente iteración
                        continue
                
                diff = (np.linalg.norm(c.x.array - c_prev.x.array) + 
                       np.linalg.norm(s.x.array - s_prev.x.array) + 
                       np.linalg.norm(i.x.array - i_prev.x.array))
                c_prev.x.array[:] = c.x.array[:]
                s_prev.x.array[:] = s.x.array[:]
                i_prev.x.array[:] = i.x.array[:]
                if diff < inner_tol:
                    warn_not_converged = False
                    convergence_failed_count = 0  # Resetear contador si converge
                    break
            
            if warn_not_converged:
                warning_msg = f"ADVERTENCIA: Las iteraciones internas no convergieron en t={t:.4f}, paso {step} (diff={diff:.2e} > tol={inner_tol:.2e})"
                print(f"⚠ {warning_msg}")
                # Guardar diagnóstico en warnings (rate-limited)
                if enable_diagnostics and diagnostic_on_warning:
                    # Evitar doble escritura si ya se guardó en este mismo step por excepción
                    if step not in diag_saved_steps:
                        if _diagnostic_rate_limit_should_save(
                            last_warning_diag_step, step, diagnostic_warning_min_step_gap
                        ):
                            try:
                                extra = {
                                    'diff': float(diff),
                                    'inner_tol': float(inner_tol),
                                    'inner_max_iter': int(inner_max_iter),
                                }
                                save_diagnostic_snapshot(
                                    c, s, i, t, step, block,
                                    reason="inner_not_converged",
                                    extra=extra
                                )
                                last_warning_diag_step = step
                                diag_saved_steps.add(step)
                            except Exception as diag_err:
                                if comm.rank == 0:
                                    print(f"⚠ [Diagnóstico] Error al guardar snapshot por warning: {diag_err}")
            
            # Proyección física a [0, 1] tras Picard, antes de propagar a c_n/s_n/i_n.
            # Mantiene las soluciones dentro del cono físico definido por las
            # capacidades de carga k_c = k_s = k_i = 1; previene blow-up por la
            # cooperación +δ i² s, +δ s² i cuando c → 0.
            if physical_clipping:
                n_c_clip, n_s_clip, n_i_clip, n_total_local = _clip_to_unit_box(c, s, i)
                should_log = (clip_log_interval > 0) and (step % clip_log_interval == 0)
                if should_log:
                    n_c_glb = comm.allreduce(n_c_clip, op=MPI.SUM)
                    n_s_glb = comm.allreduce(n_s_clip, op=MPI.SUM)
                    n_i_glb = comm.allreduce(n_i_clip, op=MPI.SUM)
                    n_total_glb = comm.allreduce(n_total_local, op=MPI.SUM)
                    if comm.rank == 0 and (n_c_glb + n_s_glb + n_i_glb) > 0 and n_total_glb > 0:
                        pct_c = 100.0 * n_c_glb / n_total_glb
                        pct_s = 100.0 * n_s_glb / n_total_glb
                        pct_i = 100.0 * n_i_glb / n_total_glb
                        print(
                            f"  [Clip] t={t:.4f} step={step} "
                            f"c:{n_c_glb} ({pct_c:.2f}%) "
                            f"s:{n_s_glb} ({pct_s:.2f}%) "
                            f"i:{n_i_glb} ({pct_i:.2f}%)"
                        )

            # CRÍTICO: Actualizar valores del paso de tiempo anterior para la siguiente iteración
            c_n.x.array[:] = c.x.array[:]
            s_n.x.array[:] = s.x.array[:]
            i_n.x.array[:] = i.x.array[:]
            
            # Graficar (solo bloque 1)
            if block == 1:
                try:
                    if use_adaptive_control:
                        plot_fields(c, s, i, block, t, u_expr, V)
                    else:
                        plot_fields(c, s, i, block, t)
                    # Limpieza inmediata después de graficar
                    gc.collect()
                except Exception as plot_error:
                    # Si falla la generación de imágenes, continuar con la ejecución
                    print(f"⚠ ADVERTENCIA: Error al generar imagen en t={t:.4f}: {plot_error}")
                    print(f"  Continuando con la simulación...")
                    # Asegurar que matplotlib se cierre correctamente
                    try:
                        plt.close('all')
                    except:
                        pass
                    gc.collect()

            # Guardar campos en alta resolución
            ts = f"{t:.3f}"
            field_to_numpy_array(c, space_size, ts, 'c', block, sample_rate=sample_rate)
            field_to_numpy_array(s, space_size, ts, 's', block, sample_rate=sample_rate)
            field_to_numpy_array(i, space_size, ts, 'i', block, sample_rate=sample_rate)
            
            # Limpieza después de guardar matrices (cada paso)
            gc.collect()

            t += dt
            step += 1
            _log_memory_profile_step(step, t)

            # Limpieza periódica adicional de memoria (más frecuente)
            if step % memory_cleanup_interval == 0:
                gc.collect()
                # Limpiar cachés de PETSc si está disponible
                cleanup_petsc_caches()
                # Limpieza más agresiva cada N pasos
                if step % (memory_cleanup_interval * 5) == 0:
                    # Forzar recolección múltiple para objetos con referencias circulares
                    for _ in range(3):
                        gc.collect()
            
            # Recrear solvers periódicamente para limpiar cachés de PETSc/FEniCSx
            if step > 0 and step % solver_recreate_interval == 0:
                if monitor_memory:
                    mem_before = get_memory_usage()
                # Recrear solvers preservando el estado
                solver_c, solver_s, solver_i, u_expr, solver_c_newton, solver_s_newton, solver_i_newton = recreate_solvers_preserving_state(
                    c, s, i, c_n, s_n, i_n, V, domain, u_expr
                )
                # Limpieza agresiva después de recrear
                cleanup_petsc_caches()
                gc.collect()
                if monitor_memory:
                    mem_after = get_memory_usage()
                    mem_freed = mem_before - mem_after
                    if mem_freed > 0:
                        print(f"  [Limpieza] Solvers recreados: liberados {mem_freed:.0f}MB")
                    else:
                        print(f"  [Limpieza] Solvers recreados (mem: {mem_after:.0f}MB)")
            
            # Sistema de checkpoint/restart
            if enable_checkpoint:
                should_save_checkpoint = False
                should_restart_now = False

                # Guardar checkpoint periódicamente
                if step > 0 and step % checkpoint_interval == 0:
                    should_save_checkpoint = True

                # Memoria alta: checkpoint con rate-limit (evita guardar en cada paso y saturar I/O)
                mem_pct = 0.0
                if monitor_memory:
                    mem_pct = get_memory_percentage()
                    if mem_pct > checkpoint_memory_threshold_pct:
                        if step - last_memory_checkpoint_step >= checkpoint_memory_min_interval:
                            should_save_checkpoint = True
                    if mem_pct > checkpoint_restart_threshold_pct:
                        should_restart_now = True

                # Guardar checkpoint si es necesario
                if should_save_checkpoint:
                    checkpoint_path = save_checkpoint(c, s, i, c_n, s_n, i_n, t, step, block)
                    print(f"  [Checkpoint] Guardado en paso {step}, t={t:.4f}")
                    if monitor_memory:
                        mem_pct = get_memory_percentage()
                        print(f"  [Checkpoint] Memoria: {mem_pct:.1f}%")
                    last_memory_checkpoint_step = step
                
                # Reiniciar si memoria excede umbral crítico
                if should_restart_now:
                    # Guardar checkpoint final antes de reiniciar
                    checkpoint_path = save_checkpoint(c, s, i, c_n, s_n, i_n, t, step, block)
                    
                    # Crear archivo de señal para indicar reinicio necesario
                    checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    restart_signal_file = os.path.join(checkpoint_dir, 'RESTART_NEEDED')
                    with open(restart_signal_file, 'w') as f:
                        f.write(f"step={step}\nt={t}\nblock={block}\n")
                    
                    print(f"\n{'='*60}")
                    print(f"⚠ Memoria crítica alcanzada ({get_memory_percentage():.1f}%)")
                    print(f"  Checkpoint guardado en paso {step}, t={t:.4f}")
                    print(f"  El proceso se reiniciará automáticamente")
                    print(f"{'='*60}\n")
                    # Terminar con código especial para indicar reinicio necesario
                    sys.exit(100)  # Código 100 = reinicio necesario
            
            # Mostrar progreso cada 10 pasos o al inicio/final
            should_print = (step % 10 == 0) or (step == 1) or (t >= T - dt)
            
            if should_print:
                # Calcular progreso
                progress_pct = min(100.0, (t / T) * 100.0)
                elapsed_time = time.time() - start_time
                
                # Estimar tiempo restante
                if step > 1 and progress_pct > 0:
                    time_per_step = elapsed_time / step
                    remaining_steps = max(0, total_steps_estimate - step)
                    estimated_remaining = time_per_step * remaining_steps
                    
                    # Formatear tiempo restante
                    if estimated_remaining < 60:
                        time_str = f"{estimated_remaining:.1f}s"
                    elif estimated_remaining < 3600:
                        time_str = f"{estimated_remaining/60:.1f}min"
                    else:
                        hours = int(estimated_remaining // 3600)
                        mins = int((estimated_remaining % 3600) // 60)
                        time_str = f"{hours}h {mins}min"
                else:
                    time_str = "calculando..."
                
                # Formatear tiempo transcurrido
                if elapsed_time < 60:
                    elapsed_str = f"{elapsed_time:.1f}s"
                elif elapsed_time < 3600:
                    elapsed_str = f"{elapsed_time/60:.1f}min"
                else:
                    hours = int(elapsed_time // 3600)
                    mins = int((elapsed_time % 3600) // 60)
                    elapsed_str = f"{hours}h {mins}min"
                
                # Información de memoria si está activado el monitoreo
                mem_str = ""
                if monitor_memory:
                    mem_mb = get_memory_usage()
                    mem_pct = get_memory_percentage()
                    if mem_pct > 0:
                        mem_str = f" | Mem: {mem_mb:.0f}MB ({mem_pct:.1f}%)"
                    else:
                        mem_str = f" | Mem: {mem_mb:.0f}MB"
                    
                    # Verificar advertencias de memoria
                    mem_warnings = check_memory_warning()
                    if mem_warnings:
                        for warning in mem_warnings:
                            print(f"⚠ {warning}")
                
                print(f"[Bloque {block}/{nb}] Paso {step:5d}/{total_steps_estimate:5d} | "
                      f"t={t:.4f}/{T:.4f} ({progress_pct:5.1f}%) | "
                      f"Tiempo: {elapsed_str} | Restante: {time_str}{mem_str}")
            
            # Mensaje final cuando termine
            if t >= T - dt:
                # Guardar checkpoint final si está habilitado
                if enable_checkpoint:
                    checkpoint_path = save_checkpoint(c, s, i, c_n, s_n, i_n, t, step, block)
                    print(f"  [Checkpoint] Guardado final en paso {step}, t={t:.4f}")
                
                total_time = time.time() - start_time
                print(f"\n{'='*60}")
                print(f"✓ Bloque {block}/{nb} completado!")
                print(f"  Total pasos: {step}")
                print(f"  Tiempo total: {total_time/60:.2f} minutos")
                print(f"{'='*60}\n")

