"""
Script centralizado para ejecutar simulaciones de escenarios.

Orden respecto al pipeline del proyecto (matrices → termo → TdC → figuras):
    Allee/PIPELINE_EJECUCION_Y_FISICA.md

Esta herramienta cubre la etapa de simulación (y correlaciones por escenario). Tras corridas
exitosas puede, opcionalmente, lanzar el postproceso termodinámico y mostrar los comandos
siguientes alineados con el mismo JSON de escenarios.

Combina las funcionalidades de:
- run_all_scenarios.py: Ejecución de escenarios (ahora usando scripts .py)
- run_scenarios_in_batches.py: Ejecución en lotes para gestión de espacio
- run_with_checkpoint.py: Sistema de checkpoint/restart automático
- manage_batches.py: Funciones de gestión de lotes
- retry_failed_scenarios.py: Re-ejecución de escenarios fallidos

Ejecuta scripts Python directamente:
- cancer_dynamics.py: Simulación principal
- correlations/correlation_fourier.py: Análisis de correlaciones

Uso:
    python run_scenarios.py                          # Ejecuta todos los escenarios
    python run_scenarios.py --scenario <nombre>       # Ejecuta solo un escenario
    python run_scenarios.py --list                   # Lista escenarios disponibles
    python run_scenarios.py --batch-mode             # Ejecuta en modo lotes automático
    python run_scenarios.py --batch-size 9 --batch-start 0  # Ejecuta lote específico
    python run_scenarios.py --scenario <nombre> --clean  # Limpia y ejecuta
    python run_scenarios.py --retry-failed           # Re-ejecuta escenarios fallidos
    python run_scenarios.py --retry-failed --from-zero  # Re-ejecuta desde cero (solo borra checkpoints)
    python run_scenarios.py --scenarios-file scenarios_v1.json  # Usa otro JSON (ruta relativa a Allee/)
    python run_scenarios.py --scenarios-file scenarios_circular_ic.json  # CI circular con/sin Hill
    # También acepta un JSON tipo steady_states_full_run.json (strong_corner.all → escenarios).
    python run_scenarios.py --run-thermodynamics     # Tras éxito total, ejecuta calculate_thermodynamic_properties --all
    python run_scenarios.py --no-pipeline-hint        # No imprimir bloque de siguientes pasos del pipeline
"""

import os
import json
import sys
import time
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Tuple, Dict, List, Optional
import logging
from datetime import datetime
import argparse
import traceback
import numpy as np

# ----------------------------------------------------------------------------
# Compatibilidad de encoding (Windows consoles suelen usar cp1252)
# Evita UnicodeEncodeError al imprimir símbolos como ✓ ✗ ⚠ ℹ
# ----------------------------------------------------------------------------
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Directorio base
BASE_DIR = Path(__file__).parent
SCENARIOS_FILE = BASE_DIR / "scenarios.json"
# Etiqueta para comentario en .env (se actualiza en main() según --scenarios-file)
_ACTIVE_SCENARIOS_JSON_LABEL = "scenarios.json"
ENV_FILE = BASE_DIR / ".env"
CANCER_DYNAMICS_SCRIPT = BASE_DIR / "cancer_dynamics.py"
CORRELATION_SCRIPT = BASE_DIR / "correlations" / "correlation_fourier.py"
THERMO_SCRIPT = BASE_DIR / "termodynamics" / "calculate_thermodynamic_properties.py"
COMPARISON_NOTEBOOK = BASE_DIR / "correlation_comparison.ipynb"  # Mantener notebook si existe
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Importar utilidades de rutas compartidas
try:
    from utils_paths import get_results_dir, is_google_drive_mounted, get_google_drive_mount_point, verify_results_dir_write_access
except ImportError:
    # Fallback si el módulo no está disponible
    def get_results_dir(base_dir=None):
        """Fallback local si utils_paths no está disponible"""
        if base_dir is None:
            base_dir = BASE_DIR
        env_results_dir = os.getenv('RESULTS_DIR')
        if env_results_dir:
            results_path = Path(env_results_dir)
            if results_path.exists():
                return results_path
        drive_mount_point = Path.home() / "googledrive"
        drive_results_dir = drive_mount_point / "Doctorado Erick Serrato" / "Resultados Paper"
        if drive_mount_point.exists():
            try:
                list(drive_mount_point.iterdir())
                drive_results_dir.mkdir(parents=True, exist_ok=True)
                return drive_results_dir
            except (OSError, PermissionError):
                pass
        return base_dir / "results"
    
    def is_google_drive_mounted():
        """Fallback local"""
        drive_mount_point = Path.home() / "googledrive"
        if not drive_mount_point.exists():
            return False
        try:
            list(drive_mount_point.iterdir())
            return True
        except (OSError, PermissionError):
            return False
    
    def get_google_drive_mount_point():
        """Fallback local"""
        return Path.home() / "googledrive"
    
    def verify_results_dir_write_access(results_dir):
        """Fallback local"""
        try:
            results_dir.mkdir(parents=True, exist_ok=True)
            test_file = results_dir / ".test_write"
            test_file.write_text("test")
            test_file.unlink()
            return True, None
        except Exception as e:
            return False, str(e)

RESULTS_DIR = get_results_dir(BASE_DIR)

# Configurar logging
log_file = LOGS_DIR / f"run_scenarios_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Código de salida que indica reinicio necesario
RESTART_EXIT_CODE = 100
MAX_RESTARTS = 50  # Límite de reinicios para evitar loops infinitos


def _staging_work_dir(scenario_name: str) -> Path:
    """Directorio local de trabajo (p. ej. bajo /tmp) para I/O intensivo."""
    root = os.environ.get("ALLEE_STAGING_DIR", "").strip()
    base = Path(root) if root else Path(tempfile.gettempdir())
    return (base / "allee_scenario_runs" / scenario_name).resolve()


def should_use_local_staging(_final_dir: Path) -> bool:
    """
    Opcional: ejecutar cancer_dynamics en /tmp (staging) y volcar al final a RESULTS_DIR.

    Por defecto todo se escribe **directamente** en ``final_dir`` (p. ej. Google Drive).

    - ALLEE_USE_LOCAL_STAGING=1|y|yes|true|on: activa staging local (menos EIO en FUSE, merge al terminar).
    - Cualquier otro valor o variable ausente: **desactivado** (salida solo en RESULTS_DIR).
    """
    raw = os.environ.get("ALLEE_USE_LOCAL_STAGING", "").strip().lower()
    return raw in ("1", "y", "yes", "true", "on")


def _seed_checkpoints_from_final(final_dir: Path, staging_dir: Path) -> None:
    src = final_dir / "checkpoints"
    if not src.is_dir():
        return
    dst = staging_dir / "checkpoints"
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _sync_checkpoints_to_final(staging_dir: Path, final_dir: Path) -> None:
    """Copia checkpoints del staging al destino final (p. ej. Drive) para poder reanudar."""
    src = staging_dir / "checkpoints"
    if not src.is_dir():
        return
    dst = final_dir / "checkpoints"
    try:
        dst.mkdir(parents=True, exist_ok=True)
        for p in src.iterdir():
            if p.is_file():
                shutil.copy2(p, dst / p.name)
    except OSError as e:
        logger.warning(f"No se pudo sincronizar checkpoints a {dst}: {e}")


def _merge_staging_tree_to_final(staging_dir: Path, final_dir: Path) -> None:
    """Copia recursivamente todo el contenido del staging sobre el directorio final."""
    if not staging_dir.is_dir():
        return
    final_dir.mkdir(parents=True, exist_ok=True)
    for p in staging_dir.rglob("*"):
        if p.is_file():
            rel = p.relative_to(staging_dir)
            out = final_dir / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)


# ============================================================================
# Funciones de gestión de lotes (de manage_batches.py)
# ============================================================================

def _looks_like_steady_states_full_run_payload(data: dict) -> bool:
    """True si ``data`` es un JSON unificado (``steady_states_full_run.json``) con ``strong_corner``."""
    if not isinstance(data, dict) or "common_params" in data:
        return False
    sc = data.get("strong_corner")
    if not isinstance(sc, dict):
        return False
    blocks = sc.get("all") or sc.get("steady_states_filtered") or []
    return isinstance(blocks, list) and len(blocks) > 0


def _corner_blocks_list(strong_corner: dict) -> List[dict]:
    for key in ("all", "steady_states_filtered", "near_corner_only"):
        blocks = strong_corner.get(key)
        if isinstance(blocks, list) and blocks:
            return blocks
    return []


def _load_reference_common_params_for_corner(data: dict) -> Dict[str, Any]:
    """
    ``common_params`` no vienen en el full-run; se toman de ``meta.scenarios_file``,
    ``scenarios_v1.json`` o un mínimo seguro.
    """
    candidates: List[Path] = []
    meta = data.get("meta")
    if isinstance(meta, dict):
        sf = meta.get("scenarios_file")
        if sf:
            candidates.append(Path(str(sf)))
    candidates.extend(
        [
            BASE_DIR / "scenarios_v1.json",
            BASE_DIR / "scenarios_v1 copy.json",
        ]
    )
    for p in candidates:
        try:
            if p.is_file():
                with open(p, "r", encoding="utf-8") as rf:
                    ref = json.load(rf)
                cp = ref.get("common_params")
                if isinstance(cp, dict):
                    return dict(cp)
        except (OSError, json.JSONDecodeError, TypeError):
            continue
    logger.warning(
        "No se encontró scenarios_v1.json (ni meta.scenarios_file); usando common_params mínimos."
    )
    return {
        "a": "0.1",
        "alle": "0.1",
        "gamma": "0.74",
        "alpha": "10.22",
        "rs": "13.12",
        "rc": "5.84",
        "delta": "5.40",
        "eta": "5.08",
        "beta": "7.6",
        "rd": "10.92",
        "D_c": "0.012",
        "D_s": "0.022",
        "D_i": "0.022",
        "dt": "0.001",
        "T": "3",
        "nodes_in_xaxis": "100",
        "nodes_in_yaxis": "100",
        "space_size": "8",
        "nb": "1",
        "sample_rate": "0.02",
        "inner_max_iter": "3",
        "inner_tol": "0.001",
        "SAVE_IMAGES": "Y",
        "MONITOR_MEMORY": "Y",
        "MEMORY_CLEANUP_INTERVAL": "5",
        "MEMORY_WARNING_THRESHOLD_MB": "0",
        "MEMORY_WARNING_THRESHOLD_PCT": "80",
        "SOLVER_RECREATE_INTERVAL": "25",
        "ENABLE_CHECKPOINT": "Y",
        "CHECKPOINT_INTERVAL": "500",
        "CHECKPOINT_MEMORY_THRESHOLD_PCT": "80",
        "CHECKPOINT_RESTART_THRESHOLD_PCT": "85",
        "mu": "1",
        "ALLEE_TYPE": "STRONG",
        "USE_ADAPTIVE_CONTROL": "N",
        "U_MAX": "0.5",
        "HILL_KC": "0.05",
        "HILL_NC": "2",
        "HILL_KI": "0.2",
        "HILL_NI": "2",
        "KU": "0.2",
        "EPS_U": "0.001",
    }


def _fmt_mu_for_scenario(mu_val) -> str:
    try:
        x = float(mu_val)
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return str(x)
    except (TypeError, ValueError):
        return str(mu_val)


def _init_ranges_from_equilibrium(ss: dict) -> Dict[str, str]:
    """Bandas de CI alrededor de (c*, s*, i*) para el PDE (valores en cadena)."""
    c = float(ss.get("c_star") or 0.0)
    s = float(ss.get("s_star") or 0.0)
    iz = float(ss.get("i_star") or 0.0)
    w = 0.02

    def fmt(x: float) -> str:
        return f"{x:.6g}"

    c_lo = max(0.0, c - w)
    c_hi = min(1.2, c + w) if c > 1e-12 else min(1.2, w)
    if c_hi <= c_lo:
        c_hi = c_lo + 0.01

    s_lo = max(0.0, s - w)
    s_hi = min(1.0, s + w)
    if s_hi <= s_lo:
        s_hi = min(1.0, s_lo + 0.01)

    if iz <= 1e-12:
        i_lo, i_hi = 0.01, 0.05
    else:
        i_lo = max(0.01, iz - w)
        i_hi = min(1.0, iz + w)
        if i_hi <= i_lo:
            i_hi = min(1.0, i_lo + 0.01)

    return {
        "C_INIT_MIN": fmt(c_lo),
        "C_INIT_MAX": fmt(c_hi),
        "S_INIT_MIN": fmt(s_lo),
        "S_INIT_MAX": fmt(s_hi),
        "I_INIT_MIN": fmt(i_lo),
        "I_INIT_MAX": fmt(i_hi),
    }


_INIT_ENV_KEYS = frozenset(
    {
        "C_INIT_MIN",
        "C_INIT_MAX",
        "S_INIT_MIN",
        "S_INIT_MAX",
        "I_INIT_MIN",
        "I_INIT_MAX",
    }
)


def _pick_steady_state_for_ic(name: str, ss_list: List[dict]) -> Optional[dict]:
    """
    Elige el elemento de ``steady_states`` coherente con el sufijo del nombre del escenario
    (p. ej. ``…_c0_s1_i0`` → equilibrio con ``target_branch == "c0_s1_i0"``).
    """
    if not ss_list:
        return None
    if len(ss_list) == 1 and isinstance(ss_list[0], dict):
        return ss_list[0]
    n = str(name or "")
    if n.endswith("_c0_s1_i0_c0_s1_i1"):
        for ss in ss_list:
            if isinstance(ss, dict) and ss.get("target_branch") == "c0_s1_i0":
                return ss
        return ss_list[0] if isinstance(ss_list[0], dict) else None
    if n.endswith("_c0_s1_i1"):
        for ss in ss_list:
            if isinstance(ss, dict) and ss.get("target_branch") == "c0_s1_i1":
                return ss
        return ss_list[-1] if len(ss_list) > 1 else ss_list[0]
    if n.endswith("_c0_s1_i0"):
        for ss in ss_list:
            if isinstance(ss, dict) and ss.get("target_branch") == "c0_s1_i0":
                return ss
        return ss_list[0] if isinstance(ss_list[0], dict) else None
    return None


def _merge_simulation_ic_from_steady_states(scenario_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Si el escenario no fija bandas de CI y el nombre indica rama esquina, rellena
    ``C_INIT_*`` / ``S_INIT_*`` / ``I_INIT_*`` desde ``(c*, s*, i*)`` en ``steady_states``.
    Así ``cancer_dynamics.py`` arranca cerca de (0,1,0) o (0,1,1) y no en el patrón por defecto
    tipo (0,0,1).
    """
    out = dict(scenario_params)
    if any(str(out.get(k, "")).strip() != "" for k in _INIT_ENV_KEYS):
        return out
    name = str(out.get("name", ""))
    raw = out.get("steady_states")
    if not isinstance(raw, list) or not raw:
        return out
    ss_list = [x for x in raw if isinstance(x, dict)]
    if not ss_list:
        return out
    ss = _pick_steady_state_for_ic(name, ss_list)
    if ss is None:
        return out
    out.update(_init_ranges_from_equilibrium(ss))
    return out


def _merge_steady_state_physics_into_flat_env(scenario_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Promueve ``steady_states[]`` a claves planas leídas por ``cancer_dynamics`` (``.env``).

    Sin esto, ``common_params`` (p. ej. ``USE_ADAPTIVE_CONTROL=N``, ``mu=1``) pisa la intención
    del JSON por rama (p. ej. ``uSi`` con ``use_adaptive_control: true`` y ``mu: 0``), y la
    simulación corre **sin** término de control ``u`` ni panel Hill en figuras.
    """
    out = dict(scenario_params)
    raw = out.get("steady_states")
    if not isinstance(raw, list) or not raw:
        return out
    ss_list = [x for x in raw if isinstance(x, dict)]
    if not ss_list:
        return out
    name = str(out.get("name", ""))
    ss = _pick_steady_state_for_ic(name, ss_list)
    if ss is None:
        ss = ss_list[0]

    # μ, tipo Allee y cinética desde el equilibrio elegido (pisan common al fusionar después)
    if ss.get("mu") is not None:
        out["mu"] = str(ss["mu"])
    if ss.get("allee_type"):
        out["ALLEE_TYPE"] = str(ss["allee_type"]).upper()
    for key in ("rc", "rs", "rd", "alpha", "beta", "delta", "eta", "gamma", "a"):
        if ss.get(key) is not None:
            out[key] = str(ss[key])

    # Control adaptativo: solo si el escenario no fija ya USE_ADAPTIVE_CONTROL a nivel plano
    has_explicit_u = "USE_ADAPTIVE_CONTROL" in scenario_params and str(
        scenario_params.get("USE_ADAPTIVE_CONTROL", "")
    ).strip() != ""
    if has_explicit_u:
        return out

    use_adaptive = bool(ss.get("use_adaptive_control"))
    hill = bool(ss.get("hill_control"))

    if hill and use_adaptive:
        out["USE_ADAPTIVE_CONTROL"] = "Y"
        u_h = ss.get("umax")
        if u_h is not None:
            out["U_MAX"] = str(u_h)
    elif use_adaptive:
        out["USE_ADAPTIVE_CONTROL"] = "Y"
        if ss.get("ku") is not None:
            out["KU"] = str(ss["ku"])
        if ss.get("eps_u") is not None:
            out["EPS_U"] = str(ss["eps_u"])
        u_m = ss.get("umax")
        if u_m is not None:
            out["U_MAX"] = str(u_m)
    else:
        out["USE_ADAPTIVE_CONTROL"] = "N"

    return out


def _corner_steady_block_to_run_scenario(block: dict, common: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convierte un elemento de ``strong_corner.all`` (name + steady_states[])
    en un escenario plano (cadenas) como en ``scenarios_v1.json``.
    """
    name = str(block.get("name") or "unnamed_scenario")
    ss_list = block.get("steady_states") or []
    if not ss_list:
        return {
            "name": name,
            "ALLEE_TYPE": str(common.get("ALLEE_TYPE", "STRONG")),
            "mu": str(common.get("mu", "1")),
            "USE_ADAPTIVE_CONTROL": "N",
        }
    ss = _pick_steady_state_for_ic(name, ss_list) or ss_list[0]
    use_adaptive = bool(ss.get("use_adaptive_control"))
    hill = bool(ss.get("hill_control"))

    scen: Dict[str, Any] = {
        "name": name,
        "ALLEE_TYPE": str(ss.get("allee_type", "STRONG")).upper(),
        "mu": _fmt_mu_for_scenario(ss.get("mu", common.get("mu", "1"))),
        "rc": str(ss.get("rc", common.get("rc", "5.84"))),
        "beta": str(ss.get("beta", common.get("beta", "7.6"))),
        "delta": str(ss.get("delta", common.get("delta", "5.40"))),
        "eta": str(ss.get("eta", common.get("eta", "5.08"))),
        "rd": str(ss.get("rd", common.get("rd", "10.92"))),
        "a": str(ss.get("a", common.get("a", "0.1"))),
    }
    scen.update(_init_ranges_from_equilibrium(ss))

    if hill and use_adaptive:
        scen["USE_ADAPTIVE_CONTROL"] = "Y"
        for k in ("HILL_KC", "HILL_NC", "HILL_KI", "HILL_NI"):
            if k in common:
                scen[k] = str(common[k])
        u_h = ss.get("umax")
        if u_h is not None:
            scen["U_MAX"] = str(u_h)
        else:
            scen["U_MAX"] = str(common.get("U_MAX", "0.5"))
    elif use_adaptive:
        scen["USE_ADAPTIVE_CONTROL"] = "Y"
        scen["KU"] = str(ss.get("ku", common.get("KU", "0.2")))
        scen["EPS_U"] = str(ss.get("eps_u", common.get("EPS_U", "0.001")))
        u_m = ss.get("umax")
        if u_m is not None:
            scen["U_MAX"] = str(u_m)
        elif "U_MAX" in common:
            scen["U_MAX"] = str(common["U_MAX"])
    else:
        scen["USE_ADAPTIVE_CONTROL"] = "N"

    return scen


def load_scenarios(scenarios_path: Optional[Path] = None) -> Tuple[Dict, List[Dict]]:
    """Carga los escenarios desde el archivo JSON.

    Soporta:
    - Formato clásico: ``{"common_params": ..., "scenarios": [...]}``.
    - Formato ``steady_states_full_run.json``: ``{"meta", "weak_grid", "strong_corner"}``;
      se toman ``strong_corner.all`` (o ``steady_states_filtered``) y ``common_params`` de
      ``meta.scenarios_file`` o ``scenarios_v1.json``.

    Args:
        scenarios_path: Ruta al JSON. Si es None, usa ``SCENARIOS_FILE`` (``scenarios.json`` en Allee/).
    """
    path = Path(scenarios_path) if scenarios_path is not None else SCENARIOS_FILE
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "common_params" in data and "scenarios" in data:
        return data["common_params"], data["scenarios"]

    if _looks_like_steady_states_full_run_payload(data):
        common = _load_reference_common_params_for_corner(data)
        blocks = _corner_blocks_list(data["strong_corner"])
        scenarios = [_corner_steady_block_to_run_scenario(b, common) for b in blocks]
        msg = (
            f"JSON tipo steady_states_full_run: {len(scenarios)} escenarios desde "
            f"strong_corner (archivo: {path.name}). common_params desde referencia v1 / meta."
        )
        print(f"Nota: {msg}")
        logger.info(msg)
        return common, scenarios

    raise ValueError(
        "El JSON no tiene formato de escenarios (common_params+scenarios) ni de "
        "steady_states_full_run (strong_corner.all / steady_states_filtered). "
        f"Claves raíz: {list(data.keys()) if isinstance(data, dict) else type(data)}."
    )

def estimate_storage_per_scenario(common_params: Dict) -> float:
    """Estima el espacio de almacenamiento necesario por escenario en GB"""
    T = float(common_params.get('T', 0.05))
    dt = float(common_params.get('dt', 0.001))
    space_size = float(common_params.get('space_size', 4))
    sample_rate = float(common_params.get('sample_rate', 0.02))
    save_images = common_params.get('SAVE_IMAGES', 'N').upper() == 'Y'
    
    # Calcular número de pasos
    total_steps = int(T / dt) + 1
    
    # Calcular tamaño de cada matriz
    sample_points = int(space_size / sample_rate) + 1
    matrix_size = sample_points * sample_points
    bytes_per_matrix = matrix_size * 8  # float64 = 8 bytes
    
    # Matrices por escenario (3 campos: c, s, i)
    matrices_per_scenario = total_steps * 3
    
    # Espacio para matrices (en bytes)
    matrices_bytes = matrices_per_scenario * bytes_per_matrix
    
    # Espacio adicional estimado
    correlation_bytes = 6 * 100 * 1024  # 6 archivos de ~100 KB
    notebooks_bytes = 2 * 5 * 1024 * 1024  # 2 notebooks de ~5 MB
    
    images_bytes = 0
    if save_images:
        images_count = int(total_steps / 10) + 1
        images_bytes = images_count * 500 * 1024  # ~500 KB por imagen
    
    total_bytes = matrices_bytes + correlation_bytes + notebooks_bytes + images_bytes
    return total_bytes / (1024**3)  # Convertir a GB

def get_available_disk_space_gb(drive_path: str = None) -> float:
    """
    Obtiene el espacio disponible en disco en GB.
    
    Args:
        drive_path: Ruta al disco a verificar (por defecto, el disco del directorio actual)
    
    Returns:
        Espacio disponible en GB
    """
    try:
        if drive_path is None:
            drive_path = str(BASE_DIR)
        total, used, free = shutil.disk_usage(drive_path)
        return free / (1024**3)
    except Exception as e:
        print(f"Advertencia: No se pudo obtener espacio en disco: {e}")
        return 0.0

def calculate_batch_size(available_gb: float, per_scenario_gb: float, safety_margin_gb: float = 2.0) -> int:
    """
    Calcula el tamaño de lote óptimo según el espacio disponible.
    
    Args:
        available_gb: Espacio disponible en GB
        per_scenario_gb: Espacio necesario por escenario en GB
        safety_margin_gb: Margen de seguridad en GB
    
    Returns:
        Número de escenarios que caben en un lote
    """
    usable_space = available_gb - safety_margin_gb
    if usable_space <= 0:
        return 1
    
    max_scenarios = int(usable_space / per_scenario_gb)
    return max(1, max_scenarios)

def divide_into_batches(scenarios: List[Dict], batch_size: int) -> List[List[Dict]]:
    """
    Divide una lista de escenarios en lotes.
    
    Args:
        scenarios: Lista de escenarios
        batch_size: Tamaño de cada lote
    
    Returns:
        Lista de lotes (cada lote es una lista de escenarios)
    """
    batches = []
    for i in range(0, len(scenarios), batch_size):
        batch = scenarios[i:i+batch_size]
        batches.append(batch)
    return batches

def check_space_before_batch(batch: List[Dict], common_params: Dict, required_margin_gb: float = 2.0) -> Tuple[bool, float, float]:
    """
    Verifica si hay suficiente espacio para ejecutar un lote.
    
    Args:
        batch: Lista de escenarios en el lote
        common_params: Parámetros comunes
        required_margin_gb: Margen de seguridad requerido en GB
    
    Returns:
        Tuple (hay_espacio, espacio_disponible_gb, espacio_necesario_gb)
    """
    per_scenario_gb = estimate_storage_per_scenario(common_params)
    required_gb = (per_scenario_gb * len(batch)) + required_margin_gb
    available_gb = get_available_disk_space_gb()
    
    has_space = available_gb >= required_gb
    return has_space, available_gb, required_gb

def print_batch_info(batches: List[List[Dict]], common_params: Dict):
    """Imprime información sobre los lotes calculados"""
    per_scenario_gb = estimate_storage_per_scenario(common_params)
    available_gb = get_available_disk_space_gb()
    
    print("="*80)
    print("INFORMACIÓN DE LOTES")
    print("="*80)
    print(f"Espacio disponible: {available_gb:.2f} GB")
    print(f"Espacio por escenario: {per_scenario_gb:.2f} GB")
    print(f"Número de lotes: {len(batches)}")
    print()
    
    for i, batch in enumerate(batches, 1):
        batch_size_gb = per_scenario_gb * len(batch)
        print(f"Lote {i}: {len(batch)} escenarios (~{batch_size_gb:.2f} GB)")
        for j, scenario in enumerate(batch, 1):
            print(f"  {j}. {scenario['name']}")
        print()

# ============================================================================
# Funciones de checkpoint/restart (de run_with_checkpoint.py)
# ============================================================================

def check_checkpoint_exists(scenario_name):
    """Verifica si existe un checkpoint para un escenario"""
    checkpoint_path = RESULTS_DIR / scenario_name / "checkpoints" / "checkpoint_latest.npz"
    return checkpoint_path.exists()

def run_script_with_checkpoint(
    scenario_name,
    script_path,
    results_dir,
    max_restarts=MAX_RESTARTS,
    checkpoint_sync_target: Optional[Path] = None,
):
    """
    Ejecuta el script Python y maneja reinicios automáticos.
    
    Args:
        scenario_name: Nombre del escenario
        script_path: Ruta del script Python a ejecutar
        results_dir: Directorio de resultados (donde se ejecuta el script)
        max_restarts: Número máximo de reinicios permitidos
        checkpoint_sync_target: Si se define, copia checkpoints/results_dir/checkpoints
            a este directorio tras cada intento (p. ej. sincronizar staging → Drive).
    
    Returns:
        tuple: (success: bool, exit_code: int, restart_count: int)
    """
    restart_count = 0
    original_cwd = os.getcwd()
    
    if not script_path.exists():
        logger.error(f"✗ Script no encontrado: {script_path}")
        return False, 1, restart_count
    
    while restart_count < max_restarts:
        logger.info(f"Ejecutando script (intento {restart_count + 1})...")
        
        try:
            # Cambiar al directorio de resultados (necesario para que el script guarde ahí)
            os.chdir(results_dir)
            
            # Ejecutar script Python directamente
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(results_dir),
                capture_output=False,  # Mostrar salida en tiempo real
                check=False  # No lanzar excepción si falla
            )
            
            # Restaurar directorio original
            os.chdir(original_cwd)

            if checkpoint_sync_target is not None:
                _sync_checkpoints_to_final(Path(results_dir), checkpoint_sync_target)
            
            # Verificar si existe archivo de señal de reinicio
            restart_signal_file = results_dir / 'checkpoints' / 'RESTART_NEEDED'
            needs_restart = restart_signal_file.exists()
            
            if needs_restart:
                restart_count += 1
                logger.info(f"⚠ Reinicio necesario (intento {restart_count}/{max_restarts})")
                # Leer información del reinicio
                try:
                    with open(restart_signal_file, 'r') as f:
                        restart_info = f.read()
                    logger.info(f"  Información: {restart_info.strip()}")
                except:
                    pass
                # Eliminar archivo de señal
                try:
                    restart_signal_file.unlink()
                except:
                    pass
                logger.info("  Esperando 5 segundos antes de reiniciar...")
                time.sleep(5)
                continue
            
            # Si llegamos aquí y el código de salida es 0, el script terminó exitosamente
            if result.returncode == 0:
                logger.info("✓ Simulación completada exitosamente")
                return True, 0, restart_count
            else:
                # Error real, no reinicio
                logger.error(f"✗ Error en la ejecución (código: {result.returncode})")
                return False, result.returncode, restart_count
                
        except Exception as e:
            # Restaurar directorio original en caso de error
            os.chdir(original_cwd)
            if checkpoint_sync_target is not None:
                try:
                    _sync_checkpoints_to_final(Path(results_dir), checkpoint_sync_target)
                except Exception:
                    pass
            logger.error(f"✗ Error inesperado: {e}")
            logger.error(traceback.format_exc())
            return False, 1, restart_count
    
    # Restaurar directorio original
    os.chdir(original_cwd)
    
    # Se alcanzó el límite de reinicios
    logger.error(f"✗ Límite de reinicios alcanzado ({max_restarts})")
    return False, 1, restart_count

# ============================================================================
# Funciones de ejecución de escenarios (de run_all_scenarios.py)
# ============================================================================

def load_system_params():
    """Carga parámetros del sistema desde .env existente que deben preservarse"""
    system_params = {}
    
    # Parámetros del sistema que NO deben sobrescribirse
    system_param_keys = ['MEDIA_DIR', 'RANDOM_SEED']
    
    if ENV_FILE.exists():
        with open(ENV_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    if key in system_param_keys:
                        system_params[key] = value
    
    return system_params

def update_env_file(common_params, scenario_params, checkpoint_max_step=None):
    """Actualiza el archivo .env con los parámetros combinados, preservando parámetros del sistema"""
    # Cargar parámetros del sistema que deben preservarse
    system_params = load_system_params()

    scenario_params = _merge_simulation_ic_from_steady_states(scenario_params)
    scenario_params = _merge_steady_state_physics_into_flat_env(scenario_params)

    # Combinar: system_params (preservados) + common_params + scenario_params
    all_params = {**common_params, **scenario_params, **system_params}
    
    # Agregar checkpoint_max_step si se especifica
    if checkpoint_max_step is not None:
        all_params['CHECKPOINT_MAX_STEP'] = str(checkpoint_max_step)
    
    with open(ENV_FILE, 'w', encoding='utf-8') as f:
        # Escribir primero los parámetros del sistema (comentados para claridad)
        if system_params:
            f.write("# Parámetros del sistema (preservados entre escenarios)\n")
            for key, value in system_params.items():
                f.write(f"{key}={value}\n")
            f.write("\n")
        
        # Escribir parámetros de simulación
        f.write(f"# Parámetros de simulación (desde {_ACTIVE_SCENARIOS_JSON_LABEL})\n")
        for key, value in all_params.items():
            if key not in system_params:  # No duplicar parámetros del sistema
                f.write(f"{key}={value}\n")

def validate_scenario_output(output_dir: Path, scenario: Dict, common_params: Dict) -> Tuple[bool, List[str]]:
    """
    Valida que el escenario haya generado los archivos esperados.
    
    Returns:
        Tuple[bool, List[str]]: (éxito, lista de advertencias)
    """
    warnings = []
    success = True
    
    # Calcular número esperado de matrices
    T = float(common_params.get('T', 0.05))
    dt = float(common_params.get('dt', 0.001))
    nb = int(common_params.get('nb', 1))
    expected_steps = int(T / dt) + 1
    expected_matrices = expected_steps * 3  # 3 campos: c, s, i
    
    matrices_dir = output_dir / 'matrices'
    correlations_dir = output_dir / 'correlations'
    
    # Verificar matrices
    if matrices_dir.exists():
        matrix_files = list(matrices_dir.glob('matrix_*.txt'))
        actual_matrices = len(matrix_files)
        
        if actual_matrices < expected_matrices * 0.9:  # Permitir 10% de tolerancia
            warning = f"Matrices insuficientes: esperadas ~{expected_matrices}, encontradas {actual_matrices}"
            warnings.append(warning)
            logger.warning(f"{scenario['name']}: {warning}")
            success = False
        elif actual_matrices != expected_matrices:
            warning = f"Matrices: esperadas ~{expected_matrices}, encontradas {actual_matrices} (diferencia: {expected_matrices - actual_matrices})"
            warnings.append(warning)
            logger.warning(f"{scenario['name']}: {warning}")
    else:
        warning = "Directorio de matrices no existe"
        warnings.append(warning)
        logger.error(f"{scenario['name']}: {warning}")
        success = False
    
    # Verificar correlaciones (deben existir si hay matrices)
    if matrices_dir.exists() and len(list(matrices_dir.glob('matrix_*.txt'))) > 0:
        if correlations_dir.exists():
            correlation_files = list(correlations_dir.glob('corr_length_*.txt'))
            expected_correlations = 6  # cs, ci, si, cc, ss, ii
            if len(correlation_files) < expected_correlations:
                warning = f"Correlaciones insuficientes: esperadas {expected_correlations}, encontradas {len(correlation_files)}"
                warnings.append(warning)
                logger.warning(f"{scenario['name']}: {warning}")
        else:
            warning = "Directorio de correlaciones no existe (pero hay matrices)"
            warnings.append(warning)
            logger.warning(f"{scenario['name']}: {warning}")
    
    return success, warnings

def run_scenario(common_params, scenario, use_checkpoint=True, max_restarts=MAX_RESTARTS, checkpoint_max_step=None):
    """Ejecuta un escenario individual con validación y logging"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Ejecutando: {scenario['name']}")
    logger.info(f"{'='*60}")
    
    print(f"\n{'='*60}")
    print(f"Ejecutando: {scenario['name']}")
    print(f"{'='*60}")
    
    # Mostrar parámetros clave
    params_info = f"  ALLEE_TYPE: {scenario.get('ALLEE_TYPE', 'N/A')}, mu: {scenario.get('mu', 'N/A')}, USE_ADAPTIVE_CONTROL: {scenario.get('USE_ADAPTIVE_CONTROL', 'N/A')}"
    print(params_info)
    logger.info(params_info)
    
    # Mostrar información de checkpoint si se especifica
    if checkpoint_max_step is not None:
        checkpoint_info = f"  Checkpoint: cargando desde antes del paso {checkpoint_max_step}"
        print(checkpoint_info)
        logger.info(f"{scenario['name']}: {checkpoint_info}")
    
    # Actualizar .env
    try:
        update_env_file(common_params, scenario, checkpoint_max_step)
        print(f"  ✓ Archivo .env actualizado")
        logger.info(f"{scenario['name']}: Archivo .env actualizado")
    except Exception as e:
        error_msg = f"Error al actualizar .env: {e}"
        print(f"  ✗ {error_msg}")
        logger.error(f"{scenario['name']}: {error_msg}", exc_info=True)
        return False
    
    # Crear directorio de resultados
    output_dir = RESULTS_DIR / scenario['name']
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        # Errno 5 suele ocurrir con mounts inestables (Google Drive/FUSE) o I/O intermitente
        # Para no abortar toda la corrida, caer a un directorio local.
        fallback_results = BASE_DIR / "results"
        fallback_output_dir = fallback_results / scenario['name']
        logger.error(f"{scenario['name']}: Error creando directorio en RESULTS_DIR={RESULTS_DIR}: {e}")
        logger.warning(f"{scenario['name']}: Usando fallback local: {fallback_output_dir}")
        print(f"  ⚠ Error creando directorio en RESULTS_DIR: {e}")
        print(f"  ⚠ Usando fallback local: {fallback_output_dir}")
        try:
            fallback_output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e2:
            error_msg = f"No se pudo crear directorio de resultados (ni en Drive ni local): {e2}"
            print(f"  ✗ {error_msg}")
            logger.error(f"{scenario['name']}: {error_msg}", exc_info=True)
            return False
        output_dir = fallback_output_dir
    final_output_dir = Path(output_dir).resolve()
    use_staging = should_use_local_staging(final_output_dir)
    run_dir = final_output_dir
    if use_staging:
        run_dir = _staging_work_dir(scenario["name"])
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        _seed_checkpoints_from_final(final_output_dir, run_dir)
        print(f"  ✓ Staging local (matrices/checkpoints): {run_dir}")
        logger.info(f"{scenario['name']}: Staging local: {run_dir}")
    print(f"  ✓ Directorio de resultados (final): {final_output_dir}")
    logger.info(f"{scenario['name']}: Directorio de resultados final: {final_output_dir}")

    original_cwd = os.getcwd()
    os.chdir(run_dir)

    cancer_dynamics_success = False
    correlation_success = False

    try:
        # Verificar si checkpoint está habilitado
        enable_checkpoint = use_checkpoint and common_params.get('ENABLE_CHECKPOINT', 'Y').upper() == 'Y'

        # Ejecutar cancer_dynamics.py
        print(f"  Ejecutando cancer_dynamics.py...")
        logger.info(f"{scenario['name']}: Ejecutando cancer_dynamics.py...")

        if not CANCER_DYNAMICS_SCRIPT.exists():
            error_msg = f"Error: No se encontró el script {CANCER_DYNAMICS_SCRIPT}"
            print(f"  ✗ {error_msg}")
            logger.error(f"{scenario['name']}: {error_msg}")
            cancer_dynamics_success = False
        else:
            if enable_checkpoint:
                print(f"  Sistema de checkpoint/restart: ACTIVADO")
                logger.info(f"{scenario['name']}: Usando sistema de checkpoint/restart")

                if check_checkpoint_exists(scenario['name']):
                    logger.info(f"✓ Checkpoint encontrado para {scenario['name']}")

                success, exit_code, restarts = run_script_with_checkpoint(
                    scenario['name'],
                    CANCER_DYNAMICS_SCRIPT,
                    run_dir,
                    max_restarts,
                    checkpoint_sync_target=final_output_dir if use_staging else None,
                )
                cancer_dynamics_success = success
                if cancer_dynamics_success:
                    print(f"  ✓ cancer_dynamics.py completado ({restarts} reinicios)")
                    logger.info(f"{scenario['name']}: cancer_dynamics.py completado exitosamente ({restarts} reinicios)")
                else:
                    error_msg = f"Error en cancer_dynamics.py (código: {exit_code}, reinicios: {restarts})"
                    print(f"  ✗ {error_msg}")
                    logger.error(f"{scenario['name']}: {error_msg}")
            else:
                try:
                    result = subprocess.run(
                        [sys.executable, str(CANCER_DYNAMICS_SCRIPT)],
                        cwd=str(run_dir),
                        check=False
                    )
                    cancer_dynamics_success = (result.returncode == 0)
                    if use_staging:
                        _sync_checkpoints_to_final(run_dir, final_output_dir)
                    if cancer_dynamics_success:
                        print(f"  ✓ cancer_dynamics.py completado")
                        logger.info(f"{scenario['name']}: cancer_dynamics.py completado exitosamente")
                    else:
                        error_msg = f"Error en cancer_dynamics.py (código: {result.returncode})"
                        print(f"  ✗ {error_msg}")
                        logger.error(f"{scenario['name']}: {error_msg}")
                except Exception as e:
                    error_msg = f"Error ejecutando cancer_dynamics.py: {e}"
                    print(f"  ✗ {error_msg}")
                    logger.error(f"{scenario['name']}: {error_msg}", exc_info=True)
                    logger.error(f"{scenario['name']}: Traceback completo:\n{traceback.format_exc()}")
                    cancer_dynamics_success = False

        # Ejecutar correlation_fourier.py después de cancer_dynamics
        if CORRELATION_SCRIPT.exists() and cancer_dynamics_success:
            print(f"  Ejecutando correlation_fourier.py...")
            logger.info(f"{scenario['name']}: Ejecutando correlation_fourier.py...")

            try:
                result = subprocess.run(
                    [sys.executable, str(CORRELATION_SCRIPT), str(run_dir)],
                    cwd=str(run_dir),
                    check=False
                )
                correlation_success = (result.returncode == 0)
                if correlation_success:
                    print(f"  ✓ correlation_fourier.py completado")
                    logger.info(f"{scenario['name']}: correlation_fourier.py completado exitosamente")
                else:
                    error_msg = f"Error en correlation_fourier.py (código: {result.returncode})"
                    print(f"  ✗ {error_msg}")
                    logger.error(f"{scenario['name']}: {error_msg}")
            except Exception as e:
                error_msg = f"Error ejecutando correlation_fourier.py: {e}"
                print(f"  ✗ {error_msg}")
                logger.error(f"{scenario['name']}: {error_msg}", exc_info=True)
                logger.error(f"{scenario['name']}: Traceback completo:\n{traceback.format_exc()}")
                correlation_success = False
        elif not CORRELATION_SCRIPT.exists():
            warning_msg = "correlation_fourier.py no encontrado, omitiendo..."
            print(f"  ⚠ {warning_msg}")
            logger.warning(f"{scenario['name']}: {warning_msg}")

        if cancer_dynamics_success and use_staging:
            try:
                _merge_staging_tree_to_final(run_dir, final_output_dir)
            except OSError as e:
                logger.error(f"{scenario['name']}: Error al volcar staging → final: {e}")

        if cancer_dynamics_success:
            validation_success, warnings = validate_scenario_output(final_output_dir, scenario, common_params)
            for warning in warnings:
                print(f"  ⚠ {warning}")
            if not validation_success:
                logger.warning(f"{scenario['name']}: Validación post-ejecución falló")

        if cancer_dynamics_success:
            print(f"  ✓ Completado: {scenario['name']}")
            logger.info(f"{scenario['name']}: Escenario completado (cancer_dynamics: ✓, correlation: {'✓' if correlation_success else '✗'})")
            return True
        else:
            logger.error(f"{scenario['name']}: Escenario falló - cancer_dynamics no completó exitosamente")
            return False

    except Exception as e:
        error_msg = f"Error inesperado en escenario: {e}"
        print(f"  ✗ {error_msg}")
        logger.error(f"{scenario['name']}: {error_msg}", exc_info=True)
        logger.error(f"{scenario['name']}: Traceback completo:\n{traceback.format_exc()}")
        return False

    finally:
        if use_staging and run_dir.resolve() != final_output_dir.resolve():
            if not cancer_dynamics_success:
                try:
                    _merge_staging_tree_to_final(run_dir, final_output_dir)
                except OSError as e:
                    logger.warning(f"{scenario['name']}: Volcado parcial staging → final falló: {e}")
            if os.environ.get("ALLEE_STAGING_CLEANUP", "Y").upper() in ("Y", "1", "YES", "TRUE"):
                shutil.rmtree(run_dir, ignore_errors=True)
        os.chdir(original_cwd)

# ============================================================================
# Funciones de retry/re-ejecución (de retry_failed_scenarios.py)
# ============================================================================

def check_scenario_status(scenario_dir: Path, scenario: Dict, common_params: Dict) -> Tuple[bool, List[str]]:
    """
    Verifica el estado de un escenario.
    
    Returns:
        Tuple[bool, List[str]]: (está_completo, lista_de_problemas)
    """
    problems = []
    is_complete = True
    
    # Calcular número esperado de matrices usando parámetros del escenario o comunes
    T = float(scenario.get('T', common_params.get('T', 0.05)))
    dt = float(scenario.get('dt', common_params.get('dt', 0.001)))
    nb = int(scenario.get('nb', common_params.get('nb', 1)))
    expected_steps = int(T / dt) + 1
    expected_matrices = expected_steps * 3 * nb  # 3 campos: c, s, i por cada bloque
    expected_correlations = 6 * nb  # cs, ci, si, cc, ss, ii por cada bloque
    
    matrices_dir = scenario_dir / 'matrices'
    correlations_dir = scenario_dir / 'correlations'
    
    # Verificar matrices
    if not matrices_dir.exists():
        problems.append("Directorio de matrices no existe")
        is_complete = False
        return is_complete, problems
    
    matrix_files = list(matrices_dir.glob('matrix_*.txt'))
    actual_matrices = len(matrix_files)
    
    if actual_matrices == 0:
        problems.append("No hay matrices generadas")
        is_complete = False
        return is_complete, problems
    
    # Solo marcar como incompleto si hay MENOS matrices de las esperadas
    # Más matrices puede indicar diferentes valores de T/dt o múltiples ejecuciones, pero no es un error
    tolerance_low = expected_matrices * 0.9
    
    if actual_matrices < tolerance_low:
        problems.append(f"Matrices insuficientes: esperadas ~{expected_matrices}, encontradas {actual_matrices}")
        is_complete = False
    
    # Verificar correlaciones
    if actual_matrices > 0:
        if not correlations_dir.exists():
            problems.append("Directorio de correlaciones no existe (pero hay matrices)")
            is_complete = False
        else:
            correlation_files = list(correlations_dir.glob('corr_length_*.txt'))
            if len(correlation_files) < expected_correlations:
                problems.append(f"Correlaciones insuficientes: esperadas {expected_correlations}, encontradas {len(correlation_files)}")
                is_complete = False
    
    # Verificar checkpoints
    checkpoint_dir = scenario_dir / 'checkpoints'
    checkpoint_file = checkpoint_dir / 'checkpoint_latest.npz'
    if checkpoint_file.exists():
        try:
            checkpoint_data = np.load(checkpoint_file)
            checkpoint_step = int(checkpoint_data['step'])
            checkpoint_t = float(checkpoint_data['t'])
            checkpoint_block = int(checkpoint_data.get('block', 1))
            problems.append(f"Checkpoint disponible: paso {checkpoint_step}, t={checkpoint_t:.4f}, bloque {checkpoint_block}")
        except Exception as e:
            problems.append(f"Checkpoint disponible pero corrupto: {e}")
    
    return is_complete, problems

def find_failed_scenarios(common_params: Dict, scenarios: List[Dict], specific_scenario: str = None) -> List[Dict]:
    """
    Identifica escenarios incompletos o fallidos.
    
    Args:
        common_params: Parámetros comunes
        scenarios: Lista de escenarios
        specific_scenario: Nombre de escenario específico a verificar (opcional)
    
    Returns:
        Lista de escenarios que necesitan re-ejecución
    """
    failed_scenarios = []
    
    print(f"\n{'='*60}")
    print("Verificando estado de escenarios")
    print(f"{'='*60}\n")
    logger.info("Verificando estado de escenarios")
    
    for scenario in scenarios:
        scenario_name = scenario['name']
        
        # Si se especificó un escenario, solo verificar ese
        if specific_scenario and scenario_name != specific_scenario:
            continue
        
        scenario_dir = RESULTS_DIR / scenario_name
        
        if not scenario_dir.exists():
            print(f"✗ {scenario_name}: Directorio no existe")
            logger.warning(f"{scenario_name}: Directorio no existe")
            failed_scenarios.append(scenario)
            continue
        
        is_complete, problems = check_scenario_status(scenario_dir, scenario, common_params)
        
        if is_complete and len(problems) == 0:
            print(f"✓ {scenario_name}: Completo")
            logger.info(f"{scenario_name}: Completo")
        else:
            print(f"✗ {scenario_name}: Incompleto")
            for problem in problems:
                print(f"  - {problem}")
            logger.warning(f"{scenario_name}: Incompleto - {', '.join(problems)}")
            failed_scenarios.append(scenario)
    
    return failed_scenarios

def remove_checkpoints_only(scenario_dir: Path) -> bool:
    """Elimina solo la carpeta checkpoints. No borra matrices ni imágenes. Devuelve True si se eliminó algo.
    Captura OSError (p. ej. I/O en Google Drive) para no abortar el batch."""
    try:
        if not scenario_dir.exists():
            return False
        checkpoints_dir = scenario_dir / 'checkpoints'
        if checkpoints_dir.exists():
            shutil.rmtree(checkpoints_dir)
            logger.info(f"Eliminados checkpoints: {checkpoints_dir}")
            return True
        return False
    except OSError as e:
        logger.warning(f"I/O al acceder a {scenario_dir}: {e}. Se continúa sin eliminar checkpoints.")
        return False


def clean_scenario_directory(scenario_dir: Path, clean_checkpoints: bool = False):
    """Limpia el directorio de un escenario (mantiene archivos de salida si existen)"""
    if not scenario_dir.exists():
        return
    
    # Mantener archivos de salida si existen (notebooks o logs)
    # Estos archivos pueden ser útiles para referencia
    
    # Eliminar subdirectorios
    subdirs_to_clean = ['matrices', 'images', 'correlations']
    if clean_checkpoints:
        subdirs_to_clean.append('checkpoints')
    
    for subdir in subdirs_to_clean:
        subdir_path = scenario_dir / subdir
        if subdir_path.exists():
            shutil.rmtree(subdir_path)
            logger.info(f"Eliminado directorio: {subdir_path}")

def clean_all_scenarios_except(keep_scenario: str, scenarios: List[Dict]) -> Tuple[int, int]:
    """
    Limpia todos los directorios de escenarios excepto uno específico.
    
    Args:
        keep_scenario: Nombre del escenario a mantener
        scenarios: Lista de todos los escenarios
    
    Returns:
        Tuple[int, int]: (escenarios_eliminados, escenarios_mantenidos)
    """
    cleaned = 0
    kept = 0
    
    print("="*80)
    print("LIMPIEZA DE ESCENARIOS")
    print("="*80)
    print(f"Escenario a mantener: {keep_scenario}")
    print()
    
    # Verificar que el escenario a mantener existe
    scenario_names = [s['name'] for s in scenarios]
    if keep_scenario not in scenario_names:
        error_msg = f"El escenario '{keep_scenario}' no existe en el archivo de escenarios cargado"
        print(f"[ERROR] {error_msg}")
        print(f"Escenarios disponibles: {', '.join(scenario_names)}")
        logger.error(error_msg)
        return cleaned, kept
    
    print(f"Total de escenarios: {len(scenarios)}")
    print()
    print("Procesando escenarios...")
    print()
    
    for scenario in scenarios:
        scenario_name = scenario['name']
        scenario_dir = RESULTS_DIR / scenario_name
        
        if scenario_name == keep_scenario:
            if scenario_dir.exists():
                print(f"  [MANTENER] {scenario_name}: Se mantiene intacto")
                logger.info(f"Manteniendo escenario: {scenario_name}")
                kept += 1
            else:
                print(f"  [INFO] {scenario_name}: Directorio no existe (se mantiene en lista)")
                kept += 1
        else:
            if scenario_dir.exists():
                try:
                    # Eliminar todo el directorio
                    shutil.rmtree(scenario_dir)
                    print(f"  [ELIMINADO] {scenario_name}")
                    logger.info(f"Eliminado escenario: {scenario_name}")
                    cleaned += 1
                except Exception as e:
                    error_msg = f"Error al eliminar {scenario_name}: {e}"
                    print(f"  [ERROR] {scenario_name}: {error_msg}")
                    logger.error(error_msg, exc_info=True)
            else:
                print(f"  [INFO] {scenario_name}: Directorio no existe, nada que limpiar")
    
    print()
    print("="*80)
    print("RESUMEN")
    print("="*80)
    print(f"Escenarios eliminados: {cleaned}")
    print(f"Escenarios mantenidos: {kept}")
    print("="*80)
    
    logger.info(f"Limpieza completada: {cleaned} eliminados, {kept} mantenidos")
    
    return cleaned, kept


def scenarios_file_arg_for_subprocess(scenarios_json_path: Path) -> str:
    """Ruta relativa a Allee/ cuando sea posible (subprocess con cwd=BASE_DIR)."""
    try:
        return str(scenarios_json_path.relative_to(BASE_DIR.resolve()))
    except ValueError:
        return str(scenarios_json_path)


def print_pipeline_next_steps(scenarios_json_path: Path) -> None:
    rel = scenarios_file_arg_for_subprocess(scenarios_json_path)
    print(f"\n{'='*60}")
    print("Pipeline Allee — siguientes pasos (PIPELINE_EJECUCION_Y_FISICA.md)")
    print(f"{'='*60}")
    print(f"  Mismo JSON en postproceso y TdC: {rel}")
    print("  1) Termodinámica efectiva:")
    print(f"       python termodynamics/calculate_thermodynamic_properties.py --scenarios-file {rel} --all")
    print("  2) Reciprocidad (ejemplo, todos los escenarios):")
    print(f"       python nonequilibrium_termodynamics/reciprocity_jacobian_analysis.py --scenarios {rel} --all-scenarios --point 0.5 0.5 0.5")
    print("  3) Flujos / σ⁺ (ajusta <NOMBRE> y --time):")
    print(f"       python nonequilibrium_termodynamics/visualize_fluxes_and_entropy_density.py --scenarios {rel} --scenario <NOMBRE> --time 1.0")
    print("  4) Figuras agregadas (revisar BASE_DIR en los scripts si no usas Drive):")
    print("       python termodynamics/generate_thermodynamic_figures.py")
    print("       python termodynamics/analyze_thermodynamic_results.py")
    print(f"{'='*60}\n")


def run_pipeline_thermodynamics(scenarios_json_path: Path) -> int:
    if not THERMO_SCRIPT.exists():
        msg = f"No se encontró {THERMO_SCRIPT}"
        print(f"⚠ {msg}")
        logger.warning(msg)
        return 1
    rel = scenarios_file_arg_for_subprocess(scenarios_json_path)
    cmd = [sys.executable, str(THERMO_SCRIPT), "--scenarios-file", rel, "--all"]
    print(f"\n{'='*60}")
    print("Pipeline: postproceso termodinámico (--all, mismo JSON que la simulación)")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}\n")
    logger.info("Ejecutando postproceso termodinámico: %s", cmd)
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    if result.returncode != 0:
        print(f"✗ calculate_thermodynamic_properties terminó con código {result.returncode}")
        logger.error("calculate_thermodynamic_properties exit code %s", result.returncode)
    else:
        print("✓ Postproceso termodinámico completado")
        logger.info("Postproceso termodinámico completado")
    return result.returncode


# ============================================================================
# Función principal
# ============================================================================

def main():
    """Función principal que maneja todos los modos de ejecución"""
    parser = argparse.ArgumentParser(
        description='Simulación espacio-tiempo (pipeline: matrices → opcional termo → TdC; ver PIPELINE_EJECUCION_Y_FISICA.md)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python run_scenarios.py                              # Ejecuta todos los escenarios
  python run_scenarios.py --scenarios-file scenarios_v1.json  # Lista de escenarios desde otro archivo
  python run_scenarios.py --scenarios-file scenarios_circular_ic.json  # Campaña con CI circular y pares uNo/uSi
  python run_scenarios.py --scenario strong_mu1_uNo_sobre_umbral  # Ejecuta solo un escenario
  python run_scenarios.py --list                       # Lista todos los escenarios disponibles
  python run_scenarios.py --status                     # Verifica estado de todos los escenarios
  python run_scenarios.py --batch-mode                  # Ejecuta en modo lotes automático
  python run_scenarios.py --batch-size 9 --batch-start 0  # Ejecuta lote específico
  python run_scenarios.py --scenario <nombre> --clean  # Limpia y ejecuta
  python run_scenarios.py --scenario <nombre> --checkpoint-step 1000  # Carga checkpoint antes del paso 1000
  python run_scenarios.py --retry-failed               # Re-ejecuta escenarios fallidos
  python run_scenarios.py --retry-failed --from-zero  # Re-ejecuta desde cero (solo borra checkpoints)
  python run_scenarios.py --clean-all-except <nombre>  # Limpia todos excepto uno
  python run_scenarios.py --no-checkpoint              # Desactiva checkpoint/restart
  python run_scenarios.py --run-thermodynamics         # Tras éxito, ejecuta calculate_thermodynamic_properties --all
  python run_scenarios.py --no-pipeline-hint           # No imprimir comandos del pipeline al final
        """
    )
    parser.add_argument(
        '--scenarios-file',
        type=str,
        metavar='PATH',
        help='Ruta al JSON de escenarios (relativa al directorio Allee/ si no es absoluta; por defecto scenarios.json)',
    )
    parser.add_argument('--scenario', '-s', type=str, help='Nombre del escenario específico a ejecutar')
    parser.add_argument('--list', '-l', action='store_true', help='Lista todos los escenarios disponibles y sale')
    parser.add_argument('--clean', action='store_true', help='Limpiar directorio del escenario antes de ejecutar')
    parser.add_argument('--from-zero', action='store_true', help='Eliminar solo checkpoints y ejecutar desde t=0. Sobrescribe matrices e imágenes sin borrarlas previamente.')
    parser.add_argument('--batch-size', type=int, help='Ejecutar solo un lote de N escenarios')
    parser.add_argument('--batch-start', type=int, default=0, help='Índice de inicio para ejecución por lotes (por defecto: 0)')
    parser.add_argument('--batch-mode', action='store_true', help='Ejecutar en modo lotes automático (ejecuta todos los lotes secuencialmente)')
    parser.add_argument('--retry-failed', action='store_true', help='Re-ejecutar solo escenarios incompletos o fallidos')
    parser.add_argument('--clean-all-except', type=str, metavar='SCENARIO', help='Limpiar todos los escenarios excepto el especificado')
    parser.add_argument('--status', action='store_true', help='Verificar y mostrar estado de todos los escenarios (sin ejecutar)')
    parser.add_argument('--yes', '-y', action='store_true', help='Ejecutar sin confirmación interactiva (útil para scripts)')
    parser.add_argument('--no-checkpoint', action='store_true', help='Desactivar sistema de checkpoint/restart')
    parser.add_argument('--max-restarts', type=int, default=MAX_RESTARTS, help=f'Número máximo de reinicios permitidos (default: {MAX_RESTARTS})')
    parser.add_argument('--checkpoint-step', type=int, metavar='STEP', help='Cargar checkpoint desde antes de este paso (evita checkpoints problemáticos)')
    parser.add_argument('--analyze-diagnostics', action='store_true', help='Analizar diagnostics/*.json generados por escenarios y mostrar un resumen')
    parser.add_argument(
        '--run-thermodynamics',
        action='store_true',
        help='Tras simulación sin fallos, ejecutar termodynamics/calculate_thermodynamic_properties.py --all con el mismo JSON',
    )
    parser.add_argument(
        '--no-pipeline-hint',
        action='store_true',
        help='No mostrar al final el bloque de comandos del pipeline (termo, TdC, figuras)',
    )
    
    args = parser.parse_args()

    if args.scenarios_file:
        raw = Path(args.scenarios_file).expanduser()
        scenarios_json_path = raw.resolve() if raw.is_absolute() else (BASE_DIR / raw).resolve()
    else:
        scenarios_json_path = SCENARIOS_FILE.resolve()

    global _ACTIVE_SCENARIOS_JSON_LABEL
    try:
        _ACTIVE_SCENARIOS_JSON_LABEL = str(scenarios_json_path.relative_to(BASE_DIR.resolve()))
    except ValueError:
        _ACTIVE_SCENARIOS_JSON_LABEL = str(scenarios_json_path)
    
    logger.info("="*60)
    logger.info("Sistema de Automatización de Simulaciones")
    logger.info("="*60)
    logger.info(f"Log guardado en: {log_file}")
    
    print("="*60)
    print("Sistema de Automatización de Simulaciones")
    print("="*60)
    print(f"Log guardado en: {log_file}")
    
    # Verificar que los scripts existen
    if not CANCER_DYNAMICS_SCRIPT.exists():
        error_msg = f"Error: No se encontró el script {CANCER_DYNAMICS_SCRIPT}"
        print(f"✗ {error_msg}")
        logger.error(error_msg)
        return 1
    
    if not CORRELATION_SCRIPT.exists():
        warning_msg = f"Advertencia: No se encontró el script {CORRELATION_SCRIPT}"
        print(f"⚠ {warning_msg}")
        print(f"  Se ejecutará solo cancer_dynamics.py")
        logger.warning(warning_msg)
    
    # Cargar configuración
    try:
        common_params, scenarios = load_scenarios(scenarios_json_path)
        info_msg = f"Cargados {len(scenarios)} escenarios desde {scenarios_json_path}"
        print(f"✓ {info_msg}")
        logger.info(info_msg)
    except Exception as e:
        error_msg = f"Error al cargar escenarios: {e}"
        print(f"✗ {error_msg}")
        logger.error(error_msg, exc_info=True)
        return 1
    
    # Verificar si estamos usando Google Drive y si está montado
    drive_mount_point = get_google_drive_mount_point()
    if RESULTS_DIR.as_posix().startswith(drive_mount_point.as_posix()):
        if not is_google_drive_mounted():
            error_msg = f"Error: Google Drive no está montado en {drive_mount_point}"
            print(f"✗ {error_msg}")
            print(f"  Ejecuta primero: bash mount_google_drive.sh")
            logger.error(error_msg)
            return 1
        
        # Verificar que el directorio de resultados existe y es accesible
        success, error_msg = verify_results_dir_write_access(RESULTS_DIR)
        if not success:
            error_msg_full = f"Error: No se puede escribir en Google Drive: {error_msg}"
            print(f"✗ {error_msg_full}")
            print(f"  Verifica que Google Drive esté montado correctamente")
            print(f"  Ruta esperada: {RESULTS_DIR}")
            logger.error(error_msg_full)
            return 1
        
        # Verificar que la estructura de carpetas existe
        doctorado_dir = drive_mount_point / "Doctorado Erick Serrato"
        resultados_dir = doctorado_dir / "Resultados Paper"
        
        if not doctorado_dir.exists():
            print(f"⚠ Advertencia: La carpeta 'Doctorado Erick Serrato' no existe en Google Drive")
            print(f"  Se creará automáticamente cuando se ejecute el primer escenario")
        elif not resultados_dir.exists():
            print(f"⚠ Advertencia: La carpeta 'Resultados Paper' no existe")
            print(f"  Se creará automáticamente cuando se ejecute el primer escenario")
        else:
            print(f"✓ Estructura de carpetas verificada en Google Drive")
            print(f"  Carpeta base: {doctorado_dir}")
            print(f"  Carpeta de resultados: {resultados_dir}")
        
        print(f"✓ Google Drive verificado: {RESULTS_DIR}")
        logger.info(f"Usando Google Drive: {RESULTS_DIR}")
        logger.info(f"Carpeta base: {doctorado_dir}")
        logger.info(f"Carpeta de resultados: {resultados_dir}")
    else:
        print(f"✓ Usando directorio local: {RESULTS_DIR}")
        logger.info(f"Usando directorio local: {RESULTS_DIR}")

    # Modo analyze-diagnostics (solo lectura, no ejecuta simulaciones)
    if args.analyze_diagnostics:
        print("="*80)
        print("ANÁLISIS DE DIAGNÓSTICOS (diagnostics/*.json)")
        print("="*80)
        print(f"Resultados en: {RESULTS_DIR}")
        print()

        def _fmt_sci(x):
            try:
                return f"{float(x):.3e}"
            except Exception:
                return "N/A"

        total_with_diags = 0
        for scenario in scenarios:
            scenario_name = scenario['name']
            scenario_dir = RESULTS_DIR / scenario_name
            diagnostics_dir = scenario_dir / "diagnostics"
            if not diagnostics_dir.exists():
                continue

            diag_files = sorted(diagnostics_dir.glob("diagnostic_*.json"))
            if not diag_files:
                continue

            total_with_diags += 1
            latest = diag_files[-1]
            try:
                with open(latest, "r", encoding="utf-8") as f:
                    data = json.load(f)

                t = data.get("t", None)
                step = data.get("step", None)
                reason = data.get("reason", "")
                c_stats = data.get("c_stats", {}) or {}
                i_stats = data.get("i_stats", {}) or {}
                u_stats = data.get("u_stats", {}) or {}
                grad_i = data.get("grad_i", {}) or {}

                print(f"- Escenario: {scenario_name}")
                print(f"  Archivo: {latest.name}")
                print(f"  reason: {reason} | t={t} | step={step}")
                if c_stats:
                    print(
                        "  c: "
                        f"min={_fmt_sci(c_stats.get('min'))} "
                        f"max={_fmt_sci(c_stats.get('max'))} "
                        f"mean={_fmt_sci(c_stats.get('mean'))}"
                    )
                if i_stats:
                    print(
                        "  i: "
                        f"min={_fmt_sci(i_stats.get('min'))} "
                        f"max={_fmt_sci(i_stats.get('max'))} "
                        f"mean={_fmt_sci(i_stats.get('mean'))}"
                    )
                if u_stats:
                    sat = u_stats.get("saturation_ratio", None)
                    sat_str = f"{sat*100:.1f}%" if isinstance(sat, (int, float)) else "N/A"
                    print(
                        "  u: "
                        f"min={_fmt_sci(u_stats.get('min'))} "
                        f"max={_fmt_sci(u_stats.get('max'))} "
                        f"mean={_fmt_sci(u_stats.get('mean'))} "
                        f"sat={sat_str}"
                    )
                if grad_i and "max_grad" in grad_i:
                    print(
                        "  |∇i|: "
                        f"max={_fmt_sci(grad_i.get('max_grad'))} "
                        f"p99={_fmt_sci(grad_i.get('p99_grad'))}"
                    )

                extra = data.get("extra", {}) or {}
                if extra.get("error_type") or extra.get("error_message"):
                    print(f"  error: {extra.get('error_type', '')} - {extra.get('error_message', '')}")
                print()

            except Exception as e:
                print(f"- Escenario: {scenario_name}")
                print(f"  ⚠ No se pudo leer {latest.name}: {e}")
                print()

        if total_with_diags == 0:
            print("No se encontraron diagnósticos. Ejecuta escenarios con ENABLE_DIAGNOSTICS=Y.")
        return 0
    
    # Listar escenarios si se solicita
    if args.list:
        print(f"\n{'='*60}")
        print("Escenarios disponibles:")
        print(f"{'='*60}")
        for i, scenario in enumerate(scenarios, 1):
            print(f"  {i}. {scenario['name']}")
        print(f"{'='*60}")
        return 0
    
    # Modo status (verificar estado sin ejecutar)
    if args.status:
        print("="*80)
        print("VERIFICACIÓN DE ESTADO DE ESCENARIOS")
        print("="*80)
        print()
        
        T = float(common_params.get('T', 0.05))
        dt = float(common_params.get('dt', 0.001))
        
        print(f"Parámetros comunes:")
        print(f"  T = {T}")
        print(f"  dt = {dt}")
        print(f"  Pasos esperados por escenario: {int(T/dt) + 1}")
        print(f"  Matrices esperadas por escenario: {(int(T/dt) + 1) * 3}")
        print()
        print("="*80)
        print()
        
        completed = []
        partial = []
        incomplete = []
        failed = []
        not_started = []
        
        for scenario in scenarios:
            scenario_dir = RESULTS_DIR / scenario['name']
            is_complete, problems = check_scenario_status(scenario_dir, scenario, common_params)
            
            # Calcular porcentaje de completitud
            T_scenario = float(scenario.get('T', common_params.get('T', 0.05)))
            dt_scenario = float(scenario.get('dt', common_params.get('dt', 0.001)))
            nb_scenario = int(scenario.get('nb', common_params.get('nb', 1)))
            expected_steps = int(T_scenario / dt_scenario) + 1
            expected_matrices = expected_steps * 3 * nb_scenario
            
            matrices_dir = scenario_dir / 'matrices'
            correlations_dir = scenario_dir / 'correlations'
            
            actual_matrices = 0
            actual_correlations = 0
            completion_percentage = 0.0
            
            if matrices_dir.exists():
                matrix_files = list(matrices_dir.glob('matrix_*.txt'))
                actual_matrices = len(matrix_files)
                if expected_matrices > 0:
                    completion_percentage = (actual_matrices / expected_matrices) * 100
            
            if correlations_dir.exists():
                correlation_files = list(correlations_dir.glob('corr_length_*.txt'))
                actual_correlations = len(correlation_files)
            
            status_str = 'COMPLETO'
            if not scenario_dir.exists():
                status_str = 'NO INICIADO'
            elif actual_matrices == 0:
                status_str = 'FALLIDO (sin matrices)'
            elif completion_percentage < 50:
                status_str = 'INCOMPLETO'
            elif completion_percentage < 95:
                status_str = 'PARCIAL'
            
            print(f"Escenario: {scenario['name']}")
            print(f"  Estado: {status_str}")
            print(f"  Matrices: {actual_matrices}/{expected_matrices} ({completion_percentage:.1f}%)")
            print(f"  Correlaciones: {actual_correlations}/{6 * nb_scenario}")
            if problems:
                for problem in problems:
                    print(f"  - {problem}")
            print()
            
            if status_str == 'COMPLETO':
                completed.append(scenario['name'])
            elif status_str == 'PARCIAL':
                partial.append(scenario['name'])
            elif status_str == 'INCOMPLETO':
                incomplete.append(scenario['name'])
            elif status_str.startswith('FALLIDO'):
                failed.append(scenario['name'])
            else:
                not_started.append(scenario['name'])
        
        print("="*80)
        print("RESUMEN")
        print("="*80)
        print(f"[OK] COMPLETOS ({len(completed)}):")
        for name in completed:
            print(f"  - {name}")
        print()
        
        print(f"[PARCIAL] PARCIALES ({len(partial)}):")
        for name in partial:
            print(f"  - {name}")
        print()
        
        print(f"[INCOMPLETO] INCOMPLETOS ({len(incomplete)}):")
        for name in incomplete:
            print(f"  - {name}")
        print()
        
        print(f"[FALLIDO] FALLIDOS ({len(failed)}):")
        for name in failed:
            print(f"  - {name}")
        print()
        
        print(f"[NO INICIADO] NO INICIADOS ({len(not_started)}):")
        for name in not_started:
            print(f"  - {name}")
        print()
        
        print("="*80)
        total = len(scenarios)
        print(f"Total: {total} escenarios")
        if total > 0:
            print(f"  Completos: {len(completed)} ({len(completed)/total*100:.1f}%)")
            print(f"  Parciales: {len(partial)} ({len(partial)/total*100:.1f}%)")
            print(f"  Incompletos: {len(incomplete)} ({len(incomplete)/total*100:.1f}%)")
            print(f"  Fallidos: {len(failed)} ({len(failed)/total*100:.1f}%)")
            print(f"  No iniciados: {len(not_started)} ({len(not_started)/total*100:.1f}%)")
        print("="*80)
        
        return 0
    
    # Modo clean-all-except (limpiar todos excepto uno)
    if args.clean_all_except:
        # Confirmar antes de limpiar (solo si no se usa --yes)
        if not args.yes:
            print(f"\n⚠ ADVERTENCIA: Se eliminarán TODOS los directorios de escenarios excepto '{args.clean_all_except}'")
            try:
                response = input("¿Continuar? (s/n): ").strip().lower()
                if response != 's':
                    print("Operación cancelada.")
                    logger.info("Operación cancelada por el usuario")
                    return 0
            except EOFError:
                print("\n⚠ Modo no interactivo detectado. Usa --yes para ejecutar sin confirmación.")
                logger.warning("Modo no interactivo detectado. Usa --yes para ejecutar sin confirmación.")
                return 0
        
        cleaned, kept = clean_all_scenarios_except(args.clean_all_except, scenarios)
        return 0
    
    # Modo retry-failed (re-ejecutar escenarios fallidos)
    if args.retry_failed:
        logger.info("="*60)
        logger.info("Script de Re-ejecución de Escenarios Fallidos")
        logger.info("="*60)
        
        print("="*60)
        print("Script de Re-ejecución de Escenarios Fallidos")
        print("="*60)
        
        # Encontrar escenarios fallidos
        failed_scenarios = find_failed_scenarios(common_params, scenarios, args.scenario)
        
        if not failed_scenarios:
            success_msg = "Todos los escenarios están completos. No hay necesidad de re-ejecutar."
            print(f"\n✓ {success_msg}")
            logger.info(success_msg)
            return 0
        
        print(f"\n{'='*60}")
        print(f"Escenarios que necesitan re-ejecución: {len(failed_scenarios)}")
        print(f"{'='*60}")
        for scenario in failed_scenarios:
            print(f"  - {scenario['name']}")
        logger.info(f"Escenarios que necesitan re-ejecución: {len(failed_scenarios)}")
        
        # Limpiar directorios si se solicita
        if args.from_zero:
            print(f"\n⚠ Eliminando solo checkpoints (ejecutar desde t=0, sin borrar matrices/imágenes)...")
            logger.info("Eliminando checkpoints de escenarios fallidos")
            for scenario in failed_scenarios:
                scenario_dir = RESULTS_DIR / scenario['name']
                if remove_checkpoints_only(scenario_dir):
                    print(f"  ✓ {scenario['name']}: checkpoints eliminados")
            print("✓ Listo")
        elif args.clean:
            print(f"\n⚠ Limpiando directorios de escenarios fallidos...")
            logger.info("Limpiando directorios de escenarios fallidos")
            for scenario in failed_scenarios:
                scenario_dir = RESULTS_DIR / scenario['name']
                clean_scenario_directory(scenario_dir, clean_checkpoints=True)
            print("✓ Directorios limpiados")
            logger.info("Directorios limpiados")
        
        # Confirmar re-ejecución (solo si no se usa --yes)
        if not args.yes and not args.scenario:
            try:
                response = input(f"\n¿Re-ejecutar {len(failed_scenarios)} escenario(s)? (s/n): ")
                if response.lower() != 's':
                    print("Operación cancelada.")
                    logger.info("Operación cancelada por el usuario")
                    return 0
            except EOFError:
                # Modo no interactivo (ej: desde Docker)
                print("\n⚠ Modo no interactivo detectado. Usa --yes para ejecutar sin confirmación.")
                logger.warning("Modo no interactivo detectado. Usa --yes para ejecutar sin confirmación.")
                return 0
        
        # Re-ejecutar escenarios fallidos
        print(f"\n{'='*60}")
        print("Re-ejecutando escenarios fallidos")
        if args.from_zero:
            print("⚠ Modo: Ejecutando desde cero (checkpoints limpiados)")
        else:
            print("ℹ Modo: Continuando desde checkpoints si existen")
        print(f"{'='*60}\n")
        logger.info("Re-ejecutando escenarios fallidos")
        
        successful = 0
        failed = 0
        use_checkpoint = not args.no_checkpoint
        
        for i, scenario in enumerate(failed_scenarios, 1):
            print(f"\n[{i}/{len(failed_scenarios)}] {scenario['name']}")
            logger.info(f"[{i}/{len(failed_scenarios)}] {scenario['name']}")
            
            if run_scenario(common_params, scenario, use_checkpoint, args.max_restarts, args.checkpoint_step):
                successful += 1
            else:
                failed += 1
        
        # Resumen final
        print(f"\n{'='*60}")
        print("Resumen de re-ejecución")
        print(f"{'='*60}")
        print(f"Total re-ejecutados: {len(failed_scenarios)}")
        print(f"✓ Exitosos: {successful}")
        print(f"✗ Fallidos: {failed}")
        print(f"{'='*60}")
        
        logger.info("="*60)
        logger.info("Resumen de re-ejecución")
        logger.info(f"Total re-ejecutados: {len(failed_scenarios)}")
        logger.info(f"Exitosos: {successful}")
        logger.info(f"Fallidos: {failed}")
        
        pipeline_thermo_rc = 0
        if args.run_thermodynamics:
            if failed > 0:
                print("\n⚠ --run-thermodynamics omitido: hubo escenarios fallidos en la re-ejecución.")
                logger.warning("--run-thermodynamics omitido por fallos en retry-failed")
            else:
                pipeline_thermo_rc = run_pipeline_thermodynamics(scenarios_json_path)
        if not args.no_pipeline_hint and successful > 0:
            print_pipeline_next_steps(scenarios_json_path)

        if failed > 0:
            warning_msg = f"Advertencia: {failed} escenario(s) fallaron durante la re-ejecución. Revisa los logs en {log_file}"
            print(f"\n⚠ {warning_msg}")
            logger.warning(warning_msg)
            return 1
        else:
            success_msg = "Todos los escenarios se re-ejecutaron exitosamente!"
            print(f"\n✓ {success_msg}")
            logger.info(success_msg)
            if args.run_thermodynamics and pipeline_thermo_rc != 0:
                return 1
            return 0
    
    # Modo lotes automático
    if args.batch_mode:
        per_scenario_gb = estimate_storage_per_scenario(common_params)
        available_gb = get_available_disk_space_gb()
        
        print(f"\nTotal de escenarios: {len(scenarios)}")
        print(f"Espacio por escenario: {per_scenario_gb:.2f} GB")
        print(f"Espacio disponible: {available_gb:.2f} GB")
        
        # Calcular tamaño de lote
        if args.batch_size:
            batch_size = args.batch_size
            print(f"\nTamaño de lote (especificado): {batch_size} escenarios")
        else:
            batch_size = 1
            print(f"\nTamaño de lote (por defecto): {batch_size} escenario por lote")
            auto_batch_size = calculate_batch_size(available_gb, per_scenario_gb)
            print(f"  (Cálculo automático basado en espacio: {auto_batch_size} escenarios)")
        
        # Dividir en lotes
        batches = divide_into_batches(scenarios, batch_size)
        print_batch_info(batches, common_params)
        
        # Confirmar ejecución
        if not args.yes:
            print("="*80)
            print("¿Ejecutar los escenarios en estos lotes automáticamente?")
            print("="*80)
            try:
                response = input("Presiona 's' para continuar, cualquier otra tecla para cancelar: ").strip().lower()
                if response != 's':
                    print("Operación cancelada.")
                    return 0
            except (EOFError, KeyboardInterrupt):
                print("\nModo no interactivo detectado. Usa --yes para ejecutar sin confirmación.")
                return 0
        
        # Ejecutar lotes secuencialmente
        print("\n" + "="*80)
        print("INICIANDO EJECUCIÓN DE LOTES")
        print("="*80)
        if args.from_zero:
            print("⚠ Modo --from-zero: Solo se eliminan checkpoints, matrices e imágenes se sobrescribirán")
        
        successful_batches = 0
        failed_batches = []
        use_checkpoint = not args.no_checkpoint
        
        for i, batch in enumerate(batches):
            batch_index = i + 1
            start_idx = i * batch_size
            
            # Verificar espacio antes de ejecutar
            has_space, available, required = check_space_before_batch(batch, common_params)
            
            if not has_space:
                print(f"\n[ADVERTENCIA] Lote {batch_index}: Espacio insuficiente")
                print(f"  Disponible: {available:.2f} GB")
                print(f"  Necesario: {required:.2f} GB")
                
                if not args.yes:
                    try:
                        response = input(f"\n¿Continuar de todas formas? (s/n): ").strip().lower()
                        if response != 's':
                            print(f"Lote {batch_index} omitido.")
                            failed_batches.append(batch_index)
                            continue
                    except (EOFError, KeyboardInterrupt):
                        print(f"\nLote {batch_index} omitido (modo no interactivo)")
                        failed_batches.append(batch_index)
                        continue
            
            # Ejecutar escenarios del lote
            print(f"\n{'='*80}")
            print(f"EJECUTANDO LOTE {batch_index}/{len(batches)}")
            print(f"{'='*80}")
            
            successful = 0
            failed = 0
            
            for scenario in batch:
                if args.from_zero:
                    remove_checkpoints_only(RESULTS_DIR / scenario['name'])
                if run_scenario(common_params, scenario, use_checkpoint, args.max_restarts, args.checkpoint_step):
                    successful += 1
                else:
                    failed += 1
                    failed_batches.append(batch_index)
            
            if successful == len(batch):
                successful_batches += 1
                new_available = get_available_disk_space_gb()
                print(f"\nEspacio disponible después del lote: {new_available:.2f} GB")
            else:
                print(f"\n[ADVERTENCIA] Lote {batch_index} tuvo fallos.")
                
                if batch_index < len(batches) and not args.yes:
                    try:
                        response = input(f"\n¿Continuar con el siguiente lote? (s/n): ").strip().lower()
                        if response != 's':
                            break
                    except (EOFError, KeyboardInterrupt):
                        break
        
        # Resumen final de lotes
        print("\n" + "="*80)
        print("RESUMEN DE EJECUCIÓN")
        print("="*80)
        print(f"Total de lotes: {len(batches)}")
        print(f"Lotes exitosos: {successful_batches}")
        print(f"Lotes fallidos: {len(failed_batches)}")
        if failed_batches:
            print(f"\nLotes fallidos: {', '.join(map(str, failed_batches))}")
        print("="*80)

        batches_ok = len(failed_batches) == 0
        pipeline_thermo_rc = 0
        if args.run_thermodynamics:
            if not batches_ok:
                print("\n⚠ --run-thermodynamics omitido: hubo lotes con fallos.")
                logger.warning("--run-thermodynamics omitido por fallos en modo lotes")
            else:
                pipeline_thermo_rc = run_pipeline_thermodynamics(scenarios_json_path)
        if not args.no_pipeline_hint and successful_batches > 0:
            print_pipeline_next_steps(scenarios_json_path)

        batch_exit = 0 if batches_ok else 1
        if args.run_thermodynamics and batches_ok and pipeline_thermo_rc != 0:
            batch_exit = 1
        return batch_exit
    
    # Filtrar escenarios si se especifica uno
    # Si se especifica un escenario, ejecutar ese y todos los siguientes
    if args.scenario:
        scenario_names = [s['name'] for s in scenarios]
        if args.scenario not in scenario_names:
            error_msg = f"Error: El escenario '{args.scenario}' no existe."
            print(f"✗ {error_msg}")
            print(f"\nEscenarios disponibles:")
            for name in scenario_names:
                print(f"  - {name}")
            logger.error(error_msg)
            return 1
        
        # Encontrar el índice del escenario especificado
        scenario_index = next((i for i, s in enumerate(scenarios) if s['name'] == args.scenario), None)
        if scenario_index is not None:
            # Ejecutar el escenario especificado y todos los siguientes
            scenarios = scenarios[scenario_index:]
            print(f"\nℹ Ejecutando escenario '{args.scenario}' y todos los siguientes ({len(scenarios)} escenarios)")
            logger.info(f"Ejecutando escenario '{args.scenario}' y todos los siguientes ({len(scenarios)} escenarios)")
        else:
            scenarios = [s for s in scenarios if s['name'] == args.scenario]
        
        # Limpiar directorio si se solicita
        scenario_dir = RESULTS_DIR / args.scenario
        if args.from_zero and scenario_dir.exists():
            # Solo checkpoints: ejecutar desde t=0, sobrescribir sin borrar
            print(f"\n⚠ Eliminando checkpoints del escenario: {args.scenario}")
            logger.info(f"Eliminando checkpoints del escenario: {args.scenario}")
            if remove_checkpoints_only(scenario_dir):
                print(f"✓ Checkpoints eliminados (matrices e imágenes se sobrescribirán)")
            else:
                print(f"ℹ No hay checkpoints que limpiar")
        elif args.clean and scenario_dir.exists():
            # Limpiar todo el directorio
            print(f"\n⚠ Limpiando directorio completo: {scenario_dir}")
            logger.info(f"Limpiando directorio completo: {scenario_dir}")
            shutil.rmtree(scenario_dir)
            print(f"✓ Directorio limpiado completamente")
            logger.info("Directorio limpiado completamente")
        elif args.clean and not scenario_dir.exists():
            print(f"ℹ El directorio {scenario_dir} no existe, nada que limpiar")
            logger.info(f"El directorio {scenario_dir} no existe")
    
    # Filtrar escenarios por lote si se especifica batch-size
    elif args.batch_size:
        total_scenarios = len(scenarios)
        start_idx = args.batch_start
        end_idx = min(start_idx + args.batch_size, total_scenarios)
        
        if start_idx >= total_scenarios:
            error_msg = f"Error: Índice de inicio ({start_idx}) es mayor o igual al número de escenarios ({total_scenarios})"
            print(f"✗ {error_msg}")
            logger.error(error_msg)
            return 1
        
        batch_scenarios = scenarios[start_idx:end_idx]
        print(f"\n{'='*60}")
        print(f"Ejecutando LOTE (índices {start_idx} a {end_idx-1})")
        print(f"{'='*60}")
        print(f"Escenarios en este lote: {len(batch_scenarios)}")
        for i, scenario in enumerate(batch_scenarios, start_idx + 1):
            print(f"  {i}. {scenario['name']}")
        print(f"{'='*60}\n")
        logger.info(f"Ejecutando lote: índices {start_idx} a {end_idx-1}, {len(batch_scenarios)} escenarios")
        
        scenarios = batch_scenarios
    
    print(f"\nScripts:")
    print(f"  - {CANCER_DYNAMICS_SCRIPT}")
    if CORRELATION_SCRIPT.exists():
        print(f"  - {CORRELATION_SCRIPT}")
    print(f"Resultados en: {RESULTS_DIR}")
    
    if args.scenario:
        print(f"\nEjecutando escenario '{args.scenario}' y todos los siguientes ({len(scenarios)} escenarios)")
        logger.info(f"Ejecutando escenario '{args.scenario}' y todos los siguientes ({len(scenarios)} escenarios)")
    else:
        print(f"\nIniciando ejecución de {len(scenarios)} escenarios...")
        logger.info(f"Iniciando ejecución de {len(scenarios)} escenarios...")
    
    if args.from_zero:
        print("⚠ Modo --from-zero: Solo se eliminan checkpoints, matrices e imágenes se sobrescribirán")
        logger.info("Modo --from-zero activo")
    
    # Ejecutar cada escenario
    successful = 0
    failed = 0
    failed_scenarios = []
    use_checkpoint = not args.no_checkpoint
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}] {scenario['name']}")
        logger.info(f"[{i}/{len(scenarios)}] {scenario['name']}")
        
        if args.from_zero:
            remove_checkpoints_only(RESULTS_DIR / scenario['name'])
        
        try:
            if run_scenario(common_params, scenario, use_checkpoint, args.max_restarts, args.checkpoint_step):
                successful += 1
            else:
                failed += 1
                failed_scenarios.append(scenario['name'])
                print(f"⚠ Escenario '{scenario['name']}' falló. Continuando con el siguiente...")
                logger.warning(f"Escenario '{scenario['name']}' falló. Continuando con el siguiente...")
        except Exception as e:
            failed += 1
            failed_scenarios.append(scenario['name'])
            error_msg = f"Error inesperado en escenario '{scenario['name']}': {e}"
            print(f"✗ {error_msg}")
            print(f"⚠ Continuando con el siguiente escenario...")
            logger.error(error_msg, exc_info=True)
            logger.warning(f"Continuando con el siguiente escenario después del error en '{scenario['name']}'")
    
    # Resumen final
    print(f"\n{'='*60}")
    print("Resumen de ejecución")
    print(f"{'='*60}")
    print(f"Total de escenarios: {len(scenarios)}")
    print(f"✓ Exitosos: {successful}")
    print(f"✗ Fallidos: {failed}")
    if failed_scenarios:
        print(f"\nEscenarios fallidos:")
        for name in failed_scenarios:
            print(f"  - {name}")
    print(f"{'='*60}")
    
    logger.info("="*60)
    logger.info("Resumen de ejecución")
    logger.info(f"Total de escenarios: {len(scenarios)}")
    logger.info(f"Exitosos: {successful}")
    logger.info(f"Fallidos: {failed}")
    if failed_scenarios:
        logger.warning(f"Escenarios fallidos: {', '.join(failed_scenarios)}")
    
    if failed > 0:
        warning_msg = f"Advertencia: {failed} escenario(s) fallaron. Revisa los logs en {log_file}"
        print(f"\n⚠ {warning_msg}")
        logger.warning(warning_msg)
    else:
        success_msg = "Todas las simulaciones se completaron exitosamente!"
        print(f"\n✓ {success_msg}")
        logger.info(success_msg)

    exit_code = 0 if failed == 0 else 1
    pipeline_thermo_rc = 0
    if args.run_thermodynamics:
        if failed > 0:
            print("\n⚠ --run-thermodynamics omitido: hubo escenarios fallidos.")
            logger.warning("--run-thermodynamics omitido por fallos en simulación")
        elif successful > 0:
            pipeline_thermo_rc = run_pipeline_thermodynamics(scenarios_json_path)
            if pipeline_thermo_rc != 0:
                exit_code = 1
    if not args.no_pipeline_hint and successful > 0:
        print_pipeline_next_steps(scenarios_json_path)
    
    # Ejecutar correlation_comparison.ipynb después de todos los escenarios (si existe)
    # Nota: Este notebook aún no tiene versión .py, se mantiene como notebook opcional
    if not args.scenario and COMPARISON_NOTEBOOK.exists() and successful > 0:
        print(f"\n{'='*60}")
        print("Ejecutando análisis de comparación de escenarios...")
        print(f"{'='*60}")
        logger.info("Ejecutando correlation_comparison.ipynb...")
        
        try:
            # Intentar usar papermill si está disponible, sino mostrar advertencia
            try:
                import papermill as pm
                comparison_output = RESULTS_DIR / "comparison_analysis.ipynb"
                pm.execute_notebook(
                    COMPARISON_NOTEBOOK,
                    comparison_output,
                    parameters={},
                    log_output=True,
                    progress_bar=True
                )
                print(f"✓ correlation_comparison.ipynb completado")
                print(f"  Resultados guardados en: {RESULTS_DIR / 'comparisons'}")
                logger.info("correlation_comparison.ipynb completado exitosamente")
            except ImportError:
                warning_msg = "papermill no está instalado, omitiendo análisis de comparación..."
                print(f"⚠ {warning_msg}")
                logger.warning(warning_msg)
        except Exception as e:
            error_msg = f"Error en correlation_comparison.ipynb: {e}"
            print(f"✗ {error_msg}")
            logger.error(error_msg, exc_info=True)
    elif args.scenario:
        info_msg = "Análisis de comparación omitido (ejecutando escenario específico)"
        print(f"\nℹ {info_msg}")
        logger.info(info_msg)
    elif not COMPARISON_NOTEBOOK.exists():
        warning_msg = "correlation_comparison.ipynb no encontrado, omitiendo análisis de comparación..."
        print(f"\n⚠ {warning_msg}")
        logger.warning(warning_msg)
    elif successful == 0:
        warning_msg = "No hay escenarios exitosos, omitiendo análisis de comparación..."
        print(f"\n⚠ {warning_msg}")
        logger.warning(warning_msg)
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main())

