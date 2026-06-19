"""
utils_paths.py

Utilidades compartidas para manejo de rutas y detección de Google Drive.
Proporciona funciones para determinar directorios de resultados y verificar montaje de Google Drive.
"""

import os
from pathlib import Path
from typing import Optional

# Layout antiguo: algunos proyectos aún tienen CSV bajo esta subcarpeta (generate_figures lo prueba como respaldo).
STEADY_STATES_EXTRACT_SUBDIR = "steady_states_extract"
# Salida del pipeline completo steady_states.py (JSON unificado) bajo Resultados Paper o RESULTS_DIR.
STEADY_STATES_RUNS_SUBDIR = "estados_estacionarios"


def get_results_dir_from_env() -> Optional[Path]:
    """
    Si ``RESULTS_DIR`` está definido, devuelve esa ruta resuelta (creando el directorio raíz).
    Si falta la variable o no se puede crear, ``None``.
    """
    env = os.getenv("RESULTS_DIR")
    if not env:
        return None
    try:
        root = Path(env).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root
    except OSError as e:
        print(f"[!] No se pudo preparar RESULTS_DIR ({env}): {e}")
        return None


def get_results_dir_cloud_only() -> Path:
    """
    Resuelve la carpeta «Resultados Paper» sin usar ``Allee/results`` como respaldo.

    Orden:
    1. ``RESULTS_DIR`` si está definida (cualquier ruta explícita).
    2. Si hay un punto de montaje de Drive activo (FUSE/rclone, p. ej. ``~/googledrive``):
       ``<mount>/Doctorado Erick Serrato/Resultados Paper``.

    Raises:
        RuntimeError: si no hay ``RESULTS_DIR`` ni Drive montado, o no se puede crear la ruta.
    """
    env = os.getenv("RESULTS_DIR")
    if env:
        try:
            p = Path(env).expanduser().resolve()
            p.mkdir(parents=True, exist_ok=True)
            return p
        except OSError as e:
            raise RuntimeError(f"RESULTS_DIR no accesible ({env}): {e}") from e

    mounted = _mounted_google_drive_mount_point()
    if mounted is None:
        raise RuntimeError(
            "No hay destino en la nube: monta Google Drive con "
            "`cd Allee && bash mount_google_drive.sh` (salida típica en "
            "~/googledrive/Doctorado Erick Serrato/Resultados Paper) "
            "o define RESULTS_DIR."
        )

    doctorado_dir = mounted / "Doctorado Erick Serrato"
    resultados_dir = doctorado_dir / "Resultados Paper"
    try:
        doctorado_dir.mkdir(parents=True, exist_ok=True)
        resultados_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"No se pudo preparar la carpeta en Drive montado: {e}") from e
    return resultados_dir.resolve()


def ensure_cloud_results_dir_ready() -> Path:
    """
    Como ``get_results_dir_cloud_only()`` pero comprueba montaje (anti ``~/googledrive`` vacío)
    y permisos de escritura.
    """
    results_dir = get_results_dir_cloud_only()
    explicit = os.getenv("RESULTS_DIR")
    drive_mp = get_google_drive_mount_point().resolve()
    try:
        resolved = results_dir.resolve()
        under_drive = resolved.as_posix().startswith(drive_mp.as_posix())
    except (OSError, ValueError):
        under_drive = False
    if not explicit and under_drive and not is_google_drive_mounted():
        raise RuntimeError(
            "La ruta cae bajo ~/googledrive pero el disco no está montado (no es FUSE). "
            "Ejecuta desde Allee: bash mount_google_drive.sh o define RESULTS_DIR."
        )
    ok, err = verify_results_dir_write_access(results_dir)
    if not ok:
        raise RuntimeError(f"No se puede escribir en {results_dir}: {err}")
    return results_dir



def ensure_steady_states_results_dir_ready() -> Path:
    """
    Carpeta para runs de steady_states/steady_states.py (JSON unificado):
    Resultados Paper o RESULTS_DIR / estados_estacionarios /

    Requiere Drive montado o RESULTS_DIR (misma logica que ensure_cloud_results_dir_ready).
    """
    base = ensure_cloud_results_dir_ready()
    sub = base / STEADY_STATES_RUNS_SUBDIR
    try:
        sub.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"No se pudo crear {sub}: {e}") from e
    return sub.resolve()


def get_steady_states_extract_dir() -> Optional[Path]:
    """
    Si ``RESULTS_DIR`` está definido, devuelve ``RESULTS_DIR/steady_states_extract/`` (creando directorios).
    Si no hay variable o falla la creación, ``None``.
    """
    env = os.getenv("RESULTS_DIR")
    if not env:
        return None
    try:
        root = Path(env).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        sub = root / STEADY_STATES_EXTRACT_SUBDIR
        sub.mkdir(parents=True, exist_ok=True)
        return sub
    except OSError as e:
        print(f"[!] No se pudo preparar {STEADY_STATES_EXTRACT_SUBDIR} bajo RESULTS_DIR: {e}")
        return None


def _candidate_google_drive_mount_points() -> list[Path]:
    """
    Devuelve una lista de puntos de montaje candidatos para Google Drive.

    Notas:
    - El proyecto históricamente usa ``~/googledrive`` (ver ``mount_google_drive.sh``: ``$HOME/googledrive``).
    - Si montaste como **root** en WSL, el punto real suele ser ``/root/googledrive``.
    - También se prueba ``/mnt/gdrive`` o ``GOOGLE_DRIVE_MOUNT_POINT``.
    """
    env_mp = os.getenv("GOOGLE_DRIVE_MOUNT_POINT") or os.getenv("GDRIVE_MOUNT_POINT")
    candidates: list[Path] = []
    if env_mp:
        candidates.append(Path(env_mp).expanduser())

    candidates.extend([
        Path.home() / "googledrive",
        # mount_google_drive.sh con usuario root en WSL → suele ser /root/googledrive
        Path("/root/googledrive"),
        Path("/mnt/gdrive"),
    ])

    # Dedupe preservando orden
    seen: set[str] = set()
    out: list[Path] = []
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out

def _mounted_google_drive_mount_point() -> Optional[Path]:
    """
    Retorna el punto de montaje detectado (si está montado), o None.

    Usamos `Path.is_mount()` para evitar falsos positivos cuando existe la carpeta
    pero Google Drive NO está montado (p.ej. directorio local vacío).
    """
    for p in _candidate_google_drive_mount_points():
        try:
            if p.exists() and p.is_mount():
                # Verificar que es accesible (un mount FUSE roto puede fallar al listar)
                try:
                    next(iter(p.iterdir()), None)
                    return p
                except (OSError, PermissionError):
                    continue
        except Exception:
            # Si el FS está en estado raro, preferimos tratarlo como "no montado"
            continue
    return None

def get_google_drive_mount_point() -> Path:
    """
    Retorna el punto de montaje de Google Drive.
    
    Returns:
        Path: Ruta preferida del punto de montaje de Google Drive (por defecto: ~/googledrive)
    """
    env_mp = os.getenv("GOOGLE_DRIVE_MOUNT_POINT") or os.getenv("GDRIVE_MOUNT_POINT")
    if env_mp:
        return Path(env_mp).expanduser()
    return Path.home() / "googledrive"

def is_google_drive_mounted() -> bool:
    """
    Verifica si Google Drive está montado y accesible.
    
    Returns:
        bool: True si Google Drive está montado y accesible, False en caso contrario
    """
    return _mounted_google_drive_mount_point() is not None

def get_results_dir(base_dir: Optional[Path] = None) -> Path:
    """
    Determina el directorio de resultados según configuración.
    
    Prioridad:
    1. Variable de entorno RESULTS_DIR (si existe y es accesible)
    2. Google Drive montado (~/googledrive/Doctorado Erick Serrato/Resultados Paper)
    3. Directorio local (base_dir/results)
    
    Args:
        base_dir: Directorio base para el fallback local (default: directorio del script)
    
    Returns:
        Path: Ruta al directorio de resultados
    """
    # Variable de entorno (prioridad máxima): crear la ruta si no existe — antes se exigía
    # exists() y si la carpeta aún no estaba en Drive/WSL, se ignoraba y se caía a local.
    env_results_dir = os.getenv('RESULTS_DIR')
    if env_results_dir:
        results_path = Path(env_results_dir).expanduser()
        try:
            results_path.mkdir(parents=True, exist_ok=True)
            return results_path.resolve()
        except OSError as e:
            print(f"[!] RESULTS_DIR no accesible ({env_results_dir}): {e}; usando predeterminado")
    
    # Verificar si Google Drive está montado
    mounted_drive = _mounted_google_drive_mount_point()
    if mounted_drive is not None:
        # Ruta en Google Drive: Doctorado Erick Serrato / Resultados Paper
        drive_results_dir = mounted_drive / "Doctorado Erick Serrato" / "Resultados Paper"
        
        try:
            # Verificar que la carpeta "Doctorado Erick Serrato" existe
            doctorado_dir = mounted_drive / "Doctorado Erick Serrato"
            if not doctorado_dir.exists():
                doctorado_dir.mkdir(parents=True, exist_ok=True)
            
            # Verificar que la carpeta "Resultados Paper" existe
            resultados_dir = doctorado_dir / "Resultados Paper"
            if not resultados_dir.exists():
                resultados_dir.mkdir(parents=True, exist_ok=True)
            
            # Crear el directorio de resultados si no existe
            drive_results_dir.mkdir(parents=True, exist_ok=True)
            
            # Verificar que se puede acceder
            if drive_results_dir.exists():
                return drive_results_dir
        except (OSError, PermissionError) as e:
            # No se pudo crear, usar fallback local
            print(f"[!] Error al acceder a Google Drive: {e}")
            print(f"  Usando directorio local como fallback")
            pass
    
    # Fallback al directorio local
    if base_dir is None:
        # Intentar determinar el directorio base desde el contexto
        # Si se llama desde un script, usar el directorio del script
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_file = frame.f_back.f_globals.get('__file__')
            if caller_file:
                base_dir = Path(caller_file).parent
            else:
                base_dir = Path.cwd()
        else:
            base_dir = Path.cwd()
    
    local_results_dir = base_dir / "results"
    return local_results_dir

def verify_results_dir_write_access(results_dir: Path) -> tuple[bool, Optional[str]]:
    """
    Verifica que se puede escribir en el directorio de resultados.
    
    Args:
        results_dir: Directorio a verificar
    
    Returns:
        tuple[bool, Optional[str]]: (éxito, mensaje_de_error)
    """
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        # Intentar escribir un archivo de prueba
        test_file = results_dir / ".test_write"
        test_file.write_text("test")
        test_file.unlink()
        return True, None
    except Exception as e:
        return False, str(e)

def get_scenario_dir(scenario_name: str, base_dir: Optional[Path] = None) -> Path:
    """
    Obtiene el directorio de un escenario específico.
    
    Args:
        scenario_name: Nombre del escenario
        base_dir: Directorio base (opcional)
    
    Returns:
        Path: Ruta al directorio del escenario
    """
    results_dir = get_results_dir(base_dir)
    return results_dir / scenario_name


def ensure_results_dir_ready(base_dir: Optional[Path] = None) -> Path:
    """
    Resuelve ``get_results_dir(base_dir)`` y comprueba escritura.

    Si la ruta queda bajo el punto de montaje de Drive configurado y el disco no está
    montado, lanza ``RuntimeError`` (misma convención que ``run_scenarios.py``).
    """
    results_dir = get_results_dir(base_dir)
    explicit = os.getenv("RESULTS_DIR")
    drive_mp = get_google_drive_mount_point().resolve()
    try:
        resolved = results_dir.resolve()
        under_drive = resolved.as_posix().startswith(drive_mp.as_posix())
    except (OSError, ValueError):
        under_drive = False
    # Si el usuario definió RESULTS_DIR, no exigimos is_mount (puede ser rclone, /mnt/g/, etc.)
    if not explicit and under_drive and not is_google_drive_mounted():
        raise RuntimeError(
            "La ruta de resultados apunta a Google Drive pero el disco no está montado. "
            "Ejecuta desde Allee: bash mount_google_drive.sh o define RESULTS_DIR local."
        )
    ok, err = verify_results_dir_write_access(results_dir)
    if not ok:
        raise RuntimeError(f"No se puede escribir en {results_dir}: {err}")
    return results_dir

