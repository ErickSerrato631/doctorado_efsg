"""
Resolución de rutas del laboratorio alineada con Models/Allee/utils_paths.py.

Prioridad (igual que los scripts):
1. Variable de entorno RESULTS_DIR si existe y el path es accesible
2. get_results_dir(ALLEE_DIR) si utils_paths está disponible
3. ALLEE_DIR / results

En Windows, si los resultados viven solo en WSL/Drive, define RESULTS_DIR apuntando a una
ruta visible para el proceso de Django (unidad montada o carpeta sincronizada).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from django.conf import settings


def get_allee_dir() -> Path:
    return Path(settings.ALLEE_DIR).resolve()


def get_lab_results_root() -> Path:
    """
    Directorio raíz donde run_scenarios crea <scenario_name>/matrices|images|...
    """
    env = (os.environ.get("RESULTS_DIR") or "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p

    allee = get_allee_dir()
    insert = str(allee)
    if insert not in sys.path:
        sys.path.insert(0, insert)
    try:
        from utils_paths import get_results_dir

        return Path(get_results_dir(allee)).resolve()
    except Exception:
        return (allee / "results").resolve()
