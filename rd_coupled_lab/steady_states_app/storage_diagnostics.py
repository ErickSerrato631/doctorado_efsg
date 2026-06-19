"""
Comprobaciones de rutas de almacenamiento (RESULTS_DIR, Drive montado, catálogo JSON de escenarios).
Solo lectura + prueba de escritura opcional en la raíz de resultados.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from django.conf import settings

from .lab_paths import get_allee_dir, get_lab_results_root


def _probe_dir(path: Path, try_write: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "is_dir": False,
        "readable": None,
        "read_error": None,
        "sample_names": None,
        "write_ok": None,
        "write_error": None,
    }
    if not path.exists():
        return out
    out["is_dir"] = path.is_dir()
    if not path.is_dir():
        return out
    try:
        sample = [p.name for p in list(path.iterdir())[:8]]
        out["readable"] = True
        out["sample_names"] = sample
    except OSError as e:
        out["readable"] = False
        out["read_error"] = str(e)
    if try_write and out.get("readable"):
        probe = path / f".lab_storage_probe_{os.getpid()}.tmp"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            out["write_ok"] = True
        except OSError as e:
            out["write_ok"] = False
            out["write_error"] = str(e)
    return out


def _probe_file(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file() if path.exists() else False,
        "readable": None,
        "read_error": None,
        "size": None,
    }
    if not path.is_file():
        return out
    try:
        out["size"] = path.stat().st_size
        with open(path, "rb") as f:
            f.read(1)
        out["readable"] = True
    except OSError as e:
        out["readable"] = False
        out["read_error"] = str(e)
    return out


def collect_storage_diagnostics() -> Dict[str, Any]:
    env_raw = (os.environ.get("RESULTS_DIR") or "").strip()
    results_root = get_lab_results_root()
    allee = get_allee_dir()
    scenarios = Path(settings.SCENARIOS_JSON_PATH)

    env_path = Path(env_raw).expanduser() if env_raw else None

    return {
        "env_results_dir_raw": env_raw or None,
        "env_path_resolved": str(env_path.resolve()) if env_path else None,
        "results_root_probe": _probe_dir(results_root, try_write=True),
        "allee_dir_probe": _probe_dir(allee, try_write=False),
        "scenarios_json_probe": _probe_file(scenarios),
        "results_root_resolved": str(results_root),
        "allee_dir_resolved": str(allee),
    }
