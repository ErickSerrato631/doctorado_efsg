#!/usr/bin/env python3
"""
Termodinamica (calculate_thermodynamic_properties) solo para la pareja de la presentacion:
  strong_mu1_uNo_bajo_umbral, strong_mu1_uSi_bajo_umbral

Fija RESULTS_DIR en el subproceso para que coincida con --results-dir si se pasa.

Ejemplo (cwd = Models/Allee):
  export RESULTS_DIR=/ruta/resultados
  python termodynamics/postprocess_presentacion_mu1_AB.py
  python termodynamics/postprocess_presentacion_mu1_AB.py --results-dir /otra/ruta
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

_ALLEE_ROOT = Path(__file__).resolve().parent.parent

SCENARIOS_PRESENTACION_MU1_AB = (
    "strong_mu1_uNo_bajo_umbral",
    "strong_mu1_uSi_bajo_umbral",
)


def _resolve_scenarios_file(allee_root: Path, override: Optional[Path]) -> Path:
    if override is not None:
        p = override.resolve()
        if not p.is_file():
            raise FileNotFoundError(f"No existe: {p}")
        return p
    for name in ("scenarios_v1.json", "scenarios.json"):
        p = (allee_root / name).resolve()
        if p.is_file():
            return p
    raise FileNotFoundError("No hay scenarios_v1.json ni scenarios.json en Allee/")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Propiedades termodinamicas (F, sigma, mu) solo escenarios mu=1 A/B."
    )
    parser.add_argument("--base-dir", type=Path, default=None, help="Raiz Models/Allee (default: auto)")
    parser.add_argument("--results-dir", type=Path, default=None, help="Directorio de resultados (RESULTS_DIR)")
    parser.add_argument("--scenarios-file", type=Path, default=None, help="JSON de escenarios")
    parser.add_argument("--block", type=int, default=1, help="Bloque (pasa a calculate_thermodynamic_properties)")
    parser.add_argument(
        "--fresh-thermo",
        action="store_true",
        help="Reenvia --fresh-thermo al calculador principal",
    )
    args = parser.parse_args()

    allee_root = args.base_dir.resolve() if args.base_dir else _ALLEE_ROOT
    sys.path.insert(0, str(allee_root))
    _rd = os.environ.get("RESULTS_DIR")
    if _rd and not Path(_rd).expanduser().exists():
        del os.environ["RESULTS_DIR"]
    from utils_paths import get_results_dir

    scenarios_path = _resolve_scenarios_file(allee_root, args.scenarios_file)
    results_dir = args.results_dir.resolve() if args.results_dir else get_results_dir(allee_root)
    results_dir.mkdir(parents=True, exist_ok=True)
    calc_script = Path(__file__).resolve().parent / "calculate_thermodynamic_properties.py"
    if not calc_script.is_file():
        print("ERROR: falta calculate_thermodynamic_properties.py")
        return 1

    env = dict(os.environ)
    env["RESULTS_DIR"] = str(results_dir)
    env.setdefault("PYTHONIOENCODING", "utf-8")

    cmd = [
        sys.executable,
        str(calc_script),
        "--scenarios",
        *SCENARIOS_PRESENTACION_MU1_AB,
        "--scenarios-file",
        str(scenarios_path),
        "--block",
        str(args.block),
    ]
    if args.fresh_thermo:
        cmd.append("--fresh-thermo")

    print(f"RESULTS_DIR     = {results_dir}")
    print(f"SCENARIOS_FILE  = {scenarios_path}")
    print(f"Escenarios      = {SCENARIOS_PRESENTACION_MU1_AB}")
    print("Ejecutando calculate_thermodynamic_properties.py ...")

    return subprocess.run(cmd, cwd=str(allee_root), env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
