#!/usr/bin/env python3
"""
Correlaciones (correlation_fourier) solo para la pareja de la presentacion:
  strong_mu1_uNo_bajo_umbral, strong_mu1_uSi_bajo_umbral

Usa common_params del JSON de escenarios para T, dt, nb, malla, etc. (como run_scenarios).

Ejemplo (cwd = Models/Allee):
  export RESULTS_DIR=/ruta/resultados
  python correlations/postprocess_presentacion_mu1_AB.py
  python correlations/postprocess_presentacion_mu1_AB.py --results-dir /otra/ruta

Si correlations/corr_length_real_inverse_nb_* ya existen (salida de correlation_fourier),
no se vuelve a ejecutar salvo que pases --force.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_ALLEE_ROOT = Path(__file__).resolve().parent.parent

SCENARIOS_PRESENTACION_MU1_AB = (
    "strong_mu1_uNo_bajo_umbral",
    "strong_mu1_uSi_bajo_umbral",
)

_CORR_PAIRS = (("c", "c"), ("c", "i"), ("i", "i"))


def _correlation_txt_outputs_ok(scen_dir: Path, block: int) -> bool:
    corr = scen_dir / "correlations"
    for a, b in _CORR_PAIRS:
        fp = corr / f"corr_length_real_inverse_nb_{block}_{a}_{b}.txt"
        if not fp.is_file():
            return False
    return True


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


def _env_from_common(common: Dict[str, Any]) -> Dict[str, str]:
    env = dict(os.environ)
    for k, v in common.items():
        env[str(k)] = str(v)
    return env


def main() -> int:
    parser = argparse.ArgumentParser(
        description="correlation_fourier solo para los dos escenarios mu=1 A/B (presentacion)."
    )
    parser.add_argument("--base-dir", type=Path, default=None, help="Raiz Models/Allee (default: auto)")
    parser.add_argument("--results-dir", type=Path, default=None, help="Sobrescribe RESULTS_DIR / utils_paths")
    parser.add_argument("--scenarios-file", type=Path, default=None, help="JSON de escenarios (default: scenarios_v1.json)")
    parser.add_argument("--block", type=int, default=1, help="nb en nombres corr_length_real_inverse_nb_*")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ejecutar correlation_fourier aunque ya existan los .txt de longitudes",
    )
    args = parser.parse_args()

    allee_root = args.base_dir.resolve() if args.base_dir else _ALLEE_ROOT
    sys.path.insert(0, str(allee_root))
    # RESULTS_DIR apuntando a ruta inexistente rompe get_results_dir (prints unicode en Windows).
    _rd = os.environ.get("RESULTS_DIR")
    if _rd and not Path(_rd).expanduser().exists():
        del os.environ["RESULTS_DIR"]
    from utils_paths import get_results_dir

    scenarios_path = _resolve_scenarios_file(allee_root, args.scenarios_file)
    with open(scenarios_path, encoding="utf-8") as f:
        data = json.load(f)
    common = data["common_params"]
    env = _env_from_common(common)
    env.setdefault("PYTHONIOENCODING", "utf-8")

    results_dir = args.results_dir.resolve() if args.results_dir else get_results_dir(allee_root)
    results_dir.mkdir(parents=True, exist_ok=True)
    script = allee_root / "correlations" / "correlation_fourier.py"
    if not script.is_file():
        print("ERROR: falta correlations/correlation_fourier.py")
        return 1

    print(f"RESULTS_DIR     = {results_dir}")
    print(f"SCENARIOS_FILE  = {scenarios_path}")
    print(f"Escenarios      = {SCENARIOS_PRESENTACION_MU1_AB}")
    print(
        f"Parametros correlacion (common_params -> env): T={common.get('T')} "
        f"dt={common.get('dt')} nb={args.block}"
    )

    rc = 0
    block = args.block
    for name in SCENARIOS_PRESENTACION_MU1_AB:
        scen_dir = (results_dir / name).resolve()
        has_matrices = (scen_dir / "matrices").is_dir()
        if not args.force and _correlation_txt_outputs_ok(scen_dir, block):
            print(f"[omitido] {name}: ya existen correlations/corr_length_real_inverse_nb_{block}_* (use --force para recalcular)")
            continue
        if not has_matrices:
            print(f"[salto] {name}: sin matrices/ y correlaciones incompletas -> {scen_dir / 'matrices'}")
            continue
        print(f"== correlation_fourier: {name} ==")
        print(f"  subprocess env: T={env.get('T')} dt={env.get('dt')} nb={block}")
        r = subprocess.run(
            [sys.executable, str(script), str(scen_dir)],
            cwd=str(scen_dir),
            env=env,
        )
        if r.returncode != 0:
            rc = r.returncode
    return rc


if __name__ == "__main__":
    sys.exit(main())
