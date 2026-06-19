"""
reciprocity_jacobian_analysis.py

Jacobiano del subsistema reaccional A_ab = ∂R_a / ∂φ_b (φ = c,s,i) y descomposición
S = (A + A^T)/2 (parte simétrica), N = (A - A^T)/2 (antisimétrica / no reciprocidad).

Usa build_reaction_equations_sympy de model_equations (misma ley que el resto del proyecto).

Ejemplos (desde el directorio Models/Allee):

  python nonequilibrium_termodynamics/reciprocity_jacobian_analysis.py --scenarios scenarios.json \\
      --scenario strong_corner_mu0_hillY_c0_s1_i0 --point 0.3 0.4 0.2

  python nonequilibrium_termodynamics/reciprocity_jacobian_analysis.py --scenarios scenarios.json \\
      --all-scenarios --point 0.5 0.5 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sympy as sp

_ALLEE_ROOT = Path(__file__).resolve().parent.parent
if str(_ALLEE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALLEE_ROOT))

from model_equations import build_reaction_equations_sympy, c_sym, s_sym, i_sym  # noqa: E402
from model_parameters import ModelParameters, load_from_scenarios_json  # noqa: E402


def reaction_jacobian_symbolic(params: ModelParameters) -> sp.Matrix:
    Rc, Rs, Ri = build_reaction_equations_sympy(params)
    F = sp.Matrix([Rc, Rs, Ri])
    return F.jacobian([c_sym, s_sym, i_sym])


def decompose_SN(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    S = 0.5 * (A + A.T)
    N = 0.5 * (A - A.T)
    return S, N


def evaluate_jacobian(
    J_sym: sp.Matrix,
    c_val: float,
    s_val: float,
    i_val: float,
) -> np.ndarray:
    subs: Dict[sp.Symbol, float] = {c_sym: c_val, s_sym: s_val, i_sym: i_val}
    out = np.zeros((3, 3), dtype=float)
    for a in range(3):
        for b in range(3):
            out[a, b] = float(J_sym[a, b].subs(subs).evalf())
    return out


def reciprocity_metrics(A: np.ndarray) -> Dict[str, float]:
    _, N = decompose_SN(A)
    n_f = np.linalg.norm(N, ord="fro")
    a_f = np.linalg.norm(A, ord="fro")
    return {
        "norm_A_fro": float(a_f),
        "norm_N_fro": float(n_f),
        "norm_S_fro": float(np.linalg.norm(0.5 * (A + A.T), ord="fro")),
        "ratio_N_over_A": float(n_f / (a_f + 1e-30)),
    }


def list_scenario_names(scenarios_file: Path) -> List[str]:
    with open(scenarios_file, encoding="utf-8") as f:
        data = json.load(f)
    return [s["name"] for s in data.get("scenarios", [])]


def run_cli() -> None:
    p = argparse.ArgumentParser(description="Jacobiano reaccional A=S+N (reciprocidad).")
    p.add_argument("--scenarios", type=Path, required=True, help="Ruta a scenarios.json")
    p.add_argument("--scenario", type=str, default=None, help="Nombre de escenario (opcional)")
    p.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Evaluar todos los escenarios del archivo en el mismo punto",
    )
    p.add_argument(
        "--point",
        type=float,
        nargs=3,
        metavar=("C", "S", "I"),
        default=[0.5, 0.5, 0.5],
        help="Punto (c, s, i) en [0,1]^3",
    )
    p.add_argument("--json", action="store_true", help="Salida en una línea JSON por escenario")
    args = p.parse_args()

    c0, s0, i0 = args.point

    if args.all_scenarios:
        names = list_scenario_names(args.scenarios)
        rows: List[Dict[str, Any]] = []
        for name in names:
            params = load_from_scenarios_json(args.scenarios, scenario_name=name)
            J_sym = reaction_jacobian_symbolic(params)
            A = evaluate_jacobian(J_sym, c0, s0, i0)
            m = reciprocity_metrics(A)
            row = {"scenario": name, "c": c0, "s": s0, "i": i0, **m}
            rows.append(row)
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            print(f"Punto (c,s,i)=({c0}, {s0}, {i0})\n")
            for row in rows:
                print(
                    f"{row['scenario']}: ||N||_F={row['norm_N_fro']:.6g}  "
                    f"||A||_F={row['norm_A_fro']:.6g}  "
                    f"||N||/||A||={row['ratio_N_over_A']:.6g}"
                )
        return

    params = load_from_scenarios_json(args.scenarios, scenario_name=args.scenario)
    J_sym = reaction_jacobian_symbolic(params)
    A = evaluate_jacobian(J_sym, c0, s0, i0)
    S, N = decompose_SN(A)
    m = reciprocity_metrics(A)

    if args.json:
        print(
            json.dumps(
                {
                    "scenario": args.scenario,
                    "point": [c0, s0, i0],
                    **m,
                    "A": A.tolist(),
                    "S": S.tolist(),
                    "N": N.tolist(),
                },
                indent=2,
            )
        )
    else:
        label = args.scenario or "(solo common_params)"
        print(f"Escenario: {label}")
        print(f"Punto (c,s,i)=({c0}, {s0}, {i0})\n")
        print("A (Jacobiano reaccional):")
        print(A)
        print("\nS (simétrica):")
        print(S)
        print("\nN (antisimétrica):")
        print(N)
        print(
            f"\n||A||_F={m['norm_A_fro']:.6g}  ||N||_F={m['norm_N_fro']:.6g}  "
            f"||N||/||A||={m['ratio_N_over_A']:.6g}"
        )


if __name__ == "__main__":
    run_cli()
