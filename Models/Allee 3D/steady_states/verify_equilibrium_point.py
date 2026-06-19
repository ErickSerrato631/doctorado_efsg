"""
Verifica si un punto (c, s, i) es estado estacionario del kinetics 3D
(Strong/Weak Allee, Hill on/off) evaluando ||F(c,s,i)||.

Uso (desde el directorio Allee):
  python steady_states/verify_equilibrium_point.py
  python steady_states/verify_equilibrium_point.py --mu 0 --c 2.14e-18 --s 1 --i 6.4e-20
  python steady_states/verify_equilibrium_point.py --no-hill --mu 1
  python steady_states/verify_equilibrium_point.py --newton-seed 0 0.99 1e-6
  python steady_states/verify_equilibrium_point.py --simplex-corners --allee STRONG --mu 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ALLEE = Path(__file__).resolve().parent.parent
if str(_ALLEE) not in sys.path:
    sys.path.insert(0, str(_ALLEE))

from steady_states.steady_states import build_numeric_3d, newton_root_3d, mu, umax


def main() -> None:
    p = argparse.ArgumentParser(description="Residuo F=0 en un punto (c,s,i)")
    p.add_argument("--mu", type=float, default=1.0)
    p.add_argument(
        "--no-hill",
        action="store_true",
        help="Sin término Hill en F_i (por defecto sí hay Hill con --umax)",
    )
    p.add_argument("--allee", choices=("STRONG", "WEAK"), default="STRONG")
    p.add_argument("--umax", type=float, default=0.5)
    p.add_argument("--c", type=float, default=2.142060090027267e-18)
    p.add_argument("--s", type=float, default=1.0)
    p.add_argument("--i", type=float, default=6.395843214321265e-20)
    p.add_argument(
        "--newton-seed",
        type=float,
        nargs=3,
        metavar=("C0", "S0", "I0"),
        help="Si se pasa, ejecuta Newton desde esta semilla y muestra ||F|| en la raíz",
    )
    p.add_argument(
        "--simplex-corners",
        action="store_true",
        help="Evalúa ||F|| en (1,0,0), (0,1,0) y (0,0,1) con los mismos --allee, --mu, --no-hill, --umax",
    )
    args = p.parse_args()

    include_hill = not args.no_hill
    override = {mu: args.mu, umax: args.umax} if include_hill else {mu: args.mu}
    f, Jsym, _ = build_numeric_3d(
        override,
        allee_type=args.allee,
        include_hill_control=include_hill,
    )

    if args.simplex_corners:
        print(f"allee={args.allee}  mu={args.mu}  hill={include_hill}  umax={args.umax if include_hill else 'N/A'}")
        for label, triple in (
            ("(1,0,0)", (1.0, 0.0, 0.0)),
            ("(0,1,0)", (0.0, 1.0, 0.0)),
            ("(0,0,1)", (0.0, 0.0, 1.0)),
        ):
            Fv = np.array(f(*triple), dtype=float).ravel()
            n = float(np.linalg.norm(Fv))
            print(f"{label}  ||F|| = {n:.6e}  F = {Fv}")
        if args.newton_seed is not None:
            r = newton_root_3d(f, Jsym, tuple(args.newton_seed))
            print(f"\nNewton desde {tuple(args.newton_seed)} -> {r}")
            if r is not None:
                Fv2 = np.array(f(*r), dtype=float).ravel()
                print(f"||F|| en raíz = {np.linalg.norm(Fv2):.6e}  F = {Fv2}")
        return

    Fv = np.array(f(args.c, args.s, args.i), dtype=float).ravel()
    n = float(np.linalg.norm(Fv))
    print(f"allee={args.allee}  mu={args.mu}  hill={include_hill}  umax={args.umax if include_hill else 'N/A'}")
    print(f"punto (c,s,i) = ({args.c}, {args.s}, {args.i})")
    print(f"||F|| = {n:.6e}")
    print(f"F     = {Fv}")

    if args.newton_seed is not None:
        r = newton_root_3d(f, Jsym, tuple(args.newton_seed))
        print(f"\nNewton desde {tuple(args.newton_seed)} -> {r}")
        if r is not None:
            Fv2 = np.array(f(*r), dtype=float).ravel()
            print(f"||F|| en raíz = {np.linalg.norm(Fv2):.6e}  F = {Fv2}")


if __name__ == "__main__":
    main()
