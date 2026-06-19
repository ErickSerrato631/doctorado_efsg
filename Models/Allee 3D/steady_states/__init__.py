"""
Paquete de estados estacionarios: núcleo simbólico/numérico, extracción desde
scenarios.json y generación de figuras (planos de fase y espectros lineales).

No importar steady_states.steady_states al cargar el paquete: evita exigir sympy
solo por ``import steady_states.generate_phase_planes`` u otros submódulos.
"""

import sys
from importlib import import_module
from typing import Any

_SS = "steady_states.steady_states"


def __getattr__(name: str) -> Any:
    sub_key = f"{__name__}.{name}"
    if sub_key in sys.modules:
        return sys.modules[sub_key]
    core = import_module(_SS)
    if hasattr(core, name):
        return getattr(core, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
