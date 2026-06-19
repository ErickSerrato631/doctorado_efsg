"""
fenics_nonequilibrium.py

Postproceso en FEniCSx: corrientes difusivas J_a = -D_a ∇φ_a, densidad
σ_loc = Σ_a J_a · ∇μ_a y total Σ = ∫ σ_loc dx.

Convenciones (documentar en publicaciones):
- "field": μ_a = φ_a  →  σ_loc = -Σ D_a ||∇φ_a||² (≤ 0).
- "reaction_derivative": μ_a = ∂R_a/∂φ_a (UFL)  →  signo depende del estado.
- "positive_proxy": σ⁺ = Σ D_a ||∇φ_a||² (coincide con la idea de disipación por
  gradientes usada en termodynamics/calculate_thermodynamic_properties.py cuando μ espacial es suave).

Requiere: ufl, dolfinx (y que params tenga D_c, D_s, D_i para flujos).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal, Optional, Tuple

_ALLEE_ROOT = Path(__file__).resolve().parent.parent
if str(_ALLEE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALLEE_ROOT))

try:
    import ufl
except ImportError:
    ufl = None  # type: ignore

try:
    from dolfinx import fem
    from dolfinx.fem import assemble_scalar, form, Function
    from dolfinx.fem.petsc import LinearProblem
    _DOLFINX = True
except ImportError:
    fem = None  # type: ignore
    assemble_scalar = None  # type: ignore
    form = None  # type: ignore
    Function = Any  # type: ignore
    LinearProblem = None  # type: ignore
    _DOLFINX = False

from model_equations import build_reaction_rates_ufl  # noqa: E402
from model_parameters import ModelParameters  # noqa: E402

MuConvention = Literal["field", "reaction_derivative", "positive_proxy"]


def _require_ufl_dolfinx() -> None:
    if ufl is None:
        raise ImportError("ufl no está instalado.")
    if (
        not _DOLFINX
        or assemble_scalar is None
        or form is None
        or fem is None
    ):
        raise ImportError("dolfinx no está instalado o es incompatible.")


def diffusive_fluxes(
    c: Any,
    s: Any,
    i: Any,
    params: ModelParameters,
) -> Tuple[Any, Any, Any]:
    if params.D_c is None or params.D_s is None or params.D_i is None:
        raise ValueError("params debe definir D_c, D_s, D_i para los flujos difusivos.")
    Jc = -params.D_c * ufl.grad(c)
    Js = -params.D_s * ufl.grad(s)
    Ji = -params.D_i * ufl.grad(i)
    return Jc, Js, Ji


def chemical_potential_expressions(
    c: Any,
    s: Any,
    i: Any,
    params: ModelParameters,
    convention: MuConvention,
) -> Tuple[Any, Any, Any]:
    if convention == "field":
        return c, s, i
    if convention == "positive_proxy":
        return c, s, i
    if convention == "reaction_derivative":
        Rc, Rs, Ri = build_reaction_rates_ufl(c, s, i, params)
        mu_c = ufl.diff(Rc, c)
        mu_s = ufl.diff(Rs, s)
        mu_i = ufl.diff(Ri, i)
        return mu_c, mu_s, mu_i
    raise ValueError(f"convention desconocida: {convention}")


def entropy_production_density_ufl(
    c: Any,
    s: Any,
    i: Any,
    params: ModelParameters,
    convention: MuConvention = "field",
) -> Any:
    """
    Densidad escalar σ_loc en cada punto (UFL).

    convention="positive_proxy" devuelve Σ D_a ||∇φ_a||² (siempre ≥ 0), no J·∇μ.
    """
    _require_ufl_dolfinx()
    Jc, Js, Ji = diffusive_fluxes(c, s, i, params)
    if convention == "positive_proxy":
        return (
            params.D_c * ufl.inner(ufl.grad(c), ufl.grad(c))
            + params.D_s * ufl.inner(ufl.grad(s), ufl.grad(s))
            + params.D_i * ufl.inner(ufl.grad(i), ufl.grad(i))
        )
    mu_c, mu_s, mu_i = chemical_potential_expressions(c, s, i, params, convention)
    gc, gs, gi = ufl.grad(mu_c), ufl.grad(mu_s), ufl.grad(mu_i)
    return ufl.inner(Jc, gc) + ufl.inner(Js, gs) + ufl.inner(Ji, gi)


def assemble_total_entropy_production(
    sigma_loc: Any,
    dx: Optional[Any] = None,
) -> float:
    """
    Σ = ∫ σ_loc dx sobre el dominio (suma MPI si aplica).
    """
    _require_ufl_dolfinx()
    measure = dx if dx is not None else ufl.dx
    return float(assemble_scalar(form(sigma_loc * measure)))


def total_entropy_production_from_functions(
    c: Any,
    s: Any,
    i: Any,
    params: ModelParameters,
    convention: MuConvention = "field",
    dx: Optional[Any] = None,
) -> float:
    """
    API cómoda: tres dolfinx.fem.Function en el mismo espacio (o mixto si los
    grad está definido de forma consistente).
    """
    sigma_loc = entropy_production_density_ufl(c, s, i, params, convention=convention)
    return assemble_total_entropy_production(sigma_loc, dx=dx)


def project_scalar_l2(expr: Any, V0: Any, dx: Optional[Any] = None) -> Any:
    """
    Proyección L² de una expresión UFL escalar sobre el espacio V0 (p. ej. DG0 para σ_loc).
    """
    _require_ufl_dolfinx()
    if LinearProblem is None or fem is None:
        raise ImportError("LinearProblem / dolfinx.fem no disponibles.")
    measure = dx if dx is not None else ufl.dx
    w = ufl.TrialFunction(V0)
    v = ufl.TestFunction(V0)
    a = ufl.inner(w, v) * measure
    L = ufl.inner(expr, v) * measure
    problem = LinearProblem(a, L, bcs=[])
    return problem.solve()


def project_entropy_density(
    c: Any,
    s: Any,
    i: Any,
    params: ModelParameters,
    mesh: Any,
    degree: int = 0,
    convention: MuConvention = "field",
    dx: Optional[Any] = None,
) -> Any:
    """
    Devuelve una Function con la densidad σ_loc(x) proyectada en DG(degree).
    """
    _require_ufl_dolfinx()
    V0 = fem.functionspace(mesh, ("DG", degree))
    sigma_loc = entropy_production_density_ufl(c, s, i, params, convention=convention)
    return project_scalar_l2(sigma_loc, V0, dx=dx)
