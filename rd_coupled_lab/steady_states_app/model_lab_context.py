"""
Contexto SymPy + catálogo de escenarios (ecuaciones y diccionario de parámetros).
Usado por el catálogo de escenarios (modal / página de detalle).
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from django.conf import settings

ALLEE_ROOT = Path(settings.ALLEE_DIR)


def build_model_equations_context(
    scenarios_file: Path,
    scenario_name: Optional[str],
    scenarios_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Carga parámetros desde scenarios_file y construye LaTeX de R_c, R_s, R_i.

    scenario_name: None → solo common_params.
    Si ``scenarios_data`` es el dict ya normalizado (common_params + scenarios), no se vuelve a abrir el archivo.

    Devuelve dict con latex_*, params_table, etc.; model_load_error si falla import/sympy.
    flash_errors: mensajes para request (p. ej. escenario inválido).
    """
    ctx: Dict[str, Any] = {
        "sympy_available": False,
        "model_load_error": None,
        "latex_Rc": None,
        "latex_Rs": None,
        "latex_Ri": None,
        "params_table": None,
        "params_source": None,
        "allee_type": None,
        "use_adaptive_control": None,
        "flash_errors": [],
        "selected_scenario_effective": (scenario_name or "").strip(),
    }

    original_cwd = os.getcwd()
    try:
        os.chdir(str(ALLEE_ROOT))
        import sympy as sp
        from model_parameters import load_from_scenarios_json, ModelParameters
        from model_equations import build_reaction_equations_sympy

        ctx["sympy_available"] = True

        data_kw: Dict[str, Any] = {}
        if scenarios_data is not None:
            data_kw["scenarios_data"] = scenarios_data

        if scenarios_data is not None or scenarios_file.exists():
            try:
                name = (scenario_name or "").strip()
                if name:
                    params = load_from_scenarios_json(
                        scenarios_file,
                        scenario_name=name,
                        load_spatial_params=True,
                        **data_kw,
                    )
                    ctx["params_source"] = f"Escenario: {name}"
                else:
                    params = load_from_scenarios_json(
                        scenarios_file,
                        scenario_name=None,
                        load_spatial_params=True,
                        **data_kw,
                    )
                    ctx["params_source"] = "Parámetros comunes (common_params)"
            except ValueError as e:
                ctx["flash_errors"].append(str(e))
                params = load_from_scenarios_json(
                    scenarios_file,
                    scenario_name=None,
                    load_spatial_params=True,
                    **data_kw,
                )
                ctx["params_source"] = "Parámetros comunes (escenario no válido)"
                ctx["selected_scenario_effective"] = ""
        else:
            params = ModelParameters(
                rc=5.84,
                rs=13.12,
                rd=10.92,
                alpha=10.22,
                delta=5.4,
                beta=7.6,
                a=0.1,
                gamma=0.74,
                eta=5.08,
                mu=1.0,
                allee_type="STRONG",
                use_adaptive_control=False,
                control_uses_hill=False,
            )
            ctx["params_source"] = "Valores por defecto (sin catálogo de escenarios accesible)"

        R_c, R_s, R_i = build_reaction_equations_sympy(params)
        ctx["latex_Rc"] = sp.latex(R_c)
        ctx["latex_Rs"] = sp.latex(R_s)
        ctx["latex_Ri"] = sp.latex(R_i)

        flat = params.to_dict(include_spatial=True)
        ctx["params_table"] = sorted(flat.items(), key=lambda x: x[0].lower())
        ctx["allee_type"] = params.allee_type
        ctx["use_adaptive_control"] = params.use_adaptive_control
    except ImportError as e:
        ctx["model_load_error"] = (
            f"No se pudieron importar model_equations / SymPy desde {ALLEE_ROOT}: {e}"
        )
    except Exception as e:
        ctx["model_load_error"] = f"{e}\n{traceback.format_exc()}"
    finally:
        try:
            os.chdir(original_cwd)
        except OSError:
            pass

    return ctx
