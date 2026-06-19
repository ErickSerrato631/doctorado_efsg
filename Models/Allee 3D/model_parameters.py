"""
model_parameters.py

Centraliza la carga y manejo de parámetros del modelo desde scenarios.json.
Proporciona una interfaz unificada para acceder a parámetros tanto para
análisis de estados estacionarios como para simulación espacial.

La fuente de verdad es scenarios.json, que puede ser generado por steady_states/steady_states.py
basado en análisis de estados estacionarios.
"""

import json
from pathlib import Path
from typing import Any, Optional, Dict, List, Literal
from dataclasses import dataclass, field


@dataclass
class ModelParameters:
    """
    Contenedor para todos los parámetros del modelo.
    
    Atributos principales:
    - Parámetros de reacción: rc, rs, rd, alpha, delta, beta, a, gamma, eta, mu
    - Parámetros de difusión: D_c, D_s, D_i (opcionales, solo para simulación espacial)
    - Tipo de Allee: WEAK o STRONG
    - Control adaptativo: use_adaptive_control, ku, eps_u, u_max
    """
    # Parámetros de reacción (requeridos)
    rc: float
    rs: float
    rd: float
    alpha: float
    delta: float
    beta: float
    a: float  # Parámetro de Allee (se llama 'alle' en scenarios.json)
    gamma: float
    eta: float
    mu: float
    
    # Parámetros de difusión (opcionales, solo para simulación espacial)
    D_c: Optional[float] = None
    D_s: Optional[float] = None
    D_i: Optional[float] = None
    
    # Tipo de efecto Allee
    allee_type: Literal['WEAK', 'STRONG'] = 'WEAK'
    
    # Control adaptativo
    use_adaptive_control: bool = False
    # Si True y use_adaptive_control: ley Hill en R_i; si False pero use_adaptive_control: ku·c/(i+ε) (cap u_max)
    control_uses_hill: bool = False
    ku: float = 0.2
    eps_u: float = 1e-3
    u_max: Optional[float] = None

    # Control tipo Hill (Opción A): u = u_max * H_act(c; Kc,nc) * H_inh(i; Ki,ni)
    hill_kc: float = 0.05
    hill_nc: float = 2.0
    hill_ki: float = 0.2
    hill_ni: float = 2.0
    
    # Parámetros de simulación espacial (opcionales)
    dt: Optional[float] = None
    T: Optional[float] = None
    nodes_in_xaxis: Optional[int] = None
    nodes_in_yaxis: Optional[int] = None
    nodes_in_zaxis: Optional[int] = None
    space_size: Optional[float] = None
    space_size_z: Optional[float] = None
    spatial_dim: int = 3
    
    # Condiciones iniciales (opcionales)
    c_init_min: Optional[float] = None
    c_init_max: Optional[float] = None
    s_init_min: Optional[float] = None
    s_init_max: Optional[float] = None
    i_init_min: Optional[float] = None
    i_init_max: Optional[float] = None
    
    def to_dict(self, include_spatial: bool = False) -> Dict:
        """
        Convierte a diccionario compatible con scenarios.json.
        
        Args:
            include_spatial: Si True, incluye parámetros espaciales
            
        Returns:
            Diccionario con formato de scenarios.json
        """
        result = {
            'rc': str(self.rc),
            'rs': str(self.rs),
            'rd': str(self.rd),
            'alpha': str(self.alpha),
            'delta': str(self.delta),
            'beta': str(self.beta),
            'alle': str(self.a),  # 'alle' en JSON, 'a' en código
            'gamma': str(self.gamma),
            'eta': str(self.eta),
            'mu': str(self.mu),
            'ALLEE_TYPE': self.allee_type,
            'USE_ADAPTIVE_CONTROL': 'Y' if self.use_adaptive_control else 'N',
        }
        
        if self.use_adaptive_control:
            result['HILL_CONTROL'] = 'Y' if self.control_uses_hill else 'N'
            result['KU'] = str(self.ku)
            result['EPS_U'] = str(self.eps_u)
            if self.u_max is not None:
                result['U_MAX'] = str(self.u_max)
            # Parámetros Hill (sólo aplican si control_uses_hill)
            result['HILL_KC'] = str(self.hill_kc)
            result['HILL_NC'] = str(self.hill_nc)
            result['HILL_KI'] = str(self.hill_ki)
            result['HILL_NI'] = str(self.hill_ni)
        
        if include_spatial:
            if self.D_c is not None:
                result['D_c'] = str(self.D_c)
            if self.D_s is not None:
                result['D_s'] = str(self.D_s)
            if self.D_i is not None:
                result['D_i'] = str(self.D_i)
            if self.dt is not None:
                result['dt'] = str(self.dt)
            if self.T is not None:
                result['T'] = str(self.T)
            if self.nodes_in_xaxis is not None:
                result['nodes_in_xaxis'] = str(self.nodes_in_xaxis)
            if self.nodes_in_yaxis is not None:
                result['nodes_in_yaxis'] = str(self.nodes_in_yaxis)
            if self.nodes_in_zaxis is not None:
                result['nodes_in_zaxis'] = str(self.nodes_in_zaxis)
            if self.space_size is not None:
                result['space_size'] = str(self.space_size)
            if self.space_size_z is not None:
                result['space_size_z'] = str(self.space_size_z)
            result['SPATIAL_DIM'] = str(self.spatial_dim)
            
            # Condiciones iniciales
            if self.c_init_min is not None:
                result['C_INIT_MIN'] = str(self.c_init_min)
            if self.c_init_max is not None:
                result['C_INIT_MAX'] = str(self.c_init_max)
            if self.s_init_min is not None:
                result['S_INIT_MIN'] = str(self.s_init_min)
            if self.s_init_max is not None:
                result['S_INIT_MAX'] = str(self.s_init_max)
            if self.i_init_min is not None:
                result['I_INIT_MIN'] = str(self.i_init_min)
            if self.i_init_max is not None:
                result['I_INIT_MAX'] = str(self.i_init_max)
        
        return result


# ============================================================================
# Carga desde scenarios.json
# ============================================================================

def _pick_steady_state_for_scenario_name(name: str, ss_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Elige steady state coherente con sufijos ``…_c0_s1_i0`` / ``…_c0_s1_i1`` del nombre."""
    if not ss_list:
        return None
    n = str(name or "")
    if n.endswith("_c0_s1_i0_c0_s1_i1"):
        for ss in ss_list:
            if isinstance(ss, dict) and ss.get("target_branch") == "c0_s1_i0":
                return ss
        return ss_list[0] if isinstance(ss_list[0], dict) else None
    if n.endswith("_c0_s1_i1"):
        for ss in ss_list:
            if isinstance(ss, dict) and ss.get("target_branch") == "c0_s1_i1":
                return ss
        return ss_list[-1] if len(ss_list) > 1 else ss_list[0]
    if n.endswith("_c0_s1_i0"):
        for ss in ss_list:
            if isinstance(ss, dict) and ss.get("target_branch") == "c0_s1_i0":
                return ss
        return ss_list[0] if isinstance(ss_list[0], dict) else None
    return ss_list[0] if isinstance(ss_list[0], dict) else None


def _promote_steady_state_into_params(
    params: Dict[str, Any],
    scenario_name: str,
    scenario_block: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Promueve física desde ``steady_states[]`` (control, μ, cinética) cuando el bloque
    del escenario no la fija a nivel plano. ``common_params`` (p. ej. USE_ADAPTIVE_CONTROL=N)
    no debe impedir leer ``use_adaptive_control: true`` en uSi.
    """
    out = dict(params)
    raw = out.get("steady_states")
    if not isinstance(raw, list) or not raw:
        return out
    ss_list = [x for x in raw if isinstance(x, dict)]
    if not ss_list:
        return out
    ss = _pick_steady_state_for_scenario_name(scenario_name, ss_list)
    if ss is None:
        return out

    block = scenario_block if isinstance(scenario_block, dict) else {}

    if "mu" not in block and ss.get("mu") is not None:
        out["mu"] = str(ss["mu"])
    if "ALLEE_TYPE" not in block and ss.get("allee_type"):
        out["ALLEE_TYPE"] = str(ss["allee_type"]).upper()
    for key in ("rc", "rs", "rd", "alpha", "beta", "delta", "eta", "gamma", "a"):
        if key not in block and ss.get(key) is not None:
            out[key] = str(ss[key])

    explicit_u = str(block.get("USE_ADAPTIVE_CONTROL", "")).strip() != ""
    if explicit_u:
        return out

    use_adaptive = bool(ss.get("use_adaptive_control"))
    hill = bool(ss.get("hill_control"))

    if hill and use_adaptive:
        out["USE_ADAPTIVE_CONTROL"] = "Y"
        out["HILL_CONTROL"] = "Y"
        u_h = ss.get("umax")
        if u_h is not None:
            out["U_MAX"] = str(u_h)
    elif use_adaptive:
        out["USE_ADAPTIVE_CONTROL"] = "Y"
        out["HILL_CONTROL"] = "N"
        if ss.get("ku") is not None:
            out["KU"] = str(ss["ku"])
        if ss.get("eps_u") is not None:
            out["EPS_U"] = str(ss["eps_u"])
        u_m = ss.get("umax")
        if u_m is not None:
            out["U_MAX"] = str(u_m)
    else:
        out["USE_ADAPTIVE_CONTROL"] = "N"
        out["HILL_CONTROL"] = "N"
    return out


def model_parameters_from_scenarios_dict(
    data: Dict[str, Any],
    scenario_name: Optional[str] = None,
    load_spatial_params: bool = False,
) -> ModelParameters:
    """
    Construye ModelParameters desde el dict ya parseado de scenarios.json
    (evita segunda lectura en disco cuando el caller ya hizo json.load).
    """
    common_params = data["common_params"]

    if scenario_name:
        scenarios = data.get("scenarios", [])
        scenario = next((s for s in scenarios if s.get("name") == scenario_name), None)
        if not scenario:
            available = [s.get("name") for s in scenarios if isinstance(s, dict) and s.get("name")]
            raise ValueError(
                f"Escenario '{scenario_name}' no encontrado en scenarios.json. "
                f"Disponibles: {available}"
            )
        params = {**common_params, **scenario}
        params = _promote_steady_state_into_params(params, scenario_name, scenario_block=scenario)
    else:
        params = common_params

    # Funciones auxiliares para conversión de tipos
    def get_float(key: str, default: float = 0.0) -> float:
        val = params.get(key)
        if val is None:
            return default
        return float(val) if isinstance(val, (int, float, str)) else default
    
    def get_int(key: str, default: int = 0) -> int:
        val = params.get(key)
        if val is None:
            return default
        return int(val) if isinstance(val, (int, float, str)) else default
    
    def get_bool(key: str, default: bool = False) -> bool:
        val = params.get(key)
        if val is None:
            return default
        if isinstance(val, str):
            return val.upper() == 'Y'
        return bool(val)
    
    u_adapt = get_bool('USE_ADAPTIVE_CONTROL')
    hill_raw = params.get('HILL_CONTROL')
    if hill_raw is None or (isinstance(hill_raw, str) and str(hill_raw).strip() == ''):
        # scenarios_v1: uSi → min-adaptativo; sin uSi/uNo en nombre y con control → Hill (legacy)
        if u_adapt and scenario_name and "uSi" in str(scenario_name):
            control_uses_hill = False
        elif u_adapt:
            control_uses_hill = True
        else:
            control_uses_hill = False
    else:
        control_uses_hill = get_bool('HILL_CONTROL')

    # Construir ModelParameters
    model_params = ModelParameters(
        rc=get_float('rc'),
        rs=get_float('rs'),
        rd=get_float('rd'),
        alpha=get_float('alpha'),
        delta=get_float('delta'),
        beta=get_float('beta'),
        a=get_float('alle', get_float('a', 0.1)),  # 'alle' o 'a'
        gamma=get_float('gamma'),
        eta=get_float('eta'),
        mu=get_float('mu'),
        allee_type=params.get('ALLEE_TYPE', 'WEAK').upper(),
        use_adaptive_control=u_adapt,
        control_uses_hill=control_uses_hill and u_adapt,
        ku=get_float('KU', 0.2),
        eps_u=get_float('EPS_U', 1e-3),
        u_max=get_float('U_MAX') if params.get('U_MAX') else None,
        hill_kc=get_float('HILL_KC', 0.05),
        hill_nc=get_float('HILL_NC', 2.0),
        hill_ki=get_float('HILL_KI', 0.2),
        hill_ni=get_float('HILL_NI', 2.0),
    )
    
    # Cargar parámetros espaciales si se solicitan
    if load_spatial_params:
        model_params.D_c = get_float('D_c')
        model_params.D_s = get_float('D_s')
        model_params.D_i = get_float('D_i')
        model_params.dt = get_float('dt')
        model_params.T = get_float('T')
        model_params.nodes_in_xaxis = get_int('nodes_in_xaxis')
        model_params.nodes_in_yaxis = get_int('nodes_in_yaxis')
        model_params.nodes_in_zaxis = get_int('nodes_in_zaxis', model_params.nodes_in_yaxis or 0)
        model_params.space_size = get_float('space_size')
        model_params.space_size_z = get_float('space_size_z', model_params.space_size or 0.0)
        model_params.spatial_dim = get_int('SPATIAL_DIM', 3)
        
        # Condiciones iniciales
        model_params.c_init_min = get_float('C_INIT_MIN')
        model_params.c_init_max = get_float('C_INIT_MAX')
        model_params.s_init_min = get_float('S_INIT_MIN')
        model_params.s_init_max = get_float('S_INIT_MAX')
        model_params.i_init_min = get_float('I_INIT_MIN')
        model_params.i_init_max = get_float('I_INIT_MAX')
    
    return model_params


def load_from_scenarios_json(
    scenarios_file: Path,
    scenario_name: Optional[str] = None,
    load_spatial_params: bool = False,
    *,
    scenarios_data: Optional[Dict[str, Any]] = None,
) -> ModelParameters:
    """
    Carga parámetros desde scenarios.json o desde un dict ya parseado.

    Si ``scenarios_data`` se pasa, no se vuelve a leer el archivo (útil para evitar I/O duplicado).

    Raises:
        FileNotFoundError: Si no hay ``scenarios_data`` y el archivo no existe.
        ValueError: Si scenario_name no se encuentra.
    """
    if scenarios_data is not None:
        return model_parameters_from_scenarios_dict(
            scenarios_data, scenario_name, load_spatial_params
        )
    if not scenarios_file.exists():
        raise FileNotFoundError(f"scenarios.json no encontrado: {scenarios_file}")
    with open(scenarios_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return model_parameters_from_scenarios_dict(data, scenario_name, load_spatial_params)


# ============================================================================
# Generación de scenarios.json desde steady_states/steady_states.py
# ============================================================================

def create_scenarios_json(
    output_file: Path,
    common_params: Dict,
    scenarios: List[Dict],
    overwrite: bool = False
) -> None:
    """
    Crea o actualiza scenarios.json con escenarios generados desde steady_states/steady_states.py.
    
    Args:
        output_file: Ruta al archivo scenarios.json
        common_params: Parámetros comunes a todos los escenarios (formato dict con strings)
        scenarios: Lista de escenarios (cada uno sobrescribe common_params)
        overwrite: Si True, sobrescribe completamente el archivo existente
        
    Ejemplo:
        common_params = {
            'rc': '5.84',
            'rs': '13.12',
            # ...
        }
        scenarios = [
            {
                'name': 'weak_mu0_test',
                'ALLEE_TYPE': 'WEAK',
                'mu': '0',
                # ...
            }
        ]
    """
    if output_file.exists() and not overwrite:
        # Cargar existente y combinar
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            # Si hay error, crear nuevo
            existing = {'common_params': {}, 'scenarios': []}
        
        # Actualizar common_params (merge, no reemplazo completo)
        if 'common_params' not in existing:
            existing['common_params'] = {}
        existing['common_params'].update(common_params)
        
        # Agregar nuevos escenarios (evitar duplicados por nombre)
        if 'scenarios' not in existing:
            existing['scenarios'] = []
        
        existing_names = {s['name'] for s in existing['scenarios']}
        for scenario in scenarios:
            if scenario['name'] not in existing_names:
                existing['scenarios'].append(scenario)
            else:
                # Actualizar escenario existente
                idx = next(
                    (i for i, s in enumerate(existing['scenarios']) 
                     if s['name'] == scenario['name']),
                    None
                )
                if idx is not None:
                    existing['scenarios'][idx].update(scenario)
                else:
                    existing['scenarios'].append(scenario)
        
        data = existing
    else:
        data = {
            'common_params': common_params,
            'scenarios': scenarios
        }
    
    # Guardar con formato legible
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ scenarios.json actualizado: {len(data['scenarios'])} escenarios")
    print(f"  Archivo: {output_file}")


def get_all_scenarios(scenarios_file: Path) -> List[str]:
    """
    Obtiene lista de nombres de todos los escenarios disponibles.
    
    Args:
        scenarios_file: Ruta al archivo scenarios.json
        
    Returns:
        Lista de nombres de escenarios
    """
    if not scenarios_file.exists():
        return []
    
    with open(scenarios_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return [s['name'] for s in data.get('scenarios', [])]


def get_common_params(scenarios_file: Path) -> Dict:
    """
    Obtiene solo los parámetros comunes.
    
    Args:
        scenarios_file: Ruta al archivo scenarios.json
        
    Returns:
        Diccionario con common_params
    """
    if not scenarios_file.exists():
        return {}
    
    with open(scenarios_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('common_params', {})

