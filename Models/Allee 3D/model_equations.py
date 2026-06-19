"""
model_equations.py

Define las ecuaciones matemáticas del modelo de dinámica de cáncer.
Proporciona ecuaciones en formato:
- SymPy: para análisis de estados estacionarios
- UFL: para simulación espacial con FEniCSx

Ecuaciones base:
- ∂c/∂t = D_c ∇²c + R_c(c, s, i)
- ∂s/∂t = D_s ∇²s + R_s(c, s, i)  
- ∂i/∂t = D_i ∇²i + R_i(c, s, i)

donde R_c, R_s, R_i son los términos de reacción (sin difusión ni tiempo).
"""

import sympy as sp
from typing import Optional, Tuple, Dict, Literal, TYPE_CHECKING, Any

# Importar ufl solo si está disponible (para FEniCSx)
try:
    import ufl
    UFL_AVAILABLE = True
except ImportError:
    UFL_AVAILABLE = False
    ufl = None

from model_parameters import ModelParameters

# Para anotaciones de tipo cuando UFL no está disponible
if TYPE_CHECKING:
    try:
        from dolfinx.fem import Function, TestFunction
        from ufl import Measure, Form
    except ImportError:
        pass


# ============================================================================
# 1. Símbolos simbólicos para SymPy
# ============================================================================

# Símbolos para análisis simbólico
c_sym, s_sym, i_sym = sp.symbols('c s i')
rc_sym, rs_sym, rd_sym = sp.symbols('rc rs rd')
alpha_sym, delta_sym, beta_sym = sp.symbols('alpha delta beta')
a_sym, gamma_sym, eta_sym, mu_sym = sp.symbols('a gamma eta mu')
ku_sym, eps_u_sym, umax_sym = sp.symbols('ku eps_u umax')


# ============================================================================
# 2. Términos de reacción individuales
# ============================================================================

def reaction_term_allee_weak(c, rc_val, a_val):
    """
    Término de efecto Allee débil: rc * c * (c - a) * (1 - c)
    
    Args:
        c: Variable de concentración (SymPy o UFL)
        rc_val: Tasa de crecimiento
        a_val: Parámetro de Allee
    """
    return rc_val * c * (c - a_val) * (1 - c)


def reaction_term_allee_strong(c, rc_val, a_val):
    """
    Término de efecto Allee fuerte: rc * c * (1 - c) * ((c - a) / (1 - a))
    
    Args:
        c: Variable de concentración (SymPy o UFL)
        rc_val: Tasa de crecimiento
        a_val: Parámetro de Allee
    """
    return rc_val * c * (1 - c) * ((c - a_val) / (1 - a_val))


def reaction_term_control_adaptive(c, i, ku_val, eps_u_val, u_max_val=None):
    """
    Término de control adaptativo: ku * c / (i + eps_u)
    
    Args:
        c, i: Variables de concentración
        ku_val: Intensidad del control
        eps_u_val: Parámetro epsilon
        u_max_val: Valor máximo (opcional)
    """
    u_raw = ku_val * c / (i + eps_u_val)
    if u_max_val is not None:
        # En UFL usar conditional, en SymPy usar Min
        if isinstance(c, sp.Basic):
            return sp.Min(u_raw, u_max_val)
        else:  # UFL
            return ufl.conditional(u_raw < u_max_val, u_raw, u_max_val)
    return u_raw


def reaction_term_control_hill(c, i, u_max_val, kc_val=0.05, nc_val=2.0, ki_val=0.2, ni_val=2.0):
    """
    Control tipo Hill (Opción A):
      u = u_max * H_act(c; Kc,nc) * H_inh(i; Ki,ni)

    H_act(x) = x^n / (K^n + x^n)
    H_inh(x) = K^n / (K^n + x^n)

    Nota: se protege contra i negativa usando max(i,0) en UFL y Piecewise en SymPy.
    """
    if isinstance(c, sp.Basic):
        i_pos = sp.Max(i, 0)
        c_pow = c**nc_val
        kc_pow = kc_val**nc_val
        h_act = c_pow / (kc_pow + c_pow)
        i_pow = i_pos**ni_val
        ki_pow = ki_val**ni_val
        h_inh = ki_pow / (ki_pow + i_pow)
        return u_max_val * h_act * h_inh
    else:
        i_pos = ufl.conditional(i > 0.0, i, 0.0)
        c_pow = c**nc_val
        kc_pow = kc_val**nc_val
        h_act = c_pow / (kc_pow + c_pow)
        i_pow = i_pos**ni_val
        ki_pow = ki_val**ni_val
        h_inh = ki_pow / (ki_pow + i_pow)
        return u_max_val * h_act * h_inh


# ============================================================================
# 3. Ecuaciones de reacción completas (SymPy)
# ============================================================================

def build_reaction_equations_sympy(
    params: ModelParameters
) -> Tuple[sp.Expr, sp.Expr, sp.Expr]:
    """
    Construye las ecuaciones de reacción en formato SymPy.
    
    Args:
        params: ModelParameters con los parámetros del modelo
        
    Returns:
        Tuple (R_c, R_s, R_i) con términos de reacción simbólicos
    """
    # Término Allee
    if params.allee_type == 'STRONG':
        allee_term = reaction_term_allee_strong(c_sym, rc_sym, a_sym)
    else:
        allee_term = reaction_term_allee_weak(c_sym, rc_sym, a_sym)
    
    # Ecuación para c
    R_c = allee_term - c_sym * (alpha_sym * s_sym**2 + beta_sym * i_sym**2)
    if params.mu > 0:
        R_c = R_c - mu_sym * c_sym * (gamma_sym * s_sym**2 + eta_sym * i_sym**2)
    
    # Ecuación para s
    R_s = (rs_sym * s_sym * (1 - s_sym) - 
           gamma_sym * c_sym**2 * s_sym + 
           delta_sym * i_sym**2 * s_sym)
    if params.mu > 0:
        R_s = R_s - (s_sym * c_sym**2 * alpha_sym * mu_sym / 2)
    
    # Ecuación para i
    R_i = (rd_sym * i_sym * (1 - i_sym) + 
           delta_sym * i_sym * s_sym**2 - 
           eta_sym * c_sym**2 * i_sym)
    if params.mu > 0:
        R_i = R_i - (i_sym * c_sym**2 * beta_sym * mu_sym / 2)
    
    # Control: ley Hill o min-adaptativa ku·c/(i+ε) (tope opcional u_max)
    if params.use_adaptive_control:
        if params.control_uses_hill:
            u_ctrl = reaction_term_control_hill(
                c_sym, i_sym,
                umax_sym if params.u_max is not None else 1.0,
                params.hill_kc, params.hill_nc, params.hill_ki, params.hill_ni
            )
        else:
            u_ctrl = reaction_term_control_adaptive(
                c_sym,
                i_sym,
                ku_sym,
                eps_u_sym,
                umax_sym if params.u_max is not None else None,
            )
        R_i = R_i + u_ctrl
    
    # Sustituir valores numéricos
    subs_dict = {
        rc_sym: params.rc,
        rs_sym: params.rs,
        rd_sym: params.rd,
        alpha_sym: params.alpha,
        delta_sym: params.delta,
        beta_sym: params.beta,
        a_sym: params.a,
        gamma_sym: params.gamma,
        eta_sym: params.eta,
        mu_sym: params.mu,
    }
    
    if params.use_adaptive_control:
        if params.control_uses_hill:
            if params.u_max is not None:
                subs_dict[umax_sym] = params.u_max
        else:
            subs_dict[ku_sym] = params.ku
            subs_dict[eps_u_sym] = params.eps_u
            if params.u_max is not None:
                subs_dict[umax_sym] = params.u_max
    
    R_c = R_c.subs(subs_dict)
    R_s = R_s.subs(subs_dict)
    R_i = R_i.subs(subs_dict)
    
    return R_c, R_s, R_i


# ============================================================================
# 4. Ecuaciones de reacción completas (UFL para FEniCSx)
# ============================================================================

def build_reaction_terms_ufl(
    c: Any,
    s: Any,
    i: Any,
    phi_c: Any,
    phi_s: Any,
    phi_i: Any,
    dx: Any,
    params: ModelParameters
) -> Tuple[Any, Any, Any]:
    """
    Construye los términos de reacción en formato UFL para FEniCSx.
    
    Args:
        c, s, i: Funciones UFL de los campos
        phi_c, phi_s, phi_i: Funciones de prueba UFL
        dx: Medida de integración UFL
        params: ModelParameters con los parámetros
        
    Returns:
        Tuple (R_c_form, R_s_form, R_i_form) con formas variacionales
    """
    if not UFL_AVAILABLE:
        raise ImportError("ufl no está disponible. Esta función requiere FEniCSx.")
    # Término Allee
    if params.allee_type == 'STRONG':
        allee_term = reaction_term_allee_strong(c, params.rc, params.a) * phi_c * dx
    else:
        allee_term = reaction_term_allee_weak(c, params.rc, params.a) * phi_c * dx
    
    # Ecuación para c
    R_c = (allee_term - 
           c * (params.alpha * s**2 + params.beta * i**2) * phi_c * dx)
    if params.mu > 0:
        R_c = R_c - params.mu * c * (params.gamma * s**2 + params.eta * i**2) * phi_c * dx
    
    # Ecuación para s
    R_s = ((params.rs * s * (1 - s) - 
            params.gamma * c**2 * s + 
            params.delta * i**2 * s) * phi_s * dx)
    if params.mu > 0:
        R_s = R_s - (s * c**2 * params.alpha * params.mu / 2) * phi_s * dx
    
    # Ecuación para i
    R_i = ((params.rd * i * (1 - i) + 
            params.delta * i * s**2 - 
            params.eta * c**2 * i) * phi_i * dx)
    if params.mu > 0:
        R_i = R_i - (i * c**2 * params.beta * params.mu / 2) * phi_i * dx
    
    if params.use_adaptive_control:
        if params.control_uses_hill:
            u_ctrl = reaction_term_control_hill(
                c, i, params.u_max if params.u_max is not None else 1.0,
                params.hill_kc, params.hill_nc, params.hill_ki, params.hill_ni
            )
        else:
            u_ctrl = reaction_term_control_adaptive(
                c, i, params.ku, params.eps_u, params.u_max
            )
        R_i = R_i + u_ctrl * phi_i * dx
    
    return R_c, R_s, R_i


def build_reaction_rates_ufl(
    c: Any,
    s: Any,
    i: Any,
    params: ModelParameters,
) -> Tuple[Any, Any, Any]:
    """
    Términos de reacción como expresiones UFL escalares R_c, R_s, R_i (sin prueba ni dx).

    Útil para ufl.diff (potencial químico local ∂R_a/∂φ_a) y postproceso tipo J·∇μ.
    """
    if not UFL_AVAILABLE:
        raise ImportError("ufl no está disponible. Esta función requiere FEniCSx.")
    if params.allee_type == 'STRONG':
        allee_term = reaction_term_allee_strong(c, params.rc, params.a)
    else:
        allee_term = reaction_term_allee_weak(c, params.rc, params.a)

    R_c = allee_term - c * (params.alpha * s**2 + params.beta * i**2)
    if params.mu > 0:
        R_c = R_c - params.mu * c * (params.gamma * s**2 + params.eta * i**2)

    R_s = (
        params.rs * s * (1 - s)
        - params.gamma * c**2 * s
        + params.delta * i**2 * s
    )
    if params.mu > 0:
        R_s = R_s - (s * c**2 * params.alpha * params.mu / 2)

    R_i = (
        params.rd * i * (1 - i)
        + params.delta * i * s**2
        - params.eta * c**2 * i
    )
    if params.mu > 0:
        R_i = R_i - (i * c**2 * params.beta * params.mu / 2)

    if params.use_adaptive_control:
        if params.control_uses_hill:
            u_ctrl = reaction_term_control_hill(
                c,
                i,
                params.u_max if params.u_max is not None else 1.0,
                params.hill_kc,
                params.hill_nc,
                params.hill_ki,
                params.hill_ni,
            )
        else:
            u_ctrl = reaction_term_control_adaptive(
                c, i, params.ku, params.eps_u, params.u_max
            )
        R_i = R_i + u_ctrl

    return R_c, R_s, R_i


# ============================================================================
# 5. Construcción de formas variacionales completas (con difusión y tiempo)
# ============================================================================

def build_variational_forms_ufl(
    c: Any,
    s: Any,
    i: Any,
    c_n: Any,
    s_n: Any,
    i_n: Any,
    phi_c: Any,
    phi_s: Any,
    phi_i: Any,
    dx: Any,
    dt: float,
    params: ModelParameters
) -> Tuple[Any, Any, Any]:
    """
    Construye las formas variacionales completas (difusión + reacción + tiempo).
    
    Args:
        c, s, i: Funciones UFL de los campos actuales
        c_n, s_n, i_n: Funciones UFL de los campos en tiempo anterior
        phi_c, phi_s, phi_i: Funciones de prueba UFL
        dx: Medida de integración UFL
        dt: Paso de tiempo
        params: ModelParameters con los parámetros (debe incluir D_c, D_s, D_i)
        
    Returns:
        Tuple (F_c, F_s, F_i) con formas variacionales completas
        
    Raises:
        ValueError: Si params no tiene parámetros de difusión
        ImportError: Si ufl no está disponible
    """
    if not UFL_AVAILABLE:
        raise ImportError("ufl no está disponible. Esta función requiere FEniCSx.")
    if params.D_c is None or params.D_s is None or params.D_i is None:
        raise ValueError(
            "params debe incluir D_c, D_s, D_i para construir formas variacionales completas"
        )
    
    # Términos de reacción
    R_c, R_s, R_i = build_reaction_terms_ufl(
        c, s, i, phi_c, phi_s, phi_i, dx, params
    )
    
    # Términos temporales y de difusión
    # Convención de la forma débil: para ∂_t φ = D ∇²φ + R, F=0 requiere
    # F = ∫(φ-φ_n)/dt·v dx + ∫D∇φ·∇v dx − ∫R·v dx. Por eso R entra con signo negativo.
    F_c = (((c - c_n) / dt) * phi_c * dx + 
           params.D_c * ufl.dot(ufl.grad(c), ufl.grad(phi_c)) * dx - 
           R_c)
    
    F_s = (((s - s_n) / dt) * phi_s * dx + 
           params.D_s * ufl.dot(ufl.grad(s), ufl.grad(phi_s)) * dx - 
           R_s)
    
    F_i = (((i - i_n) / dt) * phi_i * dx + 
           params.D_i * ufl.dot(ufl.grad(i), ufl.grad(phi_i)) * dx - 
           R_i)
    
    return F_c, F_s, F_i


# ============================================================================
# 6. Modelo reducido 2D (para análisis de estados estacionarios)
# ============================================================================

def build_reduced_model_2d_sympy(
    params: ModelParameters
) -> Tuple[sp.Expr, sp.Expr]:
    """
    Construye el modelo reducido 2D (c, s) eliminando i mediante relación algebraica.
    
    El modelo reducido asume que i puede expresarse en términos de c y s:
    i* = 1 + (delta * s² - c² * (eta + beta * mu / 2)) / rd
    
    Coherente con la eliminación algebraica de \(i\) en el equilibrio (nullclines en el plano \((c,s)\)).
    
    Args:
        params: ModelParameters con los parámetros
        
    Returns:
        Tuple (F1, F2) con las nullclines del modelo reducido
    """
    c, s = c_sym, s_sym
    
    # Relación para i* en el modelo reducido (despejada de la ecuación de i)
    # i* = 1 + (delta * s² - c² * (eta + beta * mu / 2)) / rd
    # Pero en la fórmula original se usa: 2*rd + 2*s²*delta - c²*(2*eta + beta*mu)
    # Simplificando: i* = (2*rd + 2*s²*delta - c²*(2*eta + beta*mu)) / (2*rd)
    # = 1 + (delta*s² - c²*(eta + beta*mu/2)) / rd
    
    # Construir F1 y F2 según la fórmula exacta de steady_states/steady_states.py
    # F1 = (1/4) * c * (-4*(-1+c)*(-a+c)*rc - 4*s²*alpha - 
    #      (beta*(2*rd + 2*s²*delta - c²*(2*eta + beta*mu))²)/rd² -
    #      4*mu*(s²*gamma + (eta*(2*rd + 2*s²*delta - c²*(2*eta + beta*mu))²)/(4*rd²)))
    
    inner_expr = 2 * rd_sym + 2 * s**2 * delta_sym - c**2 * (2 * eta_sym + beta_sym * mu_sym)
    
    F1 = (sp.Rational(1, 4)) * c * (
        -4 * (-1 + c) * (-a_sym + c) * rc_sym - 4 * s**2 * alpha_sym -
        (beta_sym * inner_expr**2) / rd_sym**2 -
        4 * mu_sym * (s**2 * gamma_sym + (eta_sym * inner_expr**2) / (4 * rd_sym**2))
    )
    
    # F2 = (1/4) * s * (-4*rs*(-1+s) - 4*c²*gamma - 2*c²*alpha*mu +
    #      (delta*(2*rd + 2*s²*delta - c²*(2*eta + beta*mu))²)/rd²)
    
    F2 = (sp.Rational(1, 4)) * s * (
        -4 * rs_sym * (-1 + s) - 4 * c**2 * gamma_sym - 2 * c**2 * alpha_sym * mu_sym +
        (delta_sym * inner_expr**2) / rd_sym**2
    )
    
    # Sustituir valores numéricos
    subs_dict = {
        rc_sym: params.rc,
        rs_sym: params.rs,
        rd_sym: params.rd,
        alpha_sym: params.alpha,
        delta_sym: params.delta,
        beta_sym: params.beta,
        a_sym: params.a,
        gamma_sym: params.gamma,
        eta_sym: params.eta,
        mu_sym: params.mu,
    }
    
    F1 = F1.subs(subs_dict)
    F2 = F2.subs(subs_dict)
    
    return F1, F2

