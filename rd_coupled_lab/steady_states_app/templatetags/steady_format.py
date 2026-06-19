"""
Formato de c*, s*, i* en tarjetas del dashboard: evita que floatformat:4 colapse
valores ~1e-16 a 0,0000; usa notación científica cuando hace falta.
"""

from __future__ import annotations

import math

from django import template

register = template.Library()


@register.filter
def steady_eq_star(value):
    """
    None → "—"; 0.0 → "0"; |x| muy pequeño o muy grande → formato e;
    resto → decimal sin ceros sobrantes.
    """
    if value is None:
        return "—"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(x):
        return str(x)
    if x == 0.0:
        return "0"
    ax = abs(x)
    if ax < 1e-6 or ax >= 1e4:
        return format(x, ".6e")
    s = format(x, ".8f").rstrip("0").rstrip(".")
    return s if s else "0"
