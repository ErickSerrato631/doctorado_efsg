"""
Genera Tabla I (paper): equilibrios Strong Allee para los cuatro escenarios del manuscrito.
Lee ``steady_states[]`` en ``scenarios.json`` (mismos parámetros que el Newton 3D).

Uso (desde Allee/, ver notas §1b):
  python steady_states/generate_table_control_strong_paper.py
  python steady_states/generate_table_control_strong_paper.py --also-paper-copy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ALLEE_ROOT = Path(__file__).resolve().parent.parent
if str(_ALLEE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALLEE_ROOT))

PAPER_STRONG_SCENARIO_NAMES = [
    "strong_mu0_uNo_bajo_umbral_c0_s1_i0",
    "strong_mu0_uSi_bajo_umbral_c0_s1_i0",
    "strong_mu1_uNo_bajo_umbral_c0_s1_i0",
    "strong_mu1_uSi_bajo_umbral_c0_s1_i0",
]

# Parámetros de la cinética 3D (reacción) en steady_states / build_equations_3d
KINETIC_PARAM_KEYS = [
    ("mu", "$\\mu$"),
    ("a", "$a$"),
    ("rc", "$r_c$"),
    ("rs", "$r_s$"),
    ("rd", "$r_d$"),
    ("alpha", "$\\alpha$"),
    ("beta", "$\\beta$"),
    ("delta", "$\\delta$"),
    ("gamma", "$\\gamma$"),
    ("eta", "$\\eta$"),
    ("ku", "$k_u$"),
    ("eps_u", "$\\varepsilon_u$"),
    ("umax", "$u_{\\max}$"),
]

DIFFUSION_KEYS = [
    ("D_c", "$D_c$"),
    ("D_s", "$D_s$"),
    ("D_i", "$D_i$"),
]


def _fmt_num(x: float | None, *, nd: int = 2) -> str:
    if x is None:
        return "---"
    if isinstance(x, float) and (x != x or abs(x) > 1e12):
        return "---"
    if abs(x) < 1e-10:
        return "0"
    if abs(x) >= 100 or (abs(x) < 0.01 and x != 0):
        return f"{x:.3g}"
    return f"{x:.{nd}f}"


def _fmt_coord(x: float) -> str:
    if abs(x) < 1e-6:
        return "0"
    if abs(x - 1.0) < 1e-6:
        return "1"
    return f"{x:.3f}"


def _fmt_triple(c: float, s: float, i: float) -> str:
    return f"$({_fmt_coord(c)},\\,{_fmt_coord(s)},\\,{_fmt_coord(i)})$"


def _control_label(ss: dict) -> str:
    if ss.get("hill_control"):
        return "Hill"
    if ss.get("use_adaptive_control"):
        return "min"
    return "no"


def _get_float(d: dict, key: str, default: float | None = None) -> float | None:
    v = d.get(key, default)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _find_scenario_block(by_name: dict[str, dict], target: str) -> dict | None:
    if target in by_name:
        return by_name[target]
    for name, block in by_name.items():
        if name == target or name.startswith(target + "_"):
            return block
        for ss in block.get("steady_states") or []:
            if ss.get("scenario_json_name") == target:
                return block
    return None


def _steady_for_branch(block: dict, branch: str = "c0_s1_i0") -> dict:
    for ss in block.get("steady_states") or []:
        if ss.get("target_branch") == branch:
            return ss
    raise ValueError(f"Sin rama {branch!r} en {block.get('name')}")


def _row_from_scenario(block: dict, common: dict, *, branch: str = "c0_s1_i0") -> dict[str, Any]:
    ss = _steady_for_branch(block, branch)
    row: dict[str, Any] = {
        "name": block["name"],
        "control": _control_label(ss),
        "control_mode": ss.get("control_mode", "none"),
        "allee_type": ss.get("allee_type", "STRONG"),
        "branch": ss.get("target_branch", "---"),
        "c_star": float(ss["c_star"]),
        "s_star": float(ss["s_star"]),
        "i_star": float(ss["i_star"]),
        "max_real": float(ss["max_real"]),
        "eig1_real": float(ss.get("eig1_real", 0)),
        "eig2_real": float(ss.get("eig2_real", 0)),
        "eig3_real": float(ss.get("eig3_real", 0)),
        "residual_l2": float(ss.get("residual_l2", 0)),
        "unstable": bool(ss.get("unstable", ss["max_real"] > 0)),
    }
    for key, _ in KINETIC_PARAM_KEYS:
        if key == "eps_u":
            row[key] = _get_float(ss, "eps_u", _get_float(common, "EPS_U"))
        elif key == "ku":
            row[key] = _get_float(ss, "ku", _get_float(common, "KU"))
        elif key == "umax":
            umax = ss.get("umax")
            row["umax"] = None if umax is None else float(umax)
        else:
            row[key] = _get_float(ss, key)
    for key, _ in DIFFUSION_KEYS:
        row[key] = _get_float(common, key)
    return row


def build_latex_table(rows: list[dict], common: dict, *, caption: str, label: str) -> str:
    kin_cols = [label for _, label in KINETIC_PARAM_KEYS]
    kin_keys = [k for k, _ in KINETIC_PARAM_KEYS]

    header = (
        "Escenario & $u$ & "
        + " & ".join(kin_cols)
        + " & $(c^*,s^*,i^*)$ & Rama & Re $\\lambda_{\\max}$ & "
        "$\\lambda_1$ & $\\lambda_2$ & $\\lambda_3$ \\\\"
    )
    ncol = 3 + len(kin_cols) + 5

    lines = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        r"\tiny",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        f"\\begin{{tabular}}{{l{'c' * (ncol - 1)}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    for r in rows:
        short = r["name"].replace("_c0_s1_i0", "")
        def _cell(k: str) -> str:
            if k == "umax" and r[k] is None:
                return r"$\infty$"
            return _fmt_num(r[k], nd=2 if k not in ("mu",) else 0)

        kin_vals = " & ".join(_cell(k) for k in kin_keys)
        lines.append(
            f"{short} & {r['control']} & {kin_vals} & "
            f"{_fmt_triple(r['c_star'], r['s_star'], r['i_star'])} & "
            f"\\texttt{{{r['branch']}}} & ${_fmt_num(r['max_real'])}$ & "
            f"${_fmt_num(r['eig1_real'])}$ & ${_fmt_num(r['eig2_real'])}$ & ${_fmt_num(r['eig3_real'])}$ \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table*}",
        ]
    )

    d_parts = [f"{lab}={_fmt_num(rows[0][k])}" for k, lab in DIFFUSION_KEYS]
    lines.insert(
        -2,
        r"\vspace{0.35em}\\[0.25em]"
        + r"{\footnotesize Parámetros de difusión (PDE, comunes): "
        + ", ".join(d_parts)
        + r". Allee fuerte; control: $u=0$ o $u=\min(k_u c/(i+\varepsilon_u),u_{\max})$.}",
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Tabla I paper (todos los parámetros cinéticos)")
    ap.add_argument("--scenarios-file", type=Path, default=_ALLEE_ROOT / "scenarios.json")
    ap.add_argument("--out", type=Path, default=_ALLEE_ROOT / "table_strong_allee.tex")
    ap.add_argument("--also-paper-copy", action="store_true")
    args = ap.parse_args()

    with open(args.scenarios_file, encoding="utf-8") as f:
        data = json.load(f)
    common = data.get("common_params", {})
    by_name = {s["name"]: s for s in data.get("scenarios", [])}

    rows = []
    missing = []
    for nm in PAPER_STRONG_SCENARIO_NAMES:
        block = _find_scenario_block(by_name, nm)
        if block is None:
            missing.append(nm)
            continue
        rows.append(_row_from_scenario(block, common, branch="c0_s1_i0"))
    if missing:
        print("[!] Faltan escenarios:", ", ".join(missing))
        sys.exit(1)

    caption = (
        "Equilibrios homogéneos 3D (Allee fuerte) para los cuatro escenarios del manuscrito "
        "(rama \\texttt{c0\\_s1\\_i0}). Columnas: parámetros de la cinética de reacción "
        "($\\mu$, umbral Allee $a$, tasas $r_c,r_s,r_d$, acoplamientos $\\alpha,\\beta,\\delta,\\gamma,\\eta$ "
        "y control $k_u,\\varepsilon_u,u_{\\max}$) usados en $F_c,F_s,F_i=0$; filas con/sin control mínimo adaptativo. "
        "Todos los puntos son inestables localmente ($\\mathrm{Re}\\,\\lambda_{\\max}>0$). "
        "Fuente: \\texttt{scenarios.json} (Newton 3D)."
    )
    latex = build_latex_table(rows, common, caption=caption, label="tab:control-strong")
    args.out.write_text(latex, encoding="utf-8")
    print(f"OK: {args.out}")

    if args.also_paper_copy:
        paper_tex = _ALLEE_ROOT.parent / "Paper copy" / "sections" / "05_control.tex"
        if not paper_tex.is_file():
            print(f"[!] No existe {paper_tex}")
            return
        text = paper_tex.read_text(encoding="utf-8")
        start = text.find("\\begin{table")
        end = text.find("\\end{table*}", start)
        if end < 0:
            end = text.find("\\end{table}", start)
        if start < 0 or end < 0:
            print("[!] No se encontró bloque table en 05_control.tex")
            return
        end += len("\\end{table*}") if "\\end{table*}" in text[start : end + 20] else len("\\end{table}")
        paper_tex.write_text(text[:start] + latex.rstrip() + text[end:], encoding="utf-8")
        print(f"OK: {paper_tex}")


if __name__ == "__main__":
    main()
