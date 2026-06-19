#!/usr/bin/env python3
"""
Genera biblioteca_categoria_auto.tex (categoría biblatex 'tienepdf').
Opcional: doctor_presentation.tex ya no filtra por esta categoría; sirve si quieres
reutilizar el fragmento en otro documento o filtrar la bibliografía a mano.

Uso (cwd = carpeta Presentacion/):
  python sync_biblioteca_biblatex.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

OUT_NAME = "biblioteca_categoria_auto.tex"
BIB_NAME = "References.bib"


def _bib_blocks(text: str) -> list[str]:
    parts = re.split(r"\n(?=@)", text)
    return [p for p in parts if p.lstrip().startswith("@")]


def parse_addenda(bib_path: Path) -> list[tuple[str, str]]:
    """(clave_bib, nombre_archivo_addendum) por entrada con addendum."""
    text = bib_path.read_text(encoding="utf-8")
    out: list[tuple[str, str]] = []
    for block in _bib_blocks(text):
        m_key = re.match(r"@\w+\s*\{\s*([^,\s]+)\s*,", block)
        if not m_key:
            continue
        key = m_key.group(1).strip()
        m_add = re.search(r"addendum\s*=\s*\{([^}]*)\}", block, re.I)
        if not m_add:
            continue
        fname = m_add.group(1).strip()
        if fname:
            out.append((key, fname))
    return out


def pdf_exists(bib_dir: Path, filename: str) -> bool:
    for sub in ("Biblioteca",):
        p = bib_dir / sub / filename
        if p.is_file():
            return True
    return False


def main() -> int:
    root = Path(__file__).resolve().parent
    bib = root / BIB_NAME
    if not bib.is_file():
        print(f"No se encontró {bib}", file=sys.stderr)
        return 1

    pairs = parse_addenda(bib)
    lines = [
        "% Auto-generado por sync_biblioteca_biblatex.py — no editar a mano.",
        "% Regenerar:  python sync_biblioteca_biblatex.py",
        "",
    ]
    n = 0
    for key, fname in pairs:
        if pdf_exists(root, fname):
            lines.append(f"\\addtocategory{{tienepdf}}{{{key}}}%")
            n += 1

    out_path = root / OUT_NAME
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Escrito {out_path} ({n} entrada(s) con PDF en Biblioteca/).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
