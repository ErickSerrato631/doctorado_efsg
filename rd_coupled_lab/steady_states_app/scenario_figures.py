"""
Rutas y comprobación de figuras por escenario (nulclinas / EE y estabilidad lineal).

Convención Paper/figures: steady_{name}.png, estabilidad_lineal_{name}.png
(alineado con steady_states/generate_phase_planes.py y generate_linear_spectra).

Convención simulación FEniCS (cancer_dynamics.py, SAVE_IMAGES=Y):
RESULTS_DIR/<escenario>/images/fields_block_<bloque>_step_<t>.png
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Literal, Optional

from django.conf import settings

from .scenarios_catalog import (
    load_normalized_catalog,
    pipeline_results_subdir_for_scenario,
    scenario_names_from_normalized,
)

FigureKind = Literal["steady", "estabilidad_lineal"]

FIGURE_FILENAMES: Dict[FigureKind, str] = {
    "steady": "steady_{name}.png",
    "estabilidad_lineal": "estabilidad_lineal_{name}.png",
}

ALLOWED_KINDS: FrozenSet[str] = frozenset(FIGURE_FILENAMES.keys())

# cancer_dynamics.py: f'fields_block_{block}_step_{t:.3f}.png'
FIELDS_BLOCK_PNG_RE = re.compile(
    r"^fields_block_(\d+)_step_([0-9]+(?:\.[0-9]+)?)\.png$",
    re.IGNORECASE,
)


def get_figures_dir() -> Path:
    """
    Raíz para listar y servir PNG: FIGURES_DIR si el directorio existe; si no, FIGURES_DIR_FALLBACK.
    Si ninguno existe, devuelve FIGURES_DIR (para mensajes que indiquen la ruta esperada).
    """
    primary = Path(
        getattr(settings, "FIGURES_DIR", Path(settings.ALLEE_DIR).parent / "Paper" / "figures")
    ).resolve()
    if primary.is_dir():
        return primary
    fb = getattr(settings, "FIGURES_DIR_FALLBACK", None)
    if fb:
        alt = Path(fb).resolve()
        if alt.is_dir():
            return alt
    return primary


def resolve_png_path_in_root(root: Path, basename: str) -> Optional[Path]:
    """
    Resuelve un PNG en la raíz de ``root`` por nombre exacto o igual ignorando mayúsculas
    (útil si el escenario en JSON difiere en casing del fichero en disco, p. ej. uSi vs usi).
    """
    root = root.resolve()
    bn = (basename or "").strip()
    if not bn or "/" in bn or "\\" in bn or Path(bn).name != bn:
        return None
    if not root.is_dir():
        return None
    direct = (root / bn).resolve()
    try:
        direct.relative_to(root)
    except ValueError:
        return None
    if direct.is_file():
        return direct
    bn_low = bn.lower()
    try:
        for p in root.iterdir():
            if not p.is_file():
                continue
            if p.name.lower() != bn_low:
                continue
            rp = p.resolve()
            try:
                rp.relative_to(root)
            except ValueError:
                continue
            return rp
    except OSError:
        return None
    return None


def load_scenario_names_from_json(scenarios_file: Path) -> List[str]:
    """Nombres `name` del catálogo normalizado (steady_states_full_run.json u otro JSON vía settings), en orden."""
    if not scenarios_file.is_file():
        return []
    data, err = load_normalized_catalog(scenarios_file)
    if err:
        return []
    return scenario_names_from_normalized(data)


def scenario_names_whitelist(scenarios_file: Path) -> FrozenSet[str]:
    return frozenset(load_scenario_names_from_json(scenarios_file))


def filename_for_kind(kind: FigureKind, scenario_name: str) -> str:
    return FIGURE_FILENAMES[kind].format(name=scenario_name)


def resolve_figure_path(
    kind: FigureKind,
    scenario_name: str,
    allowed_names: FrozenSet[str],
    figures_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Resuelve la ruta al PNG si el escenario está en la whitelist.
    El nombre de archivo puede coincidir ignorando mayúsculas con el del disco.
    """
    if scenario_name not in allowed_names or kind not in FIGURE_FILENAMES:
        return None
    root = (figures_dir if figures_dir is not None else get_figures_dir()).resolve()
    bn = filename_for_kind(kind, scenario_name)
    return resolve_png_path_in_root(root, bn)


def human_label_for_png_basename(basename: str, scenario_name: str) -> str:
    """Etiqueta corta para el selector (convención Allee vs nombre de archivo)."""
    nm = (scenario_name or "").strip()
    bl = basename.lower()
    if bl == filename_for_kind("steady", nm).lower():
        return "Plano de fase / nulclinas"
    if bl == filename_for_kind("estabilidad_lineal", nm).lower():
        return "Estabilidad lineal"
    m = FIELDS_BLOCK_PNG_RE.match(basename.strip())
    if m:
        return f"Simulación (campos c,s,i) — bloque {m.group(1)}, t = {m.group(2)}"
    return basename


def list_simulation_field_pngs(scenario_name: str, results_root: Path) -> List[str]:
    """
    PNG guardados por cancer_dynamics bajo ``<results_root>/<escenario>/images/``.
    Patrón: fields_block_<n>_step_<t>.png (t con decimales como en el .savefig).
    """
    nm = (scenario_name or "").strip()
    if not nm:
        return []
    images_dir = (results_root.resolve() / nm / "images")
    if not images_dir.is_dir():
        return []
    out: List[str] = []
    try:
        for p in images_dir.iterdir():
            if p.is_file() and p.suffix.lower() == ".png" and FIELDS_BLOCK_PNG_RE.match(p.name):
                out.append(p.name)
    except OSError:
        return []

    def sort_key(name: str) -> tuple:
        mm = FIELDS_BLOCK_PNG_RE.match(name)
        if not mm:
            return (0, 0.0, name.lower())
        block = int(mm.group(1))
        t = float(mm.group(2))
        return (block, -t, name.lower())

    out.sort(key=sort_key)
    return out


def resolve_simulation_field_png_path(
    scenario_name: str, basename: str, results_root: Path
) -> Optional[Path]:
    """Ruta absoluta al PNG de simulación si existe y el nombre es válido."""
    sn = (scenario_name or "").strip()
    bn = (basename or "").strip()
    if not sn or not FIELDS_BLOCK_PNG_RE.match(bn):
        return None
    images_dir = (results_root.resolve() / sn / "images").resolve()
    if not images_dir.is_dir():
        return None
    return resolve_png_path_in_root(images_dir, bn)


def list_png_filenames_for_scenario(
    scenario_name: str,
    figures_dir: Optional[Path] = None,
) -> List[str]:
    """
    Lista PNG en la raíz de FIGURES_DIR relacionados con el escenario.

    Incluye ``steady_<nombre>.png`` y ``estabilidad_lineal_<nombre>.png`` si existen,
    y cualquier otro ``*.png`` cuyo nombre contenga el identificador del escenario
    (comparación sin distinguir mayúsculas; útil si el JSON tiene ``uSi`` y el fichero ``usi``).

    Para nombres de escenario muy cortos (< 4 caracteres) solo se consideran los dos
    archivos canónicos, para evitar coincidencias espurias.
    """
    root = (figures_dir or get_figures_dir()).resolve()
    nm = (scenario_name or "").strip()
    if not root.is_dir() or not nm:
        return []
    steady_n = filename_for_kind("steady", nm)
    spec_n = filename_for_kind("estabilidad_lineal", nm)
    steady_l = steady_n.lower()
    spec_l = spec_n.lower()
    nm_l = nm.lower()
    found: set[str] = set()
    try:
        for p in root.iterdir():
            if not p.is_file() or p.suffix.lower() != ".png":
                continue
            fn = p.name
            fl = fn.lower()
            if fl == steady_l or fl == spec_l:
                found.add(fn)
                continue
            if len(nm) >= 4 and nm_l in fl:
                found.add(fn)
    except OSError:
        return []
    ordered: List[str] = []
    steady_hit = next((f for f in found if f.lower() == steady_l), None)
    if steady_hit:
        ordered.append(steady_hit)
    spec_hit = next((f for f in found if f.lower() == spec_l), None)
    if spec_hit:
        ordered.append(spec_hit)
    rest = sorted([n for n in found if n not in ordered], key=str.lower)
    ordered.extend(rest)
    return ordered


def is_png_allowed_for_scenario(
    basename: str,
    scenario_name: str,
    figures_dir: Optional[Path] = None,
) -> bool:
    """True si el basename está en el listado permitido para ese escenario."""
    bn = (basename or "").strip()
    if not bn or Path(bn).name != bn or "/" in bn or "\\" in bn:
        return False
    if not bn.lower().endswith(".png"):
        return False
    allowed = list_png_filenames_for_scenario(scenario_name, figures_dir)
    bn_l = bn.lower()
    return any(a.lower() == bn_l for a in allowed)


def catalog_figure_select_options(
    scenario_name: str,
    figures_dir: Optional[Path] = None,
    results_root: Optional[Path] = None,
    *,
    normalized_catalog: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """
    Opciones para ``<select>``: filename, label, url_kind (``paper`` | ``simulation``).

    Las PNG de simulación se buscan bajo RESULTS usando ``pipeline_folder`` del JSON
    (nombre corto de carpeta) cuando existe; Paper/figures sigue ``scenario_name``.
    """
    nm = (scenario_name or "").strip()
    nm_disk = nm
    if normalized_catalog is not None:
        nm_disk = pipeline_results_subdir_for_scenario(normalized_catalog, nm)

    out: List[Dict[str, str]] = []
    for fn in list_png_filenames_for_scenario(nm, figures_dir):
        out.append(
            {
                "filename": fn,
                "label": human_label_for_png_basename(fn, nm),
                "url_kind": "paper",
            }
        )
    if results_root is not None:
        for fn in list_simulation_field_pngs(nm_disk, results_root):
            out.append(
                {
                    "filename": fn,
                    "label": human_label_for_png_basename(fn, nm),
                    "url_kind": "simulation",
                }
            )
    return out


def list_scenario_figure_rows(
    scenario_names: List[str],
    figures_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Una fila por escenario con flags de existencia de cada tipo de figura.
    """
    root = figures_dir if figures_dir is not None else get_figures_dir()
    rows: List[Dict[str, Any]] = []
    for name in scenario_names:
        steady_bn = filename_for_kind("steady", name)
        spec_bn = filename_for_kind("estabilidad_lineal", name)
        p_steady = resolve_png_path_in_root(root, steady_bn)
        p_spec = resolve_png_path_in_root(root, spec_bn)
        rows.append(
            {
                "name": name,
                "has_phase": p_steady is not None,
                "has_spectrum": p_spec is not None,
                "phase_path": p_steady or (root / steady_bn),
                "spectrum_path": p_spec or (root / spec_bn),
            }
        )
    return rows
