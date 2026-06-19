"""
Fusiona Models/Allee/scenarios_v1.json + scenarios.json y escribe steady_states_full_run.json
( habitualmente en Google Drive / rclone ), misma ruta que usa el dashboard.

Ejemplo (WSL, tras montar Drive):

  python manage.py merge_steady_catalog_to_drive

  python manage.py merge_steady_catalog_to_drive --output ~/googledrive/'Resultados Paper'/estados_estacionarios/steady_states_full_run.json

  python manage.py merge_steady_catalog_to_drive --dry-run
"""

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from steady_states_app.merge_steady_catalog import merge_paths_and_write


def _is_local_allee_steady_catalog_write_target(p: Path) -> bool:
    """Evita sobreescribir por defecto el JSON sólo-local del repo."""
    try:
        r = p.expanduser().resolve()
        est = (Path(settings.ALLEE_DIR) / "estados_estacionarios").resolve()
        return r.name == "steady_states_full_run.json" and (est in r.parents or r.parent == est)
    except OSError:
        return False


class Command(BaseCommand):
    help = (
        "Fusiona scenarios_v1.json + scenarios.json (Models/Allee) y escribe steady_states_full_run.json "
        "(Drive por defecto: misma ruta que SCENARIOS_JSON_PATH)."
    )

    def add_arguments(self, parser):
        allee = Path(settings.ALLEE_DIR)
        parser.add_argument(
            "--v1",
            type=str,
            default=str(allee / "scenarios_v1.json"),
            help="Ruta a scenarios_v1.json (por defecto: ALLEE_DIR/scenarios_v1.json).",
        )
        parser.add_argument(
            "--main",
            type=str,
            default=str(allee / "scenarios.json"),
            help="Ruta a scenarios.json con bloques de equilibrios (por defecto: ALLEE_DIR/scenarios.json).",
        )
        parser.add_argument(
            "--output",
            type=str,
            default="",
            help=(
                "Ruta del steady_states_full_run.json de salida. "
                "Si se omite, usa settings.SCENARIOS_JSON_PATH (Drive si está configurado / encontrado)."
            ),
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="No escribe archivo; solo valida y muestra estadísticas.",
        )
        parser.add_argument(
            "--allow-local-repo-output",
            action="store_true",
            help="Permite escribir en Models/Allee/estados_estacionarios/steady_states_full_run.json (no recomendado).",
        )

    def handle(self, *args, **options):
        path_v1 = Path(options["v1"]).expanduser().resolve()
        path_main = Path(options["main"]).expanduser().resolve()
        raw_out = (options.get("output") or "").strip()
        output = Path(raw_out).expanduser().resolve() if raw_out else Path(settings.SCENARIOS_JSON_PATH).expanduser().resolve()

        if not path_v1.is_file():
            self.stderr.write(self.style.ERROR(f"No existe scenarios_v1: {path_v1}"))
            return
        if not path_main.is_file():
            self.stderr.write(self.style.ERROR(f"No existe scenarios principal: {path_main}"))
            return

        if _is_local_allee_steady_catalog_write_target(output) and not options["allow_local_repo_output"]:
            self.stderr.write(
                self.style.ERROR(
                    "La salida por defecto apunta al steady_states_full_run.json **local** del repo "
                    f"({output}). El laboratorio debe usar el JSON en Drive.\n"
                    "  → Exporta STEADY_STATES_CATALOG_JSON a la ruta en Drive, o pasa --output explícito.\n"
                    "  → Si de verdad quieres escribir en el repo: --allow-local-repo-output"
                )
            )
            return

        merged = merge_paths_and_write(path_v1, path_main, output, dry_run=options["dry_run"])

        strong = merged.get("strong_corner") or {}
        all_n = len(strong.get("all") or [])
        filt_n = len(strong.get("steady_states_filtered") or [])
        self.stdout.write(
            self.style.SUCCESS(
                f"Catálogo unificado: all={all_n} grupos, steady_states_filtered={filt_n} grupos.\n"
                f"Salida: {output}"
                + (" (dry-run, sin escritura)" if options["dry_run"] else "")
            )
        )
