"""
Script principal para generar todas las figuras (FIG. 2-5) basándose en scenarios.json.
Integra generate_phase_planes.py y generate_linear_spectra.py.

Salida: PNG bajo la carpeta de resultados (misma resolución que extract):
        ``RESULTS_DIR`` o, sin ella, Drive montado en ~/googledrive/.../Resultados Paper.
        El CSV agregado: raíz de esa carpeta, luego steady_states_extract/, luego Allee/ si existiera.
"""

import os
import sys
from pathlib import Path
import subprocess

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

_PKG_DIR = Path(__file__).resolve().parent
_ALLEE_ROOT = _PKG_DIR.parent
if str(_ALLEE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALLEE_ROOT))

try:
    from utils_paths import ensure_cloud_results_dir_ready, STEADY_STATES_EXTRACT_SUBDIR
except ImportError:
    STEADY_STATES_EXTRACT_SUBDIR = "steady_states_extract"

    def ensure_cloud_results_dir_ready():
        raise RuntimeError("Falta utils_paths en la raíz de Allee.")


def _warn_if_googledrive_path_not_mounted(results_path: Path) -> None:
    """
    ~/googledrive solo es la nube si rclone (u otro) lo ha montado.
    Si no, mkdir crea carpetas en el disco de WSL y no aparecen en drive.google.com.
    """
    try:
        gd = Path.home().resolve() / "googledrive"
        rp = results_path.resolve()
        if not gd.exists():
            return
        if gd not in rp.parents and rp != gd:
            return
        if gd.is_mount():
            print("✓ Punto de montaje ~/googledrive activo (rclone u otro FUSE).", flush=True)
            return
    except (OSError, ValueError, NotImplementedError):
        return
    print(
        "\n⚠️  ADVERTENCIA: RESULTS_DIR cae bajo ~/googledrive pero esa carpeta NO es un punto de montaje.\n"
        "   Los archivos se guardan en el disco de WSL y no suben a Google Drive en la web.\n"
        "   Opciones:\n"
        "   • Montar antes:  cd Allee && bash mount_google_drive.sh\n"
        "   • O forzar una ruta: export RESULTS_DIR='...' (p. ej. cliente Google Drive en /mnt/c/...)\n",
        flush=True,
    )


def main():
    """Función principal que genera todas las figuras."""
    print("steady_states/generate_figures_from_scenarios.py — inicio", flush=True)
    scenarios_file = _ALLEE_ROOT / 'scenarios.json'
    try:
        output_dir = ensure_cloud_results_dir_ready()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    csv_at_root = output_dir / "steady_states_scenarios.csv"
    artifact_csv = output_dir / STEADY_STATES_EXTRACT_SUBDIR / "steady_states_scenarios.csv"
    local_csv = _ALLEE_ROOT / "steady_states_scenarios.csv"

    def _pick_steady_states_csv():
        if csv_at_root.exists():
            return csv_at_root
        if artifact_csv.exists():
            return artifact_csv
        if local_csv.exists():
            return local_csv
        return None

    steady_states_csv = _pick_steady_states_csv()

    env_rd = os.environ.get("RESULTS_DIR")
    print("=" * 60, flush=True)
    print("Generador de figuras FIG. 2-5 desde scenarios.json", flush=True)
    print("=" * 60, flush=True)
    if env_rd:
        print(f"RESULTS_DIR (entorno): {env_rd}", flush=True)
    print(f"Directorio base resuelto: {output_dir}", flush=True)
    print("(Las PNG van en <directorio>/<nombre_escenario>/)", flush=True)
    _warn_if_googledrive_path_not_mounted(output_dir)
    
    # Verificar que existe scenarios.json
    if not scenarios_file.exists():
        print(f"ERROR: No se encuentra {scenarios_file}")
        sys.exit(1)
    
    # CSV: raíz de RESULTS_DIR (extract), luego layout antiguo steady_states_extract/, luego Allee/
    if steady_states_csv is None:
        print(
            "ADVERTENCIA: No hay steady_states_scenarios.csv en la raíz de resultados, "
            "ni en steady_states_extract/, ni en Allee/"
        )
        print("  Ejecutando extract (mismo entorno: Drive montado o RESULTS_DIR)...")
        try:
            result = subprocess.run(
                [sys.executable, str(_PKG_DIR / 'extract_steady_states_from_scenarios.py')],
                cwd=str(_ALLEE_ROOT),
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )
            if result.returncode != 0:
                print(f"ERROR: Error al ejecutar extract_steady_states_from_scenarios.py:")
                print(result.stderr)
                sys.exit(1)
            print("OK: Estados estacionarios calculados")
        except Exception as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        steady_states_csv = _pick_steady_states_csv()
        if steady_states_csv is None:
            print("ERROR: Tras extract sigue sin encontrarse steady_states_scenarios.csv")
            sys.exit(1)

    print(f"CSV de estados estacionarios: {steady_states_csv}", flush=True)
    
    # Crear directorio de salida si no existe
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generar phase planes (FIG. 2 y FIG. 3)
    print("\n" + "-" * 60)
    print("1. Generando phase planes (FIG. 2 y FIG. 3)...")
    print("-" * 60)
    try:
        from steady_states.generate_phase_planes import generate_phase_plane_figures
        generate_phase_plane_figures(scenarios_file, output_dir, steady_states_csv)
    except Exception as e:
        print(f"ERROR: Error al generar phase planes: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Generar espectros lineales (FIG. 4 y FIG. 5)
    print("\n" + "-" * 60)
    print("2. Generando espectros lineales (FIG. 4 y FIG. 5)...")
    print("-" * 60)
    try:
        from steady_states.generate_linear_spectra import generate_linear_spectrum_figures
        generate_linear_spectrum_figures(scenarios_file, steady_states_csv, output_dir)
    except Exception as e:
        print(f"ERROR: Error al generar espectros lineales: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Resumen
    print("\n" + "=" * 60)
    print("OK: Todas las figuras generadas exitosamente")
    print("=" * 60)
    print(f"\nFiguras guardadas bajo: {output_dir}/<nombre_escenario>/")
    print("\nFiguras generadas:")
    print("\nPhase planes (FIG. 2 y FIG. 3):")
    print("  - Una carpeta por escenario bajo el directorio de resultados resuelto (rclone o RESULTS_DIR)")
    print("  - Archivo: steady_{nombre_escenario}.png")
    print("  - Un PNG por escenario en scenarios.json (incl. hillY con USE_ADAPTIVE_CONTROL=Y)")
    print("\nEspectros lineales (FIG. 4 y FIG. 5):")
    print("  - Una carpeta por fila del CSV / escenario")
    print("  - Archivo: estabilidad_lineal_{nombre_escenario}.png")
    print("  - Un PNG por escenario presente en steady_states_scenarios.csv")


if __name__ == '__main__':
    main()

