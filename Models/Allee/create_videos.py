"""
create_videos.py

Script para crear videos a partir de imágenes de campos generadas durante las simulaciones.
Combina imágenes secuenciales en archivos de video MP4.

Uso:
    python create_videos.py --image-folder <ruta> --output <video.mp4> --pattern <patron> --steps <n> --step-size <valor>
    python create_videos.py --scenario <nombre> --field <c|s|i>  # Crear video para un escenario específico
    python create_videos.py --compare-mu1-ab  # Video comparativo A|B (strong_mu1 uNo vs uSi, bajo umbral)
"""

from __future__ import annotations

import imageio.v2 as imageio
import os
import argparse
from pathlib import Path
import sys

import numpy as np


def _pad_frame_h264_macro_block(frame: np.ndarray) -> np.ndarray:
    """Evita el resize de ffmpeg (macro_block_size=16): dimensiones multiplo de 16."""
    if frame.ndim != 3:
        return frame
    h, w = int(frame.shape[0]), int(frame.shape[1])
    h2 = (h + 15) // 16 * 16
    w2 = (w + 15) // 16 * 16
    if h2 == h and w2 == w:
        return frame
    out = np.zeros((h2, w2, frame.shape[2]), dtype=frame.dtype)
    out[:h, :w] = frame
    return out


def _report_frame_progress(
    index: int,
    total: int,
    *,
    every: int,
    time_val: float | None = None,
) -> None:
    """Imprime avance cada `every` frames (y siempre el primero y el ultimo). every<=0 desactiva."""
    if every <= 0 or total <= 0:
        return
    last = index >= total - 1
    if index > 0 and not last and (index + 1) % every != 0:
        return
    pct = 100.0 * (index + 1) / total
    extra = f"  t={time_val:.6g}" if time_val is not None else ""
    print(f"  avance {index + 1}/{total} ({pct:.1f}%){extra}", flush=True)


def _step_time_from_fields_png(path: Path) -> float | None:
    """fields_block_1_step_1.234.png -> 1.234"""
    try:
        return float(path.stem.split("_step_")[-1])
    except (ValueError, IndexError):
        return None


def _collect_fields_frames(images_dir: Path, block: int) -> dict[float, Path]:
    out: dict[float, Path] = {}
    for p in sorted(images_dir.glob(f"fields_block_{block}_step_*.png")):
        t = _step_time_from_fields_png(p)
        if t is not None:
            out[t] = p
    return out


# Escenarios presentacion mu=1 (campos c,s,i en cada PNG panel)
SCENARIOS_MU1_AB = (
    "strong_mu1_uNo_bajo_umbral",
    "strong_mu1_uSi_bajo_umbral",
)

# Importar utilidades de rutas compartidas
try:
    from utils_paths import get_results_dir, get_scenario_dir
except ImportError:
    # Fallback si el módulo no está disponible
    def get_results_dir(base_dir=None):
        """Fallback local si utils_paths no está disponible"""
        if base_dir is None:
            base_dir = Path(__file__).parent
        env_results_dir = os.getenv('RESULTS_DIR')
        if env_results_dir:
            results_path = Path(env_results_dir)
            if results_path.exists():
                return results_path
        drive_mount_point = Path.home() / "googledrive"
        drive_results_dir = drive_mount_point / "Doctorado Erick Serrato" / "Resultados Paper"
        if drive_mount_point.exists():
            try:
                list(drive_mount_point.iterdir())
                drive_results_dir.mkdir(parents=True, exist_ok=True)
                return drive_results_dir
            except (OSError, PermissionError):
                pass
        return base_dir / "results"
    
    def get_scenario_dir(scenario_name, base_dir=None):
        """Fallback local"""
        if base_dir is None:
            base_dir = Path(__file__).parent
        results_dir = get_results_dir(base_dir)
        return results_dir / scenario_name

def crear_video(
    image_folder,
    video_filename,
    pattern,
    steps,
    step_size,
    fps=10,
    progress_every: int = 100,
):
    """
    Crea un video a partir de imágenes secuenciales.
    
    Args:
        image_folder (str): Carpeta que contiene las imágenes
        video_filename (str): Nombre del archivo de video de salida
        pattern (str): Patrón para nombres de archivos (ej: 'fields_block_1._step_{step:.3f}.png')
        steps (int): Número de pasos a incluir en el video
        step_size (float): Tamaño del paso entre imágenes
        fps (int): Frames por segundo del video (default: 10)
    
    Returns:
        bool: True si el video se creó exitosamente, False en caso contrario
    """
    image_folder = os.path.expanduser(image_folder)
    video_filename = os.path.expanduser(video_filename)
    
    # Crear directorio de salida si no existe
    output_dir = os.path.dirname(video_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Verificar que la carpeta de imágenes existe
    if not os.path.exists(image_folder):
        print(f"✗ Error: La carpeta de imágenes no existe: {image_folder}")
        return False
    
    # Crear la lista de nombres de archivos de imagen
    images = [pattern.format(step=step_size * i) for i in range(1, steps)]
    
    # Verificar que hay imágenes disponibles
    available_images = []
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        if os.path.exists(image_path):
            available_images.append(image_path)
        else:
            print(f'⚠️ Imagen no encontrada: {image_path}')
    
    if not available_images:
        print(f"✗ Error: No se encontraron imágenes en {image_folder}")
        return False
    
    print(f"✓ Encontradas {len(available_images)} imágenes de {len(images)} esperadas")
    print(f"  Creando video: {video_filename}")
    if progress_every > 0:
        print(f"  (avance cada {progress_every} frames)", flush=True)

    # Crear el video
    try:
        n = len(available_images)
        with imageio.get_writer(video_filename, fps=fps) as writer:
            for idx, image_path in enumerate(available_images):
                image = _pad_frame_h264_macro_block(imageio.imread(image_path))
                writer.append_data(image)
                _report_frame_progress(idx, n, every=progress_every)
        
        print(f'✅ Video guardado como {video_filename}')
        return True
    except Exception as e:
        print(f"✗ Error al crear el video: {e}")
        return False

def create_video_for_scenario(
    scenario_name,
    field="c",
    block=1,
    results_dir=None,
    fps=10,
    progress_every: int = 100,
    t_max: float | None = 3.0,
):
    """
    Crea un video para un escenario específico desde el directorio de resultados.
    Detecta automáticamente Google Drive si está montado.
    
    Args:
        scenario_name (str): Nombre del escenario
        field (str): Campo a visualizar ('c', 's', o 'i')
        block (int): Número de bloque (default: 1)
        results_dir (str): Directorio base de resultados (None = auto-detecta Google Drive o local)
        fps (int): Frames por segundo del video (default: 10)
    
    Returns:
        bool: True si el video se creó exitosamente, False en caso contrario
    """
    if results_dir is None:
        # Usar detección automática de Google Drive
        scenario_dir = get_scenario_dir(scenario_name, Path(__file__).parent)
    else:
        scenario_dir = Path(results_dir) / scenario_name
    images_dir = scenario_dir / "images"
    
    if not images_dir.exists():
        print(f"✗ Error: No se encontró el directorio de imágenes: {images_dir}")
        return False
    
    # Patrón para buscar imágenes del campo específico
    # Las imágenes se guardan como: fields_block_{block}_step_{t:.3f}.png
    # Necesitamos encontrar todas las imágenes del bloque
    
    # Buscar todas las imágenes del bloque
    image_files = sorted(images_dir.glob(f'fields_block_{block}_step_*.png'))

    if t_max is not None:
        image_files = [
            p
            for p in image_files
            if (tv := _step_time_from_fields_png(p)) is not None and tv <= t_max + 1e-9
        ]

    if not image_files:
        print(f"✗ Error: No se encontraron imágenes para el bloque {block} en {images_dir}")
        return False

    # Crear video con todas las imágenes encontradas
    video_filename = scenario_dir / f"{scenario_name}_block_{block}_field_{field}.mp4"

    print(f"Creando video para escenario: {scenario_name}")
    print(f"  Bloque: {block}")
    if t_max is not None:
        print(f"  t <= {t_max:g} (filtro)")
    print(f"  Imágenes en el video: {len(image_files)}")
    print(f"  Video de salida: {video_filename}")
    if progress_every > 0:
        print(f"  (avance cada {progress_every} frames)", flush=True)

    try:
        n = len(image_files)
        with imageio.get_writer(str(video_filename), fps=fps) as writer:
            for idx, image_path in enumerate(image_files):
                image = _pad_frame_h264_macro_block(imageio.imread(str(image_path)))
                writer.append_data(image)
                _report_frame_progress(idx, n, every=progress_every)
        
        print(f'✅ Video guardado como {video_filename}')
        return True
    except Exception as e:
        print(f"✗ Error al crear el video: {e}")
        return False


def _match_image_size(arr_a: np.ndarray, arr_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Iguala alturas (recorte o reescala con PIL si hace falta)."""
    ha, wa = arr_a.shape[0], arr_a.shape[1]
    hb, wb = arr_b.shape[0], arr_b.shape[1]
    if ha == hb and wa == wb:
        return arr_a, arr_b
    try:
        from PIL import Image

        target_h = min(ha, hb)
        pil_a = Image.fromarray(arr_a)
        pil_b = Image.fromarray(arr_b)
        wa2 = int(wa * target_h / ha)
        wb2 = int(wb * target_h / hb)
        pil_a = pil_a.resize((wa2, target_h), Image.Resampling.LANCZOS)
        pil_b = pil_b.resize((wb2, target_h), Image.Resampling.LANCZOS)
        return np.asarray(pil_a), np.asarray(pil_b)
    except Exception:
        print("[!] Dimensiones distintas entre A y B; intentando apilar sin redimensionar (puede fallar).")
        return arr_a, arr_b


def _hstack_with_labels(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    label_a: str,
    label_b: str,
) -> np.ndarray:
    a, b = _match_image_size(arr_a, arr_b)
    h, wa, _ = a.shape
    _, wb, _ = b.shape
    try:
        from PIL import Image, ImageDraw, ImageFont

        bar = 36
        canvas_w = wa + wb + 8
        canvas_h = h + bar
        canvas = np.zeros((canvas_h, canvas_w, a.shape[2]), dtype=np.uint8)
        canvas[:] = 40
        canvas[bar : bar + h, 0:wa] = a
        canvas[bar : bar + h, wa + 8 : wa + 8 + wb] = b
        pil = Image.fromarray(canvas)
        dr = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 18)
        except OSError:
            font = ImageFont.load_default()
        dr.text((8, 8), label_a, fill=(255, 255, 255), font=font)
        dr.text((wa + 16, 8), label_b, fill=(255, 255, 255), font=font)
        return np.asarray(pil)
    except Exception:
        gap = np.zeros((h, 8, a.shape[2]), dtype=np.uint8)
        gap[:] = 30
        return np.concatenate([a, gap, b], axis=1)


def create_compare_mu1_ab_fields_video(
    output_mp4: Path,
    block: int = 1,
    results_dir: str | Path | None = None,
    fps: int = 10,
    base_dir: Path | None = None,
    label_a: str = "A: sin Hill (u=0)",
    label_b: str = "B: con Hill (u>0)",
    progress_every: int = 100,
    t_max: float | None = 3.0,
) -> bool:
    """
    Un video MP4: en cada frame, paneles c,s,i de A a la izquierda y de B a la derecha,
    alineados por el tiempo fisico del nombre fields_block_*_step_<t>.png.
    """
    base = base_dir or Path(__file__).resolve().parent
    try:
        from utils_paths import get_results_dir as _grd

        rd = Path(results_dir) if results_dir is not None else _grd(base)
    except ImportError:
        rd = Path(results_dir) if results_dir is not None else get_results_dir(base)

    scen_a, scen_b = SCENARIOS_MU1_AB
    dir_a = rd / scen_a / "images"
    dir_b = rd / scen_b / "images"
    if not dir_a.is_dir():
        print(f"[x] No existe {dir_a}")
        return False
    if not dir_b.is_dir():
        print(f"[x] No existe {dir_b}")
        return False

    map_a = _collect_fields_frames(dir_a, block)
    map_b = _collect_fields_frames(dir_b, block)
    times = sorted(set(map_a.keys()) & set(map_b.keys()))
    if not times:
        print("[x] No hay tiempos comunes entre imagenes A y B (revisa step_* en ambas carpetas).")
        return False

    if t_max is not None:
        times = [t for t in times if t <= t_max + 1e-9]
    if not times:
        print(f"[x] No quedan frames con t <= {t_max:g} (ajusta --t-max o usa --all-frames)")
        return False

    output_mp4 = Path(output_mp4)
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    n_frames = len(times)
    t_hi = f", t<={t_max:g}" if t_max is not None else ""
    print(f"Comparativa mu=1 A|B: {n_frames} frames{t_hi}, bloque {block}, fps={fps}")
    print(f"  salida: {output_mp4}")
    if progress_every > 0:
        print(f"  (avance cada {progress_every} frames)", flush=True)

    try:
        with imageio.get_writer(str(output_mp4), fps=fps) as writer:
            for idx, t in enumerate(times):
                ia = imageio.imread(str(map_a[t]))
                ib = imageio.imread(str(map_b[t]))
                frame = _pad_frame_h264_macro_block(
                    _hstack_with_labels(ia, ib, label_a, label_b)
                )
                writer.append_data(frame)
                _report_frame_progress(idx, n_frames, every=progress_every, time_val=t)
        print(f"[OK] Video guardado: {output_mp4}")
        return True
    except Exception as e:
        print(f"[x] Error al escribir video: {e}")
        return False


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description='Crea videos a partir de imágenes de campos generadas durante las simulaciones',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Crear video con parámetros específicos
  python create_videos.py --image-folder ./results/scenario1/images \\
                          --output ./results/scenario1/video.mp4 \\
                          --pattern 'fields_block_1._step_{step:.3f}.png' \\
                          --steps 400 --step-size 0.005
  
  # Crear video para un escenario específico
  python create_videos.py --scenario strong_mu1_uNo_sobre_umbral --field c --block 1
  
  # Crear video con FPS personalizado
  python create_videos.py --scenario scenario_name --fps 20

  # Video comparativo (paneles c,s,i) caso A vs B, mismo tiempo
  python create_videos.py --compare-mu1-ab --fps 12
        """
    )
    
    # Grupo de argumentos para modo manual
    parser.add_argument('--image-folder', type=str, help='Carpeta que contiene las imágenes')
    parser.add_argument('--output', type=str, help='Nombre del archivo de video de salida')
    parser.add_argument('--pattern', type=str, help='Patrón para nombres de archivos (ej: fields_block_1._step_{step:.3f}.png)')
    parser.add_argument('--steps', type=int, help='Número de pasos a incluir en el video')
    parser.add_argument('--step-size', type=float, help='Tamaño del paso entre imágenes')
    
    # Grupo de argumentos para modo escenario
    parser.add_argument('--scenario', '-s', type=str, help='Nombre del escenario (modo automático)')
    parser.add_argument('--field', type=str, choices=['c', 's', 'i'], default='c', 
                       help='Campo a visualizar: c (cáncer), s (sanas), i (inmune)')
    parser.add_argument('--block', type=int, default=1, help='Número de bloque (default: 1)')
    parser.add_argument('--results-dir', type=str, help='Directorio base de resultados (default: ./results)')
    
    # Argumentos comunes
    parser.add_argument('--fps', type=int, default=10, help='Frames por segundo del video (default: 10)')
    parser.add_argument(
        '--progress-every',
        type=int,
        default=100,
        metavar='N',
        help='Imprimir avance cada N frames (0=desactivar). Default: 100',
    )
    parser.add_argument(
        '--t-max',
        type=float,
        default=3.0,
        metavar='T',
        help='Solo incluir PNG con tiempo t<=T en el nombre step_ (default: 3). Anulado por --all-frames',
    )
    parser.add_argument(
        '--all-frames',
        action='store_true',
        help='Incluir todos los tiempos disponibles (ignora --t-max)',
    )

    parser.add_argument(
        '--compare-mu1-ab',
        action='store_true',
        help='Video lado a lado: strong_mu1_uNo_bajo_umbral vs strong_mu1_uSi_bajo_umbral (images/fields_block_*)',
    )
    parser.add_argument(
        '--output-compare',
        type=str,
        default=None,
        help='Ruta del .mp4 comparativo (default: Presentacion/figuras/campos_mu1_AB_comparativa.mp4 bajo el repo)',
    )
    
    args = parser.parse_args()
    t_max_eff = None if args.all_frames else args.t_max

    if args.compare_mu1_ab:
        if args.scenario:
            print("[x] No combines --scenario con --compare-mu1-ab")
            return 1
        base = Path(__file__).resolve().parent
        repo_root = base.parent.parent
        default_out = repo_root / "Presentacion" / "figuras" / "campos_mu1_AB_comparativa.mp4"
        out = Path(args.output_compare) if args.output_compare else default_out
        ok = create_compare_mu1_ab_fields_video(
            out,
            block=args.block,
            results_dir=args.results_dir,
            fps=args.fps,
            base_dir=base,
            progress_every=args.progress_every,
            t_max=t_max_eff,
        )
        return 0 if ok else 1
    
    # Modo escenario automático
    if args.scenario:
        return 0 if create_video_for_scenario(
            args.scenario,
            args.field,
            args.block,
            args.results_dir,
            args.fps,
            progress_every=args.progress_every,
            t_max=t_max_eff,
        ) else 1
    
    # Modo manual
    elif args.image_folder and args.output and args.pattern and args.steps and args.step_size:
        return 0 if crear_video(
            args.image_folder,
            args.output,
            args.pattern,
            args.steps,
            args.step_size,
            args.fps,
            progress_every=args.progress_every,
        ) else 1
    
    # Si no se proporcionaron argumentos suficientes
    else:
        parser.print_help()
        print("\n✗ Error: Debes proporcionar --scenario O todos los parámetros manuales (--image-folder, --output, --pattern, --steps, --step-size)")
        return 1

if __name__ == "__main__":
    sys.exit(main())

