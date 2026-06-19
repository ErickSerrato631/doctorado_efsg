"""
correlation_fourier.py

Calcula funciones de correlación cruzada y auto-correlación en el espacio de Fourier
y las transforma de vuelta al espacio real para calcular longitudes de correlación.
"""

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from scipy.ndimage import zoom
from scipy.interpolate import interp1d
import os
import sys
from pathlib import Path

# .env en la raíz Allee (el cwd suele ser el directorio del escenario al invocar desde run_scenarios)
_ALLEE_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ALLEE_ROOT / ".env")

# ============================================================================
# Configuración de directorio de trabajo
# ============================================================================

def setup_work_directory(work_dir=None):
    """
    Configura el directorio de trabajo.
    
    Args:
        work_dir (str, optional): Directorio de trabajo pasado como parámetro.
                                  Si es None, usa el directorio actual.
    
    Returns:
        str: Ruta del directorio de trabajo configurado.
    """
    if work_dir is not None:
        nueva_ruta = work_dir
        print(f"✓ Usando directorio pasado como parámetro: {nueva_ruta}")
    else:
        nueva_ruta = os.getcwd()
        print(f"✓ Ejecución manual - Usando directorio actual: {nueva_ruta}")
        print(f"  Asegúrate de estar en el directorio del escenario donde están las matrices (subcarpeta matrices/)")
    
    return nueva_ruta

# ============================================================================
# Carga de parámetros
# ============================================================================

def load_parameters():
    """Carga los parámetros desde el archivo .env."""
    T = float(os.getenv('T'))
    dt = float(os.getenv('dt'))
    nb = int(os.getenv('nb'))
    save_images = os.getenv('SAVE_IMAGES')
    
    nodes_in_xaxis = int(os.getenv('nodes_in_xaxis'))
    nodes_in_yaxis = int(os.getenv('nodes_in_yaxis'))
    space_size = float(os.getenv('space_size'))
    
    # Paso espacial coherente con los archivos guardados
    sample_rate = float(os.getenv('sample_rate', space_size / nodes_in_xaxis))
    dx = sample_rate
    
    print(f"✓ Parámetros cargados: T={T}, dt={dt}, nb={nb}, SAVE_IMAGES={save_images}")
    print(f"✓ Leyendo datos con dx={dx}, paso temporal en bucle={sample_rate}")
    
    return T, dt, nb, save_images, nodes_in_xaxis, nodes_in_yaxis, space_size, dx, sample_rate

# ============================================================================
# Validación de archivos
# ============================================================================

def validate_matrices(nueva_ruta, T, dt, nb):
    """
    Valida que existan las matrices necesarias para el procesamiento.
    
    Args:
        nueva_ruta (str): Directorio base del escenario.
        T (float): Tiempo total de simulación.
        dt (float): Paso de tiempo.
        nb (int): Número de bloques.
    
    Returns:
        bool: True si la validación es exitosa, False en caso contrario.
    """
    matrices_dir = os.path.join(nueva_ruta, 'matrices')
    expected_steps = int(T / dt) + 1
    expected_matrices_per_field = expected_steps
    expected_total_matrices = expected_steps * 3  # 3 campos: c, s, i

    print(f"\n{'='*60}")
    print("Validación de archivos de matrices")
    print(f"{'='*60}")
    print(f"Directorio de matrices: {matrices_dir}")
    print(f"Pasos de tiempo esperados: {expected_steps}")
    print(f"Matrices esperadas por campo: {expected_matrices_per_field}")
    print(f"Total de matrices esperadas: {expected_total_matrices}")

    if not os.path.exists(matrices_dir):
        error_msg = f"ERROR: El directorio de matrices no existe: {matrices_dir}"
        print(f"\n✗ {error_msg}")
        raise FileNotFoundError(error_msg)

    # Contar matrices disponibles
    matrix_files = [f for f in os.listdir(matrices_dir) if f.startswith('matrix_') and f.endswith('.txt')]
    actual_matrices = len(matrix_files)

    print(f"Matrices encontradas: {actual_matrices}")

    if actual_matrices == 0:
        error_msg = f"ERROR: No se encontraron archivos de matrices en {matrices_dir}"
        print(f"\n✗ {error_msg}")
        raise FileNotFoundError(error_msg)

    # Verificar que hay suficientes matrices (al menos 90% de las esperadas)
    if actual_matrices < expected_total_matrices * 0.9:
        warning_msg = f"ADVERTENCIA: Matrices insuficientes. Esperadas: ~{expected_total_matrices}, Encontradas: {actual_matrices}"
        print(f"\n⚠ {warning_msg}")
        print(f"  El procesamiento continuará pero puede haber pasos de tiempo faltantes.")
    else:
        print(f"\n✓ Validación exitosa: Se encontraron suficientes matrices para procesar")
        if actual_matrices != expected_total_matrices:
            print(f"  Nota: Diferencia de {expected_total_matrices - actual_matrices} matrices (esperadas vs encontradas)")

    # Verificar que existen matrices para al menos un paso de tiempo completo (c, s, i)
    field_names = ['c', 's', 'i']
    available_times = set()
    for field_name in field_names:
        for t_val in np.arange(0, T + dt, dt):
            t_str = f"{t_val:.3f}"
            matrix_file = os.path.join(matrices_dir, f"matrix_{field_name}_{t_str}_nb_{nb}.txt")
            if os.path.exists(matrix_file):
                available_times.add(t_val)

    if len(available_times) == 0:
        error_msg = f"ERROR: No se encontraron matrices válidas para ningún paso de tiempo"
        print(f"\n✗ {error_msg}")
        raise FileNotFoundError(error_msg)

    print(f"Pasos de tiempo con matrices disponibles: {len(available_times)} de {expected_steps}")
    print(f"{'='*60}\n")
    
    return True

# ============================================================================
# Funciones de procesamiento
# ============================================================================

def power_spectrum(field_name, t, block, save_image=False, save_images='N'):
    """
    Calcula el espectro de potencia 2D en el espacio de Fourier,
    pone el valor central en cero, y grafica la magnitud normalizada sin log.
    El colorbar se ajusta al alto del gráfico.
    
    Args:
        field_name (str): Nombre del campo ('c', 's', o 'i').
        t (float): Tiempo.
        block (int): Número de bloque.
        save_image (bool): Si True, guarda la imagen del espectro.
        save_images (str): 'Y' o 'N' para habilitar/deshabilitar guardado de imágenes.
    
    Returns:
        tuple: (ruta_del_archivo, magnitud_fft) o (None, None) si hay error.
    """
    # Buscar archivo en subcarpeta matrices/
    matrices_dir = os.path.join(os.getcwd(), 'matrices')
    field_file = os.path.join(matrices_dir, f"matrix_{field_name}_{t:.3f}_nb_{block}.txt")
    
    try:
        field = np.loadtxt(field_file, float)
    except FileNotFoundError:
        print(f"⚠ Archivo no encontrado: {field_file}")
        return None, None
    except Exception as e:
        print(f"Error al cargar el archivo {field_file}: {e}")
        return None, None

    # FFT 2D centrada
    fft_field = np.fft.fft2(field)
    fft_field = np.fft.fftshift(fft_field)
    fft_magnitude = np.abs(fft_field)

    # Poner el valor central en cero
    cx, cy = fft_magnitude.shape[0] // 2, fft_magnitude.shape[1] // 2
    fft_magnitude[cx, cy] = 0

    # Normalizar entre 0 y 1
    max_val = np.max(fft_magnitude)
    if max_val > 0:
        fft_magnitude /= max_val

    # Guardar imagen solo si está habilitado
    if save_image and save_images == 'Y':
        try:
            # Crear directorio de imágenes si no existe
            images_dir = os.path.join(os.getcwd(), 'images')
            os.makedirs(images_dir, exist_ok=True)
            
            # Crear figura con ejes personalizados
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(fft_magnitude, cmap='inferno', origin='lower')

            # Colorbar ajustado al alto de la imagen
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Normalized Magnitude")

            ax.set_title(f"Power Spectrum of {field_name} at t={t:.3f}, block={block}")
            ax.set_xlabel("Frequency X")
            ax.set_ylabel("Frequency Y")

            plt.tight_layout()
            image_path = os.path.join(images_dir, f'power_spectrum_{field_name}_{t:.3f}_nb_{block}.png')
            plt.savefig(image_path, dpi=300)
            plt.close()
        except Exception as save_error:
            # Si falla el guardado de imagen, continuar sin detener el análisis
            print(f"⚠ ADVERTENCIA: Error al guardar imagen de espectro de potencia para {field_name} en t={t:.3f}: {save_error}")
            print(f"  Continuando con el análisis...")
            try:
                plt.close()
            except:
                pass  # Ignorar errores al cerrar matplotlib

    return field_file, fft_magnitude

def correlation_function_fourier(dft_field1, dft_field2, field1_name, field2_name, step, block):
    """
    Calcula la correlación cruzada en el espacio de Fourier, sin transformarla al espacio real.
    Anula el máximo antes de graficar para mejorar la visualización.
    
    Args:
        dft_field1 (np.ndarray): Primer campo en el espacio de Fourier.
        dft_field2 (np.ndarray): Segundo campo en el espacio de Fourier.
        field1_name (str): Nombre del primer campo.
        field2_name (str): Nombre del segundo campo.
        step (float): Paso de tiempo.
        block (int): Número de bloque.
    
    Returns:
        np.ndarray: Función de correlación en el espacio de Fourier o None si hay error.
    """
    # Verificar que ambos campos sean válidos
    if dft_field1 is None or dft_field2 is None:
        print(f"⚠ Error: Uno de los campos es None para {field1_name} vs {field2_name} en t={step}")
        return None
    
    fft1 = dft_field1
    fft2 = dft_field2
    
    # Asegurar que tengan el mismo tamaño
    min_shape = (min(fft1.shape[0], fft2.shape[0]), min(fft1.shape[1], fft2.shape[1]))
    fft1 = fft1[:min_shape[0], :min_shape[1]]
    fft2 = fft2[:min_shape[0], :min_shape[1]]

    # Multiplicar en el dominio de Fourier
    correlation_fourier = fft1 * np.conjugate(fft2)

    # Anular el valor máximo
    max_value = np.max(correlation_fourier)
    if max_value > 0:
        correlation_fourier[correlation_fourier == max_value] = 0

    return correlation_fourier

def inverse_correlation_fft(centered_correlation, field1_name, field2_name, step, block):
    """
    Revierte el desplazamiento de la función de correlación en Fourier y la transforma de vuelta al espacio real.
    También grafica el campo centrado antes de la transformación inversa.

    Args:
        centered_correlation (np.ndarray): Función de correlación en el espacio de Fourier.
        field1_name (str): Nombre del primer campo.
        field2_name (str): Nombre del segundo campo.
        step (float): Paso de tiempo.
        block (int): Número de bloque.

    Returns:
        np.ndarray: Transformada inversa de la correlación en el espacio real o None si hay error.
    """
    # Verificar que la correlación sea válida
    if centered_correlation is None:
        print(f"⚠ Error: Correlación es None para {field1_name} vs {field2_name} en t={step}")
        return None
    
    # Aplicar la transformada inversa de Fourier y centrar el resultado
    inverse_fft = np.fft.ifft2(np.fft.ifftshift(centered_correlation)).real  # Obtener solo la parte real
    inverse_fft = np.fft.fftshift(inverse_fft)  # Recentrar la imagen en el dominio espacial

    # Verificar valores antes de normalizar
    min_val = np.min(inverse_fft)
    max_val = np.max(inverse_fft)
    
    # Normalizar entre 0 y 1
    if max_val - min_val > 1e-8:
        inverse_fft -= min_val  # Asegurar que el mínimo sea 0
        inverse_fft /= (max_val - min_val)  # Normalizar evitando división por 0
    else:
        # Si los valores son muy pequeños, retornar array de ceros
        inverse_fft = np.zeros_like(inverse_fft)
    
    return inverse_fft

def compute_correlation_length(centered_inverse_fft, dx):
    """
    Calcula la longitud de correlación a partir de un promedio radial de la transformada inversa centrada.
    
    Args:
        centered_inverse_fft (np.ndarray): Transformada inversa de la correlación centrada.
        dx (float): Resolución espacial en unidades físicas.
    
    Returns:
        float: Longitud de correlación en unidades físicas.
    """
    # Verificar que el array sea válido
    if centered_inverse_fft is None:
        print("⚠ Error: centered_inverse_fft es None")
        return 0.0
    
    ny, nx = centered_inverse_fft.shape
    x_center, y_center = nx // 2, ny // 2
    
    y, x = np.indices((ny, nx))
    r = np.sqrt((x - x_center)**2 + (y - y_center)**2).astype(int)
    
    r_max = r.max()
    radial_profile = np.zeros(r_max + 1)
    counts = np.zeros(r_max + 1)
    
    for i in range(ny):
        for j in range(nx):
            radial_profile[r[i, j]] += centered_inverse_fft[i, j]
            counts[r[i, j]] += 1
    
    radial_profile /= np.where(counts == 0, 1, counts)
    
    max_value = radial_profile[0]
    if max_value <= 0:
        print("⚠ Error: max_value <= 0 en radial_profile")
        return 0.0
    
    threshold = max_value / np.e
    correlation_length_pixels = np.argmax(radial_profile < threshold)
    
    correlation_length_real = correlation_length_pixels * dx  # Conversión a unidades físicas
    
    print('Longitud de correlación en unidades reales:', correlation_length_real)
    
    return correlation_length_real

def save_corr_real_array(array, block, field1_name, field2_name):
    """
    Guarda el array de longitudes de correlación en un archivo de texto.
    
    Args:
        array (list): Lista de [tiempo, longitud_correlacion].
        block (int): Número de bloque.
        field1_name (str): Nombre del primer campo.
        field2_name (str): Nombre del segundo campo.
    """
    length_corr = np.array(array)
    
    # Crear directorio de correlaciones si no existe
    correlations_dir = os.path.join(os.getcwd(), 'correlations')
    os.makedirs(correlations_dir, exist_ok=True)
    
    filename = f'corr_length_real_inverse_nb_{block}_{field1_name}_{field2_name}.txt'
    filepath = os.path.join(correlations_dir, filename)
    
    np.savetxt(
        filepath,
        length_corr,
        fmt="%.5f",  # Mayor precisión
        delimiter="\t",
        header="Tiempo\tLongitud_Correlacion"
    )

# ============================================================================
# Función principal de procesamiento
# ============================================================================

def process_correlations(
    nueva_ruta, T, dt, nb, dx, save_images, t_step=None,
    fieldC_name="c", fieldS_name="s", fieldI_name="i",
):
    """
    Procesa todas las correlaciones para todos los bloques y pasos de tiempo.
    
    Args:
        nueva_ruta (str): Directorio base del escenario.
        T (float): Tiempo total de simulación.
        dt (float): Paso de tiempo.
        nb (int): Número de bloques.
        dx (float): Resolución espacial.
        save_images (str): 'Y' o 'N' para habilitar/deshabilitar guardado de imágenes.
        t_step (float, optional): Incremento del bucle temporal; por defecto ``sample_rate`` o ``dt``.
        fieldC_name (str): Nombre del campo de cáncer.
        fieldS_name (str): Nombre del campo de células sanas.
        fieldI_name (str): Nombre del campo inmune.
    """
    if t_step is None:
        t_step = dt
    # Cambiar al directorio del escenario
    os.chdir(nueva_ruta)
    print(f"✓ Cambiando a directorio: {os.getcwd()}")
    
    for block in range(1, nb + 1):
        length_corr_cs = []
        length_corr_ci = []
        length_corr_si = []
        length_corr_cc = []
        length_corr_ss = []
        length_corr_ii = []
        t = 0

        while t <= T:
            print("block=", block, 'tiempo', t)
            # print('Espectro de potencias')
            field_c, dft_field_c = power_spectrum(fieldC_name, t, block, save_image=False, save_images=save_images)
            field_s, dft_field_s = power_spectrum(fieldS_name, t, block, save_image=False, save_images=save_images)
            field_i, dft_field_i = power_spectrum(fieldI_name, t, block, save_image=False, save_images=save_images)

            # Si algún campo no se pudo cargar, saltar este paso de tiempo
            if dft_field_c is None or dft_field_s is None or dft_field_i is None:
                print(f"⚠ Saltando t={t:.3f} debido a archivos faltantes")
                t += t_step
                continue

            print('Calculando funciones de correlación cruzada')
            
            correlation_func_cs = correlation_function_fourier(dft_field_c, dft_field_s, fieldC_name, fieldS_name, t, block)
            correlation_func_ci = correlation_function_fourier(dft_field_c, dft_field_i, fieldC_name, fieldI_name, t, block)
            correlation_func_si = correlation_function_fourier(dft_field_s, dft_field_i, fieldS_name, fieldI_name, t, block)

            print('Calculando funciones de auto correlación')        
            correlation_func_cc = correlation_function_fourier(dft_field_c, dft_field_c, fieldC_name, fieldC_name, t, block)
            correlation_func_ss = correlation_function_fourier(dft_field_s, dft_field_s, fieldS_name, fieldS_name, t, block)
            correlation_func_ii = correlation_function_fourier(dft_field_i, dft_field_i, fieldI_name, fieldI_name, t, block)
            
            # Verificar que las correlaciones sean válidas
            if (correlation_func_cs is None or correlation_func_ci is None or correlation_func_si is None or
                correlation_func_cc is None or correlation_func_ss is None or correlation_func_ii is None):
                print(f"⚠ Saltando t={t:.3f} debido a errores en correlaciones")
                t += t_step
                continue

            print('Obteniendo funciones de correlación cruzada en espacio real')
            inverse_fft_real_cs = inverse_correlation_fft(correlation_func_cs, fieldC_name, fieldS_name, t, block)
            inverse_fft_real_ci = inverse_correlation_fft(correlation_func_ci, fieldC_name, fieldI_name, t, block)
            inverse_fft_real_si = inverse_correlation_fft(correlation_func_si, fieldS_name, fieldI_name, t, block)
            

            print('Obteniendo funciones de auto correlación en espacio real')
            inverse_fft_real_cc = inverse_correlation_fft(correlation_func_cc, fieldC_name, fieldC_name, t, block)
            inverse_fft_real_ss = inverse_correlation_fft(correlation_func_ss, fieldS_name, fieldS_name, t, block)
            inverse_fft_real_ii = inverse_correlation_fft(correlation_func_ii, fieldI_name, fieldI_name, t, block)       
            
            # Verificar que las transformadas inversas sean válidas
            if (inverse_fft_real_cs is None or inverse_fft_real_ci is None or inverse_fft_real_si is None or
                inverse_fft_real_cc is None or inverse_fft_real_ss is None or inverse_fft_real_ii is None):
                print(f"⚠ Saltando t={t:.3f} debido a errores en transformadas inversas")
                t += t_step
                continue

            print('Obteniendo longitudes de correlación cruzada en espacio real')        
            correlation_length_cs = compute_correlation_length(inverse_fft_real_cs, dx)
            length_corr_cs.append([t, correlation_length_cs])
            
            correlation_length_ci = compute_correlation_length(inverse_fft_real_ci, dx)
            length_corr_ci.append([t, correlation_length_ci])

            correlation_length_si = compute_correlation_length(inverse_fft_real_si, dx)
            length_corr_si.append([t, correlation_length_si])
            
            
            print('Obteniendo longitudes de auto correlación en espacio real')
            correlation_length_cc = compute_correlation_length(inverse_fft_real_cc, dx)
            length_corr_cc.append([t, correlation_length_cc])
            
            correlation_length_ss = compute_correlation_length(inverse_fft_real_ss, dx)
            length_corr_ss.append([t, correlation_length_ss])

            correlation_length_ii = compute_correlation_length(inverse_fft_real_ii, dx)
            length_corr_ii.append([t, correlation_length_ii])
            
            print('--------------------------------------------------------------------')
            
            t += t_step
        
        # Guardar resultados del bloque
        save_corr_real_array(length_corr_cs, block, fieldC_name, fieldS_name)
        save_corr_real_array(length_corr_ci, block, fieldC_name, fieldI_name)
        save_corr_real_array(length_corr_si, block, fieldS_name, fieldI_name)
        save_corr_real_array(length_corr_cc, block, fieldC_name, fieldC_name)
        save_corr_real_array(length_corr_ss, block, fieldS_name, fieldS_name)
        save_corr_real_array(length_corr_ii, block, fieldI_name, fieldI_name)

# ============================================================================
# Punto de entrada principal
# ============================================================================

if __name__ == "__main__":
    # Obtener work_dir de los argumentos de línea de comandos si está disponible
    work_dir = None
    if len(sys.argv) > 1:
        work_dir = sys.argv[1]
    
    # Configurar directorio de trabajo
    nueva_ruta = setup_work_directory(work_dir)
    
    # Cargar parámetros
    T, dt, nb, save_images, nodes_in_xaxis, nodes_in_yaxis, space_size, dx, t_step = load_parameters()
    
    # Validar matrices
    validate_matrices(nueva_ruta, T, dt, nb)
    
    # Procesar correlaciones
    process_correlations(nueva_ruta, T, dt, nb, dx, save_images, t_step=t_step)



