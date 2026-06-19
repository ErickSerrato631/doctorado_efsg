# Guía Completa de Configuración y Uso - Simulaciones con FEniCSx

Esta guía te ayudará a configurar y usar el sistema completo de simulaciones de dinámicas de cáncer con FEniCSx.

**Post-proceso termodinámico** (`Allee/termodinámica/calculate_thermodynamic_properties.py`, estado de corridas, comandos WSL): ver **[Allee/README.md](Allee/README.md)**.

---

## 📋 Tabla de Contenidos

1. [Configuración Inicial](#configuración-inicial)
2. [Configuración de Google Drive](#configuración-de-google-drive)
3. [Scripts Principales](#scripts-principales)
4. [Guía de Uso](#guía-de-uso)
5. [Comandos desde PowerShell](#comandos-desde-powershell)
6. [Solución de Problemas](#solución-de-problemas)

---

## 🚀 Configuración Inicial

### Paso 1: Instalar WSL2 con Ubuntu

En PowerShell de Windows (como Administrador):

```powershell
# Verificar si WSL está instalado
wsl -l -v

# Si no está instalado, instalar WSL2
wsl --install -d Ubuntu

# Reiniciar el sistema si es necesario
```

### Paso 2: Instalar Miniconda en WSL

Abre WSL (Ubuntu) y ejecuta:

```bash
# Descargar Miniconda
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Instalar Miniconda
bash Miniconda3-latest-Linux-x86_64.sh

# Seguir las instrucciones del instalador
# Reiniciar la terminal después de instalar

# Verificar instalación
conda --version
```

### Paso 3: Crear Entorno Conda con FEniCSx

El stack recomendado es **un solo entorno `fenicsx-env`** (conda-forge): FEniCSx, MPI, PyVista y las librerías científicas que usan las simulaciones y el post-proceso (`steady_states/`, Jupyter, etc.). La definición está en **[environment.yml](environment.yml)** del directorio `Models/`.

En WSL:

```bash
# Activar conda (ajusta la ruta si usas Anaconda en lugar de Miniconda)
source ~/miniconda3/etc/profile.d/conda.sh
# o: source ~/anaconda3/etc/profile.d/conda.sh

cd /mnt/c/Users/Erick\ Serrato/Documents/Doctorado/doctorado_efsg-main/Models

# Primera instalación (crea fenicsx-env según environment.yml)
conda env create -f environment.yml

# Si el entorno ya existe y quieres sincronizar paquetes con el archivo:
# conda env update -f environment.yml --prune

conda activate fenicsx-env

# Verificar instalación
python -c "import dolfinx; print(f'FEniCSx version: {dolfinx.__version__}')"
```

**Alternativa manual** (equivalente al contenido de `environment.yml`): `conda create -n fenicsx-env python=3.11 -y`, activar, luego `conda install -c conda-forge fenics-dolfinx mpich pyvista numpy scipy matplotlib pandas sympy jupyterlab imageio python-dotenv papermill -y`.

> **`requirements.txt`** incluye también (como referencia pip) `numpy`, `scipy`, `pandas`, `sympy`, `matplotlib`, `jupyterlab`, `imageio`, `python-dotenv`, `papermill`, etc., alineados con lo que pide `steady_states/` y el post-proceso. **FEniCSx y MPI** siguen yendo solo por **`environment.yml`** (conda-forge). No sustituyas el entorno conda con un `pip install -r requirements.txt` completo sin revisar conflictos; si falta un paquete puntual dentro de `fenicsx-env`, puedes instalarlo con pip de forma selectiva.

### Paso 4: Verificar Instalación

Verifica que todo está configurado correctamente ejecutando estos comandos en WSL:

```bash
# Activar conda
source ~/miniconda3/etc/profile.d/conda.sh

# Activar entorno
conda activate fenicsx-env

# Verificar Python
python --version

# Verificar FEniCSx
python -c "import dolfinx; print(f'FEniCSx version: {dolfinx.__version__}')"

# Verificar dependencias principales (incl. análisis steady_states)
python -c "import numpy, matplotlib, scipy, pandas, sympy, jupyterlab; print('✓ Dependencias principales OK')"

# Verificar acceso al proyecto
cd /mnt/c/Users/Erick\ Serrato/Documents/Doctorado/doctorado_efsg-main/Models/Allee
ls -la run_scenarios.py

# Verificar puerto 8888 (si planeas usar Jupyter)
lsof -Pi :8888 -sTCP:LISTEN || echo "✓ Puerto 8888 disponible"
```

Si todos los comandos se ejecutan sin errores, la instalación está correcta.

---

## ☁️ Configuración de Google Drive

### Paso 1: Instalar rclone

En WSL, ejecuta estos comandos:

```bash
# Actualizar paquetes
sudo apt-get update

# Instalar rclone
sudo apt-get install -y rclone

# Verificar instalación
rclone version
```

### Paso 2: Configurar rclone con Google Drive

Ejecuta el comando de configuración:

```bash
rclone config
```

Durante la configuración interactiva, sigue estos pasos:

1. **Crear nuevo remote**: Presiona `n` y luego Enter
2. **Nombre del remote**: Escribe `gdrive` y presiona Enter
3. **Tipo de almacenamiento**: Escribe `drive` (número 15) y presiona Enter
4. **Client ID y Client Secret**: Presiona Enter para usar los valores por defecto (dos veces)
5. **Scope**: Selecciona `1` (acceso completo a Google Drive) y presiona Enter
6. **Root folder ID**: Presiona Enter para usar el valor por defecto
7. **Service Account File**: Presiona Enter para omitir
8. **Advanced config**: Presiona `n` y Enter
9. **Auto config**: Presiona `y` y Enter (se abrirá tu navegador para autenticarte)
10. **Autenticación**: Sigue las instrucciones en el navegador para autorizar el acceso
11. **Verificar configuración**: Presiona `y` y Enter
12. **Guardar**: Presiona `q` y Enter para salir

### Paso 3: Crear Directorio de Montaje

```bash
# Crear directorio donde se montará Google Drive
mkdir -p ~/googledrive

# Verificar que se creó
ls -la ~/googledrive
```

### Paso 4: Montar Google Drive

Ejecuta este comando para montar Google Drive (o usa el script `Models/Allee/mount_google_drive.sh`):

```bash
rclone mount gdrive: ~/googledrive \
    --daemon \
    --vfs-cache-mode writes \
    --vfs-cache-max-size 10G \
    --vfs-read-chunk-size 128M \
    --vfs-read-chunk-size-limit off \
    --allow-other \
    --umask 000
```

**Explicación de parámetros:**
- `--daemon`: Ejecuta en segundo plano
- `--vfs-cache-mode writes`: Cachea escrituras para mejor rendimiento
- `--vfs-cache-max-size 10G`: Límite de caché de 10GB
- `--vfs-read-chunk-size 128M`: Tamaño de lectura optimizado
- `--allow-other`: Permite acceso a otros usuarios
- `--umask 000`: Permisos permisivos para escritura

Espera unos segundos y verifica que está montado:

```bash
# Verificar montaje
mountpoint ~/googledrive

# Si está montado, deberías ver el contenido
ls ~/googledrive
```

Google Drive estará disponible en: `~/googledrive`

**Nota importante**: Los resultados se guardarán automáticamente en:
```
~/googledrive/Doctorado Erick Serrato/Resultados Paper/<escenario>/
```

### Paso 3: Verificar Montaje

```bash
# Verificar que está montado
mountpoint ~/googledrive

# Listar contenido
ls ~/googledrive

# Verificar que existe la carpeta de resultados
ls ~/googledrive/Doctorado\ Erick\ Serrato/Resultados\ Paper/
```

### Paso 5: Configurar Montaje Automático (Opcional)

Para que Google Drive se monte automáticamente cada vez que inicies WSL, agrega estos comandos a tu archivo `~/.bashrc`:

```bash
# Abrir el archivo .bashrc en un editor
nano ~/.bashrc

# O usar vim
vim ~/.bashrc
```

Agrega estas líneas al final del archivo:

```bash
# Montar Google Drive automáticamente si no está montado
if ! mountpoint -q ~/googledrive 2>/dev/null; then
    echo "Montando Google Drive..."
    rclone mount gdrive: ~/googledrive \
        --daemon \
        --vfs-cache-mode writes \
        --vfs-cache-max-size 10G \
        --vfs-read-chunk-size 128M \
        --vfs-read-chunk-size-limit off \
        --allow-other \
        --umask 000 2>/dev/null
    sleep 2
    if mountpoint -q ~/googledrive 2>/dev/null; then
        echo "✓ Google Drive montado en ~/googledrive"
    else
        echo "⚠ No se pudo montar Google Drive automáticamente"
    fi
fi
```

**Instrucciones para guardar:**
- En **nano**: Presiona `Ctrl+O` (guardar), Enter (confirmar), `Ctrl+X` (salir)
- En **vim**: Presiona `Esc`, luego escribe `:wq` y Enter

**Para aplicar los cambios:**
```bash
# Recargar .bashrc
source ~/.bashrc

# O simplemente cerrar y abrir una nueva terminal WSL
```

Ahora Google Drive se montará automáticamente cada vez que inicies WSL.

**Nota**: Si prefieres montar manualmente cada vez, puedes omitir este paso y ejecutar el comando de montaje del Paso 4 cuando lo necesites.

---

## 📜 Scripts Principales

### `run_scenarios.py` - Script Principal de Ejecución

**¿Qué hace?**
- Ejecuta simulaciones de todos los escenarios definidos en `scenarios.json`
- Gestiona ejecución en lotes para controlar espacio en disco
- Maneja sistema de checkpoint/restart automático
- Ejecuta análisis de correlaciones después de cada simulación
- Guarda resultados automáticamente en Google Drive si está montado

**Archivos que ejecuta:**
- `cancer_dynamics.py` - Simulación principal de dinámicas de cáncer
- `correlations/correlation_fourier.py` - Análisis de correlaciones en espacio de Fourier

**Ubicación**: `Models/Allee/run_scenarios.py`

### `cancer_dynamics.py` - Simulación Principal

**¿Qué hace?**
- Resuelve las ecuaciones diferenciales parciales del modelo de cáncer
- Genera matrices de campos (c, s, i) en cada paso de tiempo
- Guarda imágenes de visualización (si `SAVE_IMAGES=Y`)
- Maneja memoria y checkpoints automáticamente
- Soporta efecto Allee débil y fuerte
- Soporta control adaptativo opcional

**Salidas:**
- `matrices/matrix_<campo>_<tiempo>_nb_<bloque>.txt` - Matrices de campos
- `images/fields_block_<bloque>_step_<tiempo>.png` - Imágenes de visualización
- `checkpoints/checkpoint_latest.npz` - Checkpoints para reinicio

**Ubicación**: `Models/Allee/cancer_dynamics.py`

### `correlations/correlation_fourier.py` - Análisis de Correlaciones

**¿Qué hace?**
- Calcula funciones de correlación cruzada entre campos (c-s, c-i, s-i)
- Calcula funciones de auto-correlación (c-c, s-s, i-i)
- Calcula longitudes de correlación en espacio real
- Usa transformadas de Fourier para eficiencia

**Salidas:**
- `correlations/corr_length_real_inverse_nb_<bloque>_<campo1>_<campo2>.txt` - Longitudes de correlación

**Ubicación**: `Models/Allee/correlations/correlation_fourier.py`

Post-proceso comparativo (grids ξ(t)): `Models/Allee/correlations/correlation_comparison.py` (ejecutar desde `Allee`: `python correlations/correlation_comparison.py`).

### `create_videos.py` - Creación de Videos

**¿Qué hace?**
- Crea videos MP4 a partir de imágenes secuenciales
- Combina imágenes de campos en animaciones
- Útil para visualizar la evolución temporal

**Ubicación**: `Models/Allee/create_videos.py`

### `utils_paths.py` - Utilidades de Rutas

**¿Qué hace?**
- Detecta automáticamente Google Drive
- Determina dónde guardar resultados (Google Drive o local)
- Verifica acceso a directorios

**Ubicación**: `Models/Allee/utils_paths.py`

---

## 📖 Guía de Uso

### Ejecución Básica

#### Desde WSL (recomendado):

```bash
# Activar entorno
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fenicsx-env

# Ir al directorio
cd /mnt/c/Users/Erick\ Serrato/Documents/Doctorado/doctorado_efsg-main/Models/Allee

# Listar escenarios disponibles
python run_scenarios.py --list

# Ejecutar todos los escenarios
python run_scenarios.py

# Ejecutar un escenario específico
python run_scenarios.py --scenario strong_mu1_uNo_sobre_umbral
```

#### Desde PowerShell:

Ver sección [Comandos desde PowerShell](#comandos-desde-powershell) más abajo.

### Modos de Ejecución

#### 1. Ejecución Normal (todos los escenarios)

```bash
python run_scenarios.py
```

**Qué hace:**
- Ejecuta todos los escenarios definidos en `scenarios.json`
- Guarda resultados en Google Drive (si está montado) o localmente
- Ejecuta análisis de correlaciones después de cada simulación
- Genera logs en `logs/run_scenarios_YYYYMMDD_HHMMSS.log`

#### 2. Ejecutar Escenario Específico

```bash
python run_scenarios.py --scenario nombre_del_escenario
```

**Cuándo usar:**
- Para probar un escenario específico
- Para re-ejecutar un escenario que falló
- Para ejecutar solo un escenario de interés

#### 3. Modo Lotes (Gestión de Espacio)

```bash
# Ejecutar en modo lotes automático (ejecuta todos los lotes secuencialmente)
python run_scenarios.py --batch-mode --yes

# Ejecutar un lote específico (ejemplo: lote de 9 escenarios empezando desde el índice 0)
python run_scenarios.py --batch-size 9 --batch-start 0

# Ejecutar siguiente lote
python run_scenarios.py --batch-size 9 --batch-start 9
```

**Cuándo usar:**
- Cuando tienes espacio limitado en disco
- Para ejecutar simulaciones en etapas
- Para gestionar mejor el almacenamiento

#### 4. Re-ejecutar Escenarios Fallidos

```bash
# Verificar estado de escenarios
python run_scenarios.py --status

# Re-ejecutar solo los fallidos (continúa desde checkpoints)
python run_scenarios.py --retry-failed --yes

# Re-ejecutar desde cero (limpia checkpoints)
python run_scenarios.py --retry-failed --clean --from-zero --yes
```

**Cuándo usar:**
- Después de interrupciones
- Cuando algunos escenarios no completaron
- Para continuar desde checkpoints guardados

#### 5. Verificar Estado sin Ejecutar

```bash
python run_scenarios.py --status
```

**Qué muestra:**
- Estado de cada escenario (COMPLETO, PARCIAL, INCOMPLETO, FALLIDO, NO INICIADO)
- Porcentaje de completitud
- Número de matrices y correlaciones generadas

#### 6. Limpiar Resultados

```bash
# Limpiar un escenario específico antes de ejecutar
python run_scenarios.py --scenario nombre --clean

# Limpiar todos excepto uno específico
python run_scenarios.py --clean-all-except nombre_del_escenario_a_mantener --yes
```

### Opciones Adicionales

```bash
# Desactivar sistema de checkpoint/restart
python run_scenarios.py --no-checkpoint

# Ejecutar sin confirmación interactiva (útil para scripts)
python run_scenarios.py --yes

# Configurar máximo de reinicios
python run_scenarios.py --max-restarts 100
```

---

## 💻 Comandos desde PowerShell

### ⚠️ Antes de ejecutar escenarios: montar Google Drive (rclone)

Si quieres que los resultados se guarden en **Google Drive** (`~/googledrive/Doctorado Erick Serrato/Resultados Paper/`), **debes montar Drive en WSL antes** de lanzar `run_scenarios.py`. Si no está montado, los resultados se guardan en local (`Models/Allee/results/`) y no se suben a Drive.

**En WSL (Ubuntu), una vez por sesión:**

```bash
cd /mnt/c/Users/Erick\ Serrato/Documents/Doctorado/doctorado_efsg-main/Models/Allee
bash mount_google_drive.sh
```

O desde PowerShell (ejecuta esto antes de los comandos de abajo):

```powershell
wsl -d Ubuntu -e bash -c "cd '/mnt/c/Users/Erick Serrato/Documents/Doctorado/doctorado_efsg-main/Models/Allee' && bash mount_google_drive.sh"
```

Comprueba que está montado: `wsl -d Ubuntu -e bash -c "mountpoint ~/googledrive"` (debe salir que es un mount point).

### Comando Base (Una Línea)

```powershell
wsl -d Ubuntu -e bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate fenicsx-env && cd '/mnt/c/Users/Erick Serrato/Documents/Doctorado/doctorado_efsg-main/Models/Allee' && python run_scenarios.py"
```

### Ejemplos Específicos

#### 1. Listar Escenarios Disponibles

```powershell
wsl -d Ubuntu -e bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate fenicsx-env && cd '/mnt/c/Users/Erick Serrato/Documents/Doctorado/doctorado_efsg-main/Models/Allee' && python run_scenarios.py --list"
```

#### 2. Verificar Estado de Escenarios

```powershell
wsl -d Ubuntu -e bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate fenicsx-env && cd '/mnt/c/Users/Erick Serrato/Documents/Doctorado/doctorado_efsg-main/Models/Allee' && python run_scenarios.py --status"
```

#### 3. Ejecutar Escenario Específico

```powershell
wsl -d Ubuntu -e bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate fenicsx-env && cd '/mnt/c/Users/Erick Serrato/Documents/Doctorado/doctorado_efsg-main/Models/Allee' && python run_scenarios.py --scenario strong_mu1_uNo_sobre_umbral"
```

#### 4. Ejecutar Todos los Escenarios

```powershell
wsl -d Ubuntu -e bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate fenicsx-env && cd '/mnt/c/Users/Erick Serrato/Documents/Doctorado/doctorado_efsg-main/Models/Allee' && python run_scenarios.py"
```

#### 5. Re-ejecutar Escenarios Fallidos

```powershell
wsl -d Ubuntu -e bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate fenicsx-env && cd '/mnt/c/Users/Erick Serrato/Documents/Doctorado/doctorado_efsg-main/Models/Allee' && python run_scenarios.py --retry-failed --yes"
```

#### 6. Modo Lotes Automático

```powershell
wsl -d Ubuntu -e bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate fenicsx-env && cd '/mnt/c/Users/Erick Serrato/Documents/Doctorado/doctorado_efsg-main/Models/Allee' && python run_scenarios.py --batch-mode --yes"
```

#### 7. Limpiar Todos Excepto Uno

```powershell
wsl -d Ubuntu -e bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate fenicsx-env && cd '/mnt/c/Users/Erick Serrato/Documents/Doctorado/doctorado_efsg-main/Models/Allee' && python run_scenarios.py --clean-all-except nombre_escenario --yes"
```

### Script PowerShell Reutilizable

Puedes crear un archivo `run_scenarios.ps1` con este contenido:

```powershell
# run_scenarios.ps1
$argsString = $args -join " "
wsl -d Ubuntu -e bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate fenicsx-env && cd '/mnt/c/Users/Erick Serrato/Documents/Doctorado/doctorado_efsg-main/Models/Allee' && python run_scenarios.py $argsString"
```

Luego úsalo así:

```powershell
.\run_scenarios.ps1 --list
.\run_scenarios.ps1 --status
.\run_scenarios.ps1 --scenario nombre_escenario
.\run_scenarios.ps1 --retry-failed --yes
```

---

## 🔧 Otros Scripts Útiles

### `create_videos.py` - Crear Videos

**Uso desde WSL:**

```bash
# Crear video para un escenario específico
python create_videos.py --scenario nombre_escenario --field c --block 1

# Con FPS personalizado
python create_videos.py --scenario nombre_escenario --fps 20
```

**Uso desde PowerShell:**

```powershell
wsl -d Ubuntu -e bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate fenicsx-env && cd '/mnt/c/Users/Erick Serrato/Documents/Doctorado/doctorado_efsg-main/Models/Allee' && python create_videos.py --scenario nombre_escenario --field c"
```

### `check_scenarios_status.py` - Verificar Estado

**Uso:**

```bash
python check_scenarios_status.py
```

**Qué muestra:**
- Estado detallado de cada escenario
- Porcentajes de completitud
- Resumen por categorías

### `manage_batches.py` - Gestión de Lotes

**Uso:**

```bash
python manage_batches.py
```

**Qué hace:**
- Calcula tamaño óptimo de lotes según espacio disponible
- Muestra información sobre lotes
- Útil para planificar ejecuciones

---

## 📁 Estructura de Directorios

```
Models/Allee/
├── run_scenarios.py          # Script principal de ejecución
├── cancer_dynamics.py        # Simulación principal
├── correlations/             # Correlaciones y Fourier
│   ├── correlation_fourier.py
│   └── correlation_comparison.py
├── create_videos.py          # Creación de videos
├── utils_paths.py            # Utilidades de rutas
├── scenarios.json            # Configuración de escenarios
├── .env                      # Parámetros de simulación (se genera automáticamente)
├── logs/                     # Logs de ejecución
└── results/                  # Resultados locales (si Google Drive no está montado)

Google Drive (cuando está montado):
~/googledrive/
└── Doctorado Erick Serrato/
    └── Resultados Paper/
        ├── escenario_1/
        │   ├── matrices/          # Matrices de campos
        │   ├── images/            # Imágenes de visualización
        │   ├── correlations/      # Archivos de correlación
        │   └── checkpoints/       # Checkpoints para reinicio
        └── escenario_2/
            └── ...
```

---

## 🔍 Verificación y Diagnóstico

### Verificar que Todo Está Configurado Correctamente

Ejecuta estos comandos en WSL para verificar cada componente:

```bash
# 1. Verificar WSL (deberías estar ejecutando esto en WSL)
uname -a | grep -i microsoft && echo "✓ WSL detectado" || echo "⚠ No parece ser WSL"

# 2. Verificar Conda
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    echo "✓ Conda encontrado"
else
    echo "✗ Conda no encontrado"
fi

# 3. Verificar entorno fenicsx-env
conda env list | grep fenicsx-env && echo "✓ Entorno fenicsx-env existe" || echo "✗ Entorno no existe"

# 4. Activar entorno y verificar Python
conda activate fenicsx-env
python --version && echo "✓ Python OK"

# 5. Verificar FEniCSx
python -c "import dolfinx; print(f'✓ FEniCSx version: {dolfinx.__version__}')" || echo "✗ FEniCSx no instalado"

# 6. Verificar dependencias principales
python -c "import numpy, matplotlib, scipy, jupyterlab, dotenv; print('✓ Dependencias principales OK')" || echo "✗ Faltan dependencias"

# 7. Verificar acceso al proyecto
cd /mnt/c/Users/Erick\ Serrato/Documents/Doctorado/doctorado_efsg-main/Models/Allee
[ -f run_scenarios.py ] && echo "✓ Proyecto accesible" || echo "✗ Proyecto no accesible"

# 8. Verificar puerto 8888 (para Jupyter)
lsof -Pi :8888 -sTCP:LISTEN >/dev/null 2>&1 && echo "⚠ Puerto 8888 en uso" || echo "✓ Puerto 8888 disponible"
```

**Qué verifica cada paso:**
- ✅ Instalación de WSL
- ✅ Instalación de Conda
- ✅ Entorno fenicsx-env
- ✅ Instalación de FEniCSx
- ✅ Instalación de dependencias
- ✅ Acceso a archivos del proyecto
- ✅ Puerto 8888 disponible

### Verificar Google Drive

```bash
# Verificar que está montado
mountpoint ~/googledrive

# Verificar estructura de carpetas
ls -la ~/googledrive/Doctorado\ Erick\ Serrato/Resultados\ Paper/

# Verificar acceso de escritura
touch ~/googledrive/Doctorado\ Erick\ Serrato/Resultados\ Paper/.test
rm ~/googledrive/Doctorado\ Erick\ Serrato/Resultados\ Paper/.test
```

---

## 🐛 Solución de Problemas

### Error: "WSL no está disponible"

```powershell
# Instalar WSL2
wsl --install -d Ubuntu

# Reiniciar el sistema
```

### Error: "conda: command not found"

```bash
# En WSL, agregar conda al PATH
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Error: "fenicsx-env no existe"

```bash
source ~/miniconda3/etc/profile.d/conda.sh
cd /ruta/a/doctorado_efsg-main/Models
conda env create -f environment.yml
conda activate fenicsx-env
```

### Error: "Google Drive no está montado"

```bash
# Verificar si rclone está instalado
rclone version

# Verificar que el remote 'gdrive' existe
rclone listremotes

# Si no existe, configurarlo
rclone config

# Crear directorio de montaje si no existe
mkdir -p ~/googledrive

# Montar Google Drive
rclone mount gdrive: ~/googledrive \
    --daemon \
    --vfs-cache-mode writes \
    --vfs-cache-max-size 10G \
    --vfs-read-chunk-size 128M \
    --vfs-read-chunk-size-limit off \
    --allow-other \
    --umask 000

# Esperar unos segundos y verificar
sleep 3
mountpoint ~/googledrive
```

### Error: "No se puede escribir en Google Drive"

```bash
# Verificar permisos del directorio
ls -la ~/googledrive

# Verificar que rclone está funcionando
rclone listremotes

# Probar escritura
touch ~/googledrive/.test_write
rm ~/googledrive/.test_write && echo "✓ Escritura OK" || echo "✗ Error de escritura"

# Si falla, desmontar y re-montar
fusermount -u ~/googledrive

# Re-montar con permisos explícitos
rclone mount gdrive: ~/googledrive \
    --daemon \
    --vfs-cache-mode writes \
    --vfs-cache-max-size 10G \
    --vfs-read-chunk-size 128M \
    --vfs-read-chunk-size-limit off \
    --allow-other \
    --umask 000

# Verificar nuevamente
sleep 3
mountpoint ~/googledrive
```

### Error: "Script no encontrado"

Verifica que estás en el directorio correcto:

```bash
cd /mnt/c/Users/Erick\ Serrato/Documents/Doctorado/doctorado_efsg-main/Models/Allee
ls -la run_scenarios.py
```

### Error: "Puerto 8888 ocupado"

```bash
# En WSL, encontrar y matar el proceso
lsof -ti:8888 | xargs kill -9

# O cambiar el puerto en el script
```

### Los Resultados no se Guardan en Google Drive

1. Verifica que Google Drive está montado: `mountpoint ~/googledrive`
2. Verifica que la carpeta existe: `ls ~/googledrive/Doctorado\ Erick\ Serrato/`
3. El script creará las carpetas automáticamente si no existen
4. Si sigue fallando, revisa los logs en `logs/run_scenarios_*.log`

---

## 📝 Notas Importantes

### Sobre Google Drive

- **Montaje Manual**: Google Drive se desmonta al reiniciar WSL (a menos que configures montaje automático). Antes de cada sesión, ejecuta:
  ```bash
  rclone mount gdrive: ~/googledrive \
      --daemon \
      --vfs-cache-mode writes \
      --vfs-cache-max-size 10G \
      --vfs-read-chunk-size 128M \
      --vfs-read-chunk-size-limit off \
      --allow-other \
      --umask 000
  ```
- **Montaje Automático**: Si configuraste el montaje automático en `~/.bashrc`, Google Drive se montará automáticamente al iniciar WSL.
- **Rendimiento**: El acceso a archivos en Google Drive montado es más lento que el almacenamiento local.
- **Estructura**: Los resultados se guardan en `~/googledrive/Doctorado Erick Serrato/Resultados Paper/<escenario>/`
- **Detección Automática**: El script detecta automáticamente si Google Drive está montado y lo usa, si no, usa el directorio local.
- **Desmontar**: Para desmontar Google Drive cuando termines:
  ```bash
  fusermount -u ~/googledrive
  ```
- **Verificar Estado**: Para verificar si está montado:
  ```bash
  mountpoint ~/googledrive && echo "✓ Montado" || echo "✗ No montado"
  ```

### Sobre Checkpoints

- Los checkpoints se guardan automáticamente cada 500 pasos (configurable en `.env`)
- Si la memoria excede el umbral, el script se reinicia automáticamente desde el último checkpoint
- Los checkpoints permiten continuar simulaciones interrumpidas sin perder progreso

### Sobre la Ejecución

- Los scripts `.py` se ejecutan directamente (no requieren papermill)
- Los logs se guardan en `logs/run_scenarios_YYYYMMDD_HHMMSS.log`
- El progreso se muestra en tiempo real en la consola
- Los resultados se validan automáticamente después de cada ejecución

---

## 📚 Referencias Rápidas

### Comandos Más Usados

```bash
# Verificar estado
python run_scenarios.py --status

# Ejecutar todos
python run_scenarios.py

# Ejecutar uno específico
python run_scenarios.py --scenario nombre

# Re-ejecutar fallidos
python run_scenarios.py --retry-failed --yes

# Modo lotes
python run_scenarios.py --batch-mode --yes
```

### Rutas Importantes

- **Scripts**: `/mnt/c/Users/Erick Serrato/Documents/Doctorado/doctorado_efsg-main/Models/Allee/`
- **Google Drive**: `~/googledrive/Doctorado Erick Serrato/Resultados Paper/`
- **Logs**: `Models/Allee/logs/`
- **Configuración**: `Models/Allee/scenarios.json` y `.env`

---

¿Necesitas ayuda con alguna configuración específica? Revisa los logs en `logs/` para más detalles sobre errores.
