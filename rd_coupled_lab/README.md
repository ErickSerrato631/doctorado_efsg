# rd_coupled_lab (Django)

## Entorno Python (primera vez)

Si aparece `ModuleNotFoundError: No module named 'django'`, instala dependencias en un venv (recomendado):

```powershell
cd "C:\Users\Erick Serrato\Documents\Doctorado\doctorado_efsg-main\rd_coupled_lab"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
python manage.py migrate
```

En cada sesión nueva: `.\.venv\Scripts\Activate.ps1` y luego `python manage.py runserver` o `.\run_django_server.ps1`.

Si usas **Conda en Windows**, crea/activa un entorno e instala el mismo paquete: `pip install -r requirements.txt`.

## Servidor en Windows

Ejecuta `run_django_server.ps1` con el **mismo Python** donde instalaste dependencias (venv activado). Por defecto escucha en el puerto **8000**; si ves *That port is already in use*, cierra el otro `runserver` o cambia la variable `$port` en el script.

## Carpeta de resultados (`RESULTS_DIR`)

Los scripts en `Models/Allee` resuelven la raíz de resultados con `utils_paths.get_results_dir` y la variable de entorno **`RESULTS_DIR`** (si existe y el path es accesible). El laboratorio Django usa la misma lógica vía `steady_states_app.lab_paths.get_lab_results_root()`.

- **Windows:** si las simulaciones escriben solo dentro de WSL o en Google Drive montado en Linux, el proceso de Django en Windows **no** ve automáticamente esas rutas. Define `RESULTS_DIR` apuntando a una carpeta que **sí** lea el intérprete que ejecuta `runserver` (por ejemplo unidad de red, `WSL` montada, o carpeta sincronizada con rclone).
- **Ejemplo (PowerShell):** `$env:RESULTS_DIR = "D:\ResultadosPaper"` antes de `python manage.py runserver`.

El dashboard muestra una alerta si la ruta no existe o no es legible.

## Caché del panel de experimentos

El escaneo de escenarios en disco se cachea unos **90 s** (memoria local). Para forzar un nuevo escaneo: `GET /api/experiment-status/?refresh=1`.
