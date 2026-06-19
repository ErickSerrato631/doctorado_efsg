#!/usr/bin/env bash
# Desde Windows (PowerShell): usa .\start_lab_wsl.ps1 o .\run_django_server.ps1.
#   .\start_lab_wsl.sh en PowerShell no ejecuta bash; no verás el servidor.
# Arranque Django en WSL: localiza steady_states_full_run.json en el montaje rclone
# (Resultados paper/estados_estacionarios o My Drive/.../misma ruta) y exporta STEADY_STATES_CATALOG_JSON.
# Actualizar ese JSON desde Models/Allee (fusionar scenarios_v1 + scenarios): merge_steady_catalog_to_drive
# Sin -u: conda activate (p. ej. activate-binutils_linux-64.sh) usa variables opcionales como ADDR2LINE;
# con nounset bash aborta con «unbound variable».
set -e
# pipefail es bash (no POSIX); evita fallo si algo invocara sh/dash.
if [ -n "${BASH_VERSION:-}" ]; then
  set -o pipefail
fi

PORT="${1:-8000}"

try_catalog_at() {
  local base="$1"
  local prefix="${2:-}"
  [ -n "$base" ] || return 1
  [ -d "$base" ] || return 1
  local r f
  for r in \
    "Doctorado Erick Serrato/Resultados Paper/estados_estacionarios/steady_states_full_run.json" \
    "Doctorado Erick Serrato/Resultados paper/estados_estacionarios/steady_states_full_run.json" \
    "Resultados paper/estados_estacionarios/steady_states_full_run.json" \
    "Resultados Paper/estados_estacionarios/steady_states_full_run.json"
  do
    f="${base}/${prefix}${r}"
    if [ -r "$f" ]; then
      export STEADY_STATES_CATALOG_JSON="$f"
      echo "[rd_coupled_lab] Catálogo: $f"
      return 0
    fi
  done
  return 1
}

for base in "${GOOGLE_DRIVE_MOUNT_POINT:-}" "${HOME}/googledrive" "/root/googledrive"; do
  try_catalog_at "$base" "" && break
  try_catalog_at "$base" "My Drive/" && break
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "[rd_coupled_lab] Iniciando arranque (WSL). Puerto=${PORT}." >&2

# Preferir venv Linux del repo (recomendado; ver README.md), salvo RD_LAB_PREFER_CONDA=1 (solo Conda).
# RD_LAB_CONDA_ROOT: raíz explícita (ej. /home/tu/anaconda3). Si no, se busca Anaconda antes que Miniconda.
# RD_LAB_CONDA_ENV: nombre del env (por defecto base). Para FEniCS en miniconda: export RD_LAB_CONDA_ENV=fenicsx-env
resolve_python() {
  local conda_env="${RD_LAB_CONDA_ENV:-base}"

  if [[ -z "${RD_LAB_PREFER_CONDA:-}" ]]; then
    local venv_py="$SCRIPT_DIR/.venv/bin/python"
    if [[ -x "$venv_py" ]]; then
      echo "[rd_coupled_lab] Python: venv $venv_py" >&2
      printf '%s' "$venv_py"
      return 0
    fi
  fi

  local conda_sh=""
  if [[ -n "${RD_LAB_CONDA_ROOT:-}" && -f "${RD_LAB_CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    conda_sh="${RD_LAB_CONDA_ROOT}/etc/profile.d/conda.sh"
  else
    for p in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/mambaforge"; do
      if [[ -f "$p/etc/profile.d/conda.sh" ]]; then
        conda_sh="$p/etc/profile.d/conda.sh"
        break
      fi
    done
  fi

  if [[ -n "$conda_sh" ]]; then
    # shellcheck source=/dev/null
    source "$conda_sh"
    conda activate "$conda_env"
    if command -v python >/dev/null 2>&1; then
      echo "[rd_coupled_lab] Python: conda $(command -v python) (RD_LAB_CONDA_ENV=$conda_env)" >&2
      command -v python
      return 0
    fi
  fi

  if command -v python3 >/dev/null 2>&1; then
    echo "[rd_coupled_lab] Python: $(command -v python3) (PATH)" >&2
    command -v python3
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    echo "[rd_coupled_lab] Python: $(command -v python) (PATH)" >&2
    command -v python
    return 0
  fi
  return 1
}

PYTHON="$(resolve_python)" || {
  echo "[rd_coupled_lab] No se encontró Python. Instala Anaconda/Miniconda o python3." >&2
  exit 1
}

if ! "$PYTHON" -c "import django" >/dev/null 2>&1; then
  echo "[rd_coupled_lab] Django no está instalado para: $PYTHON" >&2
  echo "[rd_coupled_lab] Instalando dependencias desde requirements.txt..." >&2
  if ! "$PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt"; then
    echo "[rd_coupled_lab] pip install falló. Manualmente:" >&2
    echo "  cd \"$SCRIPT_DIR\" && python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements.txt" >&2
    echo "  o: pip install -r \"$SCRIPT_DIR/requirements.txt\" en el mismo entorno." >&2
    exit 1
  fi
fi

if ! "$PYTHON" -c "import django" >/dev/null 2>&1; then
  echo "[rd_coupled_lab] Tras instalar requirements, Django aún no se importa. Revisa el entorno: $PYTHON" >&2
  exit 1
fi

lab_ip="$( (hostname -I 2>/dev/null || true) | awk '{print $1}' )"
if [[ -n "$lab_ip" ]]; then
  echo "[rd_coupled_lab] Navegador en Windows: http://${lab_ip}:${PORT}/" >&2
  echo "[rd_coupled_lab] No abras http://0.0.0.0:${PORT}/ (Chrome: dirección inválida). 0.0.0.0 solo indica «escuchar en todas las interfaces»." >&2
fi
echo "[rd_coupled_lab] Dentro de WSL: http://127.0.0.1:${PORT}/" >&2
echo "[rd_coupled_lab] Django runserver 0.0.0.0:${PORT} (deja esta ventana abierta; Ctrl+C para detener)." >&2

exec "$PYTHON" manage.py runserver "0.0.0.0:${PORT}"
