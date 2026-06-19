#!/usr/bin/env bash
set -euo pipefail

# Monta Google Drive en WSL usando rclone.
# Uso:
#   bash mount_google_drive.sh
#   En Windows PowerShell NO ejecutes .\mount_google_drive.sh (abre el .txt asociado).
#   Usa:  wsl bash ./mount_google_drive.sh   o   .\mount_google_drive.ps1
#
# Variables opcionales:
# - GOOGLE_DRIVE_MOUNT_POINT / GDRIVE_MOUNT_POINT: punto de montaje (default: ~/googledrive)
# - RCLONE_REMOTE: remote de rclone (default: gdrive:)
# - VFS_CACHE_MODE / VFS_CACHE_MAX_SIZE
# - RCLONE_ALLOW_OTHER=1  → añade --allow-other (en WSL suele fallar o no aplicar; por defecto NO)
# - RCLONE_FOREGROUND=1    → monta en primer plano (sin --daemon).
#
# WSL + Docker: si ves "cannot read /proc/mounts: expected integer", NO es por --daemon:
#   en /proc/mounts aparecen líneas con espacios en las opciones (p. ej. path=C:\Program Files\...).
#   rclone < ~1.64 las parseaba mal. Solución: rclone estable reciente:
#     curl https://rclone.org/install.sh | sudo bash
#   (evita builds viejos tipo v1.60.x-DEV). Ver: rclone PR #7149 / foro rclone tema 40034.

MOUNT_POINT="${GOOGLE_DRIVE_MOUNT_POINT:-${GDRIVE_MOUNT_POINT:-$HOME/googledrive}}"
REMOTE="${RCLONE_REMOTE:-gdrive:}"
VFS_CACHE_MODE="${VFS_CACHE_MODE:-writes}"
VFS_CACHE_MAX_SIZE="${VFS_CACHE_MAX_SIZE:-10G}"

echo "== Google Drive mount (rclone) =="
echo "Mount point: ${MOUNT_POINT}"
echo "Remote:      ${REMOTE}"

if [[ -f /proc/version ]] && grep -qi microsoft /proc/version; then
  echo "Entorno:     WSL (detectado)"
fi
if [[ "${RCLONE_ALLOW_OTHER:-0}" != "1" ]]; then
  echo "Nota:        sin --allow-other (recomendado en WSL). Exporta RCLONE_ALLOW_OTHER=1 si lo necesitas."
fi

if ! command -v rclone >/dev/null 2>&1; then
  echo "✗ Error: rclone no está instalado."
  echo "  Instala en WSL: sudo apt-get update && sudo apt-get install -y rclone"
  echo "  O versión reciente: curl https://rclone.org/install.sh | sudo bash"
  exit 1
fi

echo "rclone:      $(rclone version | head -1)"

mkdir -p "${MOUNT_POINT}"

if command -v mountpoint >/dev/null 2>&1 && mountpoint -q "${MOUNT_POINT}"; then
  echo "✓ Ya está montado: ${MOUNT_POINT}"
  exit 0
fi

# FUSE exige un directorio vacío; si hay entradas, rclone suele salir con 1 y el modo --daemon
# muestra "Daemon timed out" / "daemon exited with error code 1".
if [[ -n "$(find "${MOUNT_POINT}" -mindepth 1 -maxdepth 1 2>/dev/null | head -1)" ]]; then
  echo "✗ El punto de montaje no está vacío: ${MOUNT_POINT}"
  echo "  rclone mount necesita un directorio vacío. Opciones:"
  echo "    • Mueve el contenido a otro sitio y deja vacío este directorio, o"
  echo "    • Usa otro punto de montaje, p. ej.:"
  echo "        export GOOGLE_DRIVE_MOUNT_POINT=\"\$HOME/googledrive_mnt\""
  echo "        bash $0"
  exit 1
fi

if ! rclone listremotes 2>/dev/null | grep -qx "${REMOTE}"; then
  echo "✗ Error: no existe el remote '${REMOTE}' en tu configuración de rclone."
  echo "  Remotes disponibles:"
  rclone listremotes || true
  echo
  echo "  Ejecuta: rclone config"
  exit 1
fi

ALLOW_OTHER=()
if [[ "${RCLONE_ALLOW_OTHER:-0}" == "1" ]]; then
  ALLOW_OTHER=(--allow-other)
fi

RCLONE_MOUNT_COMMON=(
  "${REMOTE}"
  "${MOUNT_POINT}"
  --vfs-cache-mode "${VFS_CACHE_MODE}"
  --vfs-cache-max-size "${VFS_CACHE_MAX_SIZE}"
  --vfs-read-chunk-size 128M
  --vfs-read-chunk-size-limit off
  --umask 000
  "${ALLOW_OTHER[@]}"
)

if [[ "${RCLONE_FOREGROUND:-0}" == "1" ]]; then
  echo "→ Montando en primer plano (sin --daemon). En otra terminal: mountpoint -q '${MOUNT_POINT}' && echo OK"
  echo "  Para detener: Ctrl+C en esta terminal."
  exec rclone mount "${RCLONE_MOUNT_COMMON[@]}"
fi

echo "→ Montando... (daemon)"
set +e
rclone mount "${RCLONE_MOUNT_COMMON[@]}" --daemon
rc=$?
set -e

if [[ "$rc" -ne 0 ]]; then
  echo "✗ rclone mount falló (código $rc)."
  echo "  Si ves \"Daemon timed out\" o \"daemon exited with error code 1\":"
  echo "    → Suele ser punto de montaje NO vacío (FUSE no monta encima de carpetas con contenido)."
  echo "      Deja vacío el directorio o define otro GOOGLE_DRIVE_MOUNT_POINT."
  echo "  Si el error es: cannot read /proc/mounts: expected integer"
  echo "    → WSL + Docker: rclone antiguo; actualiza (>= ~1.64):"
  echo "      curl https://rclone.org/install.sh | sudo bash   # en WSL: apt install -y unzip si pide unzip"
  echo "  Depuración: RCLONE_FOREGROUND=1 bash mount_google_drive.sh  (mensaje de error más claro)"
  exit "$rc"
fi

sleep 1

if command -v mountpoint >/dev/null 2>&1 && mountpoint -q "${MOUNT_POINT}"; then
  echo "✓ Montaje OK: ${MOUNT_POINT}"
else
  echo "⚠ No se pudo verificar el montaje con 'mountpoint'."
  echo "  Revisa con: mount | grep -Ei 'rclone|fuse|drive|gdrive|google'"
  echo "  O prueba: RCLONE_FOREGROUND=1 bash $0  (y comprueba en otra terminal)"
  exit 2
fi
