# Abre el laboratorio Django en el navegador predeterminado de Windows.
# El servidor corre en WSL: Chrome con 127.0.0.1 suele dar ERR_CONNECTION_REFUSED;
# hay que usar la IP de la distro (esta script la resuelve).

param(
    [int]$Port = 8000,
    [string]$WslDistro = "Ubuntu"
)

$raw = (wsl -d $WslDistro -e hostname -I 2>$null)
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($raw)) {
    Write-Host "No se pudo obtener la IP de WSL ($WslDistro)." -ForegroundColor Red
    exit 1
}

$wslIp = ($raw -replace '\s+', ' ').Trim().Split(' ')[0]
if ($wslIp -notmatch '^\d+\.\d+\.\d+\.\d+$') {
    Write-Host "IP inesperada: $raw" -ForegroundColor Red
    exit 1
}

$url = "http://${wslIp}:${Port}/"
Write-Host "Abriendo en el navegador: $url" -ForegroundColor Green
Write-Host "(Asegúrate de que .\run_django_server.ps1 esté en ejecución.)" -ForegroundColor DarkGray
Start-Process $url
