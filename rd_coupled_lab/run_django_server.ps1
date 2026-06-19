# Script PowerShell para ejecutar el servidor Django desde Windows

param(
    [int]$Port = 8000,
    [string]$WslDistro = "Ubuntu",
    # Si el arranque falla con «That port is already in use», usa -KillPort o ejecuta .\kill_wsl_port.ps1
    [switch]$KillPort = $false
)

$projectPath = $PSScriptRoot
$wslDistro = $WslDistro
$port = $Port

# Conda en WSL: nombre exacto del env (conda env list). Laboratorio FEniCS suele ser fenicsx-env; si el tuyo se llama distinto (p. ej. fenics), cámbialo aquí.
$condaEnv = "fenicsx-env"
$condaRootWsl = ""   # ejemplo: "/home/erick_errato/anaconda3" — solo si conda no está en ~/miniconda3|anaconda3|...
# $true = ignorar .venv del repo y usar siempre este entorno Conda anterior.
$wslPreferConda = $true

Write-Host "========================================" -ForegroundColor Green
Write-Host "Iniciando servidor Django" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "Proyecto: $projectPath" -ForegroundColor Cyan
Write-Host "Puerto: $port" -ForegroundColor Cyan
Write-Host ""

# Verificar que WSL está disponible
$wslCheck = wsl -l -v 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Error: WSL no está disponible" -ForegroundColor Red
    exit 1
}

Write-Host "Iniciando servidor Django..." -ForegroundColor Yellow
# WSL2: Chrome/Edge en Windows en 127.0.0.1 suele dar ERR_CONNECTION_REFUSED (el puerto está en Linux).
try {
    $wslIp = ((wsl -d $wslDistro -e hostname -I) -replace '\s+', ' ').Trim().Split(' ')[0]
    if ($wslIp -match '^\d+\.\d+\.\d+\.\d+$') {
        Write-Host "Navegador en Windows → usa: http://${wslIp}:$port" -ForegroundColor Green
        Write-Host "O ejecuta en otra ventana: .\open_django_lab_url.ps1" -ForegroundColor Cyan
    }
} catch { }
Write-Host "localhost:$port en Windows puede fallar; dentro de WSL sí: http://127.0.0.1:$port" -ForegroundColor DarkGray
Write-Host "No uses http://0.0.0.0:$port en el navegador (Chrome: ERR_ADDRESS_INVALID). 0.0.0.0 solo sirve en runserver, no como URL." -ForegroundColor Yellow
Write-Host "Probar: wsl -d $wslDistro -e curl -sI http://127.0.0.1:$port/" -ForegroundColor DarkGray
Write-Host "WSL Conda: env='$condaEnv' preferConda=$wslPreferConda" -ForegroundColor DarkGray
Write-Host ""

# Catálogo: start_lab_wsl.sh busca steady_states_full_run.json en
#   <montaje>/Resultados paper/estados_estacionarios/  (y variante con My Drive/ y "Resultados Paper")
# Montajes probados: GOOGLE_DRIVE_MOUNT_POINT, ~/googledrive, /root/googledrive (solo legible si corres como root).
# Fusionar v1 + scenarios → steady_states_full_run en Drive: .\merge_steady_catalog.ps1 (o one-liner bash -lc como en PIPELINE / merge_steady_catalog.ps1 cabecera).

$repoWsl = (wsl -d $wslDistro wslpath -a $projectPath 2>$null).Trim()
if ([string]::IsNullOrWhiteSpace($repoWsl)) {
    Write-Host "No se pudo convertir la ruta del proyecto a WSL (wslpath). Repo: $projectPath" -ForegroundColor Red
    exit 1
}

# WSL/bash fallan con CRLF (set -e\r, `{\r`). Normalizar antes de cada arranque.
$startShWin = Join-Path $PSScriptRoot 'start_lab_wsl.sh'
if (Test-Path -LiteralPath $startShWin) {
    $raw = [System.IO.File]::ReadAllText($startShWin)
    if ($raw.IndexOf([char]13) -ge 0) {
        $norm = $raw -replace "`r`n", "`n" -replace "`r", "`n"
        $utf8 = New-Object System.Text.UTF8Encoding $false
        [System.IO.File]::WriteAllText($startShWin, $norm, $utf8)
        Write-Host "[rd_coupled_lab] start_lab_wsl.sh tenía CRLF; se guardó con finales de línea LF para WSL." -ForegroundColor DarkYellow
    }
}

# cd al repo en WSL y ./script evita que rutas con espacios rompan bash -lc "bash '...'"
$envExports = "export RD_LAB_CONDA_ENV='$condaEnv'"
if (-not [string]::IsNullOrWhiteSpace($condaRootWsl)) {
    $envExports += "; export RD_LAB_CONDA_ROOT='$condaRootWsl'"
}
if ($wslPreferConda) {
    $envExports += "; export RD_LAB_PREFER_CONDA=1"
}
if ($KillPort) {
    Write-Host "[rd_coupled_lab] Liberando puerto $port en WSL (-KillPort)..." -ForegroundColor Yellow
    $killCmd = "fuser -k $port/tcp 2>/dev/null || true; lsof -t -iTCP:$port -sTCP:LISTEN 2>/dev/null | xargs -r kill -9 2>/dev/null || true"
    wsl -d $wslDistro -e bash -lc $killCmd
}

$bashRemote = "cd -- '$repoWsl' && $envExports && exec bash ./start_lab_wsl.sh $port"
wsl -d $wslDistro -e bash -lc "$bashRemote"

