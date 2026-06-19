# Script para ejecutar Jupyter Lab con FEniCSx desde WSL
# Uso: .\run_jupyter_fenicsx_wsl.ps1

$projectPath = "C:\Users\Erick Serrato\Documents\Doctorado\doctorado_efsg-main"
$wslDistro = "Ubuntu"
$port = 8888

Write-Host "========================================" -ForegroundColor Green
Write-Host "Iniciando Jupyter Lab con FEniCSx desde WSL" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "Proyecto: $projectPath" -ForegroundColor Cyan
Write-Host "Distribución WSL: $wslDistro" -ForegroundColor Cyan
Write-Host "Puerto: $port" -ForegroundColor Cyan
Write-Host ""

# Verificar que WSL está disponible
$wslCheck = wsl -l -v 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Error: WSL no está disponible o no está instalado" -ForegroundColor Red
    Write-Host "Instala WSL con: wsl --install -d Ubuntu" -ForegroundColor Yellow
    exit 1
}

# Convertir ruta de Windows a WSL
$wslProjectPath = $projectPath -replace 'C:\\', '/mnt/c/' -replace '\\', '/'

Write-Host "Verificando entorno fenicsx-env en WSL..." -ForegroundColor Yellow

# Verificar que el entorno existe
$envCheck = wsl -d $wslDistro -e bash -c "source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda env list | grep -q 'fenicsx-env' && echo 'OK' || echo 'NOT_FOUND'"
if ($envCheck -notmatch "OK") {
    Write-Host "⚠️  Advertencia: El entorno 'fenicsx-env' no se encontró" -ForegroundColor Yellow
    Write-Host "Crea el entorno ejecutando en WSL:" -ForegroundColor Yellow
    Write-Host "  conda create -n fenicsx-env python=3.11 -y" -ForegroundColor Cyan
    Write-Host "  conda activate fenicsx-env" -ForegroundColor Cyan
    Write-Host "  conda install -c conda-forge fenics-dolfinx mpich pyvista -y" -ForegroundColor Cyan
    Write-Host ""
    $continue = Read-Host "¿Deseas continuar de todos modos? (s/n)"
    if ($continue -ne "s" -and $continue -ne "S") {
        exit 1
    }
}

Write-Host "✓ Entorno verificado" -ForegroundColor Green
Write-Host ""
Write-Host "Iniciando Jupyter Lab..." -ForegroundColor Yellow
Write-Host "URL: http://localhost:$port" -ForegroundColor Cyan
Write-Host ""
Write-Host "Para detener: Presiona Ctrl+C en esta ventana" -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Green

# Ejecutar Jupyter Lab en WSL con el entorno fenicsx-env
$jupyterCommand = @"
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate fenicsx-env
cd '$wslProjectPath'
jupyter lab --ip=0.0.0.0 --port=$port --no-browser --allow-root --ServerApp.token='' --ServerApp.password=''
"@

wsl -d $wslDistro -e bash -c $jupyterCommand

