# Arranca el laboratorio Django en WSL desde Windows.
# En PowerShell NO uses .\start_lab_wsl.sh (no corre como bash); usa este .ps1 o .\run_django_server.ps1

param(
    [int]$Port = 8000
)

& "$PSScriptRoot\run_django_server.ps1" -Port $Port
