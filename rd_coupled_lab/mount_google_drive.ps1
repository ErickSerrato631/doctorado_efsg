# Ejecuta mount_google_drive.sh dentro de WSL (PowerShell no interpreta .sh).
param(
    [string]$WslDistro = "Ubuntu",
    [switch]$Foreground = $false
)

$projectPath = $PSScriptRoot
$repoWsl = (wsl -d $WslDistro wslpath -a $projectPath 2>$null).Trim()
if ([string]::IsNullOrWhiteSpace($repoWsl)) {
    Write-Host "No se pudo convertir la ruta a WSL (wslpath). Repo: $projectPath" -ForegroundColor Red
    exit 1
}

$shWin = Join-Path $PSScriptRoot "mount_google_drive.sh"
if (-not (Test-Path -LiteralPath $shWin)) {
    Write-Host "No existe mount_google_drive.sh en $projectPath" -ForegroundColor Red
    exit 1
}

# Evitar CRLF en WSL
$raw = [System.IO.File]::ReadAllText($shWin)
if ($raw.IndexOf([char]13) -ge 0) {
    $norm = $raw -replace "`r`n", "`n" -replace "`r", "`n"
    $utf8 = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($shWin, $norm, $utf8)
    Write-Host "[mount] mount_google_drive.sh tenía CRLF; normalizado a LF." -ForegroundColor DarkYellow
}

$envFg = if ($Foreground) { "export RCLONE_FOREGROUND=1; " } else { "" }
$bashRemote = "cd -- '$repoWsl' && $envFg exec bash ./mount_google_drive.sh"
Write-Host "WSL ($WslDistro): $bashRemote" -ForegroundColor Cyan
wsl -d $WslDistro -e bash -lc $bashRemote
exit $LASTEXITCODE
