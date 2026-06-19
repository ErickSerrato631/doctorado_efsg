# Libera un puerto TCP en WSL (p. ej. 8000 ocupado por runserver anterior).
param(
    [int]$Port = 8000,
    [string]$WslDistro = "Ubuntu"
)

$p = $Port
Write-Host "WSL ($WslDistro): liberando puerto TCP $p ..." -ForegroundColor Yellow

# PowerShell expande $p; bash recibe el número literal.
$cmd = "fuser -k $p/tcp 2>/dev/null || true; lsof -t -iTCP:$p -sTCP:LISTEN 2>/dev/null | xargs -r kill -9 2>/dev/null || true; ss -tlnp 2>/dev/null | grep -E ':$p\\s' || echo Puerto $p sin listener segun ss."
wsl -d $WslDistro -e bash -lc $cmd

Write-Host "Si el puerto sigue ocupado, en una terminal WSL: sudo fuser -k $p/tcp" -ForegroundColor DarkGray
