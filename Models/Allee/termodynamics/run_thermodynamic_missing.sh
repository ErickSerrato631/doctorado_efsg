#!/bin/bash
# Script para ejecutar propiedades termodinámicas de escenarios faltantes
# Ejecutar desde WSL con: bash run_thermodynamic_missing.sh

# Activar conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fenicsx-env

# Raíz Allee (padre de termodinámica)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ALLEE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ALLEE_ROOT"

# Configurar RESULTS_DIR para usar Google Drive
# Si Google Drive está montado, usar esa ruta; si no, usar local
if mountpoint -q ~/googledrive 2>/dev/null; then
    export RESULTS_DIR="$HOME/googledrive/Doctorado Erick Serrato/Resultados Paper"
    echo "✓ Usando Google Drive: $RESULTS_DIR"
else
    echo "⚠ Google Drive no está montado. Montando..."
    bash "$ALLEE_ROOT/mount_google_drive.sh"
    if mountpoint -q ~/googledrive 2>/dev/null; then
        export RESULTS_DIR="$HOME/googledrive/Doctorado Erick Serrato/Resultados Paper"
        echo "✓ Google Drive montado: $RESULTS_DIR"
    else
        echo "⚠ No se pudo montar Google Drive. Usando directorio local."
        export RESULTS_DIR="$(pwd)/results"
    fi
fi

# Escenarios que faltan
SCENARIOS=(
    "weak_mu0_uSi_bajo_umbral"
    "weak_mu1_uNo_bajo_umbral"
    "weak_mu1_uNo_sobre_umbral"
    "weak_mu1_uSi_bajo_umbral"
)

echo "=========================================="
echo "Procesando escenarios faltantes"
echo "Total: ${#SCENARIOS[@]} escenarios"
echo "=========================================="
echo ""

# Procesar cada escenario
for scenario in "${SCENARIOS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Procesando: $scenario"
    echo "=========================================="
    echo ""
    
    python "$SCRIPT_DIR/calculate_thermodynamic_properties.py" --scenario "$scenario"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Completado: $scenario"
    else
        echo ""
        echo "✗ Error en: $scenario"
        echo "Continuando con el siguiente..."
    fi
    
    echo ""
done

echo ""
echo "=========================================="
echo "Proceso completado"
echo "=========================================="

