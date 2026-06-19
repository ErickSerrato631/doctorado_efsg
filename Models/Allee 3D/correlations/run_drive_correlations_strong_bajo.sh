#!/usr/bin/env bash
# Strong bajo umbral (0,1,0): fourier si faltan .txt + grids comparativos en Drive.
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fenicsx-env
ALLEE="/mnt/c/Users/Erick Serrato/Documents/Doctorado/doctorado_efsg-main/Models/Allee"
GD="${RESULTS_DIR:-$HOME/googledrive/Doctorado Erick Serrato/Resultados Paper}"
cd "$ALLEE"

names=(
  strong_mu0_uNo_bajo_umbral_c0_s1_i0
  strong_mu0_uSi_bajo_umbral_c0_s1_i0
  strong_mu1_uNo_bajo_umbral_c0_s1_i0
  strong_mu1_uSi_bajo_umbral_c0_s1_i0
)

echo "RESULTS_DIR=$GD"
for n in "${names[@]}"; do
  scen="$GD/$n"
  if [[ ! -d "$scen/matrices" ]]; then
    echo "[skip] $n: sin matrices/"
    continue
  fi
  need=0
  if [[ $(find "$scen/correlations" -maxdepth 1 -name 'corr_length_*.txt' 2>/dev/null | wc -l) -lt 6 ]]; then
    need=1
  fi
  if [[ $need -eq 1 ]]; then
    echo "== correlation_fourier: $n =="
    export T=1 dt=0.001 nb=1 sample_rate=0.02 SAVE_IMAGES=N
    python correlations/correlation_fourier.py "$scen"
  else
    echo "[ok] $n: ya tiene 6 correlations/*.txt"
  fi
done

echo "== correlation_comparison =="
export T=1 dt=0.02
python correlations/correlation_comparison.py --results-dir "$GD" --fit-tmax 1.0 --fit-tmin 0.05

echo "Done. Per scenario: $GD/<name>/correlations/"
echo "Figures (4-case panels): $GD/comparisons/correlation_grids/corr_grid_<corr>.png"
echo "  (+ _semilogx, _loglog variants)"
