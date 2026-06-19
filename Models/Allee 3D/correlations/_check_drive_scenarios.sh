#!/usr/bin/env bash
set -euo pipefail
GD="${1:-$HOME/googledrive/Doctorado Erick Serrato/Resultados Paper}"
names=(
  strong_mu0_uNo_bajo_umbral_c0_s1_i0
  strong_mu0_uSi_bajo_umbral_c0_s1_i0
  strong_mu1_uNo_bajo_umbral_c0_s1_i0
  strong_mu1_uSi_bajo_umbral_c0_s1_i0
)
echo "RESULTS_DIR=$GD"
for n in "${names[@]}"; do
  p="$GD/$n"
  if [[ ! -d "$p" ]]; then
    echo "$n: MISSING dir"
    continue
  fi
  mc=0
  ct=0
  if [[ -d "$p/matrices" ]]; then
    mc=$(find "$p/matrices" -maxdepth 1 -name 'matrix_c_*_nb_1.txt' 2>/dev/null | wc -l)
  fi
  if [[ -d "$p/correlations" ]]; then
    ct=$(find "$p/correlations" -maxdepth 1 -name 'corr_length_*.txt' 2>/dev/null | wc -l)
  fi
  echo "$n: matrices_c=$mc corr_txt=$ct"
done
