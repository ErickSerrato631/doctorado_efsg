#!/usr/bin/env bash
# Sin -u: los scripts de `conda activate` (p. ej. binutils) usan variables opcionales como ADDR2LINE;
# con nounset bash aborta con "unbound variable".
set -eo pipefail
export RESULTS_DIR="${RESULTS_DIR:-$HOME/googledrive/Doctorado Erick Serrato/Resultados Paper}"
SCENARIO_DIR="$RESULTS_DIR/strong_mu0_uNo_bajo_umbral/matrices"
echo "RESULTS_DIR=$RESULTS_DIR"
if [[ ! -d "$SCENARIO_DIR" ]]; then
  echo "ERROR: no existe $SCENARIO_DIR"
  exit 1
fi
n=$(find "$SCENARIO_DIR" -maxdepth 1 -name 'matrix_c_*_nb_1.txt' | wc -l)
echo "Archivos matrix_c_*_nb_1.txt: $n"
cd "/mnt/c/Users/Erick Serrato/Documents/Doctorado/doctorado_efsg-main/Models/Allee"
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate fenicsx-env
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
  conda activate fenicsx-env
else
  echo "ERROR: no se encontró conda (miniconda3/anaconda3). Instala deps o ajusta el script."
  exit 1
fi
PY=python3
command -v python3 >/dev/null 2>&1 || PY=python
"$PY" nonequilibrium_termodynamics/visualize_fluxes_and_entropy_density.py \
  --scenarios scenarios_v1.json \
  --scenario strong_mu0_uNo_bajo_umbral \
  --all-times \
  --no-quiver
echo "Listo."
