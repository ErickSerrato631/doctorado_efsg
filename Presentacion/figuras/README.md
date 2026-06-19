# Figuras de la presentación

## Generación (WSL + conda)

Los scripts viven en `Models/Allee/scripts_temp/`. **Ejecútalos en WSL** con conda activo, no en PowerShell.

```bash
source ~/miniconda3/etc/profile.d/conda.sh   # o anaconda3
conda activate fenicsx-env
export RESULTS_DIR="$HOME/googledrive/Doctorado Erick Serrato/Resultados Paper"
cd "/mnt/c/Users/Erick Serrato/Documents/Doctorado/doctorado_efsg-main/Models/Allee"
python scripts_temp/export_presentacion_mu1_AB.py --prepare
```

Detalle de flags, seed sintético y post-proceso: [`Models/Allee/scripts_temp/README.md`](../../Models/Allee/scripts_temp/README.md).

## Sincronizar con rclone (desde WSL)

Sustituye `REMOTE` y `RUTA_EN_DRIVE` por tu remoto y carpeta en Drive.

```bash
rclone copy "REMOTE:RUTA_EN_DRIVE" "./figuras/" --progress --include "*.png"
```

Para espejo (cuidado con `sync` y borrados):

```bash
rclone sync "REMOTE:RUTA_EN_DRIVE" "./figuras/" --progress --include "*.png"
```

Ejecuta estos comandos desde `Presentacion/figuras/` en WSL, o usa rutas absolutas.

## Archivos usados por `doctor_presentation.tex` (carpeta `figuras/`)

El `.tex` busca cada PNG en `figuras/fields/`, `figuras/correlations/`, `figuras/termo_equilibrium/` o `figuras/` (en ese orden). El vídeo `campos_mu1_AB_comparativa.mp4` va en `figuras/` (raíz).

| Archivo | Uso |
|--------|-----|
| `fields_block_1_step_5.000_strong_mu1_uNo_bajo_umbral.png` | Campos c, s, i en t=5, caso A (sin Hill, u=0) |
| `fields_block_1_step_5.000_strong_mu1_uSi_bajo_umbral.png` | Campos c, s, i en t=5, caso B (con Hill, u>0) |
| `campos_mu1_AB_comparativa.mp4` | Vídeo comparativo (enlace al hacer clic en las imágenes de campos) |
| `corr_mu1_AB_c_i.png` | Correlación c–i (A vs B) |
| `corr_mu1_AB_i_i.png` | Correlación i–i (A vs B) |
| `thermodynamic_F_comparison.png` | Energía libre F(t) |
| `thermodynamic_sigma_comparison.png` | Producción de entropía Σ(t) |
| `thermodynamic_mu_comparison.png` | Potenciales químicos μ_c, μ_s, μ_i |

Coloca `logoUAM.png` en `Presentacion/` (mismo nivel que el `.tex`); Metropolis lo usa en la portada.
