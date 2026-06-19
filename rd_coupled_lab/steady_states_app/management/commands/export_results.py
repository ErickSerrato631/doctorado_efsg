"""
Comando de gestión Django para exportar resultados de análisis a CSV.
"""

from django.core.management.base import BaseCommand
from pathlib import Path
import pandas as pd
from steady_states_app.models import AnalysisRun, SteadyState


class Command(BaseCommand):
    help = 'Exporta resultados de análisis a CSV'

    def add_arguments(self, parser):
        parser.add_argument(
            '--analysis-id',
            type=int,
            help='ID del análisis a exportar (si no se especifica, exporta todos)'
        )
        parser.add_argument(
            '--output',
            type=str,
            default='exported_results.csv',
            help='Nombre del archivo de salida'
        )

    def handle(self, *args, **options):
        if options['analysis_id']:
            analyses = AnalysisRun.objects.filter(id=options['analysis_id'])
        else:
            analyses = AnalysisRun.objects.filter(status='completed')
        
        if not analyses.exists():
            self.stdout.write(self.style.WARNING('No hay análisis para exportar'))
            return
        
        all_data = []
        
        for analysis in analyses:
            steady_states = analysis.steady_states.all()
            
            for state in steady_states:
                all_data.append({
                    'analysis_id': analysis.id,
                    'analysis_name': analysis.name,
                    'analysis_type': analysis.analysis_type,
                    'c_star': state.c_star,
                    's_star': state.s_star,
                    'i_star': state.i_star,
                    'max_real': state.max_real,
                    'unstable': state.unstable,
                    'rc': state.rc,
                    'rs': state.rs,
                    'rd': state.rd,
                    'alpha': state.alpha,
                    'delta': state.delta,
                    'beta': state.beta,
                    'eta': state.eta,
                    'mu': state.mu,
                })
        
        if all_data:
            df = pd.DataFrame(all_data)
            output_path = Path(options['output'])
            df.to_csv(output_path, index=False)
            self.stdout.write(self.style.SUCCESS(f'✓ Exportados {len(all_data)} estados estacionarios a {output_path}'))
        else:
            self.stdout.write(self.style.WARNING('No hay datos para exportar'))

