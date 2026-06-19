"""
Comando de gestión Django para cargar escenarios desde el catálogo JSON a la base de datos.
"""

from django.core.management.base import BaseCommand
from django.conf import settings
from pathlib import Path
from steady_states_app.models import Scenario, SteadyState
from steady_states_app.scenarios_catalog import load_normalized_catalog


class Command(BaseCommand):
    help = 'Carga escenarios desde el catálogo (settings.SCENARIOS_JSON_PATH, p. ej. steady_states_full_run.json)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--file',
            type=str,
            default=None,
            help='Ruta al JSON (por defecto: settings.SCENARIOS_JSON_PATH)'
        )
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Limpiar escenarios existentes antes de cargar'
        )

    def handle(self, *args, **options):
        raw = options['file']
        file_path = Path(raw) if raw else Path(settings.SCENARIOS_JSON_PATH)
        
        if not file_path.exists():
            self.stdout.write(self.style.ERROR(f'Archivo no encontrado: {file_path}'))
            return
        
        if options['clear']:
            Scenario.objects.all().delete()
            self.stdout.write(self.style.WARNING('Escenarios existentes eliminados'))
        
        try:
            data, err = load_normalized_catalog(file_path)
            if err:
                self.stdout.write(self.style.ERROR(f'Error al leer catálogo: {err}'))
                return
            
            common_params = data.get('common_params', {})
            scenarios_list = data.get('scenarios', [])
            
            self.stdout.write(f'Cargando {len(scenarios_list)} escenarios...')
            
            for scenario_data in scenarios_list:
                scenario, created = Scenario.objects.update_or_create(
                    name=scenario_data['name'],
                    defaults={
                        'allee_type': scenario_data.get('ALLEE_TYPE', 'WEAK'),
                        'mu': float(scenario_data.get('mu', 0)),
                        'use_adaptive_control': scenario_data.get('USE_ADAPTIVE_CONTROL', 'N') == 'Y',
                        'params': scenario_data,
                        'is_active': True,
                    }
                )
                
                if created:
                    self.stdout.write(self.style.SUCCESS(f'  ✓ Creado: {scenario.name}'))
                else:
                    self.stdout.write(f'  ↻ Actualizado: {scenario.name}')
            
            self.stdout.write(self.style.SUCCESS(f'\n✓ Cargados {len(scenarios_list)} escenarios'))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error al cargar escenarios: {str(e)}'))

