"""
forms.py

Formularios Django para configuración de análisis de estados estacionarios.
"""

from django import forms
from django.core.validators import MinValueValidator, MaxValueValidator
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Row, Column, Submit, HTML, Div
from crispy_bootstrap5.bootstrap5 import FloatingField


class Complete3DForm(forms.Form):
    """
    Formulario para buscar estados estacionarios en (c, s, i).

    Equilibrio: R_c = R_s = R_i = 0; resolución numérica tipo Newton-Raphson
    sobre semillas en el espacio de las tres variables principales.
    """
    
    mu = forms.FloatField(
        label='μ (mu)',
        initial=1,
        help_text='Parámetro de control inmunológico adicional (0: sin control, 1: con control)',
        validators=[MinValueValidator(0), MaxValueValidator(1)]
    )
    
    # Control adaptativo (opcional)
    use_adaptive_control = forms.BooleanField(
        label='Incluir control adaptativo',
        initial=False,
        required=False,
        help_text='Si está activado, incluye término de control adaptativo u = ku·c/(i + ε_u) en la ecuación de i'
    )
    
    ku = forms.FloatField(
        label='ku (intensidad del control adaptativo)',
        initial=0.2,
        required=False,
        help_text='Coeficiente de intensidad del control adaptativo. Solo se usa si "Incluir control adaptativo" está activado.',
        validators=[MinValueValidator(0)]
    )
    
    eps_u = forms.FloatField(
        label='ε_u (epsilon del control adaptativo)',
        initial=0.001,
        required=False,
        help_text='Parámetro epsilon para evitar singularidad cuando i → 0. Solo se usa si el control adaptativo está activado.',
        validators=[MinValueValidator(0)]
    )
    
    u_max = forms.FloatField(
        label='u_max (valor máximo del control)',
        initial=0.5,
        required=False,
        help_text='Valor máximo permitido para el control adaptativo (None para sin límite). Opcional.',
        validators=[MinValueValidator(0)]
    )
    
    # Rangos de parámetros del modelo
    rc_vals = forms.CharField(
        label='Valores de rc (tasa de crecimiento del cáncer)',
        initial='5.0, 6.0',
        help_text='Valores separados por comas. rc controla la tasa de crecimiento del cáncer con efecto Allee.',
        widget=forms.Textarea(attrs={'rows': 2})
    )
    
    beta_vals = forms.CharField(
        label='Valores de β (beta)',
        initial='5.0, 7.0',
        help_text='Valores separados por comas. β es el coeficiente de supresión del cáncer por sistema inmune.',
        widget=forms.Textarea(attrs={'rows': 2})
    )
    
    delta_vals = forms.CharField(
        label='Valores de δ (delta)',
        initial='5.0, 7.0',
        help_text='Valores separados por comas. δ controla la interacción entre sistema inmune y células sanas.',
        widget=forms.Textarea(attrs={'rows': 2})
    )
    
    eta_vals = forms.CharField(
        label='Valores de η (eta)',
        initial='3.0, 5.0',
        help_text='Valores separados por comas. η es el coeficiente de supresión del sistema inmune por cáncer.',
        widget=forms.Textarea(attrs={'rows': 2})
    )
    
    rd_vals = forms.CharField(
        label='Valores de rd (tasa de crecimiento del sistema inmune)',
        initial='9.0, 11.0',
        help_text='Valores separados por comas. rd controla la tasa de crecimiento logístico del sistema inmune.',
        widget=forms.Textarea(attrs={'rows': 2})
    )
    
    a_vals = forms.CharField(
        label='Valores de a (parámetro Allee)',
        initial='0.1',
        help_text='Valores separados por comas. a es el parámetro del efecto Allee (umbral crítico).',
        widget=forms.Textarea(attrs={'rows': 2})
    )
    
    # Valores iniciales para Newton-Raphson
    seeds_c_min = forms.FloatField(
        label='Semillas c (mínimo)',
        initial=0.01,
        help_text='Valor mínimo para células cancerosas en los puntos iniciales del método de Newton-Raphson',
        validators=[MinValueValidator(0)]
    )
    
    seeds_c_max = forms.FloatField(
        label='Semillas c (máximo)',
        initial=2.5,
        help_text='Valor máximo para células cancerosas en los puntos iniciales',
        validators=[MinValueValidator(0)]
    )
    
    seeds_s_min = forms.FloatField(
        label='Semillas s (mínimo)',
        initial=0.01,
        help_text='Valor mínimo para células sanas en los puntos iniciales',
        validators=[MinValueValidator(0)]
    )
    
    seeds_s_max = forms.FloatField(
        label='Semillas s (máximo)',
        initial=2.5,
        help_text='Valor máximo para células sanas en los puntos iniciales',
        validators=[MinValueValidator(0)]
    )
    
    seeds_i_min = forms.FloatField(
        label='Semillas i (mínimo)',
        initial=0.01,
        help_text='Valor mínimo para sistema inmune en los puntos iniciales',
        validators=[MinValueValidator(0)]
    )
    
    seeds_i_max = forms.FloatField(
        label='Semillas i (máximo)',
        initial=2.0,
        help_text='Valor máximo para sistema inmune en los puntos iniciales',
        validators=[MinValueValidator(0)]
    )
    
    seeds_n_points = forms.IntegerField(
        label='Número de puntos en semillas',
        initial=3,
        help_text='Número de puntos por dimensión para generar la malla de valores iniciales (total = n_points³)',
        validators=[MinValueValidator(1), MaxValueValidator(10)]
    )
    
    generate_scenarios = forms.BooleanField(
        label='Generar JSON auxiliar en Allee (no es el catálogo del dashboard)',
        initial=True,
        required=False,
        help_text='Si está activado, escribe/actualiza un archivo bajo Models/Allee (el dashboard lee steady_states_full_run.json en Drive).'
    )
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.layout = Layout(
            HTML('<h4 class="mb-4">Parámetros del Modelo</h4>'),
            Row(
                Column(FloatingField('mu'), css_class='col-md-6'),
                Column(FloatingField('use_adaptive_control'), css_class='col-md-6'),
            ),
            HTML('<div id="control-fields" style="display: none;">'),
            Row(
                Column(FloatingField('ku'), css_class='col-md-6'),
                Column(FloatingField('eps_u'), css_class='col-md-6'),
            ),
            Row(
                Column(FloatingField('u_max'), css_class='col-md-6'),
            ),
            HTML('</div>'),
            HTML('<hr><h5>Rangos de Parámetros del Modelo</h5>'),
            HTML('<p class="text-muted small">Especifique los valores de parámetros a explorar. El análisis ejecutará un escaneo sistemático sobre todas las combinaciones.</p>'),
            FloatingField('rc_vals'),
            FloatingField('beta_vals'),
            FloatingField('delta_vals'),
            FloatingField('eta_vals'),
            FloatingField('rd_vals'),
            FloatingField('a_vals'),
            HTML('<hr><h5>Valores Iniciales para Newton-Raphson</h5>'),
            HTML('<p class="text-muted small">Especifique el rango de valores iniciales (semillas) para el método de Newton-Raphson. Se generará una malla de puntos iniciales en el espacio (c, s, i).</p>'),
            Row(
                Column(FloatingField('seeds_c_min'), css_class='col-md-6'),
                Column(FloatingField('seeds_c_max'), css_class='col-md-6'),
            ),
            Row(
                Column(FloatingField('seeds_s_min'), css_class='col-md-6'),
                Column(FloatingField('seeds_s_max'), css_class='col-md-6'),
            ),
            Row(
                Column(FloatingField('seeds_i_min'), css_class='col-md-6'),
                Column(FloatingField('seeds_i_max'), css_class='col-md-6'),
            ),
            FloatingField('seeds_n_points'),
            HTML('<hr>'),
            FloatingField('generate_scenarios'),
            Submit('submit', 'Ejecutar búsqueda de EE (c, s, i)', css_class='btn btn-primary mt-3 btn-lg')
        )
    
    def clean_rc_vals(self):
        vals_str = self.cleaned_data['rc_vals']
        try:
            return [float(x.strip()) for x in vals_str.split(',')]
        except ValueError:
            raise forms.ValidationError("Valores inválidos. Use números separados por comas.")
    
    def clean_beta_vals(self):
        vals_str = self.cleaned_data['beta_vals']
        try:
            return [float(x.strip()) for x in vals_str.split(',')]
        except ValueError:
            raise forms.ValidationError("Valores inválidos. Use números separados por comas.")
    
    def clean_delta_vals(self):
        vals_str = self.cleaned_data['delta_vals']
        try:
            return [float(x.strip()) for x in vals_str.split(',')]
        except ValueError:
            raise forms.ValidationError("Valores inválidos. Use números separados por comas.")
    
    def clean_eta_vals(self):
        vals_str = self.cleaned_data['eta_vals']
        try:
            return [float(x.strip()) for x in vals_str.split(',')]
        except ValueError:
            raise forms.ValidationError("Valores inválidos. Use números separados por comas.")
    
    def clean_rd_vals(self):
        vals_str = self.cleaned_data['rd_vals']
        try:
            return [float(x.strip()) for x in vals_str.split(',')]
        except ValueError:
            raise forms.ValidationError("Valores inválidos. Use números separados por comas.")
    
    def clean_a_vals(self):
        vals_str = self.cleaned_data['a_vals']
        try:
            return [float(x.strip()) for x in vals_str.split(',')]
        except ValueError:
            raise forms.ValidationError("Valores inválidos. Use números separados por comas.")
    
    def clean(self):
        cleaned_data = super().clean()
        use_control = cleaned_data.get('use_adaptive_control', False)
        
        # Si el control adaptativo está activado, ku y eps_u son requeridos
        if use_control:
            if not cleaned_data.get('ku'):
                raise forms.ValidationError({
                    'ku': 'Este campo es requerido cuando el control adaptativo está activado.'
                })
            if not cleaned_data.get('eps_u'):
                raise forms.ValidationError({
                    'eps_u': 'Este campo es requerido cuando el control adaptativo está activado.'
                })
        
        return cleaned_data
