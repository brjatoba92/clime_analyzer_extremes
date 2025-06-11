# Dependencias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import genextreme, norm
import warnings
warnings.filterwarnings('ignore')

#Mann-Kendall
try:
    from pymannkendall import original_test as mk_test
except:
    print("Instalando pymannkendall: pip install pymannkendall")
    # Implementação manual do teste Mann-Kendall
    def mk_test(data):
        n= len(data)
        s = 0
        for i in range(n-1):
            for j in range(i+1, n):
                if data[j]> data[i]:
                    s += 1
                elif data[j]< data[i]:
                    s -= 1
        
        var_s = n*(n-1)*(2*n+5)/18
        if s> 0:
            z = (s-1)/np.sqrt(var_s)
        elif s< 0:
            z = (s+1)/np.sqrt(var_s)
        else:
            z = 0
        p_value = 2*(1-stats.norm.cdf(abs(z)))
        trend = 'increasing' if z > 0 else 'decreasing' if z < 0 else 'no trend'

        class MKResult:
            def __init__(self, trend, p, z, s):
                self.trend = trend
                self.p = p
                self.z = z
                self.s = s
        return MKResult(trend, p_value, z, s)

class ClimateExtremeAnalyzer:
    """
    Sistema robusto para análise de eventos extremos climáticos
    """
    
    def __init__(self):
        self.data = None
        self.extreme_data = None
        self.gev_params = {}
        self.trends = {}
        
    def load_data(self, data_path=None, data_df=None):
        """
        Carrega dados climáticos
        """
        if data_df is not None:
            self.data = data_df
        elif data_path:
            # Carregar de arquivo
            if data_path.endswith('.csv'):
                self.data = pd.read_csv(data_path, parse_dates=['date'])
            else:
                raise ValueError("Formato de arquivo não suportado")
        else:
            # Gerar dados sintéticos para demonstração
            self.generate_synthetic_data()
            
        print(f"Dados carregados: {len(self.data)} registros")
        return self.data
    
    def generate_synthetic_data(self):
        """
        Gera dados climáticos sintéticos para demonstração
        """
        np.random.seed(42)
        
        # Gerar 30 anos de dados diários
        dates = pd.date_range('1990-01-01', '2023-12-31', freq='D')
        n_days = len(dates)
        
        # Tendência de aquecimento
        trend = np.linspace(0, 2, n_days)  # Aumento de 2°C em 30 anos
        
        # Sazonalidade
        day_of_year = dates.dayofyear
        seasonal = 10 * np.sin(2 * np.pi * day_of_year / 365.25)
        
        # Temperatura base + tendência + sazonalidade + ruído
        temperature = 20 + trend + seasonal + np.random.normal(0, 3, n_days)
        
        # Eventos extremos ocasionais
        extreme_events = np.random.exponential(0.1, n_days) * np.random.choice([0, 1], n_days, p=[0.95, 0.05])
        temperature += extreme_events * 15
        
        # Precipitação (distribuição gamma)
        precipitation = np.random.gamma(0.5, 2, n_days)
        # Eventos extremos de chuva
        extreme_rain = np.random.choice([0, 1], n_days, p=[0.98, 0.02]) * np.random.exponential(50, n_days)
        precipitation += extreme_rain
        
        self.data = pd.DataFrame({
            'date': dates,
            'temperature': temperature,
            'precipitation': precipitation,
            'year': dates.year,
            'month': dates.month
        })
        print("Dados sintéticos gerados com sucesso!")
    
    def extract_extremes(self, variable='temperature', method='block_maxima', block_size='year'):
        """
        Extrai valores extremos usando diferentes métodos
        """
        if method == 'block_maxima':
            if block_size == 'year':
                extremes = self.data.groupby('year')[variable].max()
            elif block_size == 'month':
                extremes = self.data.groupby(['year', 'month'])[variable].max()
            else:
                raise ValueError("Block size deve ser 'year' ou 'month'")
                
        elif method == 'peaks_over_threshold':
            # Método POT (Peaks Over Threshold)
            threshold = np.percentile(self.data[variable], 95)  # Top 5%
            extremes = self.data[self.data[variable] > threshold][variable]
            
        self.extreme_data = extremes
        print(f"Extraídos {len(extremes)} valores extremos usando método {method}")
        return extremes
    
    def fit_gev_distribution(self, variable='temperature'):
        """
        Ajusta distribuição GEV (Generalized Extreme Value) aos dados extremos
        """
        if self.extreme_data is None:
            self.extract_extremes(variable)
        
        # Ajustar distribuição GEV
        gev_params = genextreme.fit(self.extreme_data)
        
        self.gev_params[variable] = {
            'shape': gev_params[0],    # parâmetro de forma (xi)
            'location': gev_params[1], # parâmetro de localização (mu)
            'scale': gev_params[2]     # parâmetro de escala (sigma)
        }
        
        # Teste de ajuste Kolmogorov-Smirnov
        ks_stat, ks_p_value = stats.kstest(self.extreme_data, 
                                          lambda x: genextreme.cdf(x, *gev_params))
        
        print(f"Parâmetros GEV para {variable}:")
        print(f"  Forma (ξ): {gev_params[0]:.4f}")
        print(f"  Localização (μ): {gev_params[1]:.4f}")
        print(f"  Escala (σ): {gev_params[2]:.4f}")
        print(f"  Teste KS: estatística = {ks_stat:.4f}, p-valor = {ks_p_value:.4f}")
        
        return gev_params
    
    def calculate_return_periods(self, variable='temperature', return_periods=[2, 5, 10, 25, 50, 100]):
        """
        Calcula períodos de retorno para diferentes níveis
        """
        if variable not in self.gev_params:
            self.fit_gev_distribution(variable)
        
        params = self.gev_params[variable]
        shape, loc, scale = params['shape'], params['location'], params['scale']
        
        return_levels = {}
        
        for T in return_periods:
            # Fórmula para período de retorno usando GEV
            if abs(shape) < 1e-6:  # Gumbel (shape ≈ 0)
                level = loc - scale * np.log(-np.log(1 - 1/T))
            else:  # GEV geral
                level = loc + (scale/shape) * ((-np.log(1 - 1/T))**(-shape) - 1)
            
            return_levels[T] = level
        
        return return_levels
    
    def mann_kendall_test(self, variable='temperature', alpha=0.05):
        """
        Realiza teste de Mann-Kendall para detectar tendências
        """
        annual_data = self.data.groupby('year')[variable].mean()
        
        # Teste Mann-Kendall
        mk_result = mk_test(annual_data.values)
        
        # Sen's slope (estimativa robusta da inclinação)
        n = len(annual_data)
        slopes = []
        
        for i in range(n-1):
            for j in range(i+1, n):
                slope = (annual_data.iloc[j] - annual_data.iloc[i]) / (j - i)
                slopes.append(slope)
        
        sens_slope = np.median(slopes)
        
        self.trends[variable] = {
            'trend': mk_result.trend,
            'p_value': mk_result.p,
            'z_score': mk_result.z,
            'sens_slope': sens_slope,
            'significant': mk_result.p < alpha
        }
        
        print(f"Análise de tendência para {variable}:")
        print(f"  Tendência: {mk_result.trend}")
        print(f"  Sen's slope: {sens_slope:.6f} por ano")
        print(f"  Z-score: {mk_result.z:.4f}")
        print(f"  P-valor: {mk_result.p:.6f}")
        print(f"  Significativo (α={alpha}): {mk_result.p < alpha}")
        
        return self.trends[variable]
    
    def project_future_scenarios(self, variable='temperature', future_years=30, scenarios=None):
        """
        Gera projeções de cenários futuros baseadas em tendências
        """
        if scenarios is None:
            scenarios = {
                'Baixo': 0.5,    # Fator de multiplicação da tendência atual
                'Médio': 1.0,    # Tendência atual mantida
                'Alto': 1.5      # Tendência acelerada
            }
        
        if variable not in self.trends:
            self.mann_kendall_test(variable)
        
        current_trend = self.trends[variable]['sens_slope']
        current_year = self.data['year'].max()
        future_years_range = range(current_year + 1, current_year + future_years + 1)
        
        # Valor base (média dos últimos 5 anos)
        base_value = self.data[self.data['year'] >= current_year - 4].groupby('year')[variable].mean().mean()
        
        projections = {}
        
        for scenario_name, factor in scenarios.items():
            yearly_change = current_trend * factor
            projection = []
            
            for i, year in enumerate(future_years_range):
                projected_value = base_value + (yearly_change * (i + 1))
                projection.append(projected_value)
            
            projections[scenario_name] = {
                'years': list(future_years_range),
                'values': projection
            }
        
        return projections
