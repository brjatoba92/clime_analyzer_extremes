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
