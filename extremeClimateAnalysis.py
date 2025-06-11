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
