# Analisador de Extremos Climáticos
O Analisador de Extremos Climáticos é uma ferramenta robusta desenvolvida em Python para análise de eventos climáticos extremos, como temperaturas e precipitações extremas. Ele utiliza métodos estatísticos avançados, incluindo a distribuição GEV (Generalized Extreme Value), teste de Mann-Kendall para tendências e cálculos de períodos de retorno, para fornecer insights detalhados sobre dados climáticos. O sistema também gera visualizações gráficas e relatórios completos para facilitar a interpretação dos resultados.

# Funcionalidades
- Carregamento de Dados: Suporta dados climáticos em formato CSV ou DataFrame, com geração de dados sintéticos para demonstração.
- Extração de Extremos: Implementa métodos como Block Maxima e Peaks Over Threshold (POT) para identificar eventos extremos.
- Ajuste de Distribuição GEV: Ajusta a distribuição GEV aos dados extremos e realiza testes de qualidade de ajuste (Kolmogorov-Smirnov).
- Análise de Tendências: Utiliza o teste de Mann-Kendall e a inclinação de Sen para detectar tendências significativas.
- Períodos de Retorno: Calcula níveis de retorno para diferentes períodos (ex.: 2, 5, 10, 25, 50, 100 anos).
- Projeções Futuras: Gera cenários futuros com base em tendências observadas (baixo, médio e alto).
- Avaliação de Risco: Classifica riscos de tendências e extremos, fornecendo recomendações práticas.
- Visualizações: Cria gráficos detalhados, incluindo séries temporais, distribuições, níveis de retorno, projeções futuras, boxplots e mapas de calor mensais.
- Relatórios: Gera relatórios completos e exporta resultados em CSV.

# Dependências
As seguintes bibliotecas Python são necessárias:
- numpy
- pandas
- matplotlib
- seaborn
- scipy
- pymannkendall

Instalação das Dependências
Para instalar todas as dependências, execute:

```
pip install numpy pandas matplotlib seaborn scipy pymannkendall
```

# Como Usar

1. Clonar o Repositório
git clone https://github.com/seu-usuario/climate-extreme-analyzer.git
cd climate-extreme-analyzer

2. Estrutura de Dados

Os dados climáticos devem conter pelo menos uma coluna date (formato datetime) ou year, e colunas para variáveis como temperature e precipitation. Exemplo de CSV:
```
date,temperature,precipitation
2000-01-01,20.5,2.3
2000-01-02,21.0,0.0
```

...

3. Exemplo de Código

```
from climate_extreme_analyzer import ClimateExtremeAnalyzer
import pandas as pd
```

3.1 Inicializar o analisador
analyzer = ClimateExtremeAnalyzer()

3.2 Carregar dados (exemplo com DataFrame)
```
data = pd.DataFrame({
    'date': pd.date_range('2000-01-01', periods=100),
    'temperature': np.random.normal(20, 5, 100)
})
analyzer.load_data(data_df=data)
```

3.3 Gerar relatório completo para temperatura e precipitação
```
report = analyzer.generate_climate_report(variables=['temperature'], save_dir='./climate_plots')
```

3.4 Exportar resultados
```
analyzer.export_results(report, filename='results/climate_analysis')
```

4. Saídas

Gráficos: Salvos no diretório especificado (padrão: ./climate_plots).
Relatório CSV: Resumo dos resultados exportado para o diretório especificado (ex.: results/climate_analysis_summary.csv).
Console: Exibe estatísticas, resultados de tendências, períodos de retorno, projeções e recomendações.

Exemplo de Saída Gráfica
Os gráficos gerados incluem:

- Série temporal com tendência (Sen's slope).
- Distribuição dos extremos com ajuste GEV.
- Níveis de retorno para diferentes períodos.
- Projeções futuras para cenários baixo, médio e alto.
- Boxplot da variabilidade anual (últimos 10 anos).
- Mapa de calor mensal ao longo dos anos.

Contribuição
Contribuições são bem-vindas! Siga estas etapas para contribuir:

Faça um fork do repositório.
Crie uma branch para sua feature (git checkout -b feature/nova-funcionalidade).
Commit suas alterações (git commit -m 'Adiciona nova funcionalidade').
Envie para o repositório remoto (git push origin feature/nova-funcionalidade).
Abra um Pull Request.

Por favor, siga as diretrizes de estilo de código PEP 8 e inclua testes para novas funcionalidades.
Licença
Este projeto está licenciado sob a MIT License.

Contato
Para dúvidas ou sugestões, abra uma issue ou entre em contato com seu-email@exemplo.com.

🌍 Analisador de Extremos Climáticos - Entendendo o clima, protegendo o futuro!
