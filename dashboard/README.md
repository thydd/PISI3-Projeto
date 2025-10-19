# Spotify Insights Dashboard - Estrutura Modular

## 📋 Visão Geral

Dashboard analítico modular e escalável para exploração de dados do Spotify, desenvolvido em Python com Dash/Plotly. Este projeto foi reestruturado para permitir desenvolvimento colaborativo independente, com separação clara de responsabilidades em módulos.

## 🎯 Funcionalidades Implementadas (Versão 1.0)

### ✅ Aba: Visão Geral
- **KPIs dinâmicos**: Total de músicas, artistas, álbuns e playlists
- **Dados ausentes**: Tabela com contagem de valores faltantes
- **Estatísticas descritivas**: Métricas estatísticas das principais features
- **Hierarquia de Gêneros**: Sunburst interativo mostrando gêneros e subgêneros
- **Distribuição de Popularidade**: Violin plot por gênero
- **Tendência Temporal**: Evolução de lançamentos ao longo dos anos

### ✅ Aba: Popularidade
- **Network de Artistas**: Bubble chart dos top 30 artistas
- **Top 10 Artistas**: Ranking por popularidade média
- **Distribuição de Gêneros**: Contagem de faixas por gênero
- **Distribuição de Tons**: Análise de keys musicais (C, D, E, etc.)
- **Danceability por Gênero**: Comparativo de dançabilidade

### ✅ Aba: Audio DNA
- **Scatter 3D Interativo**: Exploração multidimensional de features
- **Heatmap de Correlação**: Matriz de correlação entre audio features
- **Gráfico Radar**: Perfil médio de features por gênero
- **Controles dinâmicos**: Seleção de eixos X, Y, Z e coloração

## 📁 Estrutura do Projeto

```
dashboard/
├── app.py                          # 🚀 Arquivo principal (limpo, ~330 linhas)
├── app_legacy.py                   # 📦 Backup do app antigo (>1000 linhas)
├── next_features.txt               # 📝 Roadmap de funcionalidades pendentes
├── requirements.txt                # 📦 Dependências Python
├── README.md                       # 📖 Este arquivo
│
├── app/                            # 🏗️ Estrutura modular do aplicativo
│   ├── __init__.py
│   ├── config.py                   # ⚙️ Configurações globais (cores, estilos)
│   │
│   ├── layouts/                    # 🎨 Layouts das abas
│   │   ├── __init__.py
│   │   ├── overview_tab.py         # ✅ Layout da Visão Geral
│   │   ├── popularity_tab.py       # ✅ Layout da Popularidade
│   │   └── audio_dna_tab.py        # ✅ Layout do Audio DNA
│   │
│   ├── callbacks/                  # 🔄 Lógica de interatividade
│   │   ├── __init__.py
│   │   ├── overview_callbacks.py   # ✅ Callbacks da Visão Geral
│   │   ├── popularity_callbacks.py # ✅ Callbacks da Popularidade
│   │   └── audio_dna_callbacks.py  # ✅ Callbacks do Audio DNA
│   │
│   └── utils/                      # 🛠️ Utilitários compartilhados
│       ├── __init__.py
│       ├── data_utils.py           # 📊 Funções de dados (load, filter, KPIs)
│       ├── model_utils.py          # 🤖 ML (Random Forest, K-Means)
│       ├── visualizations.py       # 📈 Gráficos Plotly customizados
│       └── common_components.py    # 🧩 Componentes reutilizáveis
│
└── assets/
    └── global.css                  # 🎨 Estilos personalizados
```

## 🚀 Como Executar

### Pré-requisitos
- Python 3.8+
- Dataset: `../DataSet/spotify_songs.csv` (relativo ao diretório do dashboard)

### Instalação

```bash
# 1. Navegue até o diretório do projeto
cd dashboard

# 2. Crie um ambiente virtual (opcional, mas recomendado)
python -m venv venv
.\venv\Scripts\Activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Execute o dashboard
python app.py
```

### Acesso
- Abra o navegador em: **http://127.0.0.1:8050**
- O dashboard será executado em modo debug por padrão

## 🔧 Arquitetura Modular

### Fluxo de Dados

```
┌─────────────┐
│   app.py    │  ← Ponto de entrada
└──────┬──────┘
       │
       ├──► Carrega BASE_DF (data_utils.load_dataset)
       │
       ├──► Cria layout principal
       │    ├─► Filtros globais (gênero, popularidade, ano)
       │    └─► Abas (overview, popularity, audio-dna)
       │
       └──► Registra callbacks
            ├─► overview_callbacks.register_overview_callbacks()
            ├─► popularity_callbacks.register_popularity_callbacks()
            └─► audio_dna_callbacks.register_audio_dna_callbacks()
```

### Separação de Responsabilidades

| Módulo | Responsabilidade |
|--------|------------------|
| **`app.py`** | Inicialização, layout principal, filtros globais |
| **`config.py`** | Constantes de estilo, cores, tema Bootstrap |
| **`layouts/*.py`** | Estrutura HTML/Dash dos componentes de cada aba |
| **`callbacks/*.py`** | Lógica de interatividade (Inputs → Outputs) |
| **`utils/data_utils.py`** | Carregamento e transformação de dados |
| **`utils/model_utils.py`** | Modelos de ML (classificação, clustering) |
| **`utils/visualizations.py`** | Gráficos Plotly customizados |
| **`utils/common_components.py`** | Componentes reutilizáveis (sliders, filtros) |

## 📊 Filtros Globais

Todos os gráficos e tabelas respondem dinamicamente aos seguintes filtros:

- **Gêneros**: Seleção múltipla de gêneros musicais
- **Subgêneros**: Filtragem em cascata baseada nos gêneros selecionados
- **Popularidade**: Range slider (0-100)
- **Ano de Lançamento**: Range slider baseado nos dados disponíveis

### Como os Filtros Funcionam

```python
# Em qualquer callback:
from app.utils.common_components import apply_filters

@app.callback(...)
def update_graph(genres, subgenres, popularity, years):
    filtered_df = apply_filters(BASE_DF, genres, subgenres, popularity, years)
    # Use filtered_df para gerar visualizações
    return create_figure(filtered_df)
```

## 🎨 Estilo e Tema

### Tema Visual
- **Bootstrap**: Darkly (via Bootswatch)
- **Paleta de Cores**:
  - Primary: `#4cc9f0` (Azul cyan)
  - Secondary: `#ff6dc4` (Rosa)
  - Tertiary: `#ffd60a` (Amarelo)
- **Tipografia**: Poppins, Inter, Sans-serif
- **Efeito Glass**: Cards com background semi-transparente

### Customização
Todas as constantes de estilo estão centralizadas em `app/config.py`:

```python
from app.config import (
    PRIMARY_TEXT,
    SECONDARY_TEXT,
    ACCENT_COLOR,
    CARD_BACKGROUND,
    TABLE_HEADER_STYLE,
    # ...
)
```

## 🔮 Próximas Funcionalidades

Consulte o arquivo **`next_features.txt`** para instruções detalhadas sobre como implementar:

1. **Aba: Humor & Tempo**
   - Distribuição de BPM por gênero (violin plot)
   - Mapa emocional Energia × Valência (heatmap 2D)

2. **Aba: Explorador**
   - Tabela interativa filtrável
   - Histogramas dinâmicos de features
   - Matriz de dispersão (scatter matrix)
   - Download de dados filtrados

3. **Aba: Classificação**
   - Predição de gênero musical (Random Forest)
   - Sliders para input de features
   - Tabela de probabilidades

4. **Aba: Clusters**
   - K-Means clustering (2-10 clusters)
   - Visualização PCA 2D
   - Perfil médio dos clusters
   - Amostra de músicas por cluster

## 👥 Desenvolvimento Colaborativo

### Workflow Recomendado

1. **Escolha uma funcionalidade** de `next_features.txt`
2. **Crie uma branch** dedicada:
   ```bash
   git checkout -b feature/mood-tempo-tab
   ```
3. **Implemente o layout** em `app/layouts/<nome>_tab.py`
4. **Implemente os callbacks** em `app/callbacks/<nome>_callbacks.py`
5. **Integre no app.py**:
   - Importe layout e callbacks
   - Adicione aba em `dcc.Tabs`
   - Registre callbacks
6. **Teste independentemente** a nova aba
7. **Faça commit e push**:
   ```bash
   git add .
   git commit -m "feat: adiciona aba Humor & Tempo"
   git push origin feature/mood-tempo-tab
   ```
8. **Abra Pull Request** para review

### Evitando Conflitos

- ✅ **Cada aba em arquivos separados** → sem conflitos de merge
- ✅ **IDs únicos** para componentes Dash
- ✅ **Não modifique** `app/config.py` sem coordenação
- ✅ **Use** `apply_filters()` de `common_components.py`
- ✅ **Registre callbacks** em funções `register_*_callbacks()`

## 📦 Dependências Principais

```
dash>=2.14.0
plotly>=5.17.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
```

Veja `requirements.txt` para lista completa.

## 🐛 Troubleshooting

### Dataset não encontrado
**Erro**: `DatasetNotFoundError: Dataset não encontrado em '...'`

**Solução**: Verifique se `spotify_songs.csv` está em:
```
../DataSet/spotify_songs.csv
```

### Imports não resolvidos
**Erro**: `ModuleNotFoundError: No module named 'app'`

**Solução**: Execute sempre de dentro do diretório `dashboard/`:
```bash
cd dashboard
python app.py
```

### Callbacks duplicados
**Erro**: `DuplicateCallbackOutput`

**Solução**: Cada Output ID deve ser único. Verifique se não há IDs duplicados entre abas.

## 📄 Licença

Este projeto é de uso educacional/acadêmico.

## 🤝 Contribuindo

1. Fork o repositório
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

---

**Última atualização**: Outubro 2025  
**Versão**: 1.0 (Modular)  
**Status**: ✅ Pronto para colaboração
