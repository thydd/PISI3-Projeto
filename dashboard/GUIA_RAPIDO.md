# 🚀 GUIA RÁPIDO - Spotify Insights Dashboard

## ⚡ Início Rápido (3 Passos)

### 1️⃣ Instalar Dependências
```bash
pip install -r requirements.txt
```

### 2️⃣ Executar o Dashboard
```bash
python app.py
```

### 3️⃣ Abrir no Navegador
Acesse: **http://127.0.0.1:8050**

---

## ✅ Funcionalidades Disponíveis Agora

### Aba 1: Visão Geral
- 📊 KPIs (músicas, artistas, álbuns, playlists)
- 📋 Tabela de dados ausentes
- 📈 Estatísticas descritivas
- 🌳 Sunburst de gêneros/subgêneros
- 🎻 Violin plot de popularidade
- 📅 Timeline de lançamentos

### Aba 2: Popularidade  
- 🌐 Network de top artistas
- 🏆 Top 10 artistas
- 📊 Distribuição de gêneros
- 🎵 Distribuição de tons (keys)
- 💃 Danceability por gênero

### Aba 3: Audio DNA
- 🧬 Scatter 3D interativo
- 🔥 Heatmap de correlação
- 🎯 Radar chart por gênero

---

## 🔄 Como Adicionar Nova Aba

### Exemplo: Adicionar aba "Humor & Tempo"

#### 1. Criar Layout (`app/layouts/mood_tempo_tab.py`)
```python
from dash import dcc, html

def create_mood_tempo_layout(genre_options):
    return html.Div(
        className="p-4",
        children=[
            # Seus componentes aqui
            dcc.Graph(id="mood-graph"),
        ]
    )
```

#### 2. Criar Callbacks (`app/callbacks/mood_tempo_callbacks.py`)
```python
from dash import Input, Output

def register_mood_tempo_callbacks(app, base_df):
    @app.callback(
        Output("mood-graph", "figure"),
        Input("genre-dropdown", "value"),
    )
    def update_mood(genres):
        # Sua lógica aqui
        return figure
```

#### 3. Integrar no `app.py`

**No início do arquivo:**
```python
from app.layouts.mood_tempo_tab import create_mood_tempo_layout
from app.callbacks.mood_tempo_callbacks import register_mood_tempo_callbacks
```

**Na seção de abas:**
```python
dcc.Tab(
    label="Humor & Tempo",
    value="mood",
    style=TAB_STYLE,
    selected_style=TAB_SELECTED_STYLE,
    children=[create_mood_tempo_layout(GENRE_OPTIONS)],
),
```

**Na seção de registro de callbacks:**
```python
register_mood_tempo_callbacks(app, BASE_DF)
```

#### 4. Testar
```bash
python verify_structure.py  # Verificar estrutura
python app.py               # Executar dashboard
```

---

## 📁 Estrutura Simplificada

```
dashboard/
├── app.py                  # Arquivo principal
├── app/
│   ├── config.py           # Cores, estilos, tema
│   ├── layouts/            # HTML das abas
│   │   ├── overview_tab.py
│   │   ├── popularity_tab.py
│   │   └── audio_dna_tab.py
│   ├── callbacks/          # Lógica interativa
│   │   ├── overview_callbacks.py
│   │   ├── popularity_callbacks.py
│   │   └── audio_dna_callbacks.py
│   └── utils/              # Funções auxiliares
│       ├── data_utils.py
│       ├── model_utils.py
│       ├── visualizations.py
│       └── common_components.py
└── next_features.txt       # Roadmap
```

---

## 🛠️ Funções Úteis

### Aplicar Filtros aos Dados
```python
from app.utils.common_components import apply_filters

filtered_df = apply_filters(
    BASE_DF,
    genres=["pop", "rock"],
    subgenres=None,
    popularity=[50, 100],
    years=[2010, 2020]
)
```

### Criar Gráficos
```python
from app.utils.visualizations import bar_chart, feature_3d_scatter

# Gráfico de barras
fig = bar_chart(data_series, title="Meu Gráfico", x_title="X", y_title="Y")

# Scatter 3D
fig = feature_3d_scatter(df, x="energy", y="valence", z="danceability")
```

### Criar Componentes
```python
from app.utils.common_components import slider_component

slider = slider_component(
    feature="danceability",
    label="Danceability",
    step=0.01,
    feature_ranges=FEATURE_RANGES
)
```

---

## 🔍 Verificação de Integridade

Antes de fazer commit, execute:

```bash
python verify_structure.py
```

**Deve mostrar**:
```
✅ Imports OK
✅ Dataset OK (32,833 músicas)
✅ Estrutura de arquivos OK
✅ Layouts OK
✅ Features OK
✅ Filtros OK
```

---

## 🎯 Próximas Features (Implementação Futura)

Consulte `next_features.txt` para instruções detalhadas sobre:

1. ⏳ **Humor & Tempo**: BPM distributions, mood map
2. ⏳ **Explorador**: Tabela filtrável, histogramas, scatter matrix
3. ⏳ **Classificação**: Predição de gênero com Random Forest
4. ⏳ **Clusters**: K-Means clustering com visualização PCA

---

## 💡 Dicas

### Performance
- Limite scatter plots a 1000-2000 amostras
- Use `@lru_cache` para funções pesadas

### Estilo
- Use classes Bootstrap: `row`, `col-lg-6`, `card`, etc.
- Use classes personalizadas: `glass-card` para efeito vidro

### IDs
- Use IDs descritivos e únicos: `mood-density-graph`, não `graph1`
- Evite conflitos entre abas

### Callbacks
- Sempre registre em funções `register_*_callbacks(app, ...)`
- Use `prevent_initial_call=True` quando necessário

---

## 📞 Troubleshooting

### Dataset não encontrado
```bash
# Verifique se o arquivo existe em:
ls ../DataSet/spotify_songs.csv
```

### Imports não funcionam
```bash
# Execute sempre de dentro da pasta dashboard:
cd dashboard
python app.py
```

### Erros de callbacks
- IDs duplicados? Verifique com `Ctrl+F` no código
- Output já registrado? Cada Output só pode ter 1 callback

---

## 📚 Documentação Completa

- **README.md**: Documentação detalhada
- **next_features.txt**: Guia de implementação de features
- **REESTRUTURACAO_SUMARIO.md**: Resumo da reestruturação

---

**🎉 Pronto para começar!**

Execute `python app.py` e acesse http://127.0.0.1:8050
