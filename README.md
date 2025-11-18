## 🎵 Spotify Insights Dashboard – PISI3

Dashboard analítico desenvolvido em Python (Dash/Plotly) para explorar **32.833 músicas do Spotify (1950–2020)**, com foco em visualização de dados e machine learning (classificação de gênero e clusterização de músicas).

Este repositório reúne:
- `dashboard/`: aplicação web interativa (Dash)
- `DataSet/`: dataset `spotify_songs.csv`
- `EDA/`: notebooks/scripts de análise exploratória
- `Machine_Learning/`: scripts de treinamento e exemplos de modelos

Para detalhes completos da aplicação, consulte também o `dashboard/README.md`.

---

## 🧱 Estrutura do Projeto

```bash
PISI3-Projeto/
├── dashboard/          # Dashboard interativo em Dash
├── DataSet/            # Dataset original (CSV)
├── EDA/                # Análises exploratórias (scripts)
├── Machine_Learning/   # Modelos e experimentos de ML
└── README.md           # Este arquivo
```

### `dashboard/`
- `app.py`: ponto de entrada da aplicação Dash
- `app/`: módulo da aplicação (layouts, callbacks, utils, config)
- `assets/`: estilos (por exemplo, `global.css`)
- `requirements.txt`: dependências Python específicas do dashboard
- Documentação adicional (`README.md`, `GUIA_RAPIDO.md`, etc.)

### `DataSet/`
- `spotify_songs.csv`: dataset com ~33k músicas do Spotify, incluindo
  características de áudio (danceability, energy, valence, tempo, etc.),
  gênero, popularidade e informações de identificação.

### `EDA/`
Scripts de exploração inicial do dataset (gráficos, estatísticas, estrutura).

### `Machine_Learning/`
Scripts focados em:
- Clusterização de músicas por "DNA musical" e aspectos técnicos
- Classificação de gênero e modo (maior/menor)
- Exemplos de carregamento e uso de modelos treinados

---

## 📦 Requisitos

Você precisará de:
- Python 3.10+ (recomendado)
- Pip ou gerenciador de pacotes equivalente

Dependências principais do dashboard (também listadas em `dashboard/requirements.txt`):
- `pandas>=2.1.0`
- `numpy>=1.24.0`
- `plotly>=5.18.0`
- `scikit-learn>=1.3.0`
- `dash>=2.17.0`

---

## 🚀 Como Executar o Dashboard

1. Acesse a pasta do dashboard:
	```bash
	cd dashboard
	```

2. (Opcional, mas recomendado) Crie e ative um ambiente virtual:
	```bash
	python -m venv .venv
	source .venv/bin/activate
	```

3. Instale as dependências:
	```bash
	pip install -r requirements.txt
	```

4. Verifique se o dataset existe em `../DataSet/spotify_songs.csv`.

5. Execute a aplicação:
	```bash
	python app.py
	```

6. Abra o navegador em: **http://127.0.0.1:8050**

Para uma visão mais detalhada de execução, scripts auxiliares e troubleshooting,
veja `dashboard/README.md` e `dashboard/GUIA_RAPIDO.md`.

---

## 📊 Funcionalidades Principais do Dashboard

O dashboard é organizado em **7 abas** principais:

1. **Visão Geral** – KPIs, treemap de gêneros, timeline de lançamentos, top músicas.
2. **Popularidade** – distribuição de popularidade, correlações e hierarquia de gêneros.
3. **Audio DNA** – análise profunda de características de áudio (gráficos radar, scatter 3D).
4. **Humor & Tempo** – relação entre BPM, energia, valência e emoções musicais.
5. **Explorador** – tabela interativa, histogramas dinâmicos e scatter matrix.
6. **Classificação (ML)** – predição de gênero musical com Random Forest.
7. **Clusters (K-Means)** – agrupamento de músicas similares com visualização em PCA.

Filtros globais de gênero, subgênero, popularidade e ano se aplicam a todas as abas.

Mais detalhes de cada aba estão documentados em `dashboard/README.md`.

---

## 🧪 Scripts de EDA e Machine Learning

Além do dashboard, o repositório inclui scripts em:

- `EDA/`: exploração da estrutura do dataset, gráficos de BPM, DNA emocional,
  gráficos de barras por gênero e outras análises descritivas.
- `Machine_Learning/`: experimentos de clusterização, classificação de gênero e
  modo, além de exemplos de carregamento e uso de modelos.

Esses arquivos podem ser usados como base para estudos, ajustes em modelos ou
novas análises sobre o dataset.

---

## 🤝 Contribuição

Sugestões, correções e melhorias são bem-vindas.

Possíveis contribuições:
- Novas visualizações ou abas no dashboard
- Novos modelos ou abordagens de ML na pasta `Machine_Learning/`
- Melhoria dos scripts de EDA
- Tradução e ampliação da documentação

Abra uma *issue* ou envie um *pull request* com a descrição clara da mudança.

---

## 📜 Licença

Este projeto é distribuído sob os termos da licença descrita em `LICENSE` na
raiz do repositório.

---

Desenvolvido com foco em visualização de dados, ciência de dados e música. 🎶
