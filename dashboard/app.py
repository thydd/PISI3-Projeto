"""
Spotify Insights Dashboard - Aplicativo Principal

Dashboard analítico modular para exploração de dados do Spotify.
Versão inicial com funcionalidades: Visão Geral, Popularidade e Audio DNA.

Autor: Dashboard Team
Data: 2025
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, State, dcc, html
from pathlib import Path

# Importações de configuração
from app.config import (
    BACKGROUND_GRADIENT,
    EXTERNAL_STYLESHEETS,
    FONT_FAMILY,
    GOOGLE_FONTS_URL,
    PRIMARY_TEXT,
    SECONDARY_TEXT,
    TAB_SELECTED_STYLE,
    TAB_STYLE,
)

# Importações de layouts
from app.layouts.audio_dna_tab import create_audio_dna_layout
from app.layouts.overview_tab import create_overview_layout
from app.layouts.popularity_tab import create_popularity_layout

# Importações de callbacks
from app.callbacks.audio_dna_callbacks import register_audio_dna_callbacks
from app.callbacks.overview_callbacks import register_overview_callbacks
from app.callbacks.popularity_callbacks import register_popularity_callbacks

# Importações de utilitários
from app.utils.common_components import create_range_marks
from app.utils.data_utils import load_dataset
from app.utils.model_utils import CLASSIFIER_FEATURES

# ============================================================================
# CARREGAMENTO DE DADOS E CONFIGURAÇÕES INICIAIS
# ============================================================================

BASE_DF = load_dataset()
TOTAL_SONGS = len(BASE_DF)
GENRE_OPTIONS = sorted(BASE_DF["playlist_genre"].dropna().unique())
SUBGENRE_OPTIONS = sorted(BASE_DF["playlist_subgenre"].dropna().unique())

# Configurações de popularidade
POPULARITY_MIN = int(np.floor(BASE_DF["track_popularity"].min()))
POPULARITY_MAX = int(np.ceil(BASE_DF["track_popularity"].max()))
POPULARITY_MARKS = create_range_marks(POPULARITY_MIN, POPULARITY_MAX)

# Configurações de ano
_valid_years = BASE_DF["release_year"].replace({0: np.nan}).dropna()
if _valid_years.empty:
    _valid_years = BASE_DF["track_album_release_date"].dt.year.dropna()
if _valid_years.empty:
    YEAR_MIN, YEAR_MAX = 2000, 2024
else:
    YEAR_MIN = int(_valid_years.min())
    YEAR_MAX = int(_valid_years.max())
YEAR_MARKS = create_range_marks(YEAR_MIN, YEAR_MAX)

# ============================================================================
# INICIALIZAÇÃO DO APLICATIVO DASH
# ============================================================================

app = Dash(
    __name__,
    external_stylesheets=EXTERNAL_STYLESHEETS,
    suppress_callback_exceptions=True,
    title="Spotify Insights Dashboard",
    update_title=None,
)
server = app.server

# ============================================================================
# LAYOUT PRINCIPAL
# ============================================================================

app.layout = html.Div(
    className="container-fluid py-4",
    style={
        "background": BACKGROUND_GRADIENT,
        "color": PRIMARY_TEXT,
        "minHeight": "100vh",
        "fontFamily": FONT_FAMILY,
        "paddingBottom": "3rem",
    },
    children=[
        # Google Fonts
        html.Link(rel="stylesheet", href=GOOGLE_FONTS_URL),
        # Cabeçalho
        html.Div(
            className="mb-4",
            children=[
                html.H1(
                    "Spotify Songs – Dashboard Analítico", className="fw-bold text-light"
                ),
                html.P(
                    "Explore tendências musicais, comportamento de usuários e análises avançadas de audio features.",
                    className="text-secondary",
                ),
            ],
        ),
        # Painel de Filtros
        html.Div(
            className="card bg-dark border border-secondary shadow-sm mb-4 text-light",
            children=[
                html.Div(
                    className="card-body",
                    children=[
                        html.H5("Filtros", className="card-title"),
                        html.Div(
                            className="row g-3 align-items-end",
                            children=[
                                # Filtro de Gêneros
                                html.Div(
                                    className="col-lg-3 col-md-6",
                                    children=[
                                        html.Label(
                                            "Gêneros", className="form-label text-light"
                                        ),
                                        dcc.Dropdown(
                                            id="genre-dropdown",
                                            options=[
                                                {"label": g.title(), "value": g}
                                                for g in GENRE_OPTIONS
                                            ],
                                            value=[],
                                            multi=True,
                                            placeholder="Selecione gêneros",
                                        ),
                                    ],
                                ),
                                # Filtro de Subgêneros
                                html.Div(
                                    className="col-lg-3 col-md-6",
                                    children=[
                                        html.Label(
                                            "Subgêneros",
                                            className="form-label text-light",
                                        ),
                                        dcc.Dropdown(
                                            id="subgenre-dropdown",
                                            options=[
                                                {"label": s.title(), "value": s}
                                                for s in SUBGENRE_OPTIONS
                                            ],
                                            value=[],
                                            multi=True,
                                            placeholder="Filtre subgêneros",
                                        ),
                                    ],
                                ),
                                # Filtro de Popularidade
                                html.Div(
                                    className="col-lg-3 col-md-6",
                                    children=[
                                        html.Label(
                                            "Popularidade",
                                            className="form-label text-light",
                                        ),
                                        dcc.RangeSlider(
                                            id="popularity-slider",
                                            min=POPULARITY_MIN,
                                            max=POPULARITY_MAX,
                                            step=1,
                                            value=[POPULARITY_MIN, POPULARITY_MAX],
                                            marks=POPULARITY_MARKS,
                                            tooltip={
                                                "placement": "bottom",
                                                "always_visible": False,
                                            },
                                        ),
                                        html.Div(
                                            className="small text-secondary mt-1",
                                            children=[
                                                f"Intervalo atual: {POPULARITY_MIN} – {POPULARITY_MAX}"
                                            ],
                                        ),
                                    ],
                                ),
                                # Filtro de Ano
                                html.Div(
                                    className="col-lg-3 col-md-6",
                                    children=[
                                        html.Label(
                                            "Ano de lançamento",
                                            className="form-label text-light",
                                        ),
                                        dcc.RangeSlider(
                                            id="year-slider",
                                            min=YEAR_MIN,
                                            max=YEAR_MAX,
                                            step=1,
                                            value=[YEAR_MIN, YEAR_MAX],
                                            marks=YEAR_MARKS,
                                            tooltip={
                                                "placement": "bottom",
                                                "always_visible": False,
                                            },
                                        ),
                                        html.Div(
                                            className="small text-secondary mt-1",
                                            children=[
                                                f"Intervalo atual: {YEAR_MIN} – {YEAR_MAX}"
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        # Abas do Dashboard
        dcc.Tabs(
            id="main-tabs",
            value="overview",
            className="modern-tabs bg-dark rounded shadow-sm border border-secondary text-light",
            children=[
                # Aba: Visão Geral
                dcc.Tab(
                    label="Visão Geral",
                    value="overview",
                    style=TAB_STYLE,
                    selected_style=TAB_SELECTED_STYLE,
                    children=[create_overview_layout()],
                ),
                # Aba: Popularidade
                dcc.Tab(
                    label="Popularidade",
                    value="popularity",
                    style=TAB_STYLE,
                    selected_style=TAB_SELECTED_STYLE,
                    children=[create_popularity_layout()],
                ),
                # Aba: Audio DNA
                dcc.Tab(
                    label="🧬 Audio DNA",
                    value="audio-dna",
                    style=TAB_STYLE,
                    selected_style=TAB_SELECTED_STYLE,
                    children=[create_audio_dna_layout(CLASSIFIER_FEATURES)],
                ),
                # TODO: Futuras abas serão adicionadas aqui
                # - Humor & Tempo
                # - Explorador
                # - Classificação
                # - Clusters
            ],
        ),
        # Rodapé
        html.Footer(
            className="mt-4 text-center text-secondary",
            children=[
                html.Small(
                    f"Base de dados: {Path(__file__).resolve().parent.parent / 'DataSet' / 'spotify_songs.csv'}"
                )
            ],
        ),
    ],
)

# ============================================================================
# CALLBACKS GLOBAIS
# ============================================================================


@app.callback(
    Output("subgenre-dropdown", "options"),
    Output("subgenre-dropdown", "value"),
    Input("genre-dropdown", "value"),
    State("subgenre-dropdown", "value"),
)
def update_subgenre_options(selected_genres: list[str], current_values: list[str]):
    """
    Atualiza as opções de subgêneros com base nos gêneros selecionados.

    Parameters
    ----------
    selected_genres : list[str]
        Gêneros selecionados pelo usuário.
    current_values : list[str]
        Valores atuais de subgêneros selecionados.

    Returns
    -------
    tuple
        (opções de subgêneros, valores filtrados)
    """
    if selected_genres:
        filtered_subgenres = (
            BASE_DF[BASE_DF["playlist_genre"].isin(selected_genres)][
                "playlist_subgenre"
            ]
            .dropna()
            .unique()
        )
    else:
        filtered_subgenres = SUBGENRE_OPTIONS

    options = [{"label": s.title(), "value": s} for s in sorted(filtered_subgenres)]
    if current_values:
        new_values = [v for v in current_values if v in filtered_subgenres]
    else:
        new_values = []
    return options, new_values


@app.callback(
    Output("radar-genre", "options"),
    Input("genre-dropdown", "value"),
)
def update_radar_genre_options(selected_genres: list[str]):
    """
    Atualiza opções do dropdown de gênero no radar chart.

    Parameters
    ----------
    selected_genres : list[str]
        Gêneros selecionados nos filtros.

    Returns
    -------
    list
        Lista de opções para o dropdown.
    """
    base_options = [{"label": "Todos", "value": "Todos"}]
    genre_options = [{"label": g.title(), "value": g} for g in GENRE_OPTIONS]
    return base_options + genre_options


# ============================================================================
# REGISTRO DE CALLBACKS DAS ABAS
# ============================================================================

register_overview_callbacks(app, BASE_DF, TOTAL_SONGS)
register_popularity_callbacks(app, BASE_DF)
register_audio_dna_callbacks(app, BASE_DF, CLASSIFIER_FEATURES)

# ============================================================================
# EXECUÇÃO DO SERVIDOR
# ============================================================================

if __name__ == "__main__":
    app.run(debug=True)
