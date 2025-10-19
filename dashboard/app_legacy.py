from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

Number = Union[int, float]

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, State, dcc, html
from dash.dash_table import DataTable
from dash.exceptions import PreventUpdate

from data_utils import (
    FilterState,
    compute_kpis,
    danceability_by_genre,
    descriptive_stats,
    feature_ranges,
    filter_dataframe,
    genre_distribution,
    key_distribution,
    load_dataset,
    missing_values,
    top_artists_by_popularity,
)
from model_utils import (
    CLASSIFIER_FEATURES,
    CLUSTER_FEATURES,
    apply_clustering,
    cluster_profile_table,
    train_classifier_from_df,
)
from visualizations import (
    bar_chart,
    cluster_scatter,
    feature_distribution,
    scatter_matrix,
    tempo_ridge_like,
    valence_energy_density,
    # Premium visualizations
    feature_3d_scatter,
    correlation_heatmap,
    genre_sunburst,
    timeline_release_trend,
    radar_chart_genre_profile,
    popularity_distribution_violin,
    artist_network_top,
    duration_tempo_jointplot,
)


BASE_DF = load_dataset()
TOTAL_SONGS = len(BASE_DF)
GENRE_OPTIONS = sorted(BASE_DF["playlist_genre"].dropna().unique())
SUBGENRE_OPTIONS = sorted(BASE_DF["playlist_subgenre"].dropna().unique())
FEATURE_RANGES = feature_ranges(BASE_DF, CLASSIFIER_FEATURES)
CLASSIFIER_RESULT = train_classifier_from_df(BASE_DF)
KEY_LABELS = (
    BASE_DF[["key", "key_name"]]
    .dropna()
    .drop_duplicates(subset="key")
    .set_index("key")
    .squeeze()
    .to_dict()
)
MODE_LABELS = {0: "Menor", 1: "Maior"}

POPULARITY_MIN = int(np.floor(BASE_DF["track_popularity"].min()))
POPULARITY_MAX = int(np.ceil(BASE_DF["track_popularity"].max()))
_valid_years = BASE_DF["release_year"].replace({0: np.nan}).dropna()
if _valid_years.empty:
    _valid_years = BASE_DF["track_album_release_date"].dt.year.dropna()
if _valid_years.empty:
    YEAR_MIN, YEAR_MAX = 2000, 2024
else:
    YEAR_MIN = int(_valid_years.min())
    YEAR_MAX = int(_valid_years.max())


BACKGROUND_GRADIENT = (
    "radial-gradient(circle at 20% 20%, rgba(76, 201, 240, 0.14), transparent 46%)"
    ", radial-gradient(circle at 80% -10%, rgba(255, 109, 196, 0.16), transparent 50%)"
    ", linear-gradient(135deg, #05070d 0%, #0b101a 55%, #05070d 100%)"
)
CARD_BACKGROUND = "rgba(16, 20, 26, 0.78)"
CARD_BORDER_COLOR = "rgba(255, 255, 255, 0.08)"
CARD_BORDER_STYLE = f"1px solid {CARD_BORDER_COLOR}"
CARD_SHADOW = "0 32px 65px rgba(0, 0, 0, 0.45)"
PRIMARY_TEXT = "#f8f9fa"
SECONDARY_TEXT = "#aeb6c4"
ACCENT_COLOR = "#4cc9f0"
FONT_FAMILY = "'Poppins', 'Inter', sans-serif"

TABLE_HEADER_STYLE: Dict[str, str] = {
    "backgroundColor": "rgba(33, 37, 41, 0.85)",
    "color": PRIMARY_TEXT,
    "fontWeight": "600",
    "border": CARD_BORDER_STYLE,
}
TABLE_CELL_STYLE: Dict[str, str] = {
    "backgroundColor": "rgba(12, 16, 22, 0.75)",
    "color": PRIMARY_TEXT,
    "border": CARD_BORDER_STYLE,
}

def _range_marks(min_value: Number, max_value: Number, *, vertical: bool = False) -> Dict[Number, Dict[str, str]]:
    def _format_label(value: Number) -> str:
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.2f}".rstrip("0").rstrip(".")

    def _label_style() -> Dict[str, str]:
        base = {"font-size": "12px"}
        if vertical:
            base.update({"writing-mode": "vertical-rl", "text-orientation": "upright"})
        return base

    if min_value >= max_value:
        return {min_value: {"label": _format_label(min_value), "style": _label_style()}}

    mid_value = (min_value + max_value) / 2
    values: List[Number] = [min_value]
    if not np.isclose(mid_value, min_value) and not np.isclose(mid_value, max_value):
        values.append(mid_value)
    values.append(max_value)

    marks: Dict[Number, Dict[str, str]] = {}
    for value in values:
        marks[value] = {"label": _format_label(value), "style": _label_style()}
    return marks


POPULARITY_MARKS = _range_marks(POPULARITY_MIN, POPULARITY_MAX)
YEAR_MARKS = _range_marks(YEAR_MIN, YEAR_MAX)


def apply_filters(
    genres: List[str] | None,
    subgenres: List[str] | None,
    popularity: List[int],
    years: List[int],
) -> pd.DataFrame:
    state = FilterState(
        genres=genres or [],
        subgenres=subgenres or [],
        min_popularity=int(popularity[0]),
        max_popularity=int(popularity[1]),
        year_range=(int(years[0]), int(years[1])),
    )
    return filter_dataframe(BASE_DF, state)


def slider_component(feature: str, label: str, step: float) -> html.Div:
    min_val, max_val = FEATURE_RANGES[feature]
    if np.isfinite(min_val) and np.isfinite(max_val):
        value = float(np.clip((min_val + max_val) / 2, min_val, max_val))
    else:
        value = float(min_val)
    return html.Div(
        className="col-lg-4 col-md-6",
        children=[
            html.Div(
                className="glass-card p-3 h-100",
                children=[
                    html.Label(label, className="form-label fw-semibold text-light"),
                    dcc.Slider(
                        id=f"input-{feature}",
                        min=min_val,
                        max=max_val,
                        step=step,
                        value=value,
                        marks=_range_marks(min_val, max_val),
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    html.Div(
                        className="small text-secondary mt-2",
                        children=[f"Min: {min_val:.2f} | Máx: {max_val:.2f}"],
                    ),
                ],
            ),
        ],
    )


external_stylesheets = ["https://cdn.jsdelivr.net/npm/bootswatch@5.3.3/dist/darkly/bootstrap.min.css"]
app = Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,
    title="Spotify Insights Dashboard",
    update_title=None,
)
server = app.server


tab_style = {
    "padding": "14px",
    "fontWeight": "500",
    "backgroundColor": "transparent",
    "color": SECONDARY_TEXT,
    "border": "0",
}
tab_selected_style = tab_style | {
    "background": "linear-gradient(135deg, rgba(76, 201, 240, 0.3), rgba(255, 109, 196, 0.25))",
    "color": PRIMARY_TEXT,
}

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
        html.Link(
            rel="stylesheet",
            href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap",
        ),
        html.Div(
            className="mb-4",
            children=[
                html.H1("Spotify Songs – Dashboard Analítico", className="fw-bold text-light"),
                html.P(
                    "Explore tendências musicais, comportamento de usuários e modelos de machine learning.",
                    className="text-secondary",
                ),
            ],
        ),
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
                                html.Div(
                                    className="col-lg-3 col-md-6",
                                    children=[
                                        html.Label("Gêneros", className="form-label text-light"),
                                        dcc.Dropdown(
                                            id="genre-dropdown",
                                            options=[{"label": g.title(), "value": g} for g in GENRE_OPTIONS],
                                            value=[],
                                            multi=True,
                                            placeholder="Selecione gêneros",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="col-lg-3 col-md-6",
                                    children=[
                                        html.Label("Subgêneros", className="form-label text-light"),
                                        dcc.Dropdown(
                                            id="subgenre-dropdown",
                                            options=[{"label": s.title(), "value": s} for s in SUBGENRE_OPTIONS],
                                            value=[],
                                            multi=True,
                                            placeholder="Filtre subgêneros",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="col-lg-3 col-md-6",
                                    children=[
                                        html.Label("Popularidade", className="form-label text-light"),
                                        dcc.RangeSlider(
                                            id="popularity-slider",
                                            min=POPULARITY_MIN,
                                            max=POPULARITY_MAX,
                                            step=1,
                                            value=[POPULARITY_MIN, POPULARITY_MAX],
                                            marks=POPULARITY_MARKS,
                                            tooltip={"placement": "bottom", "always_visible": False},
                                        ),
                                        html.Div(
                                            className="small text-secondary mt-1",
                                            children=[
                                                f"Intervalo atual: {POPULARITY_MIN} – {POPULARITY_MAX}"
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="col-lg-3 col-md-6",
                                    children=[
                                        html.Label("Ano de lançamento", className="form-label text-light"),
                                        dcc.RangeSlider(
                                            id="year-slider",
                                            min=YEAR_MIN,
                                            max=YEAR_MAX,
                                            step=1,
                                            value=[YEAR_MIN, YEAR_MAX],
                                            marks=YEAR_MARKS,
                                            tooltip={"placement": "bottom", "always_visible": False},
                                        ),
                                        html.Div(
                                            className="small text-secondary mt-1",
                                            children=[f"Intervalo atual: {YEAR_MIN} – {YEAR_MAX}"],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        dcc.Tabs(
            id="main-tabs",
            value="overview",
            className="modern-tabs bg-dark rounded shadow-sm border border-secondary text-light",
            children=[
                dcc.Tab(
                    label="Visão Geral",
                    value="overview",
                    style=tab_style,
                    selected_style=tab_selected_style,
                    children=[
                        html.Div(
                            className="p-4",
                            children=[
                                html.Div(id="overview-kpis", className="row g-3"),
                                html.Div(
                                    className="row g-4 mt-1",
                                    children=[
                                        html.Div(
                                            className="col-xl-6",
                                            children=[
                                                html.H5("Dados ausentes", className="mt-3"),
                                                DataTable(
                                                    id="missing-table",
                                                    data=[],
                                                    columns=[],
                                                    style_table={
                                                        "overflowX": "auto",
                                                        "backgroundColor": CARD_BACKGROUND,
                                                        "border": CARD_BORDER_STYLE,
                                                    },
                                                    style_header=TABLE_HEADER_STYLE,
                                                    style_cell=dict(TABLE_CELL_STYLE, textAlign="left"),
                                                    style_data=TABLE_CELL_STYLE,
                                                    page_size=10,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="col-xl-6",
                                            children=[
                                                html.H5("Estatísticas descritivas", className="mt-3"),
                                                DataTable(
                                                    id="descriptive-table",
                                                    data=[],
                                                    columns=[],
                                                    style_table={
                                                        "overflowX": "auto",
                                                        "backgroundColor": CARD_BACKGROUND,
                                                        "border": CARD_BORDER_STYLE,
                                                    },
                                                    style_header=TABLE_HEADER_STYLE,
                                                    style_cell=dict(TABLE_CELL_STYLE, textAlign="left"),
                                                    style_data=TABLE_CELL_STYLE,
                                                    page_size=10,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div(id="overview-summary", className="mt-4 text-secondary"),
                                # Premium visualizations
                                html.Div(
                                    className="row g-4 mt-4",
                                    children=[
                                        html.Div(
                                            className="col-xl-6",
                                            children=[
                                                html.H5("Hierarquia de Gêneros", className="mb-3"),
                                                html.Div(className="glass-card", children=[dcc.Graph(id="genre-sunburst-graph")]),
                                            ],
                                        ),
                                        html.Div(
                                            className="col-xl-6",
                                            children=[
                                                html.H5("Distribuição de Popularidade", className="mb-3"),
                                                html.Div(className="glass-card", children=[dcc.Graph(id="popularity-violin-graph")]),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="mt-4",
                                    children=[
                                        html.H5("Tendência de Lançamentos ao Longo do Tempo", className="mb-3"),
                                        html.Div(className="glass-card", children=[dcc.Graph(id="timeline-trend-graph")]),
                                    ],
                                ),
                            ],
                        )
                    ],
                ),
                dcc.Tab(
                    label="Popularidade",
                    value="popularity",
                    style=tab_style,
                    selected_style=tab_selected_style,
                    children=[
                        html.Div(
                            className="p-4",
                            children=[
                                html.Div(
                                    className="row g-4",
                                    children=[
                                        html.Div(
                                            className="col-lg-12",
                                            children=[
                                                html.H5("Network de Top Artistas", className="mb-3"),
                                                html.Div(className="glass-card", children=[dcc.Graph(id="artist-network-graph")]),
                                            ],
                                        ),
                                        html.Div(
                                            className="col-lg-6",
                                            children=[
                                                html.Div(className="glass-card", children=[dcc.Graph(id="top-artists-graph")]),
                                            ],
                                        ),
                                        html.Div(
                                            className="col-lg-6",
                                            children=[
                                                html.Div(className="glass-card", children=[dcc.Graph(id="genre-distribution-graph")]),
                                            ],
                                        ),
                                        html.Div(
                                            className="col-lg-6",
                                            children=[
                                                html.Div(className="glass-card", children=[dcc.Graph(id="key-distribution-graph")]),
                                            ],
                                        ),
                                        html.Div(
                                            className="col-lg-6",
                                            children=[
                                                html.Div(className="glass-card", children=[dcc.Graph(id="danceability-graph")]),
                                            ],
                                        ),
                                    ],
                                )
                            ],
                        )
                    ],
                ),
                dcc.Tab(
                    label="Humor & Tempo",
                    value="mood",
                    style=tab_style,
                    selected_style=tab_selected_style,
                    children=[
                        html.Div(
                            className="p-4",
                            children=[
                                html.Div(
                                    className="row g-3",
                                    children=[
                                        html.Div(
                                            className="col-lg-6",
                                            children=[
                                                html.Label(
                                                    "Selecione gêneros para comparar BPM",
                                                    className="form-label text-light",
                                                ),
                                                dcc.Dropdown(
                                                    id="tempo-genres-dropdown",
                                                    options=[
                                                        {"label": g.title(), "value": g} for g in GENRE_OPTIONS
                                                    ],
                                                    value=GENRE_OPTIONS[: min(5, len(GENRE_OPTIONS))],
                                                    multi=True,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="col-lg-6",
                                            children=[
                                                html.Label(
                                                    "Mapa emocional por gênero (Energia x Valência)",
                                                    className="form-label text-light",
                                                ),
                                                dcc.Dropdown(
                                                    id="mood-genre-dropdown",
                                                    options=[
                                                        {"label": "Todos", "value": "Todos"}
                                                    ]
                                                    + [
                                                        {"label": g.title(), "value": g}
                                                        for g in GENRE_OPTIONS
                                                    ],
                                                    value="Todos",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                dcc.Graph(id="tempo-ridge-graph", className="mt-3"),
                                html.Hr(className="my-4"),
                                dcc.Graph(id="mood-density-graph"),
                            ],
                        )
                    ],
                ),
                dcc.Tab(
                    label="Explorador",
                    value="explorer",
                    style=tab_style,
                    selected_style=tab_selected_style,
                    children=[
                        html.Div(
                            className="p-4",
                            children=[
                                html.Div(
                                    className="d-flex justify-content-between align-items-center mb-3",
                                    children=[
                                        html.H5("Amostra de faixas"),
                                        html.Button(
                                            "Baixar dados filtrados",
                                            id="download-button",
                                            className="btn btn-outline-primary",
                                        ),
                                    ],
                                ),
                                DataTable(
                                    id="explorer-table",
                                    data=[],
                                    columns=[],
                                    page_size=15,
                                    style_table={
                                        "overflowX": "auto",
                                        "backgroundColor": CARD_BACKGROUND,
                                        "border": CARD_BORDER_STYLE,
                                    },
                                    style_header=TABLE_HEADER_STYLE,
                                    style_cell=dict(TABLE_CELL_STYLE, textAlign="left"),
                                    style_data=TABLE_CELL_STYLE,
                                    filter_action="native",
                                    sort_action="native",
                                ),
                                html.Div(
                                    className="row g-3 mt-4",
                                    children=[
                                        html.Div(
                                            className="col-lg-6",
                                            children=[
                                                html.Label(
                                                    "Métrica para distribuição",
                                                    className="form-label text-light",
                                                ),
                                                dcc.Dropdown(
                                                    id="feature-select",
                                                    options=[
                                                        {"label": "Danceability", "value": "danceability"},
                                                        {"label": "Energy", "value": "energy"},
                                                        {"label": "Speechiness", "value": "speechiness"},
                                                        {"label": "Acousticness", "value": "acousticness"},
                                                        {"label": "Instrumentalness", "value": "instrumentalness"},
                                                        {"label": "Liveness", "value": "liveness"},
                                                        {"label": "Valence", "value": "valence"},
                                                        {"label": "Tempo", "value": "tempo"},
                                                        {"label": "Popularidade", "value": "track_popularity"},
                                                    ],
                                                    value="danceability",
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="col-lg-6",
                                            children=[
                                                html.Label("Colorir por", className="form-label text-light"),
                                                dcc.Dropdown(
                                                    id="color-select",
                                                    options=[
                                                        {"label": "Nenhum", "value": "none"},
                                                        {"label": "Gênero", "value": "playlist_genre"},
                                                        {"label": "Subgênero", "value": "playlist_subgenre"},
                                                    ],
                                                    value="none",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                dcc.Graph(id="feature-histogram", className="mt-3"),
                                html.Details(
                                    className="mt-4",
                                    children=[
                                        html.Summary("Matriz de dispersão das principais features"),
                                        dcc.Graph(id="scatter-matrix", className="mt-3"),
                                    ],
                                ),
                            ],
                        )
                    ],
                ),
                dcc.Tab(
                    label="Classificação",
                    value="classification",
                    style=tab_style,
                    selected_style=tab_selected_style,
                    children=[
                        html.Div(
                            className="p-4",
                            children=[
                                html.Div(
                                    className="row g-4",
                                    children=[
                                        html.Div(
                                            className="col-lg-4",
                                            children=[
                                                html.Div(
                                                    className="glass-card p-4 h-100 d-flex flex-column justify-content-center text-center",
                                                    children=[
                                                        html.Span(
                                                            "Acurácia (holdout)",
                                                            className="text-secondary text-uppercase small mb-2",
                                                        ),
                                                        html.H2(
                                                            f"{CLASSIFIER_RESULT.accuracy:.2%}",
                                                            className="fw-bold text-light mb-0",
                                                        ),
                                                    ],
                                                )
                                            ],
                                        ),
                                        html.Div(
                                            className="col-lg-8",
                                            children=[
                                                html.Div(
                                                    className="glass-card p-4 h-100",
                                                    children=[
                                                        html.H5("Relatório de classificação", className="mb-3"),
                                                        html.Pre(
                                                            CLASSIFIER_RESULT.report,
                                                            className="text-light fw-monospace mb-0",
                                                            style={
                                                                "background": "transparent",
                                                                "whiteSpace": "pre-wrap",
                                                            },
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Hr(className="my-4"),
                                html.H4("Prever gênero"),
                                html.P(
                                    "Ajuste os controles abaixo para descrever uma faixa e gere a previsão.",
                                    className="text-secondary",
                                ),
                                html.Div(
                                    className="row g-3",
                                    children=[
                                        slider_component("danceability", "Danceability", 0.01),
                                        slider_component("energy", "Energy", 0.01),
                                        slider_component("speechiness", "Speechiness", 0.01),
                                        slider_component("acousticness", "Acousticness", 0.01),
                                        slider_component("instrumentalness", "Instrumentalness", 0.01),
                                        slider_component("liveness", "Liveness", 0.01),
                                        slider_component("valence", "Valence", 0.01),
                                        slider_component("tempo", "Tempo (BPM)", 1.0),
                                        slider_component("loudness", "Loudness (dB)", 0.5),
                                        slider_component("track_popularity", "Popularidade da faixa", 1.0),
                                        slider_component("duration_ms", "Duração (ms)", 1000.0),
                                    ],
                                ),
                                html.Div(
                                    className="row g-3 mt-1",
                                    children=[
                                        html.Div(
                                            className="col-lg-6",
                                            children=[
                                                html.Div(
                                                    className="glass-card p-3 h-100",
                                                    children=[
                                                        html.Label(
                                                            "Tom (Key)",
                                                            className="form-label fw-semibold text-light",
                                                        ),
                                                        dcc.Dropdown(
                                                            id="input-key",
                                                            options=[
                                                                {
                                                                    "label": KEY_LABELS.get(k, str(k)),
                                                                    "value": k,
                                                                }
                                                                for k in range(12)
                                                            ],
                                                            value=0,
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="col-lg-6",
                                            children=[
                                                html.Div(
                                                    className="glass-card p-3 h-100",
                                                    children=[
                                                        html.Label(
                                                            "Modo",
                                                            className="form-label fw-semibold text-light",
                                                        ),
                                                        dcc.RadioItems(
                                                            id="input-mode",
                                                            options=[
                                                                {"label": label, "value": value}
                                                                for value, label in MODE_LABELS.items()
                                                            ],
                                                            value=1,
                                                            inline=True,
                                                            labelStyle={
                                                                "margin-right": "1rem",
                                                                "color": PRIMARY_TEXT,
                                                            },
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Button(
                                    "Prever gênero",
                                    id="predict-button",
                                    className="btn btn-primary mt-3",
                                ),
                                html.Div(id="prediction-output", className="glass-card p-3 mt-4"),
                                html.Div(
                                    className="glass-card p-3 mt-3",
                                    children=[
                                        DataTable(
                                            id="prediction-table",
                                            data=[],
                                            columns=[],
                                            style_table={
                                                "maxWidth": "500px",
                                                "backgroundColor": "transparent",
                                                "border": "none",
                                            },
                                            style_header=TABLE_HEADER_STYLE,
                                            style_cell=dict(TABLE_CELL_STYLE, textAlign="left"),
                                            style_data=TABLE_CELL_STYLE,
                                        ),
                                    ],
                                ),
                            ],
                        )
                    ],
                ),
                dcc.Tab(
                    label="🧬 Audio DNA",
                    value="audio-dna",
                    style=tab_style,
                    selected_style=tab_selected_style,
                    children=[
                        html.Div(
                            className="p-4",
                            children=[
                                html.H4("Análise Multidimensional de Features", className="mb-4"),
                                # 3D Scatter Plot
                                html.Div(
                                    className="row g-3 mb-4",
                                    children=[
                                        html.Div(
                                            className="col-md-3",
                                            children=[
                                                html.Label("Eixo X", className="form-label text-light"),
                                                dcc.Dropdown(
                                                    id="3d-x-axis",
                                                    options=[{"label": f.replace("_", " ").title(), "value": f} for f in CLASSIFIER_FEATURES],
                                                    value="energy",
                                                    className="dash-dropdown",
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="col-md-3",
                                            children=[
                                                html.Label("Eixo Y", className="form-label text-light"),
                                                dcc.Dropdown(
                                                    id="3d-y-axis",
                                                    options=[{"label": f.replace("_", " ").title(), "value": f} for f in CLASSIFIER_FEATURES],
                                                    value="valence",
                                                    className="dash-dropdown",
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="col-md-3",
                                            children=[
                                                html.Label("Eixo Z", className="form-label text-light"),
                                                dcc.Dropdown(
                                                    id="3d-z-axis",
                                                    options=[{"label": f.replace("_", " ").title(), "value": f} for f in CLASSIFIER_FEATURES],
                                                    value="danceability",
                                                    className="dash-dropdown",
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="col-md-3",
                                            children=[
                                                html.Label("Colorir por", className="form-label text-light"),
                                                dcc.Dropdown(
                                                    id="3d-color",
                                                    options=[
                                                        {"label": "Gênero", "value": "playlist_genre"},
                                                        {"label": "Popularidade", "value": "track_popularity"},
                                                    ],
                                                    value="playlist_genre",
                                                    className="dash-dropdown",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div(className="glass-card", children=[dcc.Graph(id="feature-3d-scatter")]),
                                # Correlation Heatmap & Radar
                                html.Div(
                                    className="row g-4 mt-4",
                                    children=[
                                        html.Div(
                                            className="col-xl-6",
                                            children=[
                                                html.Div(className="glass-card", children=[dcc.Graph(id="correlation-heatmap-graph")]),
                                            ],
                                        ),
                                        html.Div(
                                            className="col-xl-6",
                                            children=[
                                                html.Div(
                                                    className="glass-card p-3 mb-3",
                                                    children=[
                                                        html.Label("Gênero para Radar", className="form-label text-light"),
                                                        dcc.Dropdown(
                                                            id="radar-genre",
                                                            options=[{"label": "Todos", "value": "Todos"}]
                                                            + [{"label": g.title(), "value": g} for g in GENRE_OPTIONS],
                                                            value="Todos",
                                                            className="dash-dropdown",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(className="glass-card", children=[dcc.Graph(id="radar-chart-graph")]),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        )
                    ],
                ),
                dcc.Tab(
                    label="Clusters",
                    value="clusters",
                    style=tab_style,
                    selected_style=tab_selected_style,
                    children=[
                        html.Div(
                            className="p-4",
                            children=[
                                html.Div(
                                    className="row g-3 align-items-center",
                                    children=[
                                        html.Div(
                                            className="col-lg-4",
                                            children=[
                                                html.Label("Número de clusters", className="form-label text-light"),
                                                dcc.Slider(
                                                    id="cluster-slider",
                                                    min=2,
                                                    max=10,
                                                    step=1,
                                                    value=4,
                                                    marks={i: str(i) for i in range(2, 11)},
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div(id="cluster-warning", className="mt-3"),
                                dcc.Graph(id="cluster-graph", className="mt-3"),
                                html.Div(
                                    className="row g-4 mt-1",
                                    children=[
                                        html.Div(
                                            className="col-xl-6",
                                            children=[
                                                html.H5("Perfil médio por cluster"),
                                                DataTable(
                                                    id="cluster-profile-table",
                                                    data=[],
                                                    columns=[],
                                                    style_table={
                                                        "overflowX": "auto",
                                                        "backgroundColor": CARD_BACKGROUND,
                                                        "border": CARD_BORDER_STYLE,
                                                    },
                                                    style_header=TABLE_HEADER_STYLE,
                                                    style_cell=dict(TABLE_CELL_STYLE, textAlign="left"),
                                                    style_data=TABLE_CELL_STYLE,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="col-xl-6",
                                            children=[
                                                html.H5("Amostra de faixas"),
                                                DataTable(
                                                    id="cluster-sample-table",
                                                    data=[],
                                                    columns=[],
                                                    page_size=10,
                                                    style_table={
                                                        "overflowX": "auto",
                                                        "backgroundColor": CARD_BACKGROUND,
                                                        "border": CARD_BORDER_STYLE,
                                                    },
                                                    style_header=TABLE_HEADER_STYLE,
                                                    style_cell=dict(TABLE_CELL_STYLE, textAlign="left"),
                                                    style_data=TABLE_CELL_STYLE,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        )
                    ],
                ),
            ],
        ),
        dcc.Download(id="download-data"),
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


@app.callback(
    Output("subgenre-dropdown", "options"),
    Output("subgenre-dropdown", "value"),
    Input("genre-dropdown", "value"),
    State("subgenre-dropdown", "value"),
)
def update_subgenre_options(selected_genres: List[str], current_values: List[str]):
    if selected_genres:
        filtered_subgenres = (
            BASE_DF[BASE_DF["playlist_genre"].isin(selected_genres)]["playlist_subgenre"]
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
    Output("overview-kpis", "children"),
    Output("missing-table", "data"),
    Output("missing-table", "columns"),
    Output("descriptive-table", "data"),
    Output("descriptive-table", "columns"),
    Output("overview-summary", "children"),
    Input("genre-dropdown", "value"),
    Input("subgenre-dropdown", "value"),
    Input("popularity-slider", "value"),
    Input("year-slider", "value"),
)
def update_overview(genres, subgenres, popularity, years):
    filtered = apply_filters(genres, subgenres, popularity, years)

    kpis = compute_kpis(filtered)
    kpi_cards = []
    for label, value in kpis.items():
        kpi_cards.append(
            html.Div(
                className="col-sm-6 col-xl-3",
                children=[
                    html.Div(
                        className="card bg-dark border border-secondary shadow-sm",
                        children=[
                            html.Div(
                                className="card-body",
                                children=[
                                    html.Div(
                                        label.capitalize(),
                                        className="text-secondary text-uppercase small",
                                    ),
                                    html.H3(f"{value:,}", className="fw-bold"),
                                ],
                            )
                        ],
                    )
                ],
            )
        )

    missing = missing_values(filtered)
    if missing.empty:
        missing_df = pd.DataFrame({"Coluna": ["Sem dados ausentes"], "Valores faltantes": [0]})
    else:
        missing_df = missing.reset_index()
        missing_df.columns = ["Coluna", "Valores faltantes"]

    desc_cols = [
        "danceability",
        "energy",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "track_popularity",
    ]
    descriptive = descriptive_stats(filtered, desc_cols)
    descriptive = descriptive.reset_index().rename(columns={"index": "Métrica"})

    summary = html.Div(
        [
            html.Span(
                f"Total de músicas no dataset: {TOTAL_SONGS:,}. ", className="me-2"
            ),
            html.Span(
                f"Após filtros: {len(filtered):,} faixas selecionadas.",
                className="fw-semibold",
            ),
        ]
    )

    missing_data = missing_df.to_dict("records")
    missing_columns = [
        {"name": col, "id": col} for col in missing_df.columns
    ]
    desc_data = descriptive.round(3).to_dict("records")
    desc_columns = [
        {"name": col, "id": col} for col in descriptive.columns
    ]

    return kpi_cards, missing_data, missing_columns, desc_data, desc_columns, summary


@app.callback(
    Output("top-artists-graph", "figure"),
    Output("genre-distribution-graph", "figure"),
    Output("key-distribution-graph", "figure"),
    Output("danceability-graph", "figure"),
    Input("genre-dropdown", "value"),
    Input("subgenre-dropdown", "value"),
    Input("popularity-slider", "value"),
    Input("year-slider", "value"),
)
def update_popularity(genres, subgenres, popularity, years):
    filtered = apply_filters(genres, subgenres, popularity, years)

    return (
        bar_chart(
            top_artists_by_popularity(filtered, top_n=10),
            title="Top 10 artistas por popularidade média",
            x_title="Popularidade média",
            y_title="Artista",
        ),
        bar_chart(
            genre_distribution(filtered),
            title="Distribuição de músicas por gênero",
            x_title="Número de faixas",
            y_title="Gênero",
        ),
        bar_chart(
            key_distribution(filtered),
            title="Distribuição de músicas por tom",
            x_title="Número de faixas",
            y_title="Tom",
        ),
        bar_chart(
            danceability_by_genre(filtered),
            title="Danceability médio por gênero",
            x_title="Danceability",
            y_title="Gênero",
        ),
    )


@app.callback(
    Output("tempo-ridge-graph", "figure"),
    Output("mood-density-graph", "figure"),
    Input("tempo-genres-dropdown", "value"),
    Input("mood-genre-dropdown", "value"),
    Input("genre-dropdown", "value"),
    Input("subgenre-dropdown", "value"),
    Input("popularity-slider", "value"),
    Input("year-slider", "value"),
)
def update_mood(tempo_genres, mood_genre, genres, subgenres, popularity, years):
    filtered = apply_filters(genres, subgenres, popularity, years)

    tempo_genres = tempo_genres or []
    tempo_fig = tempo_ridge_like(filtered, tempo_genres)

    selected_genre = None if mood_genre in (None, "Todos") else mood_genre
    mood_fig = valence_energy_density(filtered, selected_genre)

    return tempo_fig, mood_fig


@app.callback(
    Output("explorer-table", "data"),
    Output("explorer-table", "columns"),
    Output("feature-histogram", "figure"),
    Output("scatter-matrix", "figure"),
    Input("feature-select", "value"),
    Input("color-select", "value"),
    Input("genre-dropdown", "value"),
    Input("subgenre-dropdown", "value"),
    Input("popularity-slider", "value"),
    Input("year-slider", "value"),
)
def update_explorer(feature, color, genres, subgenres, popularity, years):
    filtered = apply_filters(genres, subgenres, popularity, years)

    table_df = filtered.head(100)
    table_data = table_df.to_dict("records")
    table_columns = [
        {"name": col.replace("_", " ").title(), "id": col} for col in table_df.columns
    ]

    color_arg = None if color in (None, "none") else color
    hist_fig = feature_distribution(filtered, feature, color=color_arg)

    scatter_dims = ["danceability", "energy", "valence", "tempo", "track_popularity"]
    scatter_df = (
        filtered.sample(min(len(filtered), 1000), random_state=42)
        if len(filtered) > 1000
        else filtered
    )
    scatter_fig = scatter_matrix(scatter_df, scatter_dims, color="playlist_genre")

    return table_data, table_columns, hist_fig, scatter_fig


@app.callback(
    Output("download-data", "data"),
    Input("download-button", "n_clicks"),
    State("genre-dropdown", "value"),
    State("subgenre-dropdown", "value"),
    State("popularity-slider", "value"),
    State("year-slider", "value"),
    prevent_initial_call=True,
)
def download_filtered(n_clicks, genres, subgenres, popularity, years):
    if not n_clicks:
        raise PreventUpdate

    filtered = apply_filters(genres, subgenres, popularity, years)
    return dcc.send_data_frame(filtered.to_csv, "spotify_filtrado.csv", index=False)


@app.callback(
    Output("prediction-output", "children"),
    Output("prediction-table", "data"),
    Output("prediction-table", "columns"),
    Input("predict-button", "n_clicks"),
    State("input-danceability", "value"),
    State("input-energy", "value"),
    State("input-speechiness", "value"),
    State("input-acousticness", "value"),
    State("input-instrumentalness", "value"),
    State("input-liveness", "value"),
    State("input-valence", "value"),
    State("input-tempo", "value"),
    State("input-loudness", "value"),
    State("input-track_popularity", "value"),
    State("input-duration_ms", "value"),
    State("input-key", "value"),
    State("input-mode", "value"),
    prevent_initial_call=True,
)
def generate_prediction(
    n_clicks,
    danceability,
    energy,
    speechiness,
    acousticness,
    instrumentalness,
    liveness,
    valence,
    tempo,
    loudness,
    track_popularity,
    duration_ms,
    key,
    mode,
):
    if not n_clicks:
        raise PreventUpdate

    input_row = pd.DataFrame(
        [
            {
                "danceability": danceability,
                "energy": energy,
                "speechiness": speechiness,
                "acousticness": acousticness,
                "instrumentalness": instrumentalness,
                "liveness": liveness,
                "valence": valence,
                "tempo": tempo,
                "loudness": loudness,
                "track_popularity": track_popularity,
                "duration_ms": duration_ms,
                "key": key,
                "mode": mode,
            }
        ]
    )

    prediction = CLASSIFIER_RESULT.pipeline.predict(input_row)[0]
    proba = CLASSIFIER_RESULT.pipeline.predict_proba(input_row)[0]
    proba_df = (
        pd.DataFrame(
            {"Gênero": CLASSIFIER_RESULT.pipeline.classes_, "Probabilidade": proba}
        )
        .sort_values("Probabilidade", ascending=False)
        .reset_index(drop=True)
    )

    message = html.Div(
        className="alert alert-success",
        children=[html.Strong("Gênero previsto: "), html.Span(prediction)],
    )

    return (
        message,
        proba_df.round(4).to_dict("records"),
        [{"name": col, "id": col} for col in proba_df.columns],
    )


@app.callback(
    Output("cluster-graph", "figure"),
    Output("cluster-profile-table", "data"),
    Output("cluster-profile-table", "columns"),
    Output("cluster-sample-table", "data"),
    Output("cluster-sample-table", "columns"),
    Output("cluster-warning", "children"),
    Input("cluster-slider", "value"),
    Input("genre-dropdown", "value"),
    Input("subgenre-dropdown", "value"),
    Input("popularity-slider", "value"),
    Input("year-slider", "value"),
)
def update_clusters(n_clusters, genres, subgenres, popularity, years):
    filtered = apply_filters(genres, subgenres, popularity, years)

    if len(filtered) < 50:
        warning = html.Div(
            "É necessário ao menos 50 músicas para gerar clusters confiáveis. Ajuste os filtros.",
            className="alert alert-warning",
        )
        empty_columns: List[Dict[str, str]] = []
        empty_fig = tempo_ridge_like(filtered, [])
        empty_fig.update_layout(title="Dados insuficientes para clusterização")
        return empty_fig, [], empty_columns, [], empty_columns, warning

    result = apply_clustering(filtered, n_clusters=n_clusters)
    cluster_fig = cluster_scatter(filtered, result.pca_projection, result.clusters)

    profile_df = cluster_profile_table(result)
    profile_data = profile_df.round(3).to_dict("records")
    profile_columns = [{"name": col, "id": col} for col in profile_df.columns]

    sample_df = filtered.copy()
    sample_df["cluster"] = result.clusters
    sample_df = sample_df[
        ["track_name", "track_artist", "playlist_genre", "playlist_subgenre", "cluster"]
    ].head(200)
    sample_data = sample_df.to_dict("records")
    sample_columns = [
        {"name": "Nome", "id": "track_name"},
        {"name": "Artista", "id": "track_artist"},
        {"name": "Gênero", "id": "playlist_genre"},
        {"name": "Subgênero", "id": "playlist_subgenre"},
        {"name": "Cluster", "id": "cluster"},
    ]

    return (
        cluster_fig,
        profile_data,
        profile_columns,
        sample_data,
        sample_columns,
        None,
    )


# Premium visualizations callbacks
@app.callback(
    Output("feature-3d-scatter", "figure"),
    Input("3d-x-axis", "value"),
    Input("3d-y-axis", "value"),
    Input("3d-z-axis", "value"),
    Input("3d-color", "value"),
    Input("genre-dropdown", "value"),
    Input("subgenre-dropdown", "value"),
    Input("popularity-slider", "value"),
    Input("year-slider", "value"),
)
def update_3d_scatter(x_axis, y_axis, z_axis, color_by, genres, subgenres, popularity, years):
    """Update 3D scatter plot based on selected axes and filters."""
    filtered_df = apply_filters(genres, subgenres, popularity, years)
    return feature_3d_scatter(filtered_df, x_axis, y_axis, z_axis, color=color_by, sample_size=2000)


@app.callback(
    Output("correlation-heatmap-graph", "figure"),
    Input("genre-dropdown", "value"),
    Input("subgenre-dropdown", "value"),
    Input("popularity-slider", "value"),
    Input("year-slider", "value"),
)
def update_correlation_heatmap(genres, subgenres, popularity, years):
    """Update correlation heatmap."""
    filtered_df = apply_filters(genres, subgenres, popularity, years)
    return correlation_heatmap(filtered_df, CLASSIFIER_FEATURES)


@app.callback(
    Output("radar-chart-graph", "figure"),
    Input("radar-genre", "value"),
    Input("genre-dropdown", "value"),
    Input("subgenre-dropdown", "value"),
    Input("popularity-slider", "value"),
    Input("year-slider", "value"),
)
def update_radar_chart(selected_genre, genres, subgenres, popularity, years):
    """Update radar chart for genre profile."""
    filtered_df = apply_filters(genres, subgenres, popularity, years)
    return radar_chart_genre_profile(filtered_df, selected_genre, CLASSIFIER_FEATURES)


@app.callback(
    Output("genre-sunburst-graph", "figure"),
    Input("genre-dropdown", "value"),
    Input("subgenre-dropdown", "value"),
    Input("popularity-slider", "value"),
    Input("year-slider", "value"),
)
def update_sunburst(genres, subgenres, popularity, years):
    """Update sunburst chart for genre hierarchy."""
    filtered_df = apply_filters(genres, subgenres, popularity, years)
    return genre_sunburst(filtered_df)


@app.callback(
    Output("popularity-violin-graph", "figure"),
    Input("genre-dropdown", "value"),
    Input("subgenre-dropdown", "value"),
    Input("popularity-slider", "value"),
    Input("year-slider", "value"),
)
def update_popularity_violin(genres, subgenres, popularity, years):
    """Update popularity violin plot."""
    filtered_df = apply_filters(genres, subgenres, popularity, years)
    return popularity_distribution_violin(filtered_df)


@app.callback(
    Output("timeline-trend-graph", "figure"),
    Input("genre-dropdown", "value"),
    Input("subgenre-dropdown", "value"),
    Input("popularity-slider", "value"),
    Input("year-slider", "value"),
)
def update_timeline(genres, subgenres, popularity, years):
    """Update timeline trend chart."""
    filtered_df = apply_filters(genres, subgenres, popularity, years)
    return timeline_release_trend(filtered_df)


@app.callback(
    Output("artist-network-graph", "figure"),
    Input("genre-dropdown", "value"),
    Input("subgenre-dropdown", "value"),
    Input("popularity-slider", "value"),
    Input("year-slider", "value"),
)
def update_artist_network(genres, subgenres, popularity, years):
    """Update artist network bubble chart."""
    filtered_df = apply_filters(genres, subgenres, popularity, years)
    return artist_network_top(filtered_df, top_n=30)


if __name__ == "__main__":
    app.run(debug=False)
