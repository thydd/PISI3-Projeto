"""
Callbacks da aba "Visão Geral".

Gerencia atualizações de KPIs, tabelas de dados ausentes, estatísticas descritivas
e visualizações premium (sunburst, timeline).
"""

from typing import List

import pandas as pd
from dash import Input, Output, html
from dash.dash_table import DataTable

from app.utils.common_components import apply_filters
from app.utils.data_utils import compute_kpis, descriptive_stats, missing_values
from app.utils.visualizations import (
    genre_sunburst,
    timeline_release_trend,
)


def register_overview_callbacks(app, base_df: pd.DataFrame, total_songs: int):
    """
    Registra callbacks da aba Visão Geral.

    Parameters
    ----------
    app : Dash
        Instância do aplicativo Dash.
    base_df : pd.DataFrame
        DataFrame base completo.
    total_songs : int
        Total de músicas no dataset.
    """

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
    def update_overview(
        genres: List[str],
        subgenres: List[str],
        popularity: List[int],
        years: List[int],
    ):
        """Atualiza KPIs, tabelas e resumo da visão geral."""
        filtered = apply_filters(base_df, genres, subgenres, popularity, years)

        # KPIs
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

        # Dados ausentes
        missing = missing_values(filtered)
        if missing.empty:
            missing_df = pd.DataFrame(
                {"Coluna": ["Sem dados ausentes"], "Valores faltantes": [0]}
            )
        else:
            missing_df = missing.reset_index()
            missing_df.columns = ["Coluna", "Valores faltantes"]

        # Estatísticas descritivas
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

        # Resumo
        summary = html.Div(
            [
                html.Span(
                    f"Total de músicas no dataset: {total_songs:,}. ", className="me-2"
                ),
                html.Span(
                    f"Após filtros: {len(filtered):,} faixas selecionadas.",
                    className="fw-semibold",
                ),
            ]
        )

        missing_data = missing_df.to_dict("records")
        missing_columns = [{"name": col, "id": col} for col in missing_df.columns]
        desc_data = descriptive.round(3).to_dict("records")
        desc_columns = [{"name": col, "id": col} for col in descriptive.columns]

        return (
            kpi_cards,
            missing_data,
            missing_columns,
            desc_data,
            desc_columns,
            summary,
        )

    @app.callback(
        Output("genre-sunburst-graph", "figure"),
        Input("genre-dropdown", "value"),
        Input("subgenre-dropdown", "value"),
        Input("popularity-slider", "value"),
        Input("year-slider", "value"),
    )
    def update_sunburst(genres, subgenres, popularity, years):
        """Atualiza gráfico sunburst de hierarquia de gêneros."""
        filtered_df = apply_filters(base_df, genres, subgenres, popularity, years)
        return genre_sunburst(filtered_df)

    @app.callback(
        Output("timeline-trend-graph", "figure"),
        Input("genre-dropdown", "value"),
        Input("subgenre-dropdown", "value"),
        Input("popularity-slider", "value"),
        Input("year-slider", "value"),
    )
    def update_timeline(genres, subgenres, popularity, years):
        """Atualiza gráfico de tendência temporal de lançamentos."""
        filtered_df = apply_filters(base_df, genres, subgenres, popularity, years)
        return timeline_release_trend(filtered_df)
