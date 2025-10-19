"""
Callbacks da aba "Popularidade".

Gerencia atualizações de gráficos relacionados a artistas, gêneros, distribuição
de tons e danceability.
"""

from typing import List

import pandas as pd
from dash import Input, Output

from app.utils.common_components import apply_filters
from app.utils.data_utils import (
    danceability_by_genre,
    genre_distribution,
    key_distribution,
    top_artists_by_popularity,
)
from app.utils.visualizations import artist_network_top, bar_chart


def register_popularity_callbacks(app, base_df: pd.DataFrame):
    """
    Registra callbacks da aba Popularidade.

    Parameters
    ----------
    app : Dash
        Instância do aplicativo Dash.
    base_df : pd.DataFrame
        DataFrame base completo.
    """

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
    def update_popularity(
        genres: List[str], subgenres: List[str], popularity: List[int], years: List[int]
    ):
        """Atualiza gráficos da aba Popularidade."""
        filtered = apply_filters(base_df, genres, subgenres, popularity, years)

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
        Output("artist-network-graph", "figure"),
        Input("genre-dropdown", "value"),
        Input("subgenre-dropdown", "value"),
        Input("popularity-slider", "value"),
        Input("year-slider", "value"),
    )
    def update_artist_network(genres, subgenres, popularity, years):
        """Atualiza network de artistas."""
        filtered_df = apply_filters(base_df, genres, subgenres, popularity, years)
        return artist_network_top(filtered_df, top_n=30)
