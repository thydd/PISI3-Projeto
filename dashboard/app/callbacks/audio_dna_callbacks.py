"""
Callbacks da aba "Audio DNA".

Gerencia atualizações de visualizações multidimensionais de features de áudio,
incluindo scatter 3D e heatmap de correlação.
"""

from typing import List

import pandas as pd
from dash import Input, Output

from app.utils.common_components import apply_filters
from app.utils.visualizations import (
    correlation_heatmap,
    feature_3d_scatter,
)


def register_audio_dna_callbacks(
    app, base_df: pd.DataFrame, classifier_features: List[str]
):
    """
    Registra callbacks da aba Audio DNA.

    Parameters
    ----------
    app : Dash
        Instância do aplicativo Dash.
    base_df : pd.DataFrame
        DataFrame base completo.
    classifier_features : List[str]
        Lista de features usadas no classificador.
    """

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
    def update_3d_scatter(
        x_axis: str,
        y_axis: str,
        z_axis: str,
        color_by: str,
        genres: List[str],
        subgenres: List[str],
        popularity: List[int],
        years: List[int],
    ):
        """Atualiza scatter 3D com base nos eixos e filtros selecionados."""
        filtered_df = apply_filters(base_df, genres, subgenres, popularity, years)
        return feature_3d_scatter(
            filtered_df, x_axis, y_axis, z_axis, color=color_by, sample_size=2000
        )

    @app.callback(
        Output("correlation-heatmap-graph", "figure"),
        Input("genre-dropdown", "value"),
        Input("subgenre-dropdown", "value"),
        Input("popularity-slider", "value"),
        Input("year-slider", "value"),
    )
    def update_correlation_heatmap(genres, subgenres, popularity, years):
        """Atualiza heatmap de correlação."""
        filtered_df = apply_filters(base_df, genres, subgenres, popularity, years)
        return correlation_heatmap(filtered_df, classifier_features)
