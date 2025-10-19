"""Callbacks da aba Explorador.

Gerencia:
- Atualização da tabela interativa
- Geração de histogramas dinâmicos
- Scatter matrix multivariada
- Download de dados em CSV
"""

from dash import Input, Output, State, dcc

from app.utils.common_components import apply_filters
from app.utils.visualizations import feature_distribution, scatter_matrix


def register_explorer_callbacks(app, base_df, classifier_features):
    """
    Registra callbacks da aba Explorador.

    Parameters
    ----------
    app : Dash
        Instância do aplicativo Dash.
    base_df : pd.DataFrame
        DataFrame base com todos os dados.
    classifier_features : list
        Lista de features numéricas para análise.
    """

    @app.callback(
        Output("explorer-table", "data"),
        Output("explorer-table", "columns"),
        Input("genre-dropdown", "value"),
        Input("subgenre-dropdown", "value"),
        Input("popularity-slider", "value"),
        Input("year-slider", "value"),
    )
    def update_explorer_table(genres, subgenres, popularity, years):
        """
        Atualiza a tabela de dados com base nos filtros.

        Returns
        -------
        tuple
            (dados da tabela, definição de colunas)
        """
        # Aplica filtros
        filtered_df = apply_filters(base_df, genres, subgenres, popularity, years)

        # Limita a 100 linhas para performance
        display_df = filtered_df.head(100)

        # Seleciona colunas relevantes
        columns_to_show = [
            "track_name",
            "track_artist",
            "track_album_name",
            "playlist_genre",
            "playlist_subgenre",
            "track_popularity",
            "danceability",
            "energy",
            "valence",
            "tempo",
            "duration_ms",
        ]

        # Filtra apenas colunas existentes
        available_cols = [col for col in columns_to_show if col in display_df.columns]
        display_df = display_df[available_cols]

        # Prepara dados e colunas
        data = display_df.to_dict("records")
        columns = [
            {"name": col.replace("_", " ").title(), "id": col} for col in available_cols
        ]

        return data, columns

    @app.callback(
        Output("download-csv", "data"),
        Input("download-csv-btn", "n_clicks"),
        State("genre-dropdown", "value"),
        State("subgenre-dropdown", "value"),
        State("popularity-slider", "value"),
        State("year-slider", "value"),
        prevent_initial_call=True,
    )
    def download_csv(n_clicks, genres, subgenres, popularity, years):
        """
        Gera arquivo CSV com dados filtrados para download.

        Returns
        -------
        dict
            Dicionário com dados para download via dcc.send_data_frame.
        """
        if n_clicks > 0:
            # Aplica filtros
            filtered_df = apply_filters(base_df, genres, subgenres, popularity, years)

            # Retorna CSV
            return dcc.send_data_frame(
                filtered_df.to_csv, "spotify_filtered_data.csv", index=False
            )
        return None

    @app.callback(
        Output("dynamic-histogram", "figure"),
        Input("histogram-feature-dropdown", "value"),
        Input("histogram-color-dropdown", "value"),
        Input("genre-dropdown", "value"),
        Input("subgenre-dropdown", "value"),
        Input("popularity-slider", "value"),
        Input("year-slider", "value"),
    )
    def update_histogram(feature, color_by, genres, subgenres, popularity, years):
        """
        Atualiza histograma dinâmico baseado na feature selecionada.

        Returns
        -------
        plotly.graph_objects.Figure
            Histograma com marginal box plot.
        """
        # Aplica filtros
        filtered_df = apply_filters(base_df, genres, subgenres, popularity, years)

        # Trata "none" como None
        color_column = None if color_by == "none" else color_by

        # Gera visualização
        return feature_distribution(filtered_df, feature, color=color_column)

    @app.callback(
        Output("scatter-matrix-graph", "figure"),
        Input("genre-dropdown", "value"),
        Input("subgenre-dropdown", "value"),
        Input("popularity-slider", "value"),
        Input("year-slider", "value"),
    )
    def update_scatter_matrix(genres, subgenres, popularity, years):
        """
        Atualiza scatter matrix com features selecionadas.

        Returns
        -------
        plotly.graph_objects.Figure
            Matriz de dispersão multivariada.
        """
        # Aplica filtros
        filtered_df = apply_filters(base_df, genres, subgenres, popularity, years)

        # Limita a 1000 amostras para performance
        if len(filtered_df) > 1000:
            filtered_df = filtered_df.sample(1000, random_state=42)

        # Seleciona features principais para matriz
        dimensions = [
            "danceability",
            "energy",
            "valence",
            "tempo",
            "acousticness",
        ]

        # Gera visualização
        return scatter_matrix(filtered_df, dimensions, color="playlist_genre")
