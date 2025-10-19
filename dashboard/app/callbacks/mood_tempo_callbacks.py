"""Callbacks da aba Humor & Tempo.

Gerencia a interatividade dos gráficos de:
- Distribuição de BPM (tempo) por gênero
- Mapa de densidade emocional (Energia × Valência)
"""

from dash import Input, Output

from app.utils.common_components import apply_filters
from app.utils.visualizations import tempo_ridge_like, valence_energy_density


def register_mood_tempo_callbacks(app, base_df):
    """
    Registra callbacks da aba Humor & Tempo.

    Parameters
    ----------
    app : Dash
        Instância do aplicativo Dash.
    base_df : pd.DataFrame
        DataFrame base com todos os dados.
    """

    @app.callback(
        Output("tempo-ridge-graph", "figure"),
        Input("tempo-genres-dropdown", "value"),
        Input("genre-dropdown", "value"),
        Input("subgenre-dropdown", "value"),
        Input("popularity-slider", "value"),
        Input("year-slider", "value"),
    )
    def update_tempo_ridge(
        selected_tempo_genres, genres, subgenres, popularity, years
    ):
        """
        Atualiza o gráfico de distribuição de BPM.

        Parameters
        ----------
        selected_tempo_genres : list
            Gêneros selecionados no dropdown específico desta aba.
        genres : list
            Gêneros dos filtros globais.
        subgenres : list
            Subgêneros dos filtros globais.
        popularity : list
            Range de popularidade [min, max].
        years : list
            Range de anos [min, max].

        Returns
        -------
        plotly.graph_objects.Figure
            Gráfico violin/ridge de BPM por gênero.
        """
        # Aplica filtros globais
        filtered_df = apply_filters(base_df, genres, subgenres, popularity, years)

        # Se não houver gêneros selecionados para BPM, usa todos os disponíveis
        if not selected_tempo_genres:
            selected_tempo_genres = filtered_df["playlist_genre"].dropna().unique()

        # Gera visualização
        return tempo_ridge_like(filtered_df, selected_tempo_genres)

    @app.callback(
        Output("mood-density-graph", "figure"),
        Input("mood-genre-dropdown", "value"),
        Input("genre-dropdown", "value"),
        Input("subgenre-dropdown", "value"),
        Input("popularity-slider", "value"),
        Input("year-slider", "value"),
    )
    def update_mood_density(selected_mood_genre, genres, subgenres, popularity, years):
        """
        Atualiza o mapa de densidade emocional (Energia × Valência).

        Parameters
        ----------
        selected_mood_genre : str
            Gênero selecionado para análise emocional ("Todos" ou nome do gênero).
        genres : list
            Gêneros dos filtros globais.
        subgenres : list
            Subgêneros dos filtros globais.
        popularity : list
            Range de popularidade [min, max].
        years : list
            Range de anos [min, max].

        Returns
        -------
        plotly.graph_objects.Figure
            Heatmap 2D com quadrantes emocionais.
        """
        # Aplica filtros globais
        filtered_df = apply_filters(base_df, genres, subgenres, popularity, years)

        # Se "Todos" foi selecionado, passa None para a função
        genre_param = None if selected_mood_genre == "Todos" else selected_mood_genre

        # Gera visualização
        return valence_energy_density(filtered_df, genre_param)
