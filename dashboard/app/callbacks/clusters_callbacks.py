"""Callbacks da aba Clusters.

Gerencia a clusterização K-Means e visualizações relacionadas.
"""

from dash import Input, Output, html, dash_table

from app.utils.common_components import apply_filters
from app.utils.model_utils import apply_clustering, cluster_profile_table
from app.utils.visualizations import cluster_scatter
from app.config import PRIMARY_TEXT


def register_clusters_callbacks(app, base_df, cluster_features):
    """
    Registra callbacks da aba Clusters.

    Parameters
    ----------
    app : Dash
        Instância do aplicativo Dash.
    base_df : pd.DataFrame
        DataFrame base com todos os dados.
    cluster_features : list
        Lista de features usadas na clusterização.
    """

    @app.callback(
        Output("cluster-warning", "children"),
        Output("cluster-scatter-graph", "figure"),
        Output("cluster-profile-table", "children"),
        Output("cluster-samples-table", "children"),
        Input("n-clusters-slider", "value"),
        Input("genre-dropdown", "value"),
        Input("subgenre-dropdown", "value"),
        Input("popularity-slider", "value"),
        Input("year-slider", "value"),
    )
    def update_clusters(n_clusters, genres, subgenres, popularity, years):
        """
        Atualiza todos os componentes da aba de clusters.

        Returns
        -------
        tuple
            (warning, scatter_figure, profile_table, samples_table)
        """
        # Aplica filtros
        filtered_df = apply_filters(base_df, genres, subgenres, popularity, years)

        # Verifica se há dados suficientes
        MIN_SAMPLES = 50
        if len(filtered_df) < MIN_SAMPLES:
            warning = html.Div(
                className="alert alert-warning",
                children=[
                    html.Strong("⚠️ Dados insuficientes:"),
                    f" São necessárias pelo menos {MIN_SAMPLES} músicas para clusterização. "
                    f"Atualmente há {len(filtered_df)} músicas após aplicar os filtros. "
                    "Remova alguns filtros para obter mais dados.",
                ],
            )
            # Retorna componentes vazios
            from plotly import graph_objects as go

            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="Dados insuficientes",
                template="plotly_dark",
                paper_bgcolor="rgba(0, 0, 0, 0)",
                plot_bgcolor="rgba(0, 0, 0, 0)",
            )
            return warning, empty_fig, html.Div(), html.Div()

        # Aplica clusterização
        try:
            cluster_result = apply_clustering(filtered_df, n_clusters)

            # Cria scatter plot
            scatter_fig = cluster_scatter(
                filtered_df,
                cluster_result.pca_projection,
                cluster_result.clusters,
            )

            # Cria tabela de perfil
            profile_df = cluster_profile_table(cluster_result)
            profile_table = dash_table.DataTable(
                data=profile_df.to_dict("records"),
                columns=[{"name": col, "id": col} for col in profile_df.columns],
                style_table={
                    "overflowX": "auto",
                    "backgroundColor": "rgba(11, 16, 26, 0.6)",
                },
                style_header={
                    "backgroundColor": "rgba(11, 16, 26, 0.9)",
                    "color": PRIMARY_TEXT,
                    "fontWeight": "600",
                    "border": "1px solid rgba(76, 201, 240, 0.15)",
                    "textAlign": "center",
                },
                style_cell={
                    "backgroundColor": "transparent",
                    "color": PRIMARY_TEXT,
                    "border": "1px solid rgba(76, 201, 240, 0.1)",
                    "textAlign": "center",
                    "padding": "10px",
                    "fontFamily": "'JetBrains Mono', monospace",
                },
                style_data_conditional=[
                    {
                        "if": {"column_id": "Cluster"},
                        "fontWeight": "bold",
                        "backgroundColor": "rgba(76, 201, 240, 0.1)",
                    }
                ],
            )

            # Cria tabela de amostras
            samples_df = filtered_df.copy()
            samples_df["Cluster"] = cluster_result.clusters

            # Seleciona colunas relevantes e limita a 50 linhas
            sample_cols = [
                "Cluster",
                "track_name",
                "track_artist",
                "playlist_genre",
                "track_popularity",
            ]
            available_cols = [col for col in sample_cols if col in samples_df.columns]
            samples_df = samples_df[available_cols].head(50)

            samples_table = dash_table.DataTable(
                data=samples_df.to_dict("records"),
                columns=[
                    {"name": col.replace("_", " ").title(), "id": col}
                    for col in available_cols
                ],
                style_table={
                    "overflowX": "auto",
                    "backgroundColor": "rgba(11, 16, 26, 0.6)",
                },
                style_header={
                    "backgroundColor": "rgba(11, 16, 26, 0.9)",
                    "color": PRIMARY_TEXT,
                    "fontWeight": "600",
                    "border": "1px solid rgba(76, 201, 240, 0.15)",
                },
                style_cell={
                    "backgroundColor": "transparent",
                    "color": PRIMARY_TEXT,
                    "border": "1px solid rgba(76, 201, 240, 0.1)",
                    "textAlign": "left",
                    "padding": "8px",
                },
                style_data_conditional=[
                    {
                        "if": {"row_index": "odd"},
                        "backgroundColor": "rgba(76, 201, 240, 0.05)",
                    }
                ],
            )

            # Sem warning
            return html.Div(), scatter_fig, profile_table, samples_table

        except Exception as e:
            # Em caso de erro
            warning = html.Div(
                className="alert alert-danger",
                children=[
                    html.Strong("❌ Erro na clusterização:"),
                    f" {str(e)}",
                ],
            )
            from plotly import graph_objects as go

            empty_fig = go.Figure()
            return warning, empty_fig, html.Div(), html.Div()
