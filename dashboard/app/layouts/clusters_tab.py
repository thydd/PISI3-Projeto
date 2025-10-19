"""Layout da aba Clusters.

Esta aba oferece:
- Clusterização K-Means interativa
- Visualização PCA 2D dos clusters
- Perfil médio de cada cluster
- Tabela com amostra de músicas classificadas
"""

from dash import dcc, html


def create_clusters_layout() -> html.Div:
    """
    Cria o layout da aba Clusters.

    Returns
    -------
    html.Div
        Layout completo da aba.
    """
    return html.Div(
        className="p-4",
        children=[
            # Título
            html.H4("🎯 Clusterização de Músicas (K-Means)", className="mb-4 text-light"),
            # Controle de número de clusters
            html.Div(
                className="glass-card mb-4",
                children=[
                    html.H5("Configuração do Algoritmo", className="mb-3 text-light"),
                    html.P(
                        "K-Means agrupa músicas similares baseado em features de áudio. "
                        "Ajuste o número de clusters para explorar diferentes agrupamentos.",
                        className="text-secondary mb-4",
                    ),
                    html.Label(
                        "Número de Clusters (K)",
                        className="form-label text-light fw-semibold",
                    ),
                    dcc.Slider(
                        id="n-clusters-slider",
                        min=2,
                        max=10,
                        step=1,
                        value=5,
                        marks={i: str(i) for i in range(2, 11)},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                    html.Small(
                        "Recomendado: 4-6 clusters para melhor interpretabilidade",
                        className="text-secondary d-block mt-2",
                    ),
                ],
            ),
            # Warning de dados insuficientes
            html.Div(id="cluster-warning", className="mb-3"),
            # Visualização PCA 2D
            html.Div(
                className="glass-card mb-4",
                children=[
                    html.H5("Visualização dos Clusters (PCA 2D)", className="mb-3 text-light"),
                    html.P(
                        "Projeção 2D usando PCA (Principal Component Analysis) para visualizar "
                        "os clusters em um espaço reduzido.",
                        className="text-secondary small mb-3",
                    ),
                    dcc.Graph(id="cluster-scatter-graph"),
                ],
            ),
            # Perfil médio dos clusters
            html.Div(
                className="glass-card mb-4",
                children=[
                    html.H5("Perfil Médio de Cada Cluster", className="mb-3 text-light"),
                    html.P(
                        "Valores médios das audio features para cada cluster. "
                        "Isso ajuda a interpretar as características de cada grupo.",
                        className="text-secondary small mb-3",
                    ),
                    html.Div(id="cluster-profile-table"),
                ],
            ),
            # Amostra de músicas
            html.Div(
                className="glass-card",
                children=[
                    html.H5("Amostra de Músicas por Cluster", className="mb-3 text-light"),
                    html.P(
                        "Exemplos de músicas em cada cluster (mostrando até 50 primeiras).",
                        className="text-secondary small mb-3",
                    ),
                    html.Div(id="cluster-samples-table"),
                ],
            ),
        ],
    )
