"""
Layout da aba "Popularidade".

Contém visualizações relacionadas a artistas, gêneros, distribuição de tons
e danceability, incluindo network de artistas.
"""

from dash import dcc, html


def create_popularity_layout() -> html.Div:
    """
    Cria o layout da aba Popularidade.

    Returns
    -------
    html.Div
        Componente Dash contendo o layout completo da aba.
    """
    return html.Div(
        className="p-4",
        children=[
            html.Div(
                className="row g-4",
                children=[
                    html.Div(
                        className="col-lg-12",
                        children=[
                            html.H5("Network de Top Artistas", className="mb-3"),
                            html.Div(
                                className="glass-card",
                                children=[dcc.Graph(id="artist-network-graph")],
                            ),
                        ],
                    ),
                    html.Div(
                        className="col-lg-6",
                        children=[
                            html.Div(
                                className="glass-card",
                                children=[dcc.Graph(id="top-artists-graph")],
                            ),
                        ],
                    ),
                    html.Div(
                        className="col-lg-6",
                        children=[
                            html.Div(
                                className="glass-card",
                                children=[dcc.Graph(id="genre-distribution-graph")],
                            ),
                        ],
                    ),
                    html.Div(
                        className="col-lg-6",
                        children=[
                            html.Div(
                                className="glass-card",
                                children=[dcc.Graph(id="key-distribution-graph")],
                            ),
                        ],
                    ),
                    html.Div(
                        className="col-lg-6",
                        children=[
                            html.Div(
                                className="glass-card",
                                children=[dcc.Graph(id="danceability-graph")],
                            ),
                        ],
                    ),
                ],
            )
        ],
    )
