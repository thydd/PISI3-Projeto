"""
Layout da aba "Audio DNA".

Contém análise multidimensional de features de áudio, incluindo scatter 3D,
heatmap de correlação e gráfico de radar.
"""

from dash import dcc, html


def create_audio_dna_layout(classifier_features: list) -> html.Div:
    """
    Cria o layout da aba Audio DNA.

    Parameters
    ----------
    classifier_features : list
        Lista de features disponíveis para seleção nos eixos.

    Returns
    -------
    html.Div
        Componente Dash contendo o layout completo da aba.
    """
    return html.Div(
        className="p-4",
        children=[
            html.H4("Análise Multidimensional de Features", className="mb-4"),
            # Controles do 3D Scatter
            html.Div(
                className="row g-3 mb-4",
                children=[
                    html.Div(
                        className="col-md-3",
                        children=[
                            html.Label("Eixo X", className="form-label text-light"),
                            dcc.Dropdown(
                                id="3d-x-axis",
                                options=[
                                    {
                                        "label": f.replace("_", " ").title(),
                                        "value": f,
                                    }
                                    for f in classifier_features
                                ],
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
                                options=[
                                    {
                                        "label": f.replace("_", " ").title(),
                                        "value": f,
                                    }
                                    for f in classifier_features
                                ],
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
                                options=[
                                    {
                                        "label": f.replace("_", " ").title(),
                                        "value": f,
                                    }
                                    for f in classifier_features
                                ],
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
                                    {
                                        "label": "Popularidade",
                                        "value": "track_popularity",
                                    },
                                ],
                                value="playlist_genre",
                                className="dash-dropdown",
                            ),
                        ],
                    ),
                ],
            ),
            # 3D Scatter Plot
            html.Div(
                className="glass-card", children=[dcc.Graph(id="feature-3d-scatter")]
            ),
            # Correlation Heatmap
            html.Div(
                className="row g-4 mt-4",
                children=[
                    html.Div(
                        className="col-xl-12",
                        children=[
                            html.Div(
                                className="glass-card",
                                children=[dcc.Graph(id="correlation-heatmap-graph")],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
