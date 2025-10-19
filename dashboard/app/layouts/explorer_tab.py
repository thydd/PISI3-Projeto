"""Layout da aba Explorador.

Esta aba oferece:
- Tabela interativa com dados filtrados
- Histogramas dinâmicos de feat                            dcc.Dropdown(
                                id="histogram-color-dropdown",
                                options=[
                                    {"label": "Gênero", "value": "playlist_genre"},
                                    {"label": "Subgênero", "value": "playlist_subgenre"},
                                    {"label": "Sem cor", "value": "none"},
                                ],
                                value="none",
                                className="dash-dropdown",
                            ),riz de dispersão (scatter matrix)
- Download de dados em CSV
"""

from dash import dcc, html, dash_table


def create_explorer_layout(classifier_features: list) -> html.Div:
    """
    Cria o layout da aba Explorador.

    Parameters
    ----------
    classifier_features : list
        Lista de features numéricas disponíveis para análise.

    Returns
    -------
    html.Div
        Layout completo da aba.
    """
    # Features formatadas para exibição
    feature_options = [
        {"label": f.replace("_", " ").title(), "value": f}
        for f in classifier_features
    ]

    return html.Div(
        className="p-4",
        children=[
            # Título
            html.H4("Explorador de Dados Interativo", className="mb-4 text-light"),
            # Tabela de dados
            html.Div(
                className="glass-card mb-4",
                children=[
                    html.Div(
                        className="d-flex justify-content-between align-items-center mb-3",
                        children=[
                            html.H5("Tabela de Dados Filtrados", className="mb-0 text-light"),
                            html.Button(
                                [html.I(className="bi bi-download me-2"), "Baixar CSV"],
                                id="download-csv-btn",
                                className="btn btn-sm btn-outline-info",
                                n_clicks=0,
                            ),
                        ],
                    ),
                    dcc.Download(id="download-csv"),
                    html.P(
                        "Mostrando as primeiras 100 linhas dos dados filtrados. Use os filtros globais para refinar.",
                        className="text-secondary small mb-3",
                    ),
                    dash_table.DataTable(
                        id="explorer-table",
                        data=[],
                        columns=[],
                        page_size=20,
                        style_table={
                            "overflowX": "auto",
                            "backgroundColor": "rgba(11, 16, 26, 0.6)",
                        },
                        style_header={
                            "backgroundColor": "rgba(11, 16, 26, 0.9)",
                            "color": "#f8f9fa",
                            "fontWeight": "600",
                            "border": "1px solid rgba(76, 201, 240, 0.15)",
                            "textTransform": "uppercase",
                            "fontSize": "0.75rem",
                        },
                        style_cell={
                            "backgroundColor": "transparent",
                            "color": "#f8f9fa",
                            "border": "1px solid rgba(76, 201, 240, 0.1)",
                            "fontFamily": "'JetBrains Mono', monospace",
                            "fontSize": "0.85rem",
                            "textAlign": "left",
                            "padding": "8px",
                        },
                        style_data_conditional=[
                            {
                                "if": {"row_index": "odd"},
                                "backgroundColor": "rgba(76, 201, 240, 0.05)",
                            }
                        ],
                    ),
                ],
            ),
            # Controles para histograma
            html.Div(
                className="row g-3 mb-4",
                children=[
                    html.Div(
                        className="col-lg-6",
                        children=[
                            html.Label(
                                "Selecione uma métrica para distribuição",
                                className="form-label text-light fw-semibold",
                            ),
                            dcc.Dropdown(
                                id="histogram-feature-dropdown",
                                options=feature_options,
                                value="track_popularity",
                                className="dash-dropdown",
                            ),
                        ],
                    ),
                    html.Div(
                        className="col-lg-6",
                        children=[
                            html.Label(
                                "Colorir por",
                                className="form-label text-light fw-semibold",
                            ),
                            dcc.Dropdown(
                                id="histogram-color-dropdown",
                                options=[
                                    {"label": "Gênero", "value": "playlist_genre"},
                                    {"label": "Subgênero", "value": "playlist_subgenre"},
                                    {"label": "Sem cor", "value": "none"},
                                ],
                                value="playlist_genre",
                                className="dash-dropdown",
                            ),
                        ],
                    ),
                ],
            ),
            # Histograma dinâmico
            html.Div(
                className="glass-card mb-4",
                children=[
                    html.H5("Distribuição da Feature Selecionada", className="mb-3 text-light"),
                    dcc.Graph(id="dynamic-histogram"),
                ],
            ),
            # Scatter Matrix
            html.Div(
                className="glass-card",
                children=[
                    html.Details(
                        open=False,
                        children=[
                            html.Summary(
                                "📊 Matriz de Dispersão Multivariada (clique para expandir)",
                                className="text-light fw-semibold mb-3",
                                style={"cursor": "pointer"},
                            ),
                            html.P(
                                "Visualização de correlações entre múltiplas features (limitado a 1000 amostras para performance).",
                                className="text-secondary small mb-3",
                            ),
                            dcc.Graph(id="scatter-matrix-graph"),
                        ],
                    ),
                ],
            ),
        ],
    )
