"""
Layout da aba "Visão Geral".

Contém KPIs, dados ausentes, estatísticas descritivas e visualizações premium
de hierarquia de gêneros, distribuição de popularidade e tendência de lançamentos.
"""

from dash import dcc, html
from dash.dash_table import DataTable

from app.config import TABLE_HEADER_STYLE, TABLE_CELL_STYLE, CARD_BACKGROUND, CARD_BORDER_STYLE


def create_overview_layout() -> html.Div:
    """
    Cria o layout da aba Visão Geral.

    Returns
    -------
    html.Div
        Componente Dash contendo o layout completo da aba.
    """
    return html.Div(
        className="p-4",
        children=[
            # KPIs
            html.Div(id="overview-kpis", className="row g-3"),
            # Tabelas
            html.Div(
                className="row g-4 mt-1",
                children=[
                    html.Div(
                        className="col-xl-6",
                        children=[
                            html.H5("Dados ausentes", className="mt-3"),
                            DataTable(
                                id="missing-table",
                                data=[],
                                columns=[],
                                style_table={
                                    "overflowX": "auto",
                                    "backgroundColor": CARD_BACKGROUND,
                                    "border": CARD_BORDER_STYLE,
                                },
                                style_header=TABLE_HEADER_STYLE,
                                style_cell=dict(TABLE_CELL_STYLE, textAlign="left"),
                                style_data=TABLE_CELL_STYLE,
                                page_size=10,
                            ),
                        ],
                    ),
                    html.Div(
                        className="col-xl-6",
                        children=[
                            html.H5("Estatísticas descritivas", className="mt-3"),
                            DataTable(
                                id="descriptive-table",
                                data=[],
                                columns=[],
                                style_table={
                                    "overflowX": "auto",
                                    "backgroundColor": CARD_BACKGROUND,
                                    "border": CARD_BORDER_STYLE,
                                },
                                style_header=TABLE_HEADER_STYLE,
                                style_cell=dict(TABLE_CELL_STYLE, textAlign="left"),
                                style_data=TABLE_CELL_STYLE,
                                page_size=10,
                            ),
                        ],
                    ),
                ],
            ),
            # Resumo
            html.Div(id="overview-summary", className="mt-4 text-secondary"),
            # Visualizações Premium
            html.Div(
                className="row g-4 mt-4",
                children=[
                    html.Div(
                        className="col-xl-12",
                        children=[
                            html.H5("Hierarquia de Gêneros", className="mb-3"),
                            html.Div(
                                className="glass-card",
                                children=[dcc.Graph(id="genre-sunburst-graph")],
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="mt-4",
                children=[
                    html.H5(
                        "Tendência de Lançamentos ao Longo do Tempo", className="mb-3"
                    ),
                    html.Div(
                        className="glass-card",
                        children=[dcc.Graph(id="timeline-trend-graph")],
                    ),
                ],
            ),
        ],
    )
