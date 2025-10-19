"""Layout da aba Humor & Tempo.

Esta aba oferece visualizações focadas em:
- Distribuição de BPM (tempo) por gênero
- Mapa emocional baseado em Energia × Valência com quadrantes
"""

from dash import dcc, html


def create_mood_tempo_layout(genre_options: list) -> html.Div:
    """
    Cria o layout da aba Humor & Tempo.

    Parameters
    ----------
    genre_options : list
        Lista de gêneros disponíveis no dataset.

    Returns
    -------
    html.Div
        Layout completo da aba.
    """
    return html.Div(
        className="p-4",
        children=[
            # Título da seção
            html.H4("Análise de Humor e Tempo Musical", className="mb-4 text-light"),
            # Controles
            html.Div(
                className="row g-3 mb-4",
                children=[
                    # Seleção de gêneros para BPM
                    html.Div(
                        className="col-lg-6",
                        children=[
                            html.Label(
                                "Selecione gêneros para comparar BPM",
                                className="form-label text-light fw-semibold",
                            ),
                            dcc.Dropdown(
                                id="tempo-genres-dropdown",
                                options=[
                                    {"label": g.title(), "value": g}
                                    for g in genre_options
                                ],
                                value=genre_options[: min(5, len(genre_options))],
                                multi=True,
                                placeholder="Escolha até 5 gêneros",
                                className="dash-dropdown",
                            ),
                            html.Small(
                                "Compare a distribuição de batidas por minuto entre gêneros",
                                className="text-secondary d-block mt-2",
                            ),
                        ],
                    ),
                    # Seleção de gênero para mapa emocional
                    html.Div(
                        className="col-lg-6",
                        children=[
                            html.Label(
                                "Mapa emocional por gênero (Energia × Valência)",
                                className="form-label text-light fw-semibold",
                            ),
                            dcc.Dropdown(
                                id="mood-genre-dropdown",
                                options=[{"label": "Todos os gêneros", "value": "Todos"}]
                                + [
                                    {"label": g.title(), "value": g}
                                    for g in genre_options
                                ],
                                value="Todos",
                                className="dash-dropdown",
                            ),
                            html.Small(
                                "Explore o mapa de emoções: Feliz, Calmo, Triste, Intenso",
                                className="text-secondary d-block mt-2",
                            ),
                        ],
                    ),
                ],
            ),
            # Gráfico de BPM
            html.Div(
                className="glass-card mb-4",
                children=[
                    html.H5("Distribuição de BPM por Gênero", className="mb-3 text-light"),
                    dcc.Graph(id="tempo-ridge-graph"),
                ],
            ),
            # Divisor
            html.Hr(className="my-4 border-secondary"),
            # Gráfico de mapa emocional
            html.Div(
                className="glass-card",
                children=[
                    html.H5("Mapa Emocional: Energia × Valência", className="mb-3 text-light"),
                    html.P(
                        [
                            "Este mapa divide as músicas em 4 quadrantes emocionais: ",
                            html.Strong("Feliz", className="text-warning"),
                            " (alta energia + valência), ",
                            html.Strong("Intenso", className="text-danger"),
                            " (alta energia + baixa valência), ",
                            html.Strong("Calmo", className="text-info"),
                            " (baixa energia + valência), ",
                            html.Strong("Triste", className="text-primary"),
                            " (baixa energia + valência).",
                        ],
                        className="text-secondary small",
                    ),
                    dcc.Graph(id="mood-density-graph"),
                ],
            ),
        ],
    )
