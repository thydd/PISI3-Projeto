"""Layout da aba Classificação.

Esta aba oferece:
- Playground interativo de Machine Learning
- Predição de gênero musical usando Random Forest
- Sliders para input de audio features
- Visualização de probabilidades
"""

from dash import dcc, html

from app.utils.common_components import slider_component


def create_classification_layout(
    classifier_result, classifier_features, feature_ranges, key_labels, mode_labels
) -> html.Div:
    """
    Cria o layout da aba Classificação.

    Parameters
    ----------
    classifier_result : ClassifierResult
        Resultado do treinamento do classificador.
    classifier_features : list
        Lista de features usadas no modelo.
    feature_ranges : dict
        Dicionário com ranges (min, max) de cada feature.
    key_labels : dict
        Mapeamento de key numérica para nome (ex: 0 -> "C").
    mode_labels : dict
        Mapeamento de mode (0 -> "Menor", 1 -> "Maior").

    Returns
    -------
    html.Div
        Layout completo da aba.
    """
    # Calcula acurácia
    accuracy = classifier_result.accuracy * 100 if classifier_result else 0.0

    return html.Div(
        className="p-4",
        children=[
            # Título
            html.H4("🤖 Machine Learning: Predição de Gênero Musical", className="mb-4 text-light"),
            # Card de acurácia do modelo
            html.Div(
                className="glass-card mb-4",
                children=[
                    html.H5("📊 Performance do Modelo", className="mb-3 text-light"),
                    html.Div(
                        className="row g-3",
                        children=[
                            html.Div(
                                className="col-md-6",
                                children=[
                                    html.Div(
                                        className="kpi-card",
                                        children=[
                                            html.Div(
                                                className="kpi-label",
                                                children="Acurácia do Modelo",
                                            ),
                                            html.Div(
                                                className="kpi-value text-info",
                                                children=f"{accuracy:.1f}%",
                                            ),
                                            html.Small(
                                                "Random Forest Classifier",
                                                className="text-secondary",
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(
                                className="col-md-6",
                                children=[
                                    html.P(
                                        "Este modelo foi treinado com Random Forest para prever o gênero musical "
                                        "baseado em 13 audio features. Ajuste os parâmetros abaixo e clique em 'Prever' "
                                        "para ver a classificação.",
                                        className="text-secondary mb-0",
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            # Playground de predição
            html.Div(
                className="glass-card mb-4",
                children=[
                    html.H5("🎮 Playground Interativo", className="mb-4 text-light"),
                    html.P(
                        "Ajuste as audio features abaixo para simular uma música e ver qual gênero o modelo prevê:",
                        className="text-secondary mb-4",
                    ),
                    # Sliders para features numéricas contínuas
                    html.Div(
                        className="row g-3 mb-3",
                        children=[
                            slider_component(
                                feature="danceability",
                                label="Danceability (Dançabilidade)",
                                min_val=feature_ranges["danceability"][0],
                                max_val=feature_ranges["danceability"][1],
                                step=0.01,
                            ),
                            slider_component(
                                feature="energy",
                                label="Energy (Energia)",
                                min_val=feature_ranges["energy"][0],
                                max_val=feature_ranges["energy"][1],
                                step=0.01,
                            ),
                            slider_component(
                                feature="loudness",
                                label="Loudness (Volume)",
                                min_val=feature_ranges["loudness"][0],
                                max_val=feature_ranges["loudness"][1],
                                step=0.5,
                            ),
                            slider_component(
                                feature="speechiness",
                                label="Speechiness (Fala)",
                                min_val=feature_ranges["speechiness"][0],
                                max_val=feature_ranges["speechiness"][1],
                                step=0.01,
                            ),
                            slider_component(
                                feature="acousticness",
                                label="Acousticness (Acústico)",
                                min_val=feature_ranges["acousticness"][0],
                                max_val=feature_ranges["acousticness"][1],
                                step=0.01,
                            ),
                            slider_component(
                                feature="instrumentalness",
                                label="Instrumentalness (Instrumental)",
                                min_val=feature_ranges["instrumentalness"][0],
                                max_val=feature_ranges["instrumentalness"][1],
                                step=0.01,
                            ),
                            slider_component(
                                feature="liveness",
                                label="Liveness (Ao vivo)",
                                min_val=feature_ranges["liveness"][0],
                                max_val=feature_ranges["liveness"][1],
                                step=0.01,
                            ),
                            slider_component(
                                feature="valence",
                                label="Valence (Positividade)",
                                min_val=feature_ranges["valence"][0],
                                max_val=feature_ranges["valence"][1],
                                step=0.01,
                            ),
                            slider_component(
                                feature="tempo",
                                label="Tempo (BPM)",
                                min_val=feature_ranges["tempo"][0],
                                max_val=feature_ranges["tempo"][1],
                                step=1.0,
                            ),
                            slider_component(
                                feature="duration_ms",
                                label="Duration (ms)",
                                min_val=feature_ranges["duration_ms"][0],
                                max_val=feature_ranges["duration_ms"][1],
                                step=1000,
                            ),
                            slider_component(
                                feature="track_popularity",
                                label="Popularity (Popularidade)",
                                min_val=0,
                                max_val=100,
                                step=1,
                            ),
                        ],
                    ),
                    # Dropdowns para features categóricas
                    html.Div(
                        className="row g-3 mb-4",
                        children=[
                            html.Div(
                                className="col-lg-6",
                                children=[
                                    html.Label("Key (Tom Musical)", className="form-label text-light"),
                                    dcc.Dropdown(
                                        id="input-key",
                                        options=[
                                            {"label": label, "value": key}
                                            for key, label in sorted(key_labels.items())
                                        ],
                                        value=0,
                                        className="dash-dropdown",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="col-lg-6",
                                children=[
                                    html.Label("Mode (Escala)", className="form-label text-light"),
                                    dcc.RadioItems(
                                        id="input-mode",
                                        options=[
                                            {"label": " " + label, "value": mode}
                                            for mode, label in mode_labels.items()
                                        ],
                                        value=1,
                                        className="text-light",
                                        inline=True,
                                    ),
                                ],
                            ),
                        ],
                    ),
                    # Botão de predição
                    html.Div(
                        className="text-center mb-4",
                        children=[
                            html.Button(
                                [html.I(className="bi bi-lightning-charge me-2"), "Prever Gênero"],
                                id="predict-button",
                                className="btn btn-lg btn-info",
                                n_clicks=0,
                            ),
                        ],
                    ),
                    # Resultado da predição
                    html.Div(id="prediction-result"),
                ],
            ),
        ],
    )
