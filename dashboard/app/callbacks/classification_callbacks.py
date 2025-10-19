"""Callbacks da aba Classificação.

Gerencia a predição de gênero musical usando Random Forest.
"""

import pandas as pd
from dash import Input, Output, State, html, dash_table

from app.config import PRIMARY_TEXT, SECONDARY_TEXT


def register_classification_callbacks(app, classifier_result, classifier_features):
    """
    Registra callbacks da aba Classificação.

    Parameters
    ----------
    app : Dash
        Instância do aplicativo Dash.
    classifier_result : ClassifierResult
        Modelo treinado e metadados.
    classifier_features : list
        Lista de features usadas no modelo.
    """

    @app.callback(
        Output("prediction-result", "children"),
        Input("predict-button", "n_clicks"),
        State("input-danceability", "value"),
        State("input-energy", "value"),
        State("input-key", "value"),
        State("input-loudness", "value"),
        State("input-mode", "value"),
        State("input-speechiness", "value"),
        State("input-acousticness", "value"),
        State("input-instrumentalness", "value"),
        State("input-liveness", "value"),
        State("input-valence", "value"),
        State("input-tempo", "value"),
        State("input-duration_ms", "value"),
        State("input-track_popularity", "value"),
        prevent_initial_call=True,
    )
    def predict_genre(
        n_clicks,
        danceability,
        energy,
        key,
        loudness,
        mode,
        speechiness,
        acousticness,
        instrumentalness,
        liveness,
        valence,
        tempo,
        duration_ms,
        track_popularity,
    ):
        """
        Faz predição de gênero baseado nos inputs do usuário.

        Returns
        -------
        html.Div
            Componente com resultado da predição e tabela de probabilidades.
        """
        if n_clicks == 0 or not classifier_result:
            return html.Div()

        # Cria DataFrame com os inputs
        input_data = pd.DataFrame(
            {
                "danceability": [danceability],
                "energy": [energy],
                "key": [key],
                "loudness": [loudness],
                "mode": [mode],
                "speechiness": [speechiness],
                "acousticness": [acousticness],
                "instrumentalness": [instrumentalness],
                "liveness": [liveness],
                "valence": [valence],
                "tempo": [tempo],
                "duration_ms": [duration_ms],
                "track_popularity": [track_popularity],
            }
        )

        # Faz predição
        try:
            prediction = classifier_result.pipeline.predict(input_data)[0]
            probabilities = classifier_result.pipeline.predict_proba(input_data)[0]

            # Cria tabela de probabilidades
            prob_df = pd.DataFrame(
                {
                    "Gênero": classifier_result.pipeline.classes_,
                    "Probabilidade (%)": probabilities * 100,
                }
            ).sort_values("Probabilidade (%)", ascending=False)

            # Formata percentual
            prob_df["Probabilidade (%)"] = prob_df["Probabilidade (%)"].round(2)

            # Resultado
            return html.Div(
                children=[
                    html.Div(
                        className="alert alert-success mb-4",
                        style={"backgroundColor": "rgba(0, 245, 212, 0.15)", "border": "1px solid rgba(0, 245, 212, 0.3)"},
                        children=[
                            html.H5(
                                [
                                    html.I(className="bi bi-check-circle-fill me-2"),
                                    "Gênero Previsto:",
                                ],
                                className="text-light mb-2",
                            ),
                            html.H3(
                                prediction.title(),
                                className="text-info mb-0",
                                style={"fontWeight": "bold"},
                            ),
                        ],
                    ),
                    html.H6("Probabilidades por Gênero:", className="text-light mb-3"),
                    dash_table.DataTable(
                        data=prob_df.to_dict("records"),
                        columns=[
                            {"name": "Gênero", "id": "Gênero"},
                            {"name": "Probabilidade (%)", "id": "Probabilidade (%)"},
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
                            "padding": "10px",
                        },
                        style_data_conditional=[
                            {
                                "if": {"row_index": 0},
                                "backgroundColor": "rgba(76, 201, 240, 0.2)",
                                "fontWeight": "bold",
                            }
                        ],
                    ),
                ],
            )

        except Exception as e:
            return html.Div(
                className="alert alert-danger",
                children=[
                    html.H5("❌ Erro na predição:", className="text-danger"),
                    html.P(str(e), className="mb-0"),
                ],
            )
