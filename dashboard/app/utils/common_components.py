"""
Componentes reutilizáveis compartilhados entre diferentes partes do dashboard.

Este módulo contém funções para criar componentes UI comuns e aplicar filtros
aos dados.
"""

from __future__ import annotations

from typing import Dict, List, Union

import numpy as np
import pandas as pd
from dash import dcc, html

from app.utils.data_utils import FilterState, filter_dataframe

Number = Union[int, float]


def _range_marks(
    min_value: Number, max_value: Number, *, vertical: bool = False
) -> Dict[Number, Dict[str, str]]:
    """
    Gera marcações para um slider com formatação apropriada.

    Parameters
    ----------
    min_value : Number
        Valor mínimo do range.
    max_value : Number
        Valor máximo do range.
    vertical : bool, optional
        Se True, formata o texto verticalmente, by default False.

    Returns
    -------
    Dict[Number, Dict[str, str]]
        Dicionário com as marcações formatadas.
    """

    def _format_label(value: Number) -> str:
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.2f}".rstrip("0").rstrip(".")

    def _label_style() -> Dict[str, str]:
        base = {"font-size": "12px"}
        if vertical:
            base.update(
                {"writing-mode": "vertical-rl", "text-orientation": "upright"}
            )
        return base

    if min_value >= max_value:
        return {
            min_value: {"label": _format_label(min_value), "style": _label_style()}
        }

    mid_value = (min_value + max_value) / 2
    values: List[Number] = [min_value]
    if not np.isclose(mid_value, min_value) and not np.isclose(
        mid_value, max_value
    ):
        values.append(mid_value)
    values.append(max_value)

    marks: Dict[Number, Dict[str, str]] = {}
    for value in values:
        marks[value] = {"label": _format_label(value), "style": _label_style()}
    return marks


def apply_filters(
    base_df: pd.DataFrame,
    genres: List[str] | None,
    subgenres: List[str] | None,
    popularity: List[int],
    years: List[int],
) -> pd.DataFrame:
    """
    Aplica filtros ao DataFrame base.

    Parameters
    ----------
    base_df : pd.DataFrame
        DataFrame original.
    genres : List[str] | None
        Lista de gêneros selecionados.
    subgenres : List[str] | None
        Lista de subgêneros selecionados.
    popularity : List[int]
        Range de popularidade [min, max].
    years : List[int]
        Range de anos [min, max].

    Returns
    -------
    pd.DataFrame
        DataFrame filtrado.
    """
    state = FilterState(
        genres=genres or [],
        subgenres=subgenres or [],
        min_popularity=int(popularity[0]),
        max_popularity=int(popularity[1]),
        year_range=(int(years[0]), int(years[1])),
    )
    return filter_dataframe(base_df, state)


def slider_component(
    feature: str, label: str, min_val: float, max_val: float, step: float
) -> html.Div:
    """
    Cria um componente de slider para uma feature específica.

    Parameters
    ----------
    feature : str
        Nome da feature.
    label : str
        Label exibido para o usuário.
    min_val : float
        Valor mínimo do slider.
    max_val : float
        Valor máximo do slider.
    step : float
        Incremento do slider.

    Returns
    -------
    html.Div
        Componente Dash contendo o slider.
    """
    if np.isfinite(min_val) and np.isfinite(max_val):
        value = float(np.clip((min_val + max_val) / 2, min_val, max_val))
    else:
        value = float(min_val)

    return html.Div(
        className="col-lg-4 col-md-6",
        children=[
            html.Div(
                className="glass-card p-3 h-100",
                children=[
                    html.Label(label, className="form-label fw-semibold text-light"),
                    dcc.Slider(
                        id=f"input-{feature}",
                        min=min_val,
                        max=max_val,
                        step=step,
                        value=value,
                        marks=_range_marks(min_val, max_val),
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    html.Div(
                        className="small text-secondary mt-2",
                        children=[f"Min: {min_val:.2f} | Máx: {max_val:.2f}"],
                    ),
                ],
            ),
        ],
    )


def create_range_marks(min_value: Number, max_value: Number) -> Dict[Number, Dict[str, str]]:
    """
    Wrapper público para _range_marks.

    Parameters
    ----------
    min_value : Number
        Valor mínimo.
    max_value : Number
        Valor máximo.

    Returns
    -------
    Dict[Number, Dict[str, str]]
        Marcações formatadas para o slider.
    """
    return _range_marks(min_value, max_value)
