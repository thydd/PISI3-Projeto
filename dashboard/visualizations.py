from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def bar_chart(series: pd.Series, title: str, x_title: str, y_title: str) -> go.Figure:
    fig = px.bar(
        series.sort_values(),
        x=series.values,
        y=series.index,
        orientation="h",
        text=series.values.round(2),
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
        yaxis=dict(autorange="reversed"),
    )
    fig.update_traces(textposition="outside")
    return fig


def tempo_ridge_like(df: pd.DataFrame, genres: Iterable[str]) -> go.Figure:
    filtered = df[df["playlist_genre"].isin(genres)] if genres else df
    fig = px.violin(
        filtered,
        x="tempo",
        y="playlist_genre",
        color="playlist_genre",
        orientation="h",
        points=False,
        box=True,
    )
    fig.update_layout(
        title="Distribuição de BPM por Gênero",
        xaxis_title="BPM (Tempo)",
        yaxis_title="Gênero",
        template="plotly_white",
        showlegend=False,
    )
    return fig


def valence_energy_density(df: pd.DataFrame, genre: str | None) -> go.Figure:
    data = df if genre is None else df[df["playlist_genre"] == genre]
    if data.empty:
        return go.Figure()

    fig = px.density_heatmap(
        data,
        x="valence",
        y="energy",
        nbinsx=40,
        nbinsy=40,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        title="Mapa de Calor: Energia x Valência",
        xaxis_title="Valência (Positividade)",
        yaxis_title="Energia",
        template="plotly_white",
    )

    fig.add_shape(type="line", x0=0.5, x1=0.5, y0=0, y1=1, yref="paper", line=dict(color="white", dash="dash"))
    fig.add_shape(type="line", x0=0, x1=1, y0=0.5, y1=0.5, xref="paper", line=dict(color="white", dash="dash"))

    annotations = [
        dict(x=0.25, y=0.75, text="Intenso", showarrow=False, font=dict(color="white")),
        dict(x=0.75, y=0.75, text="Feliz", showarrow=False, font=dict(color="white")),
        dict(x=0.25, y=0.25, text="Triste", showarrow=False, font=dict(color="white")),
        dict(x=0.75, y=0.25, text="Calmo", showarrow=False, font=dict(color="white")),
    ]
    fig.update_layout(annotations=annotations)
    return fig


def feature_distribution(df: pd.DataFrame, feature: str, color: str | None = None) -> go.Figure:
    fig = px.histogram(
        df,
        x=feature,
        color=color,
        nbins=40,
        marginal="box",
        opacity=0.8,
    )
    fig.update_layout(
        title=f"Distribuição de {feature}",
        template="plotly_white",
        xaxis_title=feature,
        yaxis_title="Contagem",
    )
    return fig


def scatter_matrix(df: pd.DataFrame, dimensions: List[str], color: str) -> go.Figure:
    fig = px.scatter_matrix(df, dimensions=dimensions, color=color, opacity=0.7)
    fig.update_layout(template="plotly_white", title="Matriz de Dispersão das Features")
    return fig


def cluster_scatter(df: pd.DataFrame, reduced: np.ndarray, clusters: np.ndarray) -> go.Figure:
    fig = px.scatter(
        x=reduced[:, 0],
        y=reduced[:, 1],
        color=clusters.astype(str),
        labels={"x": "Componente 1", "y": "Componente 2", "color": "Cluster"},
        opacity=0.75,
    )
    fig.update_layout(
        title="Clusters de Músicas (PCA 2D)",
        template="plotly_white",
    )
    return fig
