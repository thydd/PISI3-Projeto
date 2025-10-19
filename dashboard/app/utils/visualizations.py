from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Enhanced theme constants
DARK_PAPER_BG = "rgba(0, 0, 0, 0)"  # Transparent to work with glass-card
DARK_PLOT_BG = "rgba(0, 0, 0, 0)"   # Transparent to work with glass-card
FONT_COLOR = "#f8f9fa"
ACCENT_PRIMARY = "#4cc9f0"
ACCENT_SECONDARY = "#ff6dc4"
ACCENT_TERTIARY = "#ffd60a"
GRID_COLOR = "rgba(255, 255, 255, 0.05)"

COLOR_PALETTE = [
    "#4cc9f0", "#ff6dc4", "#ffd60a", "#00f5d4", 
    "#9d4edd", "#06ffa5", "#f72585", "#4361ee"
]


def bar_chart(series: pd.Series, title: str, x_title: str, y_title: str) -> go.Figure:
    """Enhanced horizontal bar chart with gradient fills and modern styling."""
    fig = px.bar(
        series.sort_values(),
        x=series.values,
        y=series.index,
        orientation="h",
        text=series.values.round(2),
        color=series.values,
        color_continuous_scale=[[0, ACCENT_SECONDARY], [0.5, ACCENT_PRIMARY], [1, ACCENT_TERTIARY]],
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color=FONT_COLOR, family="Poppins")),
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_dark",
        yaxis=dict(autorange="reversed", gridcolor=GRID_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR),
        paper_bgcolor=DARK_PAPER_BG,
        plot_bgcolor=DARK_PLOT_BG,
        font=dict(color=FONT_COLOR, family="Poppins"),
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_traces(
        textposition="outside",
        textangle=0,
        textfont=dict(color=FONT_COLOR, size=11),
        insidetextanchor="middle",
        hovertemplate="<b>%{y}</b><br>%{x:.2f}<extra></extra>",
        cliponaxis=False,
        marker=dict(line=dict(color="rgba(255,255,255,0.1)", width=1)),
    )
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
        template="plotly_dark",
        showlegend=False,
        paper_bgcolor=DARK_PAPER_BG,
        plot_bgcolor=DARK_PLOT_BG,
        font=dict(color=FONT_COLOR),
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
        template="plotly_dark",
        paper_bgcolor=DARK_PAPER_BG,
        plot_bgcolor=DARK_PLOT_BG,
        font=dict(color=FONT_COLOR),
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
        template="plotly_dark",
        xaxis_title=feature,
        yaxis_title="Contagem",
        paper_bgcolor=DARK_PAPER_BG,
        plot_bgcolor=DARK_PLOT_BG,
        font=dict(color=FONT_COLOR),
    )
    return fig


def scatter_matrix(df: pd.DataFrame, dimensions: List[str], color: str) -> go.Figure:
    fig = px.scatter_matrix(df, dimensions=dimensions, color=color, opacity=0.7)
    fig.update_layout(
        template="plotly_dark",
        title="Matriz de Dispersão das Features",
        paper_bgcolor=DARK_PAPER_BG,
        plot_bgcolor=DARK_PLOT_BG,
        font=dict(color=FONT_COLOR),
    )
    return fig


def cluster_scatter(df: pd.DataFrame, reduced: np.ndarray, clusters: np.ndarray) -> go.Figure:
    fig = px.scatter(
        x=reduced[:, 0],
        y=reduced[:, 1],
        color=clusters.astype(str),
        labels={"x": "Componente 1", "y": "Componente 2", "color": "Cluster"},
        opacity=0.75,
        color_discrete_sequence=COLOR_PALETTE,
    )
    fig.update_layout(
        title="Clusters de Músicas (PCA 2D)",
        template="plotly_dark",
        paper_bgcolor=DARK_PAPER_BG,
        plot_bgcolor=DARK_PLOT_BG,
        font=dict(color=FONT_COLOR, family="Poppins"),
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
    )
    return fig


# ============================================================================
# ADVANCED NEW VISUALIZATIONS
# ============================================================================

def feature_3d_scatter(
    df: pd.DataFrame, 
    x: str, 
    y: str, 
    z: str, 
    color: Optional[str] = None,
    size: Optional[str] = None,
    sample_size: int = 2000
) -> go.Figure:
    """3D scatter plot for exploring multi-dimensional feature relationships."""
    data = df.sample(min(sample_size, len(df)), random_state=42) if len(df) > sample_size else df
    
    fig = px.scatter_3d(
        data,
        x=x,
        y=y,
        z=z,
        color=color or "playlist_genre",
        size=size,
        opacity=0.7,
        color_discrete_sequence=COLOR_PALETTE,
        hover_data=["track_name", "track_artist"],
    )
    
    fig.update_layout(
        title=f"Espaço 3D: {x.title()} × {y.title()} × {z.title()}",
        template="plotly_dark",
        paper_bgcolor=DARK_PAPER_BG,
        plot_bgcolor=DARK_PLOT_BG,
        font=dict(color=FONT_COLOR, family="Poppins"),
        scene=dict(
            xaxis=dict(backgroundcolor=DARK_PLOT_BG, gridcolor=GRID_COLOR),
            yaxis=dict(backgroundcolor=DARK_PLOT_BG, gridcolor=GRID_COLOR),
            zaxis=dict(backgroundcolor=DARK_PLOT_BG, gridcolor=GRID_COLOR),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    
    fig.update_traces(marker=dict(line=dict(width=0.5, color="rgba(255,255,255,0.1)")))
    return fig


def correlation_heatmap(df: pd.DataFrame, features: List[str]) -> go.Figure:
    """Enhanced correlation heatmap with annotations."""
    corr_matrix = df[features].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale=[[0, ACCENT_SECONDARY], [0.5, DARK_PLOT_BG], [1, ACCENT_PRIMARY]],
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=10, color=FONT_COLOR),
        hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>Correlação: %{z:.3f}<extra></extra>",
        colorbar=dict(title="Correlação", tickfont=dict(color=FONT_COLOR)),
    ))
    
    fig.update_layout(
        title="Matriz de Correlação das Audio Features",
        template="plotly_dark",
        paper_bgcolor=DARK_PAPER_BG,
        plot_bgcolor=DARK_PLOT_BG,
        font=dict(color=FONT_COLOR, family="Poppins"),
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    
    return fig


def genre_sunburst(df: pd.DataFrame) -> go.Figure:
    """Sunburst chart showing genre/subgenre hierarchy."""
    hierarchy = (
        df.groupby(["playlist_genre", "playlist_subgenre"])
        .size()
        .reset_index(name="count")
    )
    
    fig = px.sunburst(
        hierarchy,
        path=["playlist_genre", "playlist_subgenre"],
        values="count",
        color="count",
        color_continuous_scale=[[0, ACCENT_SECONDARY], [0.5, ACCENT_PRIMARY], [1, ACCENT_TERTIARY]],
        hover_data={"count": ":,"},
    )
    
    fig.update_layout(
        title="Hierarquia de Gêneros e Subgêneros",
        template="plotly_dark",
        paper_bgcolor=DARK_PAPER_BG,
        plot_bgcolor=DARK_PLOT_BG,
        font=dict(color=FONT_COLOR, family="Poppins"),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>Faixas: %{value:,}<extra></extra>",
        textfont=dict(size=12),
    )
    
    return fig


def timeline_release_trend(df: pd.DataFrame) -> go.Figure:
    """Animated timeline showing release trends over decades."""
    df_copy = df.copy()
    df_copy["release_date_parsed"] = pd.to_datetime(
        df_copy["track_album_release_date"], errors="coerce"
    )
    df_copy["release_year"] = df_copy["release_date_parsed"].dt.year
    df_copy = df_copy[df_copy["release_year"].notna()]
    
    yearly_counts = (
        df_copy.groupby(["release_year", "playlist_genre"])
        .size()
        .reset_index(name="count")
    )
    
    fig = px.line(
        yearly_counts,
        x="release_year",
        y="count",
        color="playlist_genre",
        color_discrete_sequence=COLOR_PALETTE,
        markers=True,
    )
    
    fig.update_layout(
        title="Evolução de Lançamentos por Gênero ao Longo dos Anos",
        xaxis_title="Ano",
        yaxis_title="Número de Faixas",
        template="plotly_dark",
        paper_bgcolor=DARK_PAPER_BG,
        plot_bgcolor=DARK_PLOT_BG,
        font=dict(color=FONT_COLOR, family="Poppins"),
        hovermode="x unified",
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
        legend=dict(
            title="Gênero",
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor=GRID_COLOR,
            borderwidth=1,
        ),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=6, line=dict(width=1, color="rgba(255,255,255,0.3)")),
    )
    
    return fig


def radar_chart_genre_profile(df: pd.DataFrame, genre: str, features: List[str]) -> go.Figure:
    """Radar chart showing audio profile for a specific genre."""
    genre_data = df[df["playlist_genre"] == genre] if genre != "Todos" else df
    
    avg_values = genre_data[features].mean().values
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=avg_values,
        theta=[f.replace("_", " ").title() for f in features],
        fill="toself",
        name=genre.title(),
        line=dict(color=ACCENT_PRIMARY, width=2),
        fillcolor=f"rgba(76, 201, 240, 0.3)",
        hovertemplate="<b>%{theta}</b><br>Valor: %{r:.3f}<extra></extra>",
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor=GRID_COLOR,
                color=FONT_COLOR,
            ),
            angularaxis=dict(gridcolor=GRID_COLOR, color=FONT_COLOR),
            bgcolor=DARK_PLOT_BG,
        ),
        showlegend=False,
        title=f"Perfil de Audio: {genre.title()}",
        template="plotly_dark",
        paper_bgcolor=DARK_PAPER_BG,
        font=dict(color=FONT_COLOR, family="Poppins"),
        margin=dict(l=80, r=80, t=80, b=80),
    )
    
    return fig


def popularity_distribution_violin(df: pd.DataFrame) -> go.Figure:
    """Violin plot showing popularity distribution by genre with box overlay."""
    fig = px.violin(
        df,
        x="playlist_genre",
        y="track_popularity",
        color="playlist_genre",
        box=True,
        points="outliers",
        color_discrete_sequence=COLOR_PALETTE,
    )
    
    fig.update_layout(
        title="Distribuição de Popularidade por Gênero",
        xaxis_title="Gênero",
        yaxis_title="Popularidade",
        template="plotly_dark",
        paper_bgcolor=DARK_PAPER_BG,
        plot_bgcolor=DARK_PLOT_BG,
        font=dict(color=FONT_COLOR, family="Poppins"),
        showlegend=False,
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    
    return fig


def artist_network_top(df: pd.DataFrame, top_n: int = 30) -> go.Figure:
    """Network-style bubble chart of top artists by track count and avg popularity."""
    artist_stats = (
        df.groupby("track_artist")
        .agg(
            track_count=("track_id", "count"),
            avg_popularity=("track_popularity", "mean"),
            genres=("playlist_genre", lambda x: ", ".join(x.unique()[:2])),
        )
        .reset_index()
        .nlargest(top_n, "track_count")
    )
    
    fig = px.scatter(
        artist_stats,
        x="track_count",
        y="avg_popularity",
        size="track_count",
        color="avg_popularity",
        hover_name="track_artist",
        hover_data={"track_count": ":,", "avg_popularity": ":.1f", "genres": True},
        color_continuous_scale=[[0, ACCENT_SECONDARY], [0.5, ACCENT_PRIMARY], [1, ACCENT_TERTIARY]],
        size_max=60,
    )
    
    fig.update_layout(
        title=f"Top {top_n} Artistas: Produtividade × Popularidade",
        xaxis_title="Número de Faixas no Dataset",
        yaxis_title="Popularidade Média",
        template="plotly_dark",
        paper_bgcolor=DARK_PAPER_BG,
        plot_bgcolor=DARK_PLOT_BG,
        font=dict(color=FONT_COLOR, family="Poppins"),
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
        margin=dict(l=20, r=20, t=60, b=20),
        coloraxis_colorbar=dict(title="Popularidade", tickfont=dict(color=FONT_COLOR)),
    )
    
    fig.update_traces(
        marker=dict(line=dict(width=1, color="rgba(255,255,255,0.2)")),
        hovertemplate="<b>%{hovertext}</b><br>Faixas: %{x:,}<br>Pop. média: %{y:.1f}<br>%{customdata[2]}<extra></extra>",
    )
    
    return fig


def duration_tempo_jointplot(df: pd.DataFrame, sample_size: int = 3000) -> go.Figure:
    """Joint plot showing duration vs tempo with marginal histograms."""
    data = df.sample(min(sample_size, len(df)), random_state=42)
    data = data.copy()
    data["duration_min"] = data["duration_ms"] / 60000
    
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.8, 0.2],
        row_heights=[0.2, 0.8],
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, None]]
    )
    
    # Main scatter
    scatter = go.Scatter(
        x=data["tempo"],
        y=data["duration_min"],
        mode="markers",
        marker=dict(
            size=5,
            color=data["track_popularity"],
            colorscale=[[0, ACCENT_SECONDARY], [0.5, ACCENT_PRIMARY], [1, ACCENT_TERTIARY]],
            opacity=0.6,
            line=dict(width=0.5, color="rgba(255,255,255,0.1)"),
            colorbar=dict(
                title="Popularidade",
                x=1.1,
                len=0.75,
                y=0.35,
                tickfont=dict(color=FONT_COLOR),
            ),
        ),
        hovertemplate="<b>Tempo</b>: %{x:.1f} BPM<br><b>Duração</b>: %{y:.2f} min<extra></extra>",
        showlegend=False,
    )
    fig.add_trace(scatter, row=2, col=1)
    
    # Top histogram (tempo)
    fig.add_trace(
        go.Histogram(
            x=data["tempo"],
            marker=dict(color=ACCENT_PRIMARY, opacity=0.7),
            showlegend=False,
            hovertemplate="BPM: %{x}<br>Count: %{y}<extra></extra>",
        ),
        row=1, col=1
    )
    
    # Right histogram (duration)
    fig.add_trace(
        go.Histogram(
            y=data["duration_min"],
            marker=dict(color=ACCENT_SECONDARY, opacity=0.7),
            showlegend=False,
            hovertemplate="Duração: %{y:.2f} min<br>Count: %{x}<extra></extra>",
        ),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Tempo (BPM)", gridcolor=GRID_COLOR, row=2, col=1)
    fig.update_yaxes(title_text="Duração (min)", gridcolor=GRID_COLOR, row=2, col=1)
    fig.update_xaxes(showticklabels=False, gridcolor=GRID_COLOR, row=1, col=1)
    fig.update_yaxes(showticklabels=False, gridcolor=GRID_COLOR, row=2, col=2)
    
    fig.update_layout(
        title="Relação Tempo × Duração (colorido por Popularidade)",
        template="plotly_dark",
        paper_bgcolor=DARK_PAPER_BG,
        plot_bgcolor=DARK_PLOT_BG,
        font=dict(color=FONT_COLOR, family="Poppins"),
        height=600,
        margin=dict(l=20, r=120, t=60, b=20),
    )
    
    return fig
