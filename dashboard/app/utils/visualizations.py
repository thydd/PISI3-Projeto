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

# Tooltip configuration with high contrast using app's primary accent color
HOVER_LABEL_CONFIG = dict(
    bgcolor="#033085",  # Vibrant cyan - app's primary accent color
    font=dict(
        size=14,
        family="Poppins, sans-serif",
        color="#000000"  # Pure black text for maximum contrast (8.59:1 ratio)
    ),
    bordercolor="#ffffff",  # White border for clear definition
    align="left"
)


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
        hoverlabel=HOVER_LABEL_CONFIG,
    )
    fig.update_traces(
        textposition="outside",
        textangle=0,
        textfont=dict(color=FONT_COLOR, size=11),
        insidetextanchor="middle",
        hovertemplate="<b style='color:#000000'>%{y}</b><br><span style='color:#000000'>%{x:.2f}</span><extra></extra>",
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
        hoverlabel=HOVER_LABEL_CONFIG,
    )
    return fig


def valence_energy_density(df: pd.DataFrame, genre: str | None) -> go.Figure:
    """Mapa emocional com suavização para eliminar pixelamento."""
    data = df if genre is None else df[df["playlist_genre"] == genre]
    if data.empty:
        return go.Figure()

    # Criar histograma 2D com alta resolução
    import numpy as np
    from scipy.ndimage import gaussian_filter
    
    # Criar grid de alta resolução
    bins = 100  # Muito maior que antes para suavização
    
    # Calcular histograma 2D
    hist, xedges, yedges = np.histogram2d(
        data['valence'], 
        data['energy'], 
        bins=bins, 
        range=[[0, 1], [0, 1]]
    )
    
    # Aplicar filtro gaussiano para suavização (elimina pixelamento!)
    hist_smooth = gaussian_filter(hist, sigma=2.5)  # sigma controla a suavização
    
    # Criar heatmap suavizado
    fig = go.Figure(data=go.Heatmap(
        z=hist_smooth.T,  # Transpor para orientação correta
        x=xedges[:-1],
        y=yedges[:-1],
        colorscale=[
            [0, "#020817"],      # Azul muito escuro (quase preto)
            [0.2, "#0c1e3d"],    # Azul escuro profundo
            [0.4, "#1e3a5f"],    # Azul médio escuro
            [0.6, "#2563eb"],    # Azul royal vibrante
            [0.75, "#3b82f6"],   # Azul médio brilhante
            [0.85, "#60a5fa"],   # Azul claro
            [0.95, "#93c5fd"],   # Azul muito claro
            [1, "#dbeafe"]       # Azul quase branco (alta densidade)
        ],
        hovertemplate="<b style='color:#000000'>Valência:</b> <span style='color:#000000'>%{x:.2f}</span><br>" +
                     "<b style='color:#000000'>Energia:</b> <span style='color:#000000'>%{y:.2f}</span><br>" +
                     "<b style='color:#000000'>Densidade:</b> <span style='color:#000000'>%{z:.0f}</span><extra></extra>",
        colorbar=dict(
            title=dict(
                text="Densidade",
                font=dict(size=12, color=FONT_COLOR)
            ),
            thickness=15,
            len=0.7,
            tickfont=dict(color=FONT_COLOR, size=10),
            x=1.02,
        ),
    ))
    
    fig.update_layout(
        title=dict(
            text="Mapa Emocional: Energia × Valência",
            font=dict(size=20, color=FONT_COLOR, family="Poppins"),
            x=0.5,
            xanchor="center"
        ),
        xaxis_title=dict(
            text="Valência (Positividade)",
            font=dict(size=14, color=FONT_COLOR)
        ),
        yaxis_title=dict(
            text="Energia (Intensidade)",
            font=dict(size=14, color=FONT_COLOR)
        ),
        template="plotly_dark",
        paper_bgcolor=DARK_PAPER_BG,
        plot_bgcolor=DARK_PLOT_BG,
        font=dict(color=FONT_COLOR, family="Poppins"),
        hoverlabel=HOVER_LABEL_CONFIG,
        
        # Dimensões expandidas
        height=750,  # Aumentado de 650 para 750 (+15%)
        width=None,
        
        # Eixos com proporção 1:1 (quadrado perfeito)
        xaxis=dict(
            range=[0, 1],
            gridcolor=GRID_COLOR,
            showgrid=True,
            zeroline=False,
            constrain="domain",
        ),
        yaxis=dict(
            range=[0, 1],
            gridcolor=GRID_COLOR,
            showgrid=True,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
        ),
        
        margin=dict(l=90, r=130, t=110, b=110),  # Margens ligeiramente aumentadas
    )

    # Linhas divisórias dos quadrantes
    fig.add_shape(
        type="line",
        x0=0.5, x1=0.5, y0=0, y1=1,
        line=dict(color="rgba(255, 255, 255, 0.4)", width=2, dash="dash")
    )
    fig.add_shape(
        type="line",
        x0=0, x1=1, y0=0.5, y1=0.5,
        line=dict(color="rgba(255, 255, 255, 0.4)", width=2, dash="dash")
    )

    # Labels dos quadrantes - FORA do gráfico, sem emojis, sem backgrounds
    annotations = [
        # Quadrante superior esquerdo - INTENSO
        dict(
            x=0.25, y=1.06,
            xref="x", yref="paper",
            text="<b>INTENSO</b>",
            showarrow=False,
            font=dict(color="#ef4444", size=15, family="Poppins", weight=600),
        ),
        # Quadrante superior direito - FELIZ
        dict(
            x=0.75, y=1.06,
            xref="x", yref="paper",
            text="<b>FELIZ</b>",
            showarrow=False,
            font=dict(color="#fbbf24", size=15, family="Poppins", weight=600),
        ),
        # Quadrante inferior esquerdo - TRISTE
        dict(
            x=0.25, y=-0.10,
            xref="x", yref="paper",
            text="<b>TRISTE</b>",
            showarrow=False,
            font=dict(color="#6366f1", size=15, family="Poppins", weight=600),
            
        ),
        # Quadrante inferior direito - CALMO
        dict(
            x=0.75, y=-0.10,
            xref="x", yref="paper",
            text="<b>CALMO</b>",
            showarrow=False,
            font=dict(color="#10b981", size=15, family="Poppins", weight=600),
        ),
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
        hoverlabel=HOVER_LABEL_CONFIG,
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
        hoverlabel=HOVER_LABEL_CONFIG,
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
        hoverlabel=HOVER_LABEL_CONFIG,
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
        hoverlabel=HOVER_LABEL_CONFIG,
    )
    
    fig.update_traces(
        marker=dict(line=dict(width=0.5, color="rgba(255,255,255,0.1)")),
        hovertemplate="<b style='color:#000000'>%{text}</b><br>" +
                     "<span style='color:#000000'>%{x:.2f}, %{y:.2f}, %{z:.2f}</span><extra></extra>"
    )
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
        hovertemplate="<b style='color:#000000'>%{x}</b> × <b style='color:#000000'>%{y}</b><br>" +
                     "<span style='color:#000000'>Correlação: %{z:.3f}</span><extra></extra>",
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
        hoverlabel=HOVER_LABEL_CONFIG,
    )
    
    return fig


def genre_sunburst(df: pd.DataFrame) -> go.Figure:
    """Sunburst chart showing genre/subgenre hierarchy."""
    hierarchy = (
        df.groupby(["playlist_genre", "playlist_subgenre"])
        .size()
        .reset_index(name="count")
    )
    
    # Escala de cores azul-roseada com alto contraste
    # Todos os tons são vibrantes - nenhum cinza/preto
    high_contrast_scale = [
        [0.0, "#3b82f6"],  # Azul vibrante (baixa contagem)
        [0.2, "#6366f1"],  # Índigo
        [0.4, "#8b5cf6"],  # Violeta
        [0.6, "#a855f7"],  # Roxo
        [0.8, "#d946ef"],  # Fúcsia
        [1.0, "#ec4899"],  # Rosa vibrante (alta contagem)
    ]
    
    fig = px.sunburst(
        hierarchy,
        path=["playlist_genre", "playlist_subgenre"],
        values="count",
        color="count",
        color_continuous_scale=high_contrast_scale,
        hover_data={"count": ":,"},
    )
    
    fig.update_layout(
        title="Hierarquia de Gêneros e Subgêneros",
        template="plotly_dark",
        paper_bgcolor=DARK_PAPER_BG,
        plot_bgcolor=DARK_PLOT_BG,
        font=dict(color=FONT_COLOR, family="Poppins"),
        margin=dict(l=20, r=20, t=60, b=20),
        hoverlabel=HOVER_LABEL_CONFIG,
    )
    
    fig.update_traces(
        hovertemplate="<b style='color:#000000'>%{label}</b><br>" +
                     "<span style='color:#000000'>Faixas: %{value:,}</span><extra></extra>",
        textfont=dict(size=13, color="#ffffff", family="Poppins"),  # Texto branco com tamanho aumentado
        insidetextorientation="radial",  # Orientação radial para melhor legibilidade
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
        hoverlabel=HOVER_LABEL_CONFIG,
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
        hovertemplate="<b style='color:#000000'>%{theta}</b><br>" +
                     "<span style='color:#000000'>Valor: %{r:.3f}</span><extra></extra>",
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
        hoverlabel=HOVER_LABEL_CONFIG,
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
        hoverlabel=HOVER_LABEL_CONFIG,
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
        hoverlabel=HOVER_LABEL_CONFIG,
    )
    
    fig.update_traces(
        marker=dict(line=dict(width=1, color="rgba(255,255,255,0.2)")),
        hovertemplate="<b style='color:#000000'>%{hovertext}</b><br>" +
                     "<span style='color:#000000'>Faixas: %{x:,}<br>Pop. média: %{y:.1f}<br>%{customdata[2]}</span><extra></extra>",
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
        hovertemplate="<b style='color:#000000'>Tempo</b>: %{x:.1f} BPM<br>" +
                     "<b style='color:#000000'>Duração</b>: %{y:.2f} min<extra></extra>",
        showlegend=False,
    )
    fig.add_trace(scatter, row=2, col=1)
    
    # Top histogram (tempo)
    fig.add_trace(
        go.Histogram(
            x=data["tempo"],
            marker=dict(color=ACCENT_PRIMARY, opacity=0.7),
            showlegend=False,
            hovertemplate="<span style='color:#000000'>BPM: %{x}<br>Count: %{y}</span><extra></extra>",
        ),
        row=1, col=1
    )
    
    # Right histogram (duration)
    fig.add_trace(
        go.Histogram(
            y=data["duration_min"],
            marker=dict(color=ACCENT_SECONDARY, opacity=0.7),
            showlegend=False,
            hovertemplate="<span style='color:#000000'>Duração: %{y:.2f} min<br>Count: %{x}</span><extra></extra>",
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
        hoverlabel=HOVER_LABEL_CONFIG,
    )
    
    return fig
