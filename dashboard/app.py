from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from data_utils import (
    FilterState,
    compute_kpis,
    danceability_by_genre,
    descriptive_stats,
    feature_ranges,
    filter_dataframe,
    genre_distribution,
    key_distribution,
    load_dataset,
    missing_values,
    subgenre_distribution,
    top_artists_by_popularity,
)
from model_utils import (
    CLASSIFIER_FEATURES,
    CLUSTER_FEATURES,
    apply_clustering,
    cluster_profile_table,
    train_classifier_from_df,
)
from visualizations import (
    bar_chart,
    cluster_scatter,
    feature_distribution,
    scatter_matrix,
    tempo_ridge_like,
    valence_energy_density,
)

st.set_page_config(
    page_title="Spotify Insights Dashboard",
    page_icon="🎧",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    return load_dataset()


def _get_filter_bounds(df: pd.DataFrame) -> Tuple[int, int, int, int]:
    min_pop = int(df["track_popularity"].min())
    max_pop = int(df["track_popularity"].max())
    years = df["release_year"].dropna()
    if years.empty:
        min_year = int(df["track_album_release_date"].dt.year.min())
        max_year = int(df["track_album_release_date"].dt.year.max())
    else:
        min_year = int(years.min())
        max_year = int(years.max())
    return min_pop, max_pop, min_year, max_year


def sidebar_filters(df: pd.DataFrame) -> FilterState:
    st.sidebar.header("Filtros")
    genres = sorted(df["playlist_genre"].dropna().unique())
    subgenres = sorted(df["playlist_subgenre"].dropna().unique())

    selected_genres = st.sidebar.multiselect("Gêneros", genres, default=[])

    filtered_subgenres_source = (
        df[df["playlist_genre"].isin(selected_genres)]["playlist_subgenre"].dropna().unique()
        if selected_genres
        else subgenres
    )
    selected_subgenres = st.sidebar.multiselect(
        "Subgêneros", sorted(filtered_subgenres_source), default=[]
    )

    min_pop, max_pop, min_year, max_year = _get_filter_bounds(df)
    popularity = st.sidebar.slider(
        "Popularidade (0-100)", min_value=min_pop, max_value=max_pop, value=(min_pop, max_pop)
    )

    year_range = st.sidebar.slider(
        "Ano de lançamento",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
    )

    st.sidebar.caption("Os filtros acima impactam todas as seções do dashboard.")

    return FilterState(
        genres=selected_genres,
        subgenres=selected_subgenres,
        min_popularity=popularity[0],
        max_popularity=popularity[1],
        year_range=year_range,
    )


def download_button(df: pd.DataFrame, label: str, filename: str) -> None:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    st.download_button(label=label, data=buffer.getvalue(), file_name=filename, mime="text/csv")


@st.cache_resource(show_spinner=True)
def get_classifier_model():
    base_df = get_dataset()
    return train_classifier_from_df(base_df)


def render_overview_tab(df: pd.DataFrame, filtered: pd.DataFrame) -> None:
    st.subheader("Visão Geral")

    kpis = compute_kpis(filtered)
    cols = st.columns(len(kpis))
    for col, (label, value) in zip(cols, kpis.items()):
        col.metric(label.capitalize(), f"{value:,}")

    st.markdown("---")
    with st.expander("Dados ausentes por coluna"):
        miss = missing_values(filtered)
        if miss.empty:
            st.success("Nenhum dado ausente nas colunas selecionadas.")
        else:
            st.dataframe(miss.to_frame("Valores faltantes"))

    with st.expander("Estatísticas descritivas das principais métricas"):
        num_cols = [
            "danceability",
            "energy",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
            "track_popularity",
        ]
        stats = descriptive_stats(filtered, num_cols)
        st.dataframe(stats)

    st.markdown("---")
    st.write(
        "O dataset completo possui"
        f" {len(df):,} músicas. Os filtros aplicados reduzem a análise para"
        f" {len(filtered):,} faixas."
    )


def render_popularity_tab(filtered: pd.DataFrame) -> None:
    st.subheader("Popularidade & Comportamento")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            bar_chart(
                top_artists_by_popularity(filtered, top_n=10),
                title="Top 10 Artistas por Popularidade Média",
                x_title="Popularidade média",
                y_title="Artista",
            ),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            bar_chart(
                genre_distribution(filtered),
                title="Distribuição de Músicas por Gênero",
                x_title="Número de faixas",
                y_title="Gênero",
            ),
            use_container_width=True,
        )

    left2, right2 = st.columns(2)
    with left2:
        st.plotly_chart(
            bar_chart(
                key_distribution(filtered),
                title="Distribuição de Músicas por Tom",
                x_title="Número de faixas",
                y_title="Tom",
            ),
            use_container_width=True,
        )
    with right2:
        st.plotly_chart(
            bar_chart(
                danceability_by_genre(filtered),
                title="Danceability Médio por Gênero",
                x_title="Danceability médio",
                y_title="Gênero",
            ),
            use_container_width=True,
        )


def render_mood_tab(filtered: pd.DataFrame) -> None:
    st.subheader("Humor & Tempo")
    genres_available = sorted(filtered["playlist_genre"].dropna().unique())
    selected = st.multiselect(
        "Selecione gêneros para comparar BPM",
        options=genres_available,
        default=genres_available[:5],
    )
    st.plotly_chart(tempo_ridge_like(filtered, selected), use_container_width=True)

    st.markdown("---")
    mood_genre = st.selectbox(
        "Escolha um gênero para o mapa emocional (Energia x Valência)",
        options=["Todos"] + genres_available,
    )
    chosen_genre = None if mood_genre == "Todos" else mood_genre
    fig = valence_energy_density(filtered, chosen_genre)
    if fig.data:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Seleção sem dados suficientes para plotagem.")


def render_explorer_tab(filtered: pd.DataFrame) -> None:
    st.subheader("Explorador de Faixas")
    st.dataframe(filtered.head(100), use_container_width=True)
    download_button(filtered, "Baixar dados filtrados", "spotify_filtrado.csv")

    st.markdown("---")
    feature = st.selectbox(
        "Selecione uma métrica para visualizar a distribuição",
        options=[
            "danceability",
            "energy",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
            "track_popularity",
        ],
    )
    color_col = st.selectbox(
        "Colorir por",
        options=[None, "playlist_genre", "playlist_subgenre"],
        index=0,
    )
    st.plotly_chart(feature_distribution(filtered, feature, color=color_col), use_container_width=True)

    with st.expander("Matriz de Dispersão das Features Principais"):
        dims = ["danceability", "energy", "valence", "tempo", "track_popularity"]
        sample = filtered.sample(min(len(filtered), 1000), random_state=42) if len(filtered) > 1000 else filtered
        st.plotly_chart(scatter_matrix(sample, dims, color="playlist_genre"), use_container_width=True)


def render_model_tab(df: pd.DataFrame) -> None:
    st.subheader("Classificador de Gênero Musical")
    result = get_classifier_model()

    metric_col, report_col = st.columns([1, 2])
    metric_col.metric("Acurácia (holdout)", f"{result.accuracy:.2%}")
    with report_col:
        st.code(result.report, language="text")

    st.markdown("### Faça sua previsão")
    ranges = feature_ranges(df, CLASSIFIER_FEATURES)
    key_labels = (
        df[["key", "key_name"]]
        .dropna()
        .drop_duplicates(subset="key")
        .set_index("key")
        .squeeze()
        .to_dict()
    )
    mode_labels = {0: "Menor", 1: "Maior"}

    with st.form("predict_form"):
        inputs: Dict[str, float] = {}

        col1, col2, col3 = st.columns(3)
        with col1:
            inputs["danceability"] = st.slider("Danceability", *ranges["danceability"], step=0.01)
            inputs["energy"] = st.slider("Energy", *ranges["energy"], step=0.01)
            inputs["speechiness"] = st.slider("Speechiness", *ranges["speechiness"], step=0.01)
            inputs["acousticness"] = st.slider("Acousticness", *ranges["acousticness"], step=0.01)
            inputs["instrumentalness"] = st.slider("Instrumentalness", *ranges["instrumentalness"], step=0.01)
        with col2:
            inputs["liveness"] = st.slider("Liveness", *ranges["liveness"], step=0.01)
            inputs["valence"] = st.slider("Valence", *ranges["valence"], step=0.01)
            tempo_min, tempo_max = ranges["tempo"]
            inputs["tempo"] = st.slider("Tempo (BPM)", tempo_min, tempo_max, step=1.0)
            loud_min, loud_max = ranges["loudness"]
            inputs["loudness"] = st.slider("Loudness", loud_min, loud_max, step=0.5)
            inputs["track_popularity"] = st.slider(
                "Popularidade da Faixa", *ranges["track_popularity"], step=1.0
            )
        with col3:
            duration_min, duration_max = ranges["duration_ms"]
            inputs["duration_ms"] = st.slider(
                "Duração (ms)", duration_min, duration_max, step=1000.0
            )
            inputs["key"] = st.selectbox(
                "Tom (Key)",
                options=list(range(12)),
                format_func=lambda x: key_labels.get(int(x), str(x)),
            )
            inputs["mode"] = st.selectbox(
                "Modo", options=[0, 1], format_func=lambda x: mode_labels.get(int(x), str(x))
            )

        submitted = st.form_submit_button("Prever gênero")

    if submitted:
        input_df = pd.DataFrame([inputs])
        prediction = result.pipeline.predict(input_df)[0]
        proba = result.pipeline.predict_proba(input_df)[0]
        proba_df = pd.DataFrame(
            {"Gênero": result.pipeline.classes_, "Probabilidade": np.round(proba, 4)}
        ).sort_values("Probabilidade", ascending=False)

        st.success(f"Gênero previsto: **{prediction}**")
        st.dataframe(proba_df, use_container_width=True)


def render_clusters_tab(filtered: pd.DataFrame) -> None:
    st.subheader("Análise de Clusters")

    if len(filtered) < 50:
        st.info("É necessário ao menos 50 músicas para gerar clusters confiáveis. Ajuste os filtros.")
        return

    n_clusters = st.slider("Número de clusters", min_value=2, max_value=10, value=4)
    result = apply_clustering(filtered, n_clusters=n_clusters)

    st.plotly_chart(cluster_scatter(filtered, result.pca_projection, result.clusters), use_container_width=True)

    profile = cluster_profile_table(result)
    st.dataframe(profile, use_container_width=True)

    with st.expander("Amostra de faixas por cluster"):
        temp = filtered.copy()
        temp["cluster"] = result.clusters
        st.dataframe(temp[[
            "track_name",
            "track_artist",
            "playlist_genre",
            "playlist_subgenre",
            "cluster",
        ]].head(200), use_container_width=True)


def main() -> None:
    df = get_dataset()
    filters = sidebar_filters(df)
    filtered_df = filter_dataframe(df, filters)

    st.title("Spotify Songs – Dashboard Analítico")
    st.caption("Explore tendências musicais, comportamento de usuários e modelos de machine learning.")

    tabs = st.tabs([
        "Visão Geral",
        "Popularidade",
        "Humor & Tempo",
        "Explorador",
        "Classificação",
        "Clusters",
    ])

    with tabs[0]:
        render_overview_tab(df, filtered_df)
    with tabs[1]:
        render_popularity_tab(filtered_df)
    with tabs[2]:
        render_mood_tab(filtered_df)
    with tabs[3]:
        render_explorer_tab(filtered_df)
    with tabs[4]:
        render_model_tab(df)
    with tabs[5]:
        render_clusters_tab(filtered_df)

    st.sidebar.markdown("---")
    st.sidebar.write("Dashboard desenvolvido para o projeto PISI3.")
    repo_path = Path(__file__).resolve().parent.parent
    st.sidebar.write(f"Base de dados: `{repo_path / 'DataSet' / 'spotify_songs.csv'}`")


if __name__ == "__main__":
    main()
