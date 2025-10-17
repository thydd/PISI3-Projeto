from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DATASET_PATH = Path(__file__).resolve().parent.parent / "DataSet" / "spotify_songs.csv"


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the Spotify dataset is missing."""


@lru_cache(maxsize=1)
def load_dataset() -> pd.DataFrame:
    """Load the Spotify songs dataset from disk.

    Returns
    -------
    pd.DataFrame
        Dataset with original columns and helper columns for analytics.
    """
    if not DATASET_PATH.exists():
        raise DatasetNotFoundError(
            f"Dataset não encontrado em '{DATASET_PATH}'. "
            "Verifique se o arquivo 'spotify_songs.csv' está disponível."
        )

    df = pd.read_csv(DATASET_PATH)

    # Pré-processamento leve para facilitar análises posteriores.
    df["track_album_release_date"] = pd.to_datetime(
        df["track_album_release_date"], errors="coerce"
    )
    df["release_year"] = df["track_album_release_date"].dt.year
    df["key_name"] = df["key"].map(
        {
            0: "C",
            1: "C#",
            2: "D",
            3: "D#",
            4: "E",
            5: "F",
            6: "F#",
            7: "G",
            8: "G#",
            9: "A",
            10: "A#",
            11: "B",
        }
    )
    return df


def compute_kpis(df: pd.DataFrame) -> Dict[str, int]:
    """Return quick KPI style metrics for the overview section."""
    return {
        "songs": len(df),
        "artists": df["track_artist"].nunique(),
        "albums": df["track_album_name"].nunique(),
        "playlists": df["playlist_name"].nunique(),
    }


def missing_values(df: pd.DataFrame) -> pd.Series:
    """Counts of missing values per column (non-zero only)."""
    missing = df.isna().sum()
    return missing[missing > 0].sort_values(ascending=False)


def top_artists_by_popularity(df: pd.DataFrame, top_n: int = 10) -> pd.Series:
    """Average popularity of the top N artists."""
    return (
        df.groupby("track_artist")["track_popularity"].mean()
        .sort_values(ascending=False)
        .head(top_n)
    )


def genre_distribution(df: pd.DataFrame) -> pd.Series:
    return df["playlist_genre"].value_counts().sort_values(ascending=False)


def subgenre_distribution(df: pd.DataFrame) -> pd.Series:
    return df["playlist_subgenre"].value_counts().sort_values(ascending=False)


def danceability_by_genre(df: pd.DataFrame) -> pd.Series:
    return (
        df.groupby("playlist_genre")["danceability"].mean()
        .sort_values(ascending=False)
    )


def key_distribution(df: pd.DataFrame) -> pd.Series:
    return df["key_name"].value_counts().sort_values(ascending=False)


def tempo_stats(df: pd.DataFrame) -> pd.Series:
    return df.groupby("playlist_genre")["tempo"].describe()["mean"].sort_values()


@dataclass
class FilterState:
    genres: List[str]
    subgenres: List[str]
    min_popularity: int
    max_popularity: int
    year_range: Tuple[int, int]


def filter_dataframe(df: pd.DataFrame, state: FilterState) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)

    if state.genres:
        mask &= df["playlist_genre"].isin(state.genres)
    if state.subgenres:
        mask &= df["playlist_subgenre"].isin(state.subgenres)
    mask &= df["track_popularity"].between(state.min_popularity, state.max_popularity)

    if not df["release_year"].isna().all():
        mask &= df["release_year"].between(state.year_range[0], state.year_range[1])

    return df.loc[mask].copy()


def feature_ranges(df: pd.DataFrame, feature_names: List[str]) -> Dict[str, Tuple[float, float]]:
    ranges: Dict[str, Tuple[float, float]] = {}
    for name in feature_names:
        col = df[name].dropna()
        if col.empty:
            ranges[name] = (0.0, 1.0)
        else:
            ranges[name] = (float(col.min()), float(col.max()))
    return ranges


def descriptive_stats(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    return df[columns].describe().T
