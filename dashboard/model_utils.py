from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


CLASSIFIER_FEATURES: List[str] = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
    "track_popularity",
]


def prepare_classifier_pipeline() -> Pipeline:
    numeric_features = CLASSIFIER_FEATURES
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)]
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    clf = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    return clf


@dataclass
class ClassifierResult:
    pipeline: Pipeline
    accuracy: float
    report: str


def train_classifier_from_df(df: pd.DataFrame, target_column: str = "playlist_genre") -> ClassifierResult:
    clf = prepare_classifier_pipeline()

    X = df[CLASSIFIER_FEATURES]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, zero_division=0)
    return ClassifierResult(pipeline=clf, accuracy=accuracy, report=report)


@dataclass
class ClusterResult:
    clusters: np.ndarray
    kmeans: KMeans
    pca_projection: np.ndarray
    feature_means: pd.DataFrame


CLUSTER_FEATURES: List[str] = [
    "valence",
    "energy",
    "danceability",
    "tempo",
    "acousticness",
]


def apply_clustering(df: pd.DataFrame, n_clusters: int = 4) -> ClusterResult:
    features = df[CLUSTER_FEATURES]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=42)
    clusters = kmeans.fit_predict(scaled)

    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(scaled)

    feature_means = (
        pd.concat([features.reset_index(drop=True), pd.Series(clusters, name="cluster")], axis=1)
        .groupby("cluster")
        .mean()
    )

    return ClusterResult(
        clusters=clusters,
        kmeans=kmeans,
        pca_projection=reduced,
        feature_means=feature_means,
    )


def cluster_profile_table(cluster_result: ClusterResult) -> pd.DataFrame:
    df = cluster_result.feature_means.copy()
    return df.reset_index().rename(columns={"index": "Cluster"})
