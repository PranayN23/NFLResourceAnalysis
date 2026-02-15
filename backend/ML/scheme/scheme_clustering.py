"""
K-means clustering of offensive schemes using tendency metrics.
Use scheme features (motion rate, play action rate, shotgun rate, under center rate,
no huddle rate, air yards per att) to assign each team-season (or team) to a cluster
for downstream use (e.g. labeling coaches, predicting player stats with scheme).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    from .config import SCHEME_DATA_DIR, SCHEME_FEATURE_COLUMNS
except ImportError:
    import os
    SCHEME_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
    SCHEME_FEATURE_COLUMNS = [
        "motion_rate",
        "play_action_rate",
        "shotgun_rate",
        "under_center_rate",
        "no_huddle_rate",
        "air_yards_per_att",
    ]


def get_available_features(df: pd.DataFrame) -> list[str]:
    """Return feature columns that exist in the DataFrame."""
    return [c for c in SCHEME_FEATURE_COLUMNS if c in df.columns]


def prepare_scheme_matrix(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    fill_missing: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """
    Prepare numeric matrix for clustering.
    Returns (df with cluster_id column, feature matrix, list of features used).
    """
    feature_columns = feature_columns or get_available_features(df)
    if not feature_columns:
        raise ValueError(
            f"None of the scheme columns {SCHEME_FEATURE_COLUMNS} found in DataFrame. "
            f"Columns: {list(df.columns)}"
        )
    X = df[feature_columns].astype(float)
    if fill_missing:
        # Fill NaN with median, but if median is NaN (all values missing), fill with 0
        medians = X.median()
        X = X.fillna(medians.fillna(0))
    return df, X.values, feature_columns


def fit_kmeans(
    df: pd.DataFrame,
    n_clusters: int = 5,
    feature_columns: list[str] | None = None,
    random_state: int = 42,
    standardize: bool = True,
) -> tuple[pd.DataFrame, KMeans, StandardScaler | None]:
    """
    Fit K-means on scheme features and add cluster_id to dataframe.
    Returns (df with cluster_id, fitted KMeans, scaler if standardize=True).
    """
    _, X, feats = prepare_scheme_matrix(df, feature_columns=feature_columns)
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    df = df.copy()
    df["scheme_cluster"] = labels
    return df, km, scaler


def run_clustering(
    scheme_csv_path: str | Path,
    n_clusters: int = 5,
    output_path: str | Path | None = None,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, Path]:
    """
    Load scheme CSV, fit K-means, save labeled data and optional summary.
    Returns (labeled DataFrame, path to saved CSV).
    """
    path = Path(scheme_csv_path)
    if not path.is_absolute():
        # Resolve relative to cwd first; if not found, try SCHEME_DATA_DIR with filename only
        if not path.exists():
            path = Path(SCHEME_DATA_DIR) / path.name
    df = pd.read_csv(path)
    feats = feature_columns or get_available_features(df)
    df_labeled, km, scaler = fit_kmeans(df, n_clusters=n_clusters, feature_columns=feats)
    out_path = output_path or path.parent / f"{path.stem}_clustered.csv"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_labeled.to_csv(out_path, index=False)
    # Save cluster centers for interpretation
    centers_path = out_path.parent / f"{out_path.stem}_centers.csv"
    if scaler is not None and feats:
        # Inverse transform centers to original scale for readability
        centers_orig = scaler.inverse_transform(km.cluster_centers_)
        pd.DataFrame(centers_orig, columns=feats, index=range(n_clusters)).to_csv(centers_path)
    return df_labeled, out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="K-means cluster offensive schemes")
    parser.add_argument("input", nargs="?", default="scheme_from_pbp.csv", help="Scheme CSV path")
    parser.add_argument("-k", "--clusters", type=int, default=5, help="Number of clusters")
    parser.add_argument("-o", "--output", help="Output CSV path")
    args = parser.parse_args()
    out_df, out_path = run_clustering(args.input, n_clusters=args.clusters, output_path=args.output)
    print(f"Clustered {len(out_df)} rows into {args.clusters} clusters. Saved to {out_path}.")
