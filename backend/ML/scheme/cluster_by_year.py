"""
Run K-means clustering separately for each season's schemes file.

For each {year}_schemes.csv in SCHEME_DATA_DIR:
- Fit KMeans on the standard scheme feature columns
- Overwrite/insert `scheme_cluster` in that year's CSV
- Save cluster centers to {year}_schemes_cluster_centers.csv

By clustering per year, you avoid mixing distributions across eras,
especially as new metrics (like Sharp + personnel) are added.

Usage (from repo root):

    python -m backend.ML.scheme.cluster_by_year
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

try:
    from .config import SCHEME_DATA_DIR, SCHEME_FEATURE_COLUMNS
    from .scheme_clustering import fit_kmeans
except ImportError:  # pragma: no cover
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
    from scheme_clustering import fit_kmeans  # type: ignore


def _list_year_scheme_files() -> Dict[int, Path]:
    """Return mapping of year -> {year}_schemes.csv path."""
    base = Path(SCHEME_DATA_DIR)
    out: Dict[int, Path] = {}
    for p in base.glob("*_schemes.csv"):
        stem = p.stem
        if stem.startswith("sharp_"):
            continue
        try:
            year = int(stem.split("_")[0])
        except ValueError:
            continue
        out[year] = p
    return out


def cluster_year_schemes(year: int, n_clusters: int = 4) -> Path:
    """
    Cluster a single year's schemes file in-place.

    Returns path to the updated {year}_schemes.csv.
    """
    base = Path(SCHEME_DATA_DIR)
    csv_path = base / f"{year}_schemes.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    # Only use features that actually exist
    feature_cols = [c for c in SCHEME_FEATURE_COLUMNS if c in df.columns]
    if not feature_cols:
        raise ValueError(
            f"No clustering features found for {year}. "
            f"Tried: {SCHEME_FEATURE_COLUMNS}. Columns: {list(df.columns)}"
        )

    df_clustered, km, _ = fit_kmeans(
        df,
        n_clusters=n_clusters,
        feature_columns=feature_cols,
        random_state=42,
        standardize=True,
    )

    # Overwrite the yearly CSV with the new cluster labels
    df_clustered.to_csv(csv_path, index=False)

    # Save cluster centers in original scale for interpretation
    centers_path = base / f"{year}_schemes_cluster_centers.csv"
    pd.DataFrame(km.cluster_centers_, columns=feature_cols).to_csv(centers_path, index=False)

    print(
        f"[cluster_by_year] Year {year}: clustered {len(df_clustered)} rows "
        f"into {n_clusters} clusters using {feature_cols}. "
        f"Wrote labels to {csv_path} and centers to {centers_path}."
    )
    return csv_path


def cluster_all_years(n_clusters: int = 4) -> None:
    years = sorted(_list_year_scheme_files().keys())
    for yr in years:
        cluster_year_schemes(yr, n_clusters=n_clusters)


if __name__ == "__main__":
    cluster_all_years()

