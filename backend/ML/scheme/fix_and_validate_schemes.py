"""
Fix and validate scheme data, then re-cluster using all available metrics.

Fixes:
1. Cluster centers: inverse transform from standardized to original scale
2. Formation rates: ensure they sum correctly (play_action + dropback + run = 100% per formation)
3. Derive play_action_rate from formation rates where Sharp data is missing
4. Use ALL available metrics for clustering (not just the basic 6)
5. Check for read option data (if available in PBP)

Usage (from repo root):
    python -m backend.ML.scheme.fix_and_validate_schemes
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    from .config import (
        SCHEME_DATA_DIR,
        SCHEME_FEATURE_COLUMNS,
        FORMATION_FEATURE_COLUMNS,
        DOWN_FEATURE_COLUMNS,
    )
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
    FORMATION_FEATURE_COLUMNS = [
        "under_center_play_action_rate",
        "under_center_dropback_rate",
        "under_center_run_rate",
        "shotgun_play_action_rate",
        "shotgun_dropback_rate",
        "shotgun_run_rate",
    ]
    DOWN_FEATURE_COLUMNS = [
        "down_1_pass_rate",
        "down_1_run_rate",
        "down_2_pass_rate",
        "down_2_run_rate",
        "down_3_pass_rate",
        "down_3_run_rate",
    ]


def _get_all_clustering_features() -> List[str]:
    """Return comprehensive list of all possible clustering features."""
    personnel_cols = [
        "personnel_01_rate",
        "personnel_10_rate",
        "personnel_11_rate",
        "personnel_12_rate",
        "personnel_13_rate",
        "personnel_20_rate",
        "personnel_21_rate",
        "personnel_22_rate",
        "personnel_31_rate",
    ]
    return (
        SCHEME_FEATURE_COLUMNS
        + FORMATION_FEATURE_COLUMNS
        + DOWN_FEATURE_COLUMNS
        + personnel_cols
    )


def _validate_formation_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and fix formation-specific rates.
    
    For under center: play_action + dropback + run should = 100% of under center plays.
    For shotgun: play_action + dropback + run should = 100% of shotgun plays.
    
    If they don't sum correctly, normalize them.
    """
    df = df.copy()
    
    # Under center rates
    uc_cols = [
        "under_center_play_action_rate",
        "under_center_pass_rate",
        "under_center_run_rate",
    ]
    if all(c in df.columns for c in uc_cols):
        uc_sum = df[uc_cols].sum(axis=1)
        # Normalize so they sum to 100% (if sum > 0)
        for col in uc_cols:
            df[col] = df[col] / uc_sum.replace(0, float("nan")) * 100
            df[col] = df[col].fillna(0)
    
    # Shotgun rates
    sg_cols = [
        "shotgun_play_action_rate",
        "shotgun_pass_rate",
        "shotgun_run_rate",
    ]
    if all(c in df.columns for c in sg_cols):
        sg_sum = df[sg_cols].sum(axis=1)
        # Normalize so they sum to 100% (if sum > 0)
        for col in sg_cols:
            df[col] = df[col] / sg_sum.replace(0, float("nan")) * 100
            df[col] = df[col].fillna(0)
    
    return df


def _derive_play_action_from_formation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive overall play_action_rate from formation-specific rates if missing.
    
    play_action_rate = (uc_pa_rate * uc_rate + sg_pa_rate * sg_rate) / 100
    """
    df = df.copy()
    
    if "play_action_rate" in df.columns and df["play_action_rate"].notna().all():
        # Already have it, skip
        return df
    
    uc_pa_col = "under_center_play_action_rate"
    sg_pa_col = "shotgun_play_action_rate"
    uc_rate_col = "under_center_rate"
    sg_rate_col = "shotgun_rate"
    
    if all(c in df.columns for c in [uc_pa_col, sg_pa_col, uc_rate_col, sg_rate_col]):
        # Weighted average: (uc_pa * uc_rate + sg_pa * sg_rate) / 100
        df["play_action_rate"] = (
            df[uc_pa_col] * df[uc_rate_col] / 100
            + df[sg_pa_col] * df[sg_rate_col] / 100
        )
        print(f"[fix_and_validate] Derived play_action_rate from formation rates")
    
    return df


def _get_available_features_for_year(df: pd.DataFrame, year: int) -> List[str]:
    """
    Get ALL available numeric features for clustering, excluding only metadata columns.
    
    Includes features even if some teams have zero values - we want to use all available
    data for clustering. Only excludes columns that are completely empty (all NaN).
    """
    exclude_cols = {
        "team_abbr",
        "season",
        "n_plays",
        "team_nickname",
        "scheme_cluster",
        "index",
        "index_x",
        "index_y",
    }
    
    # Get all numeric columns that aren't excluded
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    available = [c for c in numeric_cols if c not in exclude_cols]
    
    # Include ALL features that have at least some valid (non-NaN) data
    # Don't filter out features just because some teams have zero values
    valid_features = []
    for col in available:
        # Only exclude if ALL values are NaN (completely missing data)
        if df[col].notna().sum() > 0:
            valid_features.append(col)
    
    return valid_features


def fix_and_cluster_year(year: int, n_clusters: int = 4) -> Path:
    """
    Fix data issues, validate, and re-cluster a single year.
    """
    base = Path(SCHEME_DATA_DIR)
    csv_path = base / f"{year}_schemes.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    
    df = pd.read_csv(csv_path)
    
    # 1) Validate and fix formation rates
    df = _validate_formation_rates(df)
    
    # 2) Derive play_action_rate if missing
    df = _derive_play_action_from_formation(df)
    
    # 3) Get all available features for this year
    feature_cols = _get_available_features_for_year(df, year)
    
    if not feature_cols:
        raise ValueError(f"No valid clustering features found for {year}")
    
    print(f"[fix_and_validate] Year {year}: using {len(feature_cols)} features: {feature_cols[:10]}...")
    
    # 4) Prepare feature matrix
    X = df[feature_cols].astype(float).fillna(0)
    
    # Fill remaining NaN with median
    medians = X.median()
    X = X.fillna(medians.fillna(0))
    
    # 5) Standardize and cluster
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    
    df["scheme_cluster"] = labels
    
    # 6) Save updated CSV
    df.to_csv(csv_path, index=False)
    
    # 7) Save cluster centers RELATIVE TO MEAN (centered)
    # Convert centers back to original scale, then subtract mean to show deviation from average
    centers_orig = scaler.inverse_transform(km.cluster_centers_)
    feature_means = X.mean().values
    centers_relative = centers_orig - feature_means
    
    centers_path = base / f"{year}_schemes_cluster_centers.csv"
    centers_df = pd.DataFrame(
        centers_relative,
        columns=feature_cols,
        index=range(n_clusters)
    )
    # Add a note in the first row explaining these are relative to mean
    centers_df.to_csv(centers_path, index=False)
    
    # Also save a note file explaining the format
    note_path = base / f"{year}_schemes_cluster_centers_README.txt"
    with open(note_path, "w") as f:
        f.write(
            f"Cluster Centers for {year} - RELATIVE TO MEAN\n"
            f"==============================================\n\n"
            f"Values show how each cluster differs from the league average.\n"
            f"- Positive values = above average\n"
            f"- Negative values = below average\n"
            f"- Zero = exactly at league average\n\n"
            f"To get absolute values, add the mean for each feature:\n"
        )
        for i, col in enumerate(feature_cols):
            f.write(f"  {col}: mean = {feature_means[i]:.2f}\n")
    
    print(
        f"[fix_and_validate] Year {year}: clustered {len(df)} rows into {n_clusters} clusters. "
        f"Centers saved RELATIVE TO MEAN to {centers_path} (note: {note_path})"
    )
    
    return csv_path


def fix_and_cluster_all_years(n_clusters: int = 4) -> None:
    """Fix and re-cluster all years."""
    base = Path(SCHEME_DATA_DIR)
    years = sorted([
        int(p.stem.split("_")[0])
        for p in base.glob("*_schemes.csv")
        if not p.stem.startswith("sharp_")
    ])
    
    for yr in years:
        fix_and_cluster_year(yr, n_clusters=n_clusters)


if __name__ == "__main__":
    fix_and_cluster_all_years()
