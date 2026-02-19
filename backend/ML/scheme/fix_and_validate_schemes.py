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


def balanced_kmeans(X: np.ndarray, n_clusters: int, min_cluster_size: int = 4, max_iter: int = 100, random_state: int = 42, feature_weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    K-means with minimum cluster size constraint (allows natural imbalance).
    
    Strategy: Use regular K-means for correctness, then post-process to ensure
    no cluster has fewer than min_cluster_size teams. This preserves natural
    scheme groupings while preventing tiny clusters.
    
    Args:
        X: Feature matrix
        n_clusters: Number of clusters
        min_cluster_size: Minimum teams per cluster (default 4)
        max_iter: Max iterations for refinement
        random_state: Random seed
        feature_weights: Optional array of weights for each feature (shape: n_features).
                        Features with higher weights have more influence in clustering.
                        If None, all features weighted equally (default 1.0).
    
    Returns:
        labels: cluster assignments
        centers: cluster centers (in weighted space)
    """
    np.random.seed(random_state)
    n_samples, n_features = X.shape
    
    # Apply feature weights if provided
    if feature_weights is not None:
        if len(feature_weights) != n_features:
            raise ValueError(f"feature_weights length ({len(feature_weights)}) must match number of features ({n_features})")
        X_weighted = X * feature_weights[np.newaxis, :]
    else:
        X_weighted = X
    
    # Step 1: Use regular K-means to find natural groupings (correctness first)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_weighted)
    centers = km.cluster_centers_
    
    # Step 2: Post-process to enforce minimum cluster size while maintaining n_clusters
    for iteration in range(max_iter):
        cluster_counts = np.bincount(labels, minlength=n_clusters)
        small_clusters = np.where(cluster_counts < min_cluster_size)[0]
        
        if len(small_clusters) == 0:
            break  # All clusters meet minimum size
        
        # Reassign points from small clusters to nearest larger cluster
        # Use weighted distances if weights were applied
        distances = np.sqrt(((X_weighted[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(axis=2))
        
        # Process small clusters, updating counts as we go
        # Sort small clusters by size (smallest first) to handle worst cases first
        small_cluster_sizes = [(c, cluster_counts[c]) for c in small_clusters]
        small_cluster_sizes.sort(key=lambda x: x[1])
        
        for small_cluster, current_size in small_cluster_sizes:
            if cluster_counts[small_cluster] >= min_cluster_size:
                continue  # Already fixed by previous reassignments
            
            # Find points in this small cluster
            points_in_small = np.where(labels == small_cluster)[0].copy()
            
            # If cluster is too small (< min_cluster_size), move ALL points to other clusters
            # This ensures we don't have tiny clusters
            for point_idx in points_in_small:
                point_distances = distances[point_idx]
                
                # Find nearest cluster that's >= min_cluster_size (or largest if all are small)
                valid_clusters = [
                    i for i in range(n_clusters)
                    if i != small_cluster and cluster_counts[i] >= min_cluster_size
                ]
                
                if valid_clusters:
                    nearest_valid = min(valid_clusters, key=lambda i: point_distances[i])
                    labels[point_idx] = nearest_valid
                    cluster_counts[small_cluster] -= 1
                    cluster_counts[nearest_valid] += 1
                else:
                    # If all clusters are small, assign to largest
                    largest_cluster = np.argmax(cluster_counts)
                    if largest_cluster != small_cluster:
                        labels[point_idx] = largest_cluster
                        cluster_counts[small_cluster] -= 1
                        cluster_counts[largest_cluster] += 1
        
        # Final pass: ensure all clusters have at least min_cluster_size teams
        cluster_counts = np.bincount(labels, minlength=n_clusters)
        still_small = np.where(cluster_counts < min_cluster_size)[0]
        
        if len(still_small) > 0:
            # Redistribute from largest clusters to small ones
            large_clusters = np.where(cluster_counts >= min_cluster_size + 1)[0]  # Clusters that can spare teams
            
            for small_cluster in still_small:
                need = min_cluster_size - cluster_counts[small_cluster]
                if need <= 0:
                    continue
                
                # Take teams from largest clusters
                for _ in range(need):
                    if len(large_clusters) == 0:
                        # If no large clusters, take from largest overall
                        largest_cluster = np.argmax(cluster_counts)
                        if largest_cluster != small_cluster and cluster_counts[largest_cluster] > 0:
                            points_in_largest = np.where(labels == largest_cluster)[0]
                            if len(points_in_largest) > 0:
                                # Move point closest to small cluster's center
                                point_distances = distances[points_in_largest, small_cluster]
                                closest_idx = points_in_largest[np.argmin(point_distances)]
                                labels[closest_idx] = small_cluster
                                cluster_counts[largest_cluster] -= 1
                                cluster_counts[small_cluster] += 1
                    else:
                        # Take from a large cluster
                        source_cluster = large_clusters[0]
                        points_in_source = np.where(labels == source_cluster)[0]
                        if len(points_in_source) > 0:
                            # Move point closest to small cluster's center
                            point_distances = distances[points_in_source, small_cluster]
                            closest_idx = points_in_source[np.argmin(point_distances)]
                            labels[closest_idx] = small_cluster
                            cluster_counts[source_cluster] -= 1
                            cluster_counts[small_cluster] += 1
                            
                            # Update large_clusters if source became too small
                            if cluster_counts[source_cluster] < min_cluster_size + 1:
                                large_clusters = large_clusters[large_clusters != source_cluster]
        
        # Update centers based on new assignments (in weighted space)
        new_centers = np.zeros_like(centers)
        for i in range(n_clusters):
            mask = labels == i
            if mask.sum() > 0:
                new_centers[i] = X_weighted[mask].mean(axis=0)
            else:
                new_centers[i] = centers[i]  # Keep old center if cluster is empty
        
        centers = new_centers
    
    return labels, centers


def get_feature_weights_for_year(year: int) -> Dict[str, float]:
    """
    Get feature weights based on year and available data.
    
    Strategy:
    - 2025: Motion/play action available (Sharp), emphasize: motion_rate, play_action_rate,
            under_center_rate, no_huddle_rate (2x) + personnel (1.5x)
    - 2022-2024: NO motion/play action (not reliable), emphasize: under_center_rate,
                 no_huddle_rate (2x) + personnel (1.5x)
    - 2015-2021: NO motion/play action, NO personnel, emphasize: no_huddle_rate,
                 shotgun_rate, down_1_pass_rate (2x)
    
    Returns dict mapping feature names to weights (default 1.0 if not specified).
    """
    weights: Dict[str, float] = {}
    
    if year == 2025:
        # 2025: Motion and play action available from Sharp Football
        # Goal: Group Shanahan offenses (SF, LAR, GB, MIA) better - aim for 2 clusters max
        # They share: high motion (59-67), play action (14-21), personnel 11 (39-58)
        # Note: shotgun_play_action_rate is 0.0 for all, so don't weight it
        weights.update({
            "motion_rate": 1.75,  # All Shanahan teams have high motion (59-67)
            "play_action_rate": 2.5,  # Strong emphasis on play action (key Shanahan trait)
            "under_center_rate": 1.0,  # Minimize - GB/MIA have much lower rates (28-38 vs 47-60)
            "no_huddle_rate": 1.0,  # Minimize - varies too much (3.7-8.5)
            "personnel_11_rate": 2.0,  # Increase - Shanahan teams use 11 personnel heavily (39-58)
            "personnel_12_rate": 1.0,  # Minimize - GB has much higher (33) than others (9-11)
        })
        
        # Personnel data available for 2025, weight it more
        weights.update({
            "personnel_11_rate": 1.5,
            "personnel_12_rate": 1.5,
            "personnel_13_rate": 1.5,
            "personnel_21_rate": 1.5,
            "personnel_22_rate": 1.5,
            "personnel_10_rate": 1.5,
            "personnel_20_rate": 1.5,
            "personnel_01_rate": 1.5,
            "personnel_31_rate": 1.5,
        })
    elif year >= 2022:
        # 2022-2024: NO motion/play action (not reliable), but personnel available
        weights.update({
            "under_center_rate": 2.0,
            "no_huddle_rate": 2.0,
        })
        
        # Personnel data available for 2022-2024, weight it more
        weights.update({
            "personnel_11_rate": 1.5,
            "personnel_12_rate": 1.5,
            "personnel_13_rate": 1.5,
            "personnel_21_rate": 1.5,
            "personnel_22_rate": 1.5,
            "personnel_10_rate": 1.5,
            "personnel_20_rate": 1.5,
            "personnel_01_rate": 1.5,
            "personnel_31_rate": 1.5,
        })
    else:
        # 2015-2021: NO motion/play action, NO personnel
        weights.update({
            "no_huddle_rate": 2.0,
            "shotgun_rate": 2.0,
            "down_1_pass_rate": 2.0,
        })
    
    # All other features default to 1.0 (equal weight)
    return weights


def fix_and_cluster_year(year: int, n_clusters: int = 4, feature_weights: Dict[str, float] | None = None) -> Path:
    """
    Fix data issues, validate, and re-cluster a single year.
    
    Args:
        year: Season year
        n_clusters: Number of clusters (default 4)
        feature_weights: Optional dict mapping feature names to weights.
                        Features with higher weights have more influence.
                        Example: {"motion_rate": 2.0, "play_action_rate": 1.5, ...}
                        If None, all features weighted equally.
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
    
    # 5) Standardize and cluster with balanced K-means
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Build feature weights array: use provided weights, or year-specific defaults
    if feature_weights is None:
        feature_weights = get_feature_weights_for_year(year)
    
    weights_array = np.array([feature_weights.get(col, 1.0) for col in feature_cols])
    if weights_array.max() > 1.0:
        weighted_features = [col for col in feature_cols if feature_weights.get(col, 1.0) > 1.0]
        print(f"[fix_and_validate] Year {year}: applying feature weights (min={weights_array.min():.2f}, max={weights_array.max():.2f})")
        print(f"[fix_and_validate] Year {year}: weighted features: {weighted_features[:5]}...")
    
    # Use balanced K-means with minimum cluster size (allows natural imbalance)
    # Minimum 4 teams per cluster to avoid tiny clusters, but allows natural groupings
    labels, centers_scaled = balanced_kmeans(
        X_scaled, 
        n_clusters=n_clusters, 
        min_cluster_size=4, 
        random_state=42,
        feature_weights=weights_array
    )
    
    # Convert centers back to original scale for saving
    # If weights were applied, unweight the centers first (they're in weighted scaled space)
    if weights_array is not None:
        centers_unweighted = centers_scaled / weights_array[np.newaxis, :]
    else:
        centers_unweighted = centers_scaled
    centers_orig = scaler.inverse_transform(centers_unweighted)
    
    df["scheme_cluster"] = labels
    
    # Check cluster balance
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    print(f"[fix_and_validate] Year {year}: cluster sizes: {dict(cluster_counts)}")
    
    # 6) Put scheme_cluster last and save
    personnel_cols = [c for c in df.columns if c.startswith("personnel_")]
    core_cols = [c for c in df.columns if c not in personnel_cols and c != "scheme_cluster"]
    col_order = core_cols + personnel_cols + ["scheme_cluster"]
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]
    df.to_csv(csv_path, index=False)
    
    # 7) Save cluster centers RELATIVE TO MEAN (centered)
    # centers_orig already computed above from balanced_kmeans
    feature_means = X.mean().values
    centers_relative = centers_orig - feature_means
    
    centers_path = base / f"{year}_schemes_cluster_centers.csv"
    centers_df = pd.DataFrame(
        centers_relative,
        columns=feature_cols,
        index=range(n_clusters)
    )
    # Label which row is which cluster (first column)
    centers_df.insert(0, "cluster", range(n_clusters))
    centers_df.to_csv(centers_path, index=False)
    
    # Also save a note file explaining the format
    note_path = base / f"{year}_schemes_cluster_centers_README.txt"
    with open(note_path, "w") as f:
        f.write(
            f"Cluster Centers for {year} - RELATIVE TO MEAN\n"
            f"==============================================\n\n"
            f"The first column 'cluster' is the cluster ID (0, 1, 2, 3). Each row is that cluster's center.\n\n"
            f"Feature values show how each cluster differs from the league average for that year.\n"
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


def fix_and_cluster_all_years(n_clusters: int = 4, feature_weights: Dict[str, float] | None = None) -> None:
    """
    Fix and re-cluster all years.
    
    Args:
        n_clusters: Number of clusters (default 4)
        feature_weights: Optional dict mapping feature names to weights.
                        Example: {"motion_rate": 2.0, "play_action_rate": 1.5}
    """
    base = Path(SCHEME_DATA_DIR)
    years = sorted([
        int(p.stem.split("_")[0])
        for p in base.glob("*_schemes.csv")
        if not p.stem.startswith("sharp_")
    ])
    
    for yr in years:
        fix_and_cluster_year(yr, n_clusters=n_clusters, feature_weights=feature_weights)


if __name__ == "__main__":
    fix_and_cluster_all_years()
