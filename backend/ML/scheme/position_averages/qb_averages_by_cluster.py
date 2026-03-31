"""
Calculate QB averages by scheme cluster for each year and across all years.

For each year:
1. Merge QB data with scheme clusters
2. Calculate average QB stats for each cluster
3. Calculate cluster averages relative to overall mean (like cluster centers)

Outputs:
- {year}_qb_averages_by_cluster.csv: Per-year QB averages by cluster
- all_years_qb_averages_by_cluster.csv: Combined QB averages by cluster
- {year}_qb_cluster_centers.csv: Cluster averages relative to mean (per year)
- all_years_qb_cluster_centers.csv: Combined cluster averages relative to mean

Usage (from repo root):
    python -m backend.ML.scheme.position_averages.qb_averages_by_cluster
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Paths
SCHEME_DATA_DIR = Path(__file__).parent.parent / "data"
QB_PATH = Path(__file__).parent.parent.parent / "QB.csv"
OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Team name to abbreviation mapping
TEAM_TO_ABBR = {
    "Cardinals": "ARI",
    "Falcons": "ATL",
    "Ravens": "BAL",
    "Bills": "BUF",
    "Panthers": "CAR",
    "Bears": "CHI",
    "Bengals": "CIN",
    "Browns": "CLE",
    "Cowboys": "DAL",
    "Broncos": "DEN",
    "Lions": "DET",
    "Packers": "GB",
    "Texans": "HOU",
    "Colts": "IND",
    "Jaguars": "JAX",
    "Chiefs": "KC",
    "Raiders": "LV",
    "Chargers": "LAC",
    "Rams": "LAR",
    "Dolphins": "MIA",
    "Vikings": "MIN",
    "Patriots": "NE",
    "Saints": "NO",
    "Giants": "NYG",
    "Jets": "NYJ",
    "Eagles": "PHI",
    "Steelers": "PIT",
    "49ers": "SF",
    "Seahawks": "SEA",
    "Buccaneers": "TB",
    "Titans": "TEN",
    "Commanders": "WAS",
}


def load_qb_data() -> pd.DataFrame:
    """Load and clean QB data."""
    df = pd.read_csv(QB_PATH)
    
    # Convert numeric columns
    numeric_cols = [
        "Year", "age", "Cap_Space", "adjusted_value", "Net EPA", "accuracy_percent",
        "aimed_passes", "attempts", "avg_depth_of_target", "avg_time_to_throw",
        "bats", "big_time_throws", "btt_rate", "completion_percent", "completions",
        "declined_penalties", "def_gen_pressures", "drop_rate", "dropbacks", "drops",
        "first_downs", "grades_hands_fumble", "grades_offense", "grades_pass",
        "grades_run", "hit_as_threw", "interceptions", "passing_snaps", "penalties",
        "pressure_to_sack_rate", "qb_rating", "sack_percent", "sacks", "scrambles",
        "spikes", "thrown_aways", "touchdowns", "turnover_worthy_plays", "twp_rate",
        "yards", "ypa", "weighted_grade", "weighted_average_grade",
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Normalize team names
    df["team_abbr"] = df["Team"].map(TEAM_TO_ABBR)
    df = df[df["team_abbr"].notna()].copy()
    
    # Rename Year to season for consistency
    df = df.rename(columns={"Year": "season"})
    
    return df


def load_scheme_clusters() -> pd.DataFrame:
    """Load scheme cluster assignments for all years."""
    rows = []
    for p in sorted(SCHEME_DATA_DIR.glob("[0-9][0-9][0-9][0-9]_schemes.csv")):
        year = int(p.stem.split("_")[0])
        df = pd.read_csv(p)
        if "scheme_cluster" not in df.columns:
            continue
        df = df[["team_abbr", "season", "scheme_cluster"]].copy()
        df["season"] = df["season"].astype(int)
        # Ensure scheme_cluster is integer
        df["scheme_cluster"] = df["scheme_cluster"].astype(int)
        rows.append(df)
    
    if not rows:
        return pd.DataFrame()
    
    return pd.concat(rows, ignore_index=True)


def get_numeric_qb_columns(df: pd.DataFrame) -> list[str]:
    """Get list of numeric QB stat columns."""
    exclude_cols = {
        "player_id", "player", "Team", "team_abbr", "season", "position", "position_x",
        "franchise_id", "Win %", "scheme_cluster",  # Exclude scheme_cluster from stats
    }
    
    numeric_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if df[col].dtype in [np.int64, np.float64] or pd.api.types.is_numeric_dtype(df[col]):
            # Only include columns with meaningful variance
            if df[col].notna().sum() > 0:
                numeric_cols.append(col)
    
    return numeric_cols


def calculate_cluster_averages(
    merged: pd.DataFrame, 
    stat_cols: list[str],
    relative_to_mean: bool = False
) -> pd.DataFrame:
    """
    Calculate average QB stats by cluster.
    
    If relative_to_mean=True, returns averages relative to overall mean (like cluster centers).
    """
    cluster_stats = []
    
    # Ensure scheme_cluster is integer
    merged = merged.copy()
    merged["scheme_cluster"] = merged["scheme_cluster"].astype(int)
    
    for cluster in sorted(merged["scheme_cluster"].unique()):
        cluster_data = merged[merged["scheme_cluster"] == cluster]
        
        stats = {"scheme_cluster": cluster}
        
        for col in stat_cols:
            values = cluster_data[col].dropna()
            if len(values) > 0:
                mean_val = values.mean()
                if relative_to_mean:
                    # Calculate relative to overall mean
                    overall_mean = merged[col].dropna().mean()
                    stats[col] = mean_val - overall_mean
                else:
                    stats[col] = mean_val
            else:
                stats[col] = np.nan
        
        cluster_stats.append(stats)
    
    return pd.DataFrame(cluster_stats)


def main():
    print("Loading QB data...")
    qb_df = load_qb_data()
    print(f"  Loaded {len(qb_df)} QB rows")
    
    print("\nLoading scheme clusters...")
    scheme_df = load_scheme_clusters()
    if scheme_df.empty:
        print("  No scheme cluster data found. Run rebuild_all_schemes first.")
        return
    print(f"  Loaded {len(scheme_df)} team-year cluster assignments")
    
    # Merge QB data with scheme clusters
    print("\nMerging QB data with scheme clusters...")
    merged = qb_df.merge(
        scheme_df,
        on=["team_abbr", "season"],
        how="inner",
    )
    # Ensure scheme_cluster is integer after merge
    merged["scheme_cluster"] = merged["scheme_cluster"].astype(int)
    print(f"  Merged dataset: {len(merged)} QB-season rows")
    
    if merged.empty:
        print("  No overlapping data. Exiting.")
        return
    
    # Get numeric stat columns
    stat_cols = get_numeric_qb_columns(merged)
    print(f"\nFound {len(stat_cols)} numeric QB stat columns")
    
    # Process per year
    years = sorted(merged["season"].unique())
    print(f"\nProcessing {len(years)} years: {years}")
    
    all_year_averages = []
    all_year_centers = []
    
    for year in years:
        year_data = merged[merged["season"] == year].copy()
        if year_data.empty:
            continue
        
        print(f"\n  Year {year}:")
        print(f"    {len(year_data)} QB rows, {len(year_data['scheme_cluster'].unique())} clusters")
        
        # Calculate averages by cluster (relative to mean)
        cluster_avgs = calculate_cluster_averages(year_data, stat_cols, relative_to_mean=True)
        cluster_avgs["season"] = year
        # Ensure scheme_cluster is integer and matches cluster
        cluster_avgs["scheme_cluster"] = cluster_avgs["scheme_cluster"].astype(int)
        cluster_avgs["cluster"] = cluster_avgs["scheme_cluster"]  # cluster = scheme_cluster (0,1,2,3)
        cluster_avgs = cluster_avgs[["cluster", "season", "scheme_cluster"] + stat_cols]
        
        # Save per-year averages (relative to mean)
        out_file = OUTPUT_DIR / f"{year}_qb_averages_by_cluster.csv"
        cluster_avgs.to_csv(out_file, index=False)
        print(f"    Saved {out_file}")
        
        # Calculate cluster centers (relative to mean)
        cluster_centers = calculate_cluster_averages(year_data, stat_cols, relative_to_mean=True)
        cluster_centers["season"] = year
        # Ensure scheme_cluster is integer and matches cluster
        cluster_centers["scheme_cluster"] = cluster_centers["scheme_cluster"].astype(int)
        cluster_centers["cluster"] = cluster_centers["scheme_cluster"]  # cluster = scheme_cluster (0,1,2,3)
        # Reorder: cluster first, then season, scheme_cluster, then stats
        cluster_centers = cluster_centers[["cluster", "season", "scheme_cluster"] + stat_cols]
        
        # Save per-year centers
        centers_file = OUTPUT_DIR / f"{year}_qb_cluster_centers.csv"
        cluster_centers.to_csv(centers_file, index=False)
        print(f"    Saved {centers_file}")
        
        all_year_averages.append(cluster_avgs)
        all_year_centers.append(cluster_centers)
    
    # Combine all years
    if all_year_averages:
        print("\nCombining all years...")
        
        # Combined averages - calculate relative to overall mean
        all_data = merged.copy()
        combined_avgs_list = []
        clusters = sorted([int(c) for c in all_data["scheme_cluster"].unique()])
        
        for cluster in clusters:
            cluster_data = all_data[all_data["scheme_cluster"] == cluster]
            stats = {"cluster": cluster}
            
            for col in stat_cols:
                cluster_values = cluster_data[col].dropna()
                overall_values = all_data[col].dropna()
                
                if len(cluster_values) > 0 and len(overall_values) > 0:
                    cluster_mean = cluster_values.mean()
                    overall_mean = overall_values.mean()
                    stats[col] = cluster_mean - overall_mean  # Relative to mean
                else:
                    stats[col] = np.nan
            
            combined_avgs_list.append(stats)
        
        combined_avgs = pd.DataFrame(combined_avgs_list)
        cols = ["cluster"] + stat_cols
        combined_avgs = combined_avgs[[c for c in cols if c in combined_avgs.columns]]
        
        out_file = OUTPUT_DIR / "all_years_qb_averages_by_cluster.csv"
        combined_avgs.to_csv(out_file, index=False)
        print(f"  Saved {out_file}")
        
        # Combined centers (relative to overall mean across all years)
        all_data = merged.copy()
        combined_centers = []
        clusters = sorted([int(c) for c in all_data["scheme_cluster"].unique()])
        
        for cluster in clusters:
            cluster_data = all_data[all_data["scheme_cluster"] == cluster]
            stats = {"cluster": cluster}
            
            for col in stat_cols:
                cluster_values = cluster_data[col].dropna()
                overall_values = all_data[col].dropna()
                
                if len(cluster_values) > 0 and len(overall_values) > 0:
                    cluster_mean = cluster_values.mean()
                    overall_mean = overall_values.mean()
                    stats[col] = cluster_mean - overall_mean
                else:
                    stats[col] = np.nan
            
            combined_centers.append(stats)
        
        combined_centers_df = pd.DataFrame(combined_centers)
        # Reorder columns: cluster first, then stats
        cols = ["cluster"] + stat_cols
        combined_centers_df = combined_centers_df[[c for c in cols if c in combined_centers_df.columns]]
        centers_file = OUTPUT_DIR / "all_years_qb_cluster_centers.csv"
        combined_centers_df.to_csv(centers_file, index=False)
        print(f"  Saved {centers_file}")
    
    print("\n✓ QB averages by cluster calculation complete!")


if __name__ == "__main__":
    main()
