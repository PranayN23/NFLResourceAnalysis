"""
Plot EPA metrics (from nfl_epa.csv) vs scheme cluster for each year.

Generates plots for multiple EPA metrics:
1. Net EPA plots (for all years): {year}_net_epa_by_cluster.png
2. Offensive EPA per play plots (when available): {year}_offensive_epa_by_cluster.png
3. EPA/Play plots (when available): {year}_epa_per_play_by_cluster.png
4. Total EPA plots (when available): {year}_total_epa_by_cluster.png
5. Success Rate plots (when available): {year}_success_rate_by_cluster.png
6. EPA/Pass plots (when available): {year}_epa_per_pass_by_cluster.png
7. EPA/Rush plots (when available): {year}_epa_per_rush_by_cluster.png

For each season, creates plots: x = scheme cluster, y = EPA metric, with each
team as a labeled point.

Clustering note: scheme_cluster is computed per year (K-means is fit on that
year's data only). Cluster 0 in 2019 is not comparable to cluster 0 in 2020.

Usage (from repo root):
    python -m backend.ML.scheme.plot_epa_by_cluster
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SCHEME_DATA_DIR = Path(__file__).parent / "data"
EPA_PATH = Path(__file__).parent.parent / "nfl_epa.csv"
FIG_DIR = Path(__file__).parent / "data" / "figs"

EPA_TEAM_TO_ABBR = {
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


def load_epa() -> pd.DataFrame:
    """Load all EPA data (all available metrics)."""
    df = pd.read_csv(EPA_PATH)
    df["Team"] = df["Team"].astype(str).str.strip()
    df["team_abbr"] = df["Team"].map(EPA_TEAM_TO_ABBR)
    df = df.rename(columns={"Year": "season"})
    
    # Return all available EPA columns
    cols = ["season", "team_abbr"]
    
    # Add all EPA-related columns that exist
    epa_cols = ["Net EPA", "Offensive EPA per play", "EPA/Play", "Total EPA", "Success Rate", "EPA/Pass", "EPA/Rush"]
    for col in epa_cols:
        if col in df.columns:
            cols.append(col)
    
    # Handle Offensive EPA alias
    if "Offensive EPA" in df.columns and "Offensive EPA per play" not in df.columns:
        df = df.rename(columns={"Offensive EPA": "Offensive EPA per play"})
        if "Offensive EPA per play" not in cols:
            cols.append("Offensive EPA per play")
    
    result = df[cols].copy()
    result = result.dropna(subset=["team_abbr"])
    
    if result.empty:
        raise ValueError("No EPA data found in nfl_epa.csv")
    
    return result


def load_schemes_with_cluster() -> pd.DataFrame:
    rows = []
    for p in sorted(SCHEME_DATA_DIR.glob("[0-9][0-9][0-9][0-9]_schemes.csv")):
        year = int(p.stem.split("_")[0])
        df = pd.read_csv(p)
        if "scheme_cluster" not in df.columns:
            continue
        df = df[["team_abbr", "season", "scheme_cluster"]].copy()
        df["season"] = df["season"].astype(int)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def plot_single_year(merged: pd.DataFrame, year: int, epa_col: str, label: str, filename_suffix: str, show_zero_line: bool = True) -> None:
    """Plot EPA by cluster for a single year."""
    sub = merged[merged["season"] == year].copy()
    sub = sub[sub[epa_col].notna()].copy()  # Only plot teams with EPA data
    
    if sub.empty:
        return
    
    sub["scheme_cluster"] = sub["scheme_cluster"].astype(int)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    clusters = sorted(sub["scheme_cluster"].unique())
    colors = plt.cm.tab10([c / max(clusters) if clusters else 0 for c in clusters])
    
    for i, c in enumerate(clusters):
        mask = sub["scheme_cluster"] == c
        x = sub.loc[mask, "scheme_cluster"]
        y = sub.loc[mask, epa_col]
        ax.scatter(x, y, c=[colors[i]], label=f"Cluster {c}", alpha=0.7, s=40, edgecolors="white", linewidths=0.3)
        for _, row in sub[mask].iterrows():
            ax.annotate(
                row["team_abbr"],
                (row["scheme_cluster"], row[epa_col]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                fontweight="normal",
            )
    
    ax.set_xlabel("Scheme cluster (per-year K-means)")
    ax.set_ylabel(label)
    ax.set_title(f"{year} — {label} by scheme cluster")
    ax.set_xticks(clusters)
    if show_zero_line:
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    out = FIG_DIR / f"{year}_{filename_suffix}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def plot_epa_by_cluster() -> None:
    epa = load_epa()
    schemes = load_schemes_with_cluster()
    if schemes.empty:
        print("No scheme files with scheme_cluster found. Run rebuild_all_schemes first.")
        return

    merged = epa.merge(
        schemes,
        on=["season", "team_abbr"],
        how="inner",
    )
    if merged.empty:
        print("No overlapping team-years between EPA and scheme data.")
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    years = sorted(merged["season"].unique())
    has_offensive_epa = "Offensive EPA per play" in merged.columns
    has_net_epa = "Net EPA" in merged.columns
    
    # Define all metrics to plot
    metrics_to_plot = []
    
    if has_net_epa:
        metrics_to_plot.append(("Net EPA", "Net EPA", "net_epa_by_cluster", True))
    
    if has_offensive_epa:
        metrics_to_plot.append(("Offensive EPA per play", "Offensive EPA per play", "offensive_epa_by_cluster", True))
    
    # Check for SumerSports metrics
    if "EPA/Play" in merged.columns:
        metrics_to_plot.append(("EPA/Play", "EPA/Play", "epa_per_play_by_cluster", True))
    if "Total EPA" in merged.columns:
        metrics_to_plot.append(("Total EPA", "Total EPA", "total_epa_by_cluster", True))
    if "Success Rate" in merged.columns:
        metrics_to_plot.append(("Success Rate", "Success Rate", "success_rate_by_cluster", False))  # No zero line for percentage
    if "EPA/Pass" in merged.columns:
        metrics_to_plot.append(("EPA/Pass", "EPA/Pass", "epa_per_pass_by_cluster", True))
    if "EPA/Rush" in merged.columns:
        metrics_to_plot.append(("EPA/Rush", "EPA/Rush", "epa_per_rush_by_cluster", True))
    
    # Generate plots for each metric
    for label, col_name, filename_suffix, show_zero in metrics_to_plot:
        print(f"\nGenerating {label} plots...")
        years_with_data = sorted(merged[merged[col_name].notna()]["season"].unique())
        
        if not years_with_data:
            print(f"  No {label} data available. Skipping.")
            continue
        
        # Individual year plots
        for year in years_with_data:
            plot_single_year(merged, year, col_name, label, filename_suffix, show_zero_line=show_zero)
        
        # Combined boxplot
        fig, ax = plt.subplots(figsize=(8, 5))
        clusters = sorted(merged["scheme_cluster"].unique())
        data = [merged.loc[merged["scheme_cluster"] == c, col_name].dropna().values for c in clusters]
        valid_data = [d for d in data if len(d) > 0]
        valid_clusters = [c for c in clusters if len(merged.loc[merged["scheme_cluster"] == c, col_name].dropna()) > 0]
        
        if valid_data:
            ax.boxplot(valid_data, 
                      tick_labels=[str(int(c)) for c in valid_clusters], 
                      patch_artist=True)
            ax.set_xlabel("Scheme cluster")
            ax.set_ylabel(label)
            ax.set_title(f"{label} by scheme cluster (all years combined)")
            if show_zero:
                ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            plt.tight_layout()
            out_all = FIG_DIR / f"all_years_{filename_suffix}.png"
            plt.savefig(out_all, dpi=150)
            plt.close()
            print(f"Saved {out_all}")
    
    print(f"\nSummary:")
    for label, col_name, _, _ in metrics_to_plot:
        if col_name in merged.columns:
            years_with_data = sorted(merged[merged[col_name].notna()]["season"].unique())
            print(f"  {label} plots: {len(years_with_data)} years ({sorted(years_with_data)})")


if __name__ == "__main__":
    plot_epa_by_cluster()
