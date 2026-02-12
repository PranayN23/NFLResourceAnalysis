"""
Plot scheme clusters by season to visualize how teams group.

For each yearly schemes CSV (e.g. 2015_schemes.csv ... 2025_schemes.csv),
this script creates a scatter plot with:
- X-axis: motion_rate
- Y-axis: play_action_rate
- Color: scheme_cluster
- Label: team_abbr

Outputs PNGs under:
  backend/ML/scheme/data/figs/{year}_clusters_motion_playaction.png

Usage (from backend/ML):

    python -m scheme.plot_scheme_clusters

You can also restrict to specific years:

    python -m scheme.plot_scheme_clusters 2019 2020 2021 2022 2023 2024 2025
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

try:
    from .config import SCHEME_DATA_DIR
except ImportError:  # pragma: no cover
    import os

    SCHEME_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _available_year_files() -> list[int]:
    """Detect which *_schemes.csv files exist in SCHEME_DATA_DIR."""
    base = Path(SCHEME_DATA_DIR)
    years: list[int] = []
    for p in base.glob("*_schemes.csv"):
        # Ignore Sharp-derived or other non-year files if any sneak in
        stem = p.stem
        if stem.startswith("sharp_"):
            continue
        try:
            year = int(stem.split("_")[0])
        except ValueError:
            continue
        years.append(year)
    return sorted(set(years))


def plot_year_clusters(year: int) -> Path | None:
    """
    Plot a cluster scatter for a given season year.

    Returns the path to the PNG, or None if required columns are missing.
    """
    base = Path(SCHEME_DATA_DIR)
    csv_path = base / f"{year}_schemes.csv"
    if not csv_path.exists():
        print(f"[plot_scheme_clusters] No schemes CSV for year {year}: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    # Require basic columns
    required = {"team_abbr", "scheme_cluster"}
    if not required.issubset(df.columns):
        print(
            f"[plot_scheme_clusters] Skipping {year}: missing columns {required - set(df.columns)}"
        )
        return None

    # Choose axes; fall back if motion/play_action not present.
    x_candidates = ["motion_rate", "shotgun_rate", "down_1_pass_rate"]
    y_candidates = ["play_action_rate", "air_yards_per_att", "down_2_pass_rate"]

    x_col = next((c for c in x_candidates if c in df.columns), None)
    y_col = next((c for c in y_candidates if c in df.columns), None)

    if x_col is None or y_col is None:
        print(
            f"[plot_scheme_clusters] Skipping {year}: couldn't find suitable x/y columns "
            f"(tried x={x_candidates}, y={y_candidates})"
        )
        return None

    # Drop rows with missing cluster or axes to avoid NaN issues in plotting/legend
    df = df.dropna(subset=[x_col, y_col, "scheme_cluster"]).copy()
    if df.empty:
        print(f"[plot_scheme_clusters] Skipping {year}: no valid rows after dropping NaNs.")
        return None

    fig, ax = plt.subplots(figsize=(9, 7))

    scatter = ax.scatter(
        df[x_col],
        df[y_col],
        c=df["scheme_cluster"],
        cmap="tab10",
        edgecolor="k",
        alpha=0.8,
    )

    # Label each point with team abbreviation
    for _, row in df.iterrows():
        ax.text(
            row[x_col],
            row[y_col],
            str(row["team_abbr"]),
            fontsize=8,
            ha="center",
            va="center",
        )

    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.set_title(f"Offensive Scheme Clusters {year}")

    # Legend for clusters (build manually to avoid NaN issues)
    clusters = sorted({int(c) for c in df["scheme_cluster"].dropna().unique()})
    legend_handles: list[Line2D] = []
    for c in clusters:
        color = scatter.cmap(scatter.norm(c))
        legend_handles.append(
            Line2D(
                [], [], marker="o", linestyle="", color=color, label=f"Cluster {c}", markersize=8
            )
        )
    if legend_handles:
        ax.legend(handles=legend_handles, title="Scheme Cluster")

    out_dir = base / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{year}_clusters_{x_col}_vs_{y_col}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[plot_scheme_clusters] Saved cluster plot for {year} to {out_path}")
    return out_path


def plot_clusters_for_years(years: Iterable[int]) -> list[Path]:
    """Plot clusters for a list of years; returns list of created PNG paths."""
    outputs: list[Path] = []
    for yr in years:
        out = plot_year_clusters(int(yr))
        if out is not None:
            outputs.append(out)
    return outputs


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        yrs = [int(x) for x in sys.argv[1:]]
    else:
        yrs = _available_year_files()
        if not yrs:
            raise SystemExit(
                "[plot_scheme_clusters] No *_schemes.csv files found in SCHEME_DATA_DIR"
            )

    paths = plot_clusters_for_years(yrs)
    print(f"[plot_scheme_clusters] Generated {len(paths)} plots.")

