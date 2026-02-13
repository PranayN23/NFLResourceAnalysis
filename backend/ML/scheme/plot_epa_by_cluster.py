"""
Plot Net EPA (from nfl_epa.csv) vs scheme cluster for each year.

For each season, creates one plot: x = scheme cluster, y = Net EPA, with each
team as a labeled point. Saves to data/figs/{year}_net_epa_by_cluster.png.

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
    df = pd.read_csv(EPA_PATH)
    df["Team"] = df["Team"].astype(str).str.strip()
    df["team_abbr"] = df["Team"].map(EPA_TEAM_TO_ABBR)
    df = df.rename(columns={"Year": "season"})
    return df[["season", "team_abbr", "Net EPA"]].dropna(subset=["team_abbr"])


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

    # One plot per year: Net EPA (y) vs scheme cluster (x), team labels
    years = sorted(merged["season"].unique())
    for year in years:
        sub = merged[merged["season"] == year].copy()
        sub["scheme_cluster"] = sub["scheme_cluster"].astype(int)

        fig, ax = plt.subplots(figsize=(10, 6))
        clusters = sorted(sub["scheme_cluster"].unique())
        colors = plt.cm.tab10([c / max(clusters) if clusters else 0 for c in clusters])

        for i, c in enumerate(clusters):
            mask = sub["scheme_cluster"] == c
            x = sub.loc[mask, "scheme_cluster"]
            y = sub.loc[mask, "Net EPA"]
            ax.scatter(x, y, c=[colors[i]], label=f"Cluster {c}", alpha=0.8, s=80, edgecolors="white", linewidths=0.5)
            for _, row in sub[mask].iterrows():
                ax.annotate(
                    row["team_abbr"],
                    (row["scheme_cluster"], row["Net EPA"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                )

        ax.set_xlabel("Scheme cluster (per-year K-means)")
        ax.set_ylabel("Net EPA")
        ax.set_title(f"{year} — Net EPA by scheme cluster")
        ax.set_xticks(clusters)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")
        plt.tight_layout()
        out = FIG_DIR / f"{year}_net_epa_by_cluster.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved {out}")

    # Optional: one combined boxplot (all years)
    fig, ax = plt.subplots(figsize=(8, 5))
    clusters = sorted(merged["scheme_cluster"].unique())
    data = [merged.loc[merged["scheme_cluster"] == c, "Net EPA"].values for c in clusters]
    ax.boxplot(data, tick_labels=[str(int(c)) for c in clusters], patch_artist=True)
    ax.set_xlabel("Scheme cluster")
    ax.set_ylabel("Net EPA")
    ax.set_title("Net EPA by scheme cluster (all years combined)")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_all = FIG_DIR / "all_years_net_epa_by_cluster.png"
    plt.savefig(out_all, dpi=150)
    plt.close()
    print(f"Saved {out_all}")


if __name__ == "__main__":
    plot_epa_by_cluster()
