"""
Merge SumerSports offensive personnel tendency data into yearly scheme CSVs.

Expected Sumer CSV format (one file per season, exported by you from
https://sumersports.com/teams/offensive/personnel-tendency/):

- Columns: at least ['Team', 'Season', 'Personnel', 'Rate', ...]
- 'Personnel' is a string like '11', '12', '21' etc.
- 'Rate' is the usage rate for that personnel package (either as %, e.g. '69.8%',
  or decimal like 0.698).

Usage (from backend/ML):

  python -m scheme.merge_sumer_personnel 2022 2023 2024 2025

For each year Y this will:
- Read scheme/data/Y_schemes.csv
- Read scheme/data/sumer_personnel_Y.csv
- Add columns:
    personnel_11_rate, personnel_12_rate, personnel_21_rate
- Write to scheme/data/Y_schemes_with_personnel.csv
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from .config import SCHEME_DATA_DIR
except ImportError:
    import os

    SCHEME_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _normalize_rate(series: pd.Series) -> pd.Series:
    """
    Convert Sumer 'Rate' column to percentage (0-100).
    Handles values like '69.8%', '69.8', or 0.698.
    """
    s = series.astype(str).str.strip()
    # If it ends with %, strip and parse
    is_pct = s.str.endswith("%")
    s_pct = (
        s[is_pct]
        .str.replace("%", "", regex=False)
        .replace("", "0")
        .astype(float)
    )
    # Non-%: could be 0-1 or 0-100. Heuristic: if <= 1, treat as fraction.
    s_non = (
        s[~is_pct]
        .replace("", "0")
        .astype(float)
    )
    s_non = s_non.where(s_non > 1, s_non * 100.0)
    # Combine back
    out = pd.Series(index=series.index, dtype=float)
    out.loc[is_pct] = s_pct
    out.loc[~is_pct] = s_non
    return out.fillna(0.0)


def merge_sumer_personnel_for_year(year: int) -> Path:
    """
    Merge Sumer personnel usage into Y_schemes.csv for a given year.

    Expects:
      - scheme/data/{year}_schemes.csv
      - scheme/data/sumer_personnel_{year}.csv
    """
    base_dir = Path(SCHEME_DATA_DIR)
    schemes_path = base_dir / f"{year}_schemes.csv"
    sumer_path = base_dir / f"sumer_personnel_{year}.csv"

    if not schemes_path.exists():
        raise FileNotFoundError(f"Scheme file not found: {schemes_path}")
    if not sumer_path.exists():
        raise FileNotFoundError(
            f"Sumer personnel file not found: {sumer_path}\n"
            f"Please export the table from SumerSports to CSV and save it as that path."
        )

    schemes = pd.read_csv(schemes_path)
    sumer = pd.read_csv(sumer_path)

    # Basic sanity checks
    if "Team" not in sumer.columns or "Personnel" not in sumer.columns or "Rate" not in sumer.columns:
        raise ValueError(
            f"Sumer CSV {sumer_path} must have at least 'Team', 'Personnel', and 'Rate' columns."
        )

    # Map full team name to nickname (last word), then to abbr via schemes' team_abbr
    # We build a mapping from nickname -> abbr from the schemes file
    if "team_abbr" not in schemes.columns:
        raise ValueError(f"'team_abbr' column not found in {schemes_path}")

    # For schemes, we also need a consistent nickname to merge on
    schemes["team_nickname"] = schemes["team_abbr"]  # temporary, will replace below

    # Build nickname -> abbr mapping from a standard list
    nickname_to_abbr = {
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

    # For Sumer, get nickname as last word of full team name
    sumer["team_nickname"] = sumer["Team"].astype(str).str.split().str[-1]
    # Map to abbr
    sumer["team_abbr"] = sumer["team_nickname"].map(nickname_to_abbr)
    sumer = sumer.dropna(subset=["team_abbr"]).copy()

    # Filter to this season if 'Season' column exists
    if "Season" in sumer.columns:
        sumer = sumer[sumer["Season"] == year]

    # Normalize rate and pivot to columns per personnel group
    sumer["rate_pct"] = _normalize_rate(sumer["Rate"])
    # Keep the common personnel groupings we care about
    # You can extend this list if you want more groups.
    wanted_personnel = ["10", "11", "12", "13", "20", "21", "22", "01", "31"]
    sumer_sub = sumer[sumer["Personnel"].isin(wanted_personnel)].copy()

    if sumer_sub.empty:
        raise ValueError(
            f"No matching personnel rows (any of {wanted_personnel}) found in {sumer_path}"
        )

    pivot = sumer_sub.pivot_table(
        index="team_abbr",
        columns="Personnel",
        values="rate_pct",
        aggfunc="max",
    ).reset_index()

    # Rename personnel columns to more descriptive names
    rename_map = {
        "10": "personnel_10_rate",
        "11": "personnel_11_rate",
        "12": "personnel_12_rate",
        "13": "personnel_13_rate",
        "20": "personnel_20_rate",
        "21": "personnel_21_rate",
        "22": "personnel_22_rate",
        "01": "personnel_01_rate",
        "31": "personnel_31_rate",
    }
    pivot = pivot.rename(columns={k: v for k, v in rename_map.items() if k in pivot.columns})

    # Merge into schemes on team_abbr + season
    merged = schemes.merge(
        pivot,
        on="team_abbr",
        how="left",
    )

    # Fill missing personnel rates with 0 (if any)
    for col in [v for v in rename_map.values() if v in merged.columns]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)

    out_path = base_dir / f"{year}_schemes_with_personnel.csv"
    merged.to_csv(out_path, index=False)
    return out_path


def merge_sumer_personnel(years: Iterable[int]) -> list[Path]:
    out_paths: list[Path] = []
    for yr in years:
        out_paths.append(merge_sumer_personnel_for_year(int(yr)))
    return out_paths


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m scheme.merge_sumer_personnel 2022 2023 2024 2025")
    yrs = [int(x) for x in sys.argv[1:]]
    paths = merge_sumer_personnel(yrs)
    for p in paths:
        print(f"Saved merged schemes with personnel to {p}")

