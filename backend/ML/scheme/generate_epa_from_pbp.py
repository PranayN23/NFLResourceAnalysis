"""
Generate Offensive EPA per play data from nflfastR play-by-play data.

Offensive EPA per play = Average EPA when team has the ball (offensive plays only)

This script reads PBP data (from local CSVs or nfl_data_py) and calculates
team-season Offensive EPA per play, then adds it to nfl_epa.csv.

Usage (from repo root):
    python -m backend.ML.scheme.generate_epa_from_pbp --year 2025
    python -m backend.ML.scheme.generate_epa_from_pbp --year 2025 --update-csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .config import SCHEME_DATA_DIR
except ImportError:
    import os
    SCHEME_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Directory for locally downloaded nflfastR pbp CSVs
RAW_PBP_DIR = Path(SCHEME_DATA_DIR).parent / "raw_pbp"
RAW_PBP_DIR.mkdir(parents=True, exist_ok=True)

# Path to main EPA CSV
EPA_PATH = Path(__file__).parent.parent / "nfl_epa.csv"

# Team name mapping (full names as in nfl_epa.csv)
TEAM_ABBR_TO_NAME = {
    "ARI": "Cardinals",
    "ATL": "Falcons",
    "BAL": "Ravens",
    "BUF": "Bills",
    "CAR": "Panthers",
    "CHI": "Bears",
    "CIN": "Bengals",
    "CLE": "Browns",
    "DAL": "Cowboys",
    "DEN": "Broncos",
    "DET": "Lions",
    "GB": "Packers",
    "HOU": "Texans",
    "IND": "Colts",
    "JAX": "Jaguars",
    "KC": "Chiefs",
    "LV": "Raiders",
    "LAC": "Chargers",
    "LAR": "Rams",
    "MIA": "Dolphins",
    "MIN": "Vikings",
    "NE": "Patriots",
    "NO": "Saints",
    "NYG": "Giants",
    "NYJ": "Jets",
    "PHI": "Eagles",
    "PIT": "Steelers",
    "SF": "49ers",
    "SEA": "Seahawks",
    "TB": "Buccaneers",
    "TEN": "Titans",
    "WAS": "Commanders",
}


def load_pbp_for_year(year: int) -> pd.DataFrame:
    """Load play-by-play for a specific year."""
    local_path = RAW_PBP_DIR / f"pbp_{year}.csv"
    if local_path.exists():
        print(f"Loading local PBP: {local_path}")
        return pd.read_csv(local_path)
    
    # Fallback to nfl_data_py
    try:
        import nfl_data_py as nfl
        print(f"Loading PBP from nfl_data_py for {year}")
        df = nfl.import_pbp_data([year])
        return pd.DataFrame(df) if hasattr(df, "to_pandas") else df
    except ImportError:
        try:
            import nflreadpy
            print(f"Loading PBP from nflreadpy for {year}")
            df = nflreadpy.load_pbp(seasons=[year])
            return df.to_pandas() if hasattr(df, "to_pandas") else pd.DataFrame(df)
        except ImportError:
            raise FileNotFoundError(
                f"No local PBP file found at {local_path} and neither nfl_data_py nor nflreadpy are installed. "
                f"Download pbp_{year}.csv from nflfastR GitHub or install nfl_data_py."
            )


def calculate_offensive_epa(pbp: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Calculate Offensive EPA per play for each team in a given season.
    
    Offensive EPA per play = Average EPA when team has the ball (offensive plays only)
    
    Returns DataFrame with columns: Year, Team, Offensive EPA per play
    """
    if "epa" not in pbp.columns:
        raise ValueError("PBP data must have 'epa' column. Use nflfastR enriched PBP data.")
    
    # Filter to regular offensive plays (exclude special teams, penalties, etc.)
    pbp = pbp[
        pbp["play_type"].isin(["pass", "run"]) &
        pbp["epa"].notna() &
        pbp["posteam"].notna() &
        (pbp["posteam"] != "")
    ].copy()
    
    if pbp.empty:
        raise ValueError(f"No valid plays found for {year}")
    
    # Calculate offensive EPA per play (when team has the ball)
    off_epa = (
        pbp.groupby(["posteam", "season"])["epa"]
        .mean()
        .reset_index()
        .rename(columns={"posteam": "team", "epa": "Offensive EPA per play"})
    )
    
    # Map team abbreviations to full names
    off_epa["Team"] = off_epa["team"].map(TEAM_ABBR_TO_NAME)
    off_epa = off_epa[off_epa["Team"].notna()].copy()
    
    # Select and rename columns
    result = off_epa[["season", "Team", "Offensive EPA per play"]].copy()
    result = result.rename(columns={"season": "Year"})
    result = result.sort_values("Offensive EPA per play", ascending=False)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate Offensive EPA per play from PBP data")
    parser.add_argument("--year", type=int, required=True, help="Year to calculate EPA for")
    parser.add_argument("--update-csv", action="store_true", help="Add/update Offensive EPA per play column in nfl_epa.csv")
    args = parser.parse_args()
    
    year = args.year
    
    print(f"Calculating Offensive EPA per play for {year}...")
    pbp = load_pbp_for_year(year)
    epa_df = calculate_offensive_epa(pbp, year)
    
    print(f"\nOffensive EPA per play for {year}:")
    print(epa_df.to_string(index=False))
    
    if args.update_csv:
        # Load existing EPA data
        if EPA_PATH.exists():
            existing = pd.read_csv(EPA_PATH)
            # Remove existing entries for this year's offensive EPA
            if "Offensive EPA per play" in existing.columns:
                existing.loc[existing["Year"] == year, "Offensive EPA per play"] = None
            
            # Merge new offensive EPA data
            existing = existing.merge(
                epa_df[["Year", "Team", "Offensive EPA per play"]],
                on=["Year", "Team"],
                how="outer",
                suffixes=("", "_new")
            )
            
            # Update Offensive EPA per play column
            if "Offensive EPA per play_new" in existing.columns:
                existing["Offensive EPA per play"] = existing["Offensive EPA per play_new"].fillna(existing.get("Offensive EPA per play", None))
                existing = existing.drop(columns=["Offensive EPA per play_new"])
            
            # Ensure Offensive EPA per play column exists
            if "Offensive EPA per play" not in existing.columns:
                existing["Offensive EPA per play"] = None
            
            # Fill in the new values
            for _, row in epa_df.iterrows():
                mask = (existing["Year"] == row["Year"]) & (existing["Team"] == row["Team"])
                existing.loc[mask, "Offensive EPA per play"] = row["Offensive EPA per play"]
            
            epa_df = existing
        else:
            print(f"Creating new EPA file at {EPA_PATH}")
            # If file doesn't exist, create it with just offensive EPA
            epa_df = epa_df.rename(columns={"Offensive EPA per play": "Offensive EPA per play"})
        
        epa_df = epa_df.sort_values(["Year", "Offensive EPA per play"], ascending=[True, False], na_position="last")
        epa_df.to_csv(EPA_PATH, index=False)
        print(f"\nUpdated {EPA_PATH} with {year} Offensive EPA per play data")
    else:
        print(f"\nUse --update-csv to add this data to {EPA_PATH}")


if __name__ == "__main__":
    main()
