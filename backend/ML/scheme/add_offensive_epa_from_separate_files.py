"""
Extract Offensive EPA per play from separate EPA CSV files and add to nfl_epa.csv.

The separate EPA files (EPA 2019.csv, EPA 2020.csv, etc.) have "EPA/play" column
which is offensive EPA per play. This script extracts that data and adds it to
the main nfl_epa.csv file.

Usage (from repo root):
    python -m backend.ML.scheme.add_offensive_epa_from_separate_files
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Path to main EPA CSV
EPA_PATH = Path(__file__).parent.parent / "nfl_epa.csv"

# Path to separate EPA files
EPA_DATA_DIR = Path(__file__).parent.parent / "data"

# Team abbreviation to full name mapping
ABBR_TO_NAME = {
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


def main():
    # Load main EPA file
    if not EPA_PATH.exists():
        print(f"Error: {EPA_PATH} not found. Please create it first.")
        return
    
    main_df = pd.read_csv(EPA_PATH)
    print(f"Loaded {EPA_PATH} with {len(main_df)} rows")
    
    # Find all separate EPA files
    epa_files = sorted(EPA_DATA_DIR.glob("EPA [0-9][0-9][0-9][0-9].csv"))
    if not epa_files:
        print(f"No separate EPA files found in {EPA_DATA_DIR}")
        return
    
    print(f"Found {len(epa_files)} separate EPA files")
    
    # Extract offensive EPA from each file
    offensive_epa_rows = []
    for epa_file in epa_files:
        year = int(epa_file.stem.split()[-1])
        print(f"Processing {epa_file.name} (year {year})...")
        
        df = pd.read_csv(epa_file)
        
        # Check if "EPA/play" column exists
        if "EPA/play" not in df.columns:
            print(f"  Warning: {epa_file.name} does not have 'EPA/play' column. Skipping.")
            continue
        
        # Extract team abbreviation and EPA/play
        for _, row in df.iterrows():
            abbr = row.get("Abbr", None)
            if pd.isna(abbr):
                continue
            
            team_name = ABBR_TO_NAME.get(abbr, None)
            if team_name is None:
                print(f"  Warning: Unknown team abbreviation {abbr} in {epa_file.name}")
                continue
            
            epa_per_play = row["EPA/play"]
            if pd.isna(epa_per_play):
                continue
            
            offensive_epa_rows.append({
                "Year": year,
                "Team": team_name,
                "Offensive EPA per play": float(epa_per_play)
            })
    
    if not offensive_epa_rows:
        print("No offensive EPA data extracted.")
        return
    
    offensive_epa_df = pd.DataFrame(offensive_epa_rows)
    print(f"\nExtracted {len(offensive_epa_df)} offensive EPA entries")
    
    # Merge with main dataframe
    if "Offensive EPA per play" not in main_df.columns:
        main_df["Offensive EPA per play"] = None
    
    # Update existing rows or add new ones
    for _, row in offensive_epa_df.iterrows():
        mask = (main_df["Year"] == row["Year"]) & (main_df["Team"] == row["Team"])
        if mask.any():
            main_df.loc[mask, "Offensive EPA per play"] = row["Offensive EPA per play"]
        else:
            # Add new row if team-year doesn't exist
            new_row = {
                "Year": row["Year"],
                "Team": row["Team"],
                "Net EPA": None,  # Keep Net EPA if it exists
                "Offensive EPA per play": row["Offensive EPA per play"]
            }
            main_df = pd.concat([main_df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Sort by Year and Offensive EPA per play
    main_df = main_df.sort_values(["Year", "Offensive EPA per play"], ascending=[True, False], na_position="last")
    
    # Save updated file
    main_df.to_csv(EPA_PATH, index=False)
    print(f"\nUpdated {EPA_PATH} with Offensive EPA per play data")
    print(f"Total rows: {len(main_df)}")
    print(f"Rows with Offensive EPA per play: {main_df['Offensive EPA per play'].notna().sum()}")


if __name__ == "__main__":
    main()
