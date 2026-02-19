"""
Manually add 2025 SumerSports offensive EPA data.

This script adds the 2025 data that was scraped from SumerSports.
You can manually add other years (2022-2024) by editing this script or running
the scraper when you have network access.

Usage (from repo root):
    python -m backend.ML.scheme.add_2025_sumer_epa
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

EPA_PATH = Path(__file__).parent.parent / "nfl_epa.csv"

# 2025 data from SumerSports (manually extracted)
DATA_2025 = [
    {"Team": "Patriots", "EPA/Play": 0.13, "Total EPA": 140.69, "Success Rate": 0.4693, "EPA/Pass": 0.29, "EPA/Rush": -0.03},
    {"Team": "Rams", "EPA/Play": 0.13, "Total EPA": 136.87, "Success Rate": 0.5046, "EPA/Pass": 0.23, "EPA/Rush": -0.01},
    {"Team": "Bills", "EPA/Play": 0.12, "Total EPA": 132.57, "Success Rate": 0.4769, "EPA/Pass": 0.17, "EPA/Rush": 0.08},
    {"Team": "Packers", "EPA/Play": 0.10, "Total EPA": 101.44, "Success Rate": 0.4673, "EPA/Pass": 0.23, "EPA/Rush": -0.03},
    {"Team": "Cowboys", "EPA/Play": 0.08, "Total EPA": 94.83, "Success Rate": 0.4598, "EPA/Pass": 0.16, "EPA/Rush": -0.02},
    {"Team": "Lions", "EPA/Play": 0.07, "Total EPA": 78.79, "Success Rate": 0.4468, "EPA/Pass": 0.17, "EPA/Rush": -0.06},
    {"Team": "Bears", "EPA/Play": 0.07, "Total EPA": 80.50, "Success Rate": 0.4524, "EPA/Pass": 0.08, "EPA/Rush": 0.06},
    {"Team": "49ers", "EPA/Play": 0.07, "Total EPA": 77.67, "Success Rate": 0.4741, "EPA/Pass": 0.16, "EPA/Rush": -0.04},
    {"Team": "Colts", "EPA/Play": 0.07, "Total EPA": 69.55, "Success Rate": 0.4597, "EPA/Pass": 0.06, "EPA/Rush": 0.07},
    {"Team": "Broncos", "EPA/Play": 0.04, "Total EPA": 42.38, "Success Rate": 0.4341, "EPA/Pass": 0.08, "EPA/Rush": -0.02},
    {"Team": "Chiefs", "EPA/Play": 0.03, "Total EPA": 32.20, "Success Rate": 0.4373, "EPA/Pass": 0.04, "EPA/Rush": 0.02},
    {"Team": "Jaguars", "EPA/Play": 0.03, "Total EPA": 30.46, "Success Rate": 0.4373, "EPA/Pass": 0.06, "EPA/Rush": -0.01},
    {"Team": "Ravens", "EPA/Play": 0.03, "Total EPA": 25.36, "Success Rate": 0.4374, "EPA/Pass": -0.04, "EPA/Rush": 0.08},
    {"Team": "Steelers", "EPA/Play": 0.02, "Total EPA": 20.22, "Success Rate": 0.4342, "EPA/Pass": 0.03, "EPA/Rush": 0.01},
    {"Team": "Seahawks", "EPA/Play": 0.02, "Total EPA": 16.57, "Success Rate": 0.4532, "EPA/Pass": 0.11, "EPA/Rush": -0.07},
    {"Team": "Eagles", "EPA/Play": 0.01, "Total EPA": 12.98, "Success Rate": 0.4238, "EPA/Pass": 0.05, "EPA/Rush": -0.02},
    {"Team": "Giants", "EPA/Play": 0.01, "Total EPA": 7.44, "Success Rate": 0.4157, "EPA/Pass": -0.01, "EPA/Rush": 0.03},
    {"Team": "Commanders", "EPA/Play": 0.0, "Total EPA": -0.01, "Success Rate": 0.4505, "EPA/Pass": -0.01, "EPA/Rush": 0.01},
    {"Team": "Bengals", "EPA/Play": -0.01, "Total EPA": -10.46, "Success Rate": 0.4513, "EPA/Pass": -0.03, "EPA/Rush": 0.02},
    {"Team": "Buccaneers", "EPA/Play": -0.01, "Total EPA": -11.61, "Success Rate": 0.4213, "EPA/Pass": -0.02, "EPA/Rush": 0.01},
    {"Team": "Cardinals", "EPA/Play": -0.02, "Total EPA": -17.75, "Success Rate": 0.4302, "EPA/Pass": 0.0, "EPA/Rush": -0.06},
    {"Team": "Dolphins", "EPA/Play": -0.02, "Total EPA": -16.04, "Success Rate": 0.4173, "EPA/Pass": -0.01, "EPA/Rush": -0.03},
    {"Team": "Texans", "EPA/Play": -0.02, "Total EPA": -19.33, "Success Rate": 0.4090, "EPA/Pass": 0.04, "EPA/Rush": -0.09},
    {"Team": "Falcons", "EPA/Play": -0.02, "Total EPA": -26.02, "Success Rate": 0.4323, "EPA/Pass": -0.01, "EPA/Rush": -0.04},
    {"Team": "Chargers", "EPA/Play": -0.04, "Total EPA": -42.33, "Success Rate": 0.4247, "EPA/Pass": -0.06, "EPA/Rush": -0.02},
    {"Team": "Panthers", "EPA/Play": -0.04, "Total EPA": -44.31, "Success Rate": 0.4243, "EPA/Pass": -0.05, "EPA/Rush": -0.04},
    {"Team": "Saints", "EPA/Play": -0.09, "Total EPA": -98.53, "Success Rate": 0.4149, "EPA/Pass": -0.08, "EPA/Rush": -0.12},
    {"Team": "Vikings", "EPA/Play": -0.12, "Total EPA": -116.28, "Success Rate": 0.4193, "EPA/Pass": -0.19, "EPA/Rush": -0.03},
    {"Team": "Jets", "EPA/Play": -0.13, "Total EPA": -136.22, "Success Rate": 0.4010, "EPA/Pass": -0.19, "EPA/Rush": -0.07},
    {"Team": "Titans", "EPA/Play": -0.16, "Total EPA": -162.39, "Success Rate": 0.3715, "EPA/Pass": -0.22, "EPA/Rush": -0.07},
    {"Team": "Browns", "EPA/Play": -0.19, "Total EPA": -195.47, "Success Rate": 0.3447, "EPA/Pass": -0.28, "EPA/Rush": -0.05},
    {"Team": "Raiders", "EPA/Play": -0.21, "Total EPA": -203.57, "Success Rate": 0.3734, "EPA/Pass": -0.19, "EPA/Rush": -0.26},
]


def main():
    # Load existing EPA file
    if EPA_PATH.exists():
        df = pd.read_csv(EPA_PATH)
        print(f"Loaded existing EPA file with {len(df)} rows")
    else:
        df = pd.DataFrame(columns=["Year", "Team", "Net EPA"])
        print("Creating new EPA file")
    
    # Add 2025 data
    year = 2025
    for row_data in DATA_2025:
        team = row_data["Team"]
        mask = (df["Year"] == year) & (df["Team"] == team)
        
        if mask.any():
            # Update existing row
            idx = df[mask].index[0]
            for col in ["EPA/Play", "Total EPA", "Success Rate", "EPA/Pass", "EPA/Rush"]:
                df.loc[idx, col] = row_data[col]
            # Also update "Offensive EPA per play" if it matches "EPA/Play"
            if "Offensive EPA per play" in df.columns:
                df.loc[idx, "Offensive EPA per play"] = row_data["EPA/Play"]
        else:
            # Add new row
            new_row = {
                "Year": year,
                "Team": team,
                "EPA/Play": row_data["EPA/Play"],
                "Total EPA": row_data["Total EPA"],
                "Success Rate": row_data["Success Rate"],
                "EPA/Pass": row_data["EPA/Pass"],
                "EPA/Rush": row_data["EPA/Rush"],
            }
            if "Net EPA" in df.columns:
                new_row["Net EPA"] = None
            if "Offensive EPA per play" in df.columns:
                new_row["Offensive EPA per play"] = row_data["EPA/Play"]
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Sort and save
    df = df.sort_values(["Year", "EPA/Play"], ascending=[True, False], na_position="last")
    df.to_csv(EPA_PATH, index=False)
    print(f"\n✓ Updated {EPA_PATH}")
    print(f"  Total rows: {len(df)}")
    print(f"  2025 rows with EPA/Play: {len(df[(df['Year'] == 2025) & (df['EPA/Play'].notna())])}")


if __name__ == "__main__":
    main()
