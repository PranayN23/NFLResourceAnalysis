"""
Scrape offensive EPA data from SumerSports for 2022-2025.

Scrapes: EPA/Play, Total EPA, Success Rate, EPA/Pass, EPA/Rush
from https://sumersports.com/teams/offensive/

Usage (from repo root):
    python -m backend.ML.scheme.scrape_sumer_offensive_epa
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Path to main EPA CSV
EPA_PATH = Path(__file__).parent.parent / "nfl_epa.csv"

# Team abbreviation mapping (SumerSports uses various formats)
SUMER_TO_STANDARD = {
    "NE": "Patriots",
    "LA": "Rams",  # SumerSports uses LA for Rams
    "LAR": "Rams",
    "BUF": "Bills",
    "GB": "Packers",
    "DAL": "Cowboys",
    "DET": "Lions",
    "CHI": "Bears",
    "SF": "49ers",
    "IND": "Colts",
    "DEN": "Broncos",
    "KC": "Chiefs",
    "JAX": "Jaguars",
    "BLT": "Ravens",  # SumerSports uses BLT
    "BAL": "Ravens",
    "PIT": "Steelers",
    "SEA": "Seahawks",
    "PHI": "Eagles",
    "NYG": "Giants",
    "WAS": "Commanders",
    "CIN": "Bengals",
    "TB": "Buccaneers",
    "ARZ": "Cardinals",  # SumerSports uses ARZ
    "ARI": "Cardinals",
    "MIA": "Dolphins",
    "HST": "Texans",  # SumerSports uses HST
    "HOU": "Texans",
    "ATL": "Falcons",
    "LAC": "Chargers",
    "CAR": "Panthers",
    "NO": "Saints",
    "MIN": "Vikings",
    "NYJ": "Jets",
    "TEN": "Titans",
    "CLV": "Browns",  # SumerSports uses CLV
    "CLE": "Browns",
    "LV": "Raiders",
}


def parse_team_name(team_text: str) -> str | None:
    """Parse team name from SumerSports format (e.g., '1.New England Patriots' or 'Los Angeles Rams')."""
    # Remove leading number and period if present
    team_text = re.sub(r'^\d+\.\s*', '', team_text).strip()
    
    # Map to standard team names
    team_mapping = {
        "New England Patriots": "Patriots",
        "Los Angeles Rams": "Rams",
        "Buffalo Bills": "Bills",
        "Green Bay Packers": "Packers",
        "Dallas Cowboys": "Cowboys",
        "Detroit Lions": "Lions",
        "Chicago Bears": "Bears",
        "San Francisco 49ers": "49ers",
        "Indianapolis Colts": "Colts",
        "Denver Broncos": "Broncos",
        "Kansas City Chiefs": "Chiefs",
        "Jacksonville Jaguars": "Jaguars",
        "Baltimore Ravens": "Ravens",
        "Pittsburgh Steelers": "Steelers",
        "Seattle Seahawks": "Seahawks",
        "Philadelphia Eagles": "Eagles",
        "New York Giants": "Giants",
        "Washington Commanders": "Commanders",
        "Cincinnati Bengals": "Bengals",
        "Tampa Bay Buccaneers": "Buccaneers",
        "Arizona Cardinals": "Cardinals",
        "Miami Dolphins": "Dolphins",
        "Houston Texans": "Texans",
        "Atlanta Falcons": "Falcons",
        "Los Angeles Chargers": "Chargers",
        "Carolina Panthers": "Panthers",
        "New Orleans Saints": "Saints",
        "Minnesota Vikings": "Vikings",
        "New York Jets": "Jets",
        "Tennessee Titans": "Titans",
        "Cleveland Browns": "Browns",
        "Las Vegas Raiders": "Raiders",
    }
    
    return team_mapping.get(team_text)


def scrape_sumer_offensive_epa(year: int) -> pd.DataFrame:
    """
    Scrape offensive EPA data from SumerSports for a given year.
    
    Returns DataFrame with columns: Team, EPA/Play, Total EPA, Success Rate, EPA/Pass, EPA/Rush
    """
    url = f"https://sumersports.com/teams/offensive/?season={year}"
    print(f"Scraping {url}...")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except Exception as e:
        raise Exception(f"Failed to fetch {url}: {e}")
    
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Find the table with offensive stats
    table = soup.find("table")
    if not table:
        raise ValueError(f"No table found on {url}")
    
    rows = []
    for tr in table.find_all("tr")[1:]:  # Skip header row
        cells = tr.find_all(["td", "th"])
        if len(cells) < 6:
            continue
        
        # Extract team name from first cell
        team_text = cells[0].get_text(strip=True)
        team_name = parse_team_name(team_text)
        
        if not team_name:
            print(f"Warning: Could not parse team name from: {team_text}")
            continue
        
        # Extract stats (columns: Team, Season, EPA/Play, Total EPA, Success %, EPA/Pass, EPA/Rush, ...)
        try:
            # Column indices: 0=Team, 1=Season, 2=EPA/Play, 3=Total EPA, 4=Success %, 5=EPA/Pass, 6=EPA/Rush
            epa_per_play = float(cells[2].get_text(strip=True))
            total_epa = float(cells[3].get_text(strip=True))
            success_rate_str = cells[4].get_text(strip=True).replace("%", "")
            success_rate = float(success_rate_str) / 100.0
            epa_per_pass = float(cells[5].get_text(strip=True))
            epa_per_rush = float(cells[6].get_text(strip=True))
            
            rows.append({
                "Year": year,
                "Team": team_name,
                "EPA/Play": epa_per_play,
                "Total EPA": total_epa,
                "Success Rate": success_rate,
                "EPA/Pass": epa_per_pass,
                "EPA/Rush": epa_per_rush,
            })
        except (ValueError, IndexError) as e:
            print(f"Warning: Error parsing row for {team_name}: {e}")
            continue
    
    if not rows:
        raise ValueError(f"No data extracted from {url}")
    
    return pd.DataFrame(rows)


def main():
    years = [2022, 2023, 2024, 2025]
    all_data = []
    
    for year in years:
        try:
            df = scrape_sumer_offensive_epa(year)
            all_data.append(df)
            print(f"✓ Scraped {len(df)} teams for {year}")
        except Exception as e:
            print(f"✗ Failed to scrape {year}: {e}")
            continue
    
    if not all_data:
        print("No data scraped. Exiting.")
        return
    
    # Combine all years
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal rows scraped: {len(combined)}")
    
    # Load existing EPA file
    if EPA_PATH.exists():
        existing = pd.read_csv(EPA_PATH)
        print(f"Loaded existing EPA file with {len(existing)} rows")
        
        # Merge new data
        # For each year, update or add rows
        for _, row in combined.iterrows():
            mask = (existing["Year"] == row["Year"]) & (existing["Team"] == row["Team"])
            if mask.any():
                # Update existing row
                for col in ["EPA/Play", "Total EPA", "Success Rate", "EPA/Pass", "EPA/Rush"]:
                    if col in row:
                        existing.loc[mask, col] = row[col]
            else:
                # Add new row
                new_row = {"Year": row["Year"], "Team": row["Team"]}
                if "Net EPA" in existing.columns:
                    new_row["Net EPA"] = None
                if "Offensive EPA per play" in existing.columns:
                    new_row["Offensive EPA per play"] = row.get("EPA/Play", None)
                for col in ["EPA/Play", "Total EPA", "Success Rate", "EPA/Pass", "EPA/Rush"]:
                    new_row[col] = row.get(col, None)
                existing = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
        
        # Also update "Offensive EPA per play" if it matches "EPA/Play"
        if "EPA/Play" in existing.columns and "Offensive EPA per play" in existing.columns:
            mask = existing["Offensive EPA per play"].isna() & existing["EPA/Play"].notna()
            existing.loc[mask, "Offensive EPA per play"] = existing.loc[mask, "EPA/Play"]
        
        combined = existing
    else:
        # Create new file
        print("Creating new EPA file")
        combined = combined.copy()
        combined["Net EPA"] = None
    
    # Sort and save
    combined = combined.sort_values(["Year", "EPA/Play"], ascending=[True, False], na_position="last")
    combined.to_csv(EPA_PATH, index=False)
    print(f"\n✓ Updated {EPA_PATH}")
    print(f"  Total rows: {len(combined)}")
    print(f"  Years with EPA/Play: {sorted(combined[combined['EPA/Play'].notna()]['Year'].unique())}")


if __name__ == "__main__":
    main()
