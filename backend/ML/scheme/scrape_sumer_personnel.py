"""
Scrape offensive personnel tendency data from SumerSports and save to CSV.

This replaces the manual "export CSV" step for:
  https://sumersports.com/teams/offensive/personnel-tendency/

For each requested season it will:
- Download the personnel tendency table for that season
- Parse it into a DataFrame
- Write scheme/data/sumer_personnel_{season}.csv

Usage (from backend/ML):

  python -m scheme.scrape_sumer_personnel 2022 2023 2024 2025

After that, you can run:

  python -m scheme.merge_sumer_personnel 2022 2023 2024 2025
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    from .config import SCHEME_DATA_DIR
except ImportError:  # pragma: no cover
    import os

    SCHEME_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


SUMER_BASE_URL = "https://sumersports.com/teams/offensive/personnel-tendency/"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _parse_sumer_table(table) -> pd.DataFrame:
    """Parse the main SumerSports personnel table into a DataFrame."""
    # Headers
    headers: list[str] = []
    thead = table.find("thead")
    if thead:
        header_row = thead.find("tr")
        if header_row:
            for th in header_row.find_all(["th", "td"]):
                headers.append(th.get_text(strip=True))
    if not headers:
        first_tr = table.find("tr")
        if first_tr:
            for cell in first_tr.find_all(["th", "td"]):
                headers.append(cell.get_text(strip=True))

    rows = []
    tbody = table.find("tbody") or table
    for tr in tbody.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if not cells:
            continue
        # Some rows are ranking/number prefix; ignore if lengths don't match
        if len(cells) == len(headers):
            row = dict(zip(headers, cells))
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return _clean_sumer_df(df)


def _clean_sumer_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize SumerSports personnel table.

    We ensure it has at least:
      - Team
      - Season (int)
      - Personnel
      - Rate  (string/number; merge_sumer_personnel will normalize)
    """
    # Build a case-insensitive mapping from normalized name -> original
    norm_map = {c.lower().strip().replace(" ", "").replace(".", ""): c for c in df.columns}

    def col_like(target: str) -> str | None:
        key = target.lower().strip().replace(" ", "").replace(".", "")
        for k, orig in norm_map.items():
            if key == k or key in k or k in key:
                return orig
        return None

    team_col = col_like("team") or "Team"
    season_col = col_like("season") or "Season"
    pers_col = col_like("personnel") or "Personnel"
    rate_col = col_like("rate") or "Rate"

    rename = {}
    if team_col in df.columns:
        rename[team_col] = "Team"
    if season_col in df.columns:
        rename[season_col] = "Season"
    if pers_col in df.columns:
        rename[pers_col] = "Personnel"
    if rate_col in df.columns:
        rename[rate_col] = "Rate"

    df = df.rename(columns=rename)

    # Best-effort type conversion
    if "Season" in df.columns:
        df["Season"] = pd.to_numeric(df["Season"], errors="coerce").astype("Int64")

    return df


def scrape_sumer_personnel(season: int, session: requests.Session | None = None) -> pd.DataFrame:
    """
    Scrape SumerSports personnel tendency table for a given logical season.

    NOTE on year alignment:
      - As of early 2026, SumerSports labels the most recent season as 2025,
        while our Sharp-based scheme data uses 2026.
      - To align these, when you request season=2026 we actually fetch
        Sumer's 2025 page, but we keep the Season column as 2026 so it
        lines up with your 2026 scheme data.

    Returns a DataFrame with at least columns: Team, Season, Personnel, Rate.
    """
    session = session or _session()

    # Map our logical season to the Sumer query season.
    # For now we only special-case 2026 -> 2025 based on the site's labeling.
    sumer_season = 2025 if season == 2026 else season

    # Sumer supports a ?season=YYYY query parameter for this page
    url = f"{SUMER_BASE_URL}?season={sumer_season}"
    resp = session.get(url, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")
    if not table:
        raise ValueError(f"Could not find personnel table on Sumer page for season {season} ({url})")

    df = _parse_sumer_table(table)
    if df.empty:
        raise ValueError(f"Sumer personnel table for season {season} parsed as empty DataFrame")

    # Force the Season column to our logical season so downstream code
    # (like merge_sumer_personnel) can join on your scheme year.
    df["Season"] = season

    return df


def save_sumer_personnel(season: int) -> Path:
    """Scrape a season and save to scheme/data/sumer_personnel_{season}.csv."""
    df = scrape_sumer_personnel(season)
    out_path = Path(SCHEME_DATA_DIR) / f"sumer_personnel_{season}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def scrape_many_seasons(seasons: Iterable[int]) -> list[Path]:
    """Scrape and save multiple seasons; returns list of output paths."""
    paths: list[Path] = []
    for yr in seasons:
        paths.append(save_sumer_personnel(int(yr)))
    return paths


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python -m scheme.scrape_sumer_personnel 2022 2023 2024 2025"
        )

    years = [int(x) for x in sys.argv[1:]]
    outputs = scrape_many_seasons(years)
    for p in outputs:
        print(f"Saved Sumer personnel CSV to {p}")

