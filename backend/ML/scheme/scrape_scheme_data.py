"""
Scrape offensive scheme/tendency data from Sharp Football Analysis and (optionally) nfelo.
Outputs CSVs with motion rate, play action rate, shotgun rate, under center rate, no huddle rate,
and air yards per attempt for use in scheme clustering.
"""
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    from .config import SCHEME_DATA_DIR, SHARP_TEAM_TO_ABBR
except ImportError:
    from config import SCHEME_DATA_DIR, SHARP_TEAM_TO_ABBR

SHARP_URL = "https://www.sharpfootballanalysis.com/stats-nfl/nfl-offensive-tendencies-stats/"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _session():
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def scrape_sharp_football(session=None) -> pd.DataFrame:
    """
    Scrape NFL offensive tendencies from Sharp Football Analysis.
    Returns a DataFrame with: team_abbr, team_name, motion_rate, play_action_rate,
    air_yards_per_att, shotgun_rate, no_huddle_rate. under_center_rate = 100 - shotgun_rate.
    """
    session = session or _session()
    r = session.get(SHARP_URL, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Try to find a proper HTML table first
    table = soup.find("table")
    if table:
        return _parse_sharp_table(table)

    # Fallback: look for markdown-style table in page text (e.g. |Team|Motion Rate|...)
    text = soup.get_text()
    return _parse_sharp_text_table(text)


def _parse_sharp_table(table) -> pd.DataFrame:
    """Parse Sharp Football HTML table."""
    headers = []
    thead = table.find("thead")
    if thead:
        for th in thead.find_all(["th", "td"]):
            headers.append(th.get_text(strip=True).replace("\n", " "))
    if not headers:
        for cell in table.find("tr").find_all(["th", "td"]):
            headers.append(cell.get_text(strip=True))

    # Normalize header names to our schema
    header_map = {
        "team": "team_name",
        "motion rate": "motion_rate",
        "play action rate": "play_action_rate",
        "airyards/att": "air_yards_per_att",
        "air yards/att": "air_yards_per_att",
        "shotgun rate": "shotgun_rate",
        "nohuddle rate": "no_huddle_rate",
        "no huddle rate": "no_huddle_rate",
    }
    normalized = []
    for h in headers:
        key = h.lower().strip()
        normalized.append(header_map.get(key, h))

    rows = []
    tbody = table.find("tbody") or table
    for tr in tbody.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if len(cells) != len(normalized):
            continue
        row = dict(zip(normalized, cells))
        if row.get("team_name") and row["team_name"].strip():
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return _clean_sharp_df(df)


def _parse_sharp_text_table(text: str) -> pd.DataFrame:
    """Fallback: parse table from page text (pipe-separated or tab-separated lines)."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # Find a line that looks like header: Team, Motion, Play Action, ...
    data_lines = []
    for i, line in enumerate(lines):
        # Match rows that look like "Rams|63.3|21.1|8.3|39.8|7.4" or "Rams  63.3  21.1 ..."
        if "|" in line:
            parts = [p.strip() for p in line.split("|")]
        else:
            parts = line.split()
        if len(parts) >= 5 and parts[0] in SHARP_TEAM_TO_ABBR:
            # First column is team name, rest are numbers
            try:
                nums = [float(re.sub(r"[^\d.-]", "", p)) for p in parts[1:6]]
                data_lines.append([parts[0]] + nums)
            except (ValueError, IndexError):
                continue

    if not data_lines:
        return pd.DataFrame()

    df = pd.DataFrame(
        data_lines,
        columns=[
            "team_name",
            "motion_rate",
            "play_action_rate",
            "air_yards_per_att",
            "shotgun_rate",
            "no_huddle_rate",
        ],
    )
    return _clean_sharp_df(df)


def _clean_sharp_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and types, add team_abbr and under_center_rate."""
    # Ensure we have expected columns (allow different casing from site)
    col_lower = {c.lower().replace(" ", "_"): c for c in df.columns}
    rename = {}
    for target in [
        "motion_rate",
        "play_action_rate",
        "air_yards_per_att",
        "shotgun_rate",
        "no_huddle_rate",
    ]:
        for k, v in col_lower.items():
            if target in k or k in target:
                rename[v] = target
                break
    if "team_name" not in df.columns and "team" in [c.lower() for c in df.columns]:
        rename[[c for c in df.columns if c.lower() == "team"][0]] = "team_name"
    df = df.rename(columns=rename)

    for col in ["motion_rate", "play_action_rate", "shotgun_rate", "no_huddle_rate", "air_yards_per_att"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")

    df["team_abbr"] = df["team_name"].map(SHARP_TEAM_TO_ABBR)
    # Drop rows that didn't map to a known team
    df = df.dropna(subset=["team_abbr"])
    df["under_center_rate"] = 100.0 - df["shotgun_rate"].fillna(0)
    return df


def save_sharp_scheme(season: int | None = None) -> Path:
    """
    Scrape Sharp Football and save to CSV.
    Sharp's main page typically shows current season; season is used only for filename.
    """
    df = scrape_sharp_football()
    if df.empty:
        raise ValueError("Sharp Football scrape returned no data")
    year = season or pd.Timestamp.now().year
    out_path = Path(SCHEME_DATA_DIR) / f"sharp_scheme_{year}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def load_scheme_csv(path: str | Path) -> pd.DataFrame:
    """Load a scheme CSV (Sharp or combined) for clustering."""
    return pd.read_csv(path)


if __name__ == "__main__":
    out = save_sharp_scheme()
    print(f"Saved Sharp Football scheme data to {out}")
