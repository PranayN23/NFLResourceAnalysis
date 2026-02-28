"""
team_epa_builder.py
Build a team-season dataset for the Net EPA / Win % prediction model.

Sources (in priority):
  1. nflpowerrankings.csv  — split off/def EPA, QB adj, nfelo, wins, Pythag
  2. TeamTendencies.csv    — scheme: PROE, aDOT, pass rate, shotgun, formations, TOP
  3. nfl_epa.csv           — Net EPA (off-def) ground truth
  4. nfl_win.csv           — Win % target
  5. Position PFF CSVs     — mean grade / position group per team-year (supplementary)
"""

import os
import re
import pandas as pd
import numpy as np

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
ML_DIR = os.path.join(ROOT, "backend", "ML")
DATA_DIR = os.path.join(ML_DIR, "data")
OUT_DIR = os.path.dirname(__file__)
os.makedirs(OUT_DIR, exist_ok=True)

# ── Team name normalisation map (abbr → full name used in nfl_epa / nfl_win) ──
ABBR_TO_NAME = {
    "ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills",
    "CAR": "Panthers", "CHI": "Bears", "CIN": "Bengals", "CLE": "Browns",
    "DAL": "Cowboys", "DEN": "Broncos", "DET": "Lions", "GB":  "Packers",
    "HOU": "Texans",  "IND": "Colts",  "JAX": "Jaguars", "KC":  "Chiefs",
    "LAC": "Chargers","LAR": "Rams",   "MIA": "Dolphins", "MIN": "Vikings",
    "NE":  "Patriots","NO":  "Saints", "NYG": "Giants",  "NYJ": "Jets",
    "OAK": "Raiders", "PHI": "Eagles", "PIT": "Steelers","SEA": "Seahawks",
    "SF":  "49ers",   "TB":  "Buccaneers","TEN": "Titans","WAS": "Commanders",
}

NAME_TO_ABBR = {v: k for k, v in ABBR_TO_NAME.items()}
# Alias: Raiders moved from Oakland
NAME_TO_ABBR["Raiders"] = "OAK"

# ── 1. Load nflpowerrankings.csv ─────────────────────────────────────────────
def load_power_rankings():
    path = os.path.join(DATA_DIR, "nflpowerrankings.csv")
    df = pd.read_csv(path)

    # The CSV has duplicate column names (Play, Pass, Rush appear twice)
    # pandas will rename them Play, Pass, Rush, Play.1, Pass.1, Rush.1
    rename = {
        "Team":    "abbr",
        "Season":  "year",
        "nfelo":   "nfelo",
        "QB Adj":  "qb_adj",
        "Value":   "value",
        "WoW":     "week_over_week",
        "YTD":     "ytd_value",
        "Play":    "off_epa",
        "Pass":    "off_pass_epa",
        "Rush":    "off_rush_epa",
        "Play.1":  "def_epa",
        "Pass.1":  "def_pass_epa",
        "Rush.1":  "def_rush_epa",
        "Dif":     "point_diff",
        "Wins":    "wins",
        "Pythag":  "pythag_wins",
        "Elo":     "elo",
        "Film":    "film",
    }
    df = df.rename(columns=rename)
    df["year"] = df["year"].astype(int)

    # Map abbr → team name
    df["team"] = df["abbr"].map(ABBR_TO_NAME)

    keep = ["team", "abbr", "year", "nfelo", "qb_adj", "value",
            "off_epa", "off_pass_epa", "off_rush_epa",
            "def_epa", "def_pass_epa", "def_rush_epa",
            "point_diff", "wins", "pythag_wins", "film"]
    df = df[keep].copy()
    df = df.dropna(subset=["team"])
    return df


# ── 2. Load TeamTendencies.csv ───────────────────────────────────────────────
def load_tendencies():
    path = os.path.join(DATA_DIR, "TeamTendencies.csv")
    df = pd.read_csv(path)
    df = df.rename(columns={"Team": "abbr", "Season": "year"})
    df["year"] = df["year"].astype(int)

    # Parse TOP (time of possession) "MM:SS" → decimal minutes
    if "TOP" in df.columns:
        def parse_top(v):
            try:
                parts = str(v).split(":")
                return int(parts[0]) + int(parts[1]) / 60
            except Exception:
                return np.nan
        df["top_min"] = df["TOP"].apply(parse_top)

    keep = ["abbr", "year", "Pass Rate", "Rush Rate", "PROE", "aDOT",
            "Shotgun", "No Huddle", "11", "12", "top_min"]
    # Use only columns that exist
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    df = df.rename(columns={
        "Pass Rate": "pass_rate",
        "Rush Rate": "rush_rate",
        "aDOT":      "adot",
        "PROE":      "proe",
        "Shotgun":   "shotgun_pct",
        "No Huddle": "no_huddle_pct",
        "11":        "personnel_11_pct",
        "12":        "personnel_12_pct",
    })
    return df


# ── 3. Load nfl_epa.csv ──────────────────────────────────────────────────────
def load_net_epa():
    path = os.path.join(ML_DIR, "nfl_epa.csv")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Year": "year", "Team": "team_raw", "Net EPA": "net_epa"})
    df["year"] = df["year"].astype(int)
    df["team_raw"] = df["team_raw"].str.strip()
    df["abbr"] = df["team_raw"].map(NAME_TO_ABBR)
    return df[["abbr", "year", "net_epa"]].dropna(subset=["abbr"])


# ── 4. Load nfl_win.csv ──────────────────────────────────────────────────────
def load_win_pct():
    path = os.path.join(ML_DIR, "nfl_win.csv")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Team": "team_raw", "Win %": "win_pct", "Year": "year"})
    df["year"] = df["year"].astype(int)
    df["team_raw"] = df["team_raw"].str.strip()
    df["abbr"] = df["team_raw"].map(NAME_TO_ABBR)

    # Parse win pct: "82.40%" → 0.824
    def parse_pct(v):
        try:
            return float(str(v).replace("%", "")) / 100.0
        except Exception:
            return np.nan
    df["win_pct"] = df["win_pct"].apply(parse_pct)
    return df[["abbr", "year", "win_pct"]].dropna(subset=["abbr"])


# ── 4b. Load Granular EPA (Success Rates) ────────────────────────────────────
def load_granular_epa():
    frames = []
    for year in range(2010, 2026):
        path = os.path.join(DATA_DIR, f"EPA {year}.csv")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                # Columns: Team, Abbr, EPA/play, Success Rate (SR), Dropback EPA, Dropback SR, Rush EPA, Rush SR, Net
                
                # Parse percentages
                for col in ["Success Rate (SR)", "Dropback SR", "Rush SR"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r"%", "", regex=True), errors="coerce") / 100.0
                
                # Team abbreviation matching
                if "Abbr" in df.columns:
                    df["abbr"] = df["Abbr"].str.strip()
                elif "Team" in df.columns:
                    df["abbr"] = df["Team"].str.strip().map(NAME_TO_ABBR).fillna(df["Team"].str.strip())
                else:
                    continue
                    
                df["year"] = int(year)
                
                keep = ["abbr", "year"]
                rename_dict = {}
                if "Success Rate (SR)" in df.columns:
                    keep.append("Success Rate (SR)")
                    rename_dict["Success Rate (SR)"] = "success_rate"
                if "Dropback SR" in df.columns:
                    keep.append("Dropback SR")
                    rename_dict["Dropback SR"] = "dropback_sr"
                if "Rush SR" in df.columns:
                    keep.append("Rush SR")
                    rename_dict["Rush SR"] = "rush_sr"
                
                df = df[keep].rename(columns=rename_dict)
                frames.append(df)
            except Exception as e:
                print(f"  [warn] Could not load EPS {year}.csv: {e}")
                
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=["abbr", "year", "success_rate", "dropback_sr", "rush_sr"])


# ── 5. Load position PFF grades (supplementary) ──────────────────────────────
POSITION_FILES = {
    "qb_grade":   ("QB.csv", ["player", "Team", "Year", "grades_pass", "grades_offense"]),
    "rb_grade":   ("HB.csv", ["player", "Team", "Year", "grades_offense", "grades_run"]),
    "wr_grade":   ("WR.csv", ["player", "Team", "Year", "grades_offense", "grades_pass_route"]),
    "te_grade":   ("TightEnds", None),   # handled separately
    "ol_grade":   ("OL_Pranay_Transformers", None),
    "edge_grade": ("ED.csv", ["player", "Team", "Year", "grades_defense", "grades_pass_rush_defense"]),
    "idl_grade":  ("DI.csv", ["player", "Team", "Year", "grades_defense", "grades_run_defense"]),
    "lb_grade":   ("LB.csv", ["player", "Team", "Year", "grades_defense", "grades_coverage_defense"]),
    "cb_grade":   ("CB.csv", ["player", "Team", "Year", "grades_coverage_defense", "grades_defense"]),
    "s_grade":    ("S.csv",  ["player", "Team", "Year", "grades_coverage_defense", "grades_defense"]),
}

def load_position_grades():
    """
    For each position, compute mean primary grade per Team-Year.
    Returns a wide DataFrame indexed by (abbr, year).
    """
    frames = []

    simple_positions = {
        "qb_grade":   (os.path.join(ML_DIR, "QB.csv"), "grades_offense"),
        "rb_grade":   (os.path.join(ML_DIR, "HB.csv"), "grades_offense"),
        "wr_grade":   (os.path.join(ML_DIR, "WR.csv"), "grades_offense"),
        "edge_grade": (os.path.join(ML_DIR, "ED.csv"), "grades_defense"),
        "idl_grade":  (os.path.join(ML_DIR, "DI.csv"), "grades_defense"),
        "lb_grade":   (os.path.join(ML_DIR, "LB.csv"), "grades_defense"),
        "cb_grade":   (os.path.join(ML_DIR, "CB.csv"), "grades_defense"),
        "s_grade":    (os.path.join(ML_DIR, "S.csv"),  "grades_defense"),
    }

    for grade_col, (fpath, grade_key) in simple_positions.items():
        if not os.path.exists(fpath):
            continue
        try:
            df = pd.read_csv(fpath, low_memory=False)
            # Normalise column names
            df.columns = [c.strip() for c in df.columns]
            team_col = next((c for c in df.columns if c.lower() in ("team", "team_name")), None)
            year_col = next((c for c in df.columns if c.lower() in ("year", "season")), None)
            if not team_col or not year_col or grade_key not in df.columns:
                continue
            df = df.rename(columns={team_col: "team_raw", year_col: "year"})
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df[grade_key] = pd.to_numeric(df[grade_key], errors="coerce")
            df = df.dropna(subset=["team_raw", "year", grade_key])
            agg = df.groupby(["team_raw", "year"])[grade_key].mean().reset_index()
            agg["abbr"] = agg["team_raw"].str.strip().map(NAME_TO_ABBR)
            agg = agg.dropna(subset=["abbr"])
            agg["year"] = agg["year"].astype(int)
            agg = agg[["abbr", "year", grade_key]].rename(columns={grade_key: grade_col})
            frames.append(agg)
        except Exception as e:
            print(f"  [warn] Could not load {fpath}: {e}")

    # OL: use OLPFF.csv
    ol_path = os.path.join(ML_DIR, "OLPFF.csv")
    if os.path.exists(ol_path):
        try:
            df = pd.read_csv(ol_path, low_memory=False)
            df.columns = [c.strip() for c in df.columns]
            team_col = next((c for c in df.columns if c.lower() in ("team", "team_name")), None)
            year_col = next((c for c in df.columns if c.lower() in ("year", "season")), None)
            grade_key = next((c for c in df.columns if "grades_offense" in c.lower()), None)
            if team_col and year_col and grade_key:
                df = df.rename(columns={team_col: "team_raw", year_col: "year"})
                df["year"] = pd.to_numeric(df["year"], errors="coerce")
                df[grade_key] = pd.to_numeric(df[grade_key], errors="coerce")
                df = df.dropna(subset=["team_raw", "year", grade_key])
                agg = df.groupby(["team_raw", "year"])[grade_key].mean().reset_index()
                agg["abbr"] = agg["team_raw"].str.strip().map(NAME_TO_ABBR)
                agg = agg.dropna(subset=["abbr"])
                agg["year"] = agg["year"].astype(int)
                agg = agg[["abbr", "year", grade_key]].rename(columns={grade_key: "ol_grade"})
                frames.append(agg)
        except Exception as e:
            print(f"  [warn] Could not load OL grades: {e}")

    if not frames:
        return pd.DataFrame(columns=["abbr", "year"])

    # Merge all position frames into one wide table
    from functools import reduce
    merged = reduce(lambda a, b: pd.merge(a, b, on=["abbr", "year"], how="outer"), frames)
    return merged


# ── 6. Build lagged features ──────────────────────────────────────────────────
def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["abbr", "year"]).copy()

    lag_cols = [
        "off_epa", "off_pass_epa", "off_rush_epa",
        "def_epa", "def_pass_epa", "def_rush_epa",
        "net_epa", "wins", "win_pct", "nfelo", "qb_adj",
        "pythag_wins",
        "pass_rate", "proe", "adot", "shotgun_pct",
        "qb_grade", "ol_grade", "rb_grade", "wr_grade",
        "edge_grade", "idl_grade", "lb_grade", "cb_grade", "s_grade",
        "success_rate", "dropback_sr", "rush_sr"
    ]

    for col in lag_cols:
        if col in df.columns:
            df[f"lag_{col}"] = df.groupby("abbr")[col].shift(1)

    # 2-year EPA trend: (year-1 EPA) - (year-2 EPA)
    if "net_epa" in df.columns:
        df["lag2_net_epa"] = df.groupby("abbr")["net_epa"].shift(2)
        df["epa_trend_2yr"] = df["lag_net_epa"] - df["lag2_net_epa"]

    if "nfelo" in df.columns:
        df["lag2_nfelo"] = df.groupby("abbr")["nfelo"].shift(2)
        df["nfelo_trend"] = df["lag_nfelo"] - df["lag2_nfelo"]

    return df


# ── 7. Create prediction targets ──────────────────────────────────────────────
def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["abbr", "year"]).copy()

    if "net_epa" in df.columns:
        df["next_net_epa"] = df.groupby("abbr")["net_epa"].shift(-1)
    if "win_pct" in df.columns:
        df["next_win_pct"] = df.groupby("abbr")["win_pct"].shift(-1)
    if "wins" in df.columns:
        df["next_wins"] = df.groupby("abbr")["wins"].shift(-1)

    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def build():
    print("Loading nflpowerrankings.csv …")
    base = load_power_rankings()
    print(f"  {len(base)} rows, {base['year'].min()}–{base['year'].max()}")

    print("Loading TeamTendencies.csv …")
    tend = load_tendencies()
    print(f"  {len(tend)} rows, {tend['year'].min()}–{tend['year'].max()}")

    print("Loading nfl_epa.csv …")
    epa = load_net_epa()
    print(f"  {len(epa)} rows")

    print("Loading nfl_win.csv …")
    win = load_win_pct()
    print(f"  {len(win)} rows")

    print("Loading position PFF grades …")
    grades = load_position_grades()
    print(f"  {len(grades)} team-year rows with position grades")

    print("Loading granular EPA (Success Rates) …")
    granular_epa = load_granular_epa()
    print(f"  {len(granular_epa)} team-year rows with granular EPA")

    # Merge
    print("\nMerging …")
    df = pd.merge(base, tend, on=["abbr", "year"], how="left")
    df = pd.merge(df, epa,   on=["abbr", "year"], how="left")
    df = pd.merge(df, win,   on=["abbr", "year"], how="left")
    if not grades.empty:
        df = pd.merge(df, grades, on=["abbr", "year"], how="left")
    if not granular_epa.empty:
        df = pd.merge(df, granular_epa, on=["abbr", "year"], how="left")

    print(f"  Pre-lag shape: {df.shape}")

    # Lags + targets
    df = add_lags(df)
    df = add_targets(df)

    # Drop rows with no lag data (first year per team) or no targets (last year)
    df = df.dropna(subset=["lag_off_epa", "lag_net_epa"])

    out_path = os.path.join(OUT_DIR, "team_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"\n✅ Saved {len(df)} team-season rows → {out_path}")

    # Quick summary
    target_coverage = df[["next_net_epa", "next_win_pct"]].notna().sum()
    print(f"   next_net_epa coverage : {target_coverage['next_net_epa']}")
    print(f"   next_win_pct coverage : {target_coverage['next_win_pct']}")
    print(f"   Year range            : {df['year'].min()}–{df['year'].max()}")
    print(f"   Teams                 : {df['abbr'].nunique()}")

    return df


if __name__ == "__main__":
    build()
