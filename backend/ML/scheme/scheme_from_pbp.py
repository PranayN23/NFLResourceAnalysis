"""
Build offensive scheme metrics from nflfastR/nfl_data_py play-by-play data.
Computes motion_rate, play_action_rate, shotgun_rate, under_center_rate, no_huddle_rate
(and optional PROE/personnel) by team-season for use in clustering and 10-year analysis.
"""
from pathlib import Path

import pandas as pd

try:
    from .config import SCHEME_DATA_DIR
except ImportError:
    import os
    SCHEME_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Prefer nfl_data_py (already used in this repo); fallback to nflreadpy
try:
    import nfl_data_py as nfl
    _HAS_NFL_DATA = True
except ImportError:
    nfl = None
    _HAS_NFL_DATA = False

try:
    import nflreadpy
    _HAS_NFLREADPY = True
except ImportError:
    nflreadpy = None
    _HAS_NFLREADPY = False


def load_pbp(seasons: list[int]):
    """Load play-by-play for given seasons; use nfl_data_py or nflreadpy."""
    if _HAS_NFL_DATA:
        df = nfl.import_pbp_data(seasons)
        return pd.DataFrame(df) if hasattr(df, "to_pandas") else df
    if _HAS_NFLREADPY:
        df = nflreadpy.load_pbp(seasons=seasons)
        if hasattr(df, "to_pandas"):
            return df.to_pandas()
        return pd.DataFrame(df)
    raise ImportError("Install nfl_data_py or nflreadpy: pip install nfl_data_py (or nflreadpy)")


def _safe_agg(series: pd.Series) -> float:
    """Mean of numeric series, or 0 if empty/invalid."""
    if series is None or series.empty:
        return 0.0
    return float(series.mean())


def build_scheme_metrics_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate PBP to team-season scheme metrics.
    pbp must have: posteam, season, and (if present) shotgun, no_huddle, play_type, desc.
    """
    # Filter to offensive plays (we need posteam and season)
    needed = ["posteam", "season"]
    if not all(c in pbp.columns for c in needed):
        raise ValueError(f"PBP must have columns {needed}. Got: {list(pbp.columns)}")
    # Drop rows with no posteam (e.g. special teams or missing)
    pbp = pbp.loc[pbp["posteam"].notna() & (pbp["posteam"] != "")].copy()

    # Shotgun rate: column 'shotgun' 0/1, or infer from formation/desc if missing
    if "shotgun" in pbp.columns:
        pbp["_shotgun"] = pbp["shotgun"].fillna(0).astype(float)
    else:
        pbp["_shotgun"] = 0.0

    # No huddle
    if "no_huddle" in pbp.columns:
        pbp["_no_huddle"] = pbp["no_huddle"].fillna(0).astype(float)
    else:
        pbp["_no_huddle"] = 0.0

    # Play action: often in 'desc' as "play action" or a dedicated column
    if "pass_attempt" in pbp.columns and "play_action" in pbp.columns:
        pbp["_play_action_pass"] = ((pbp["pass_attempt"] == 1) & (pbp["play_action"] == 1)).astype(float)
    elif "play_action" in pbp.columns:
        pbp["_play_action_pass"] = pbp["play_action"].fillna(0).astype(float)
    else:
        desc = pbp.get("desc", pd.Series(dtype=object))
        pbp["_play_action_pass"] = desc.astype(str).str.lower().str.contains("play action", na=False).astype(float)

    # Pass attempts (for play action rate = play action passes / pass attempts)
    if "pass_attempt" in pbp.columns:
        pbp["_pass_attempt"] = pbp["pass_attempt"].fillna(0)
    else:
        pbp["_pass_attempt"] = (pbp.get("play_type", pd.Series(dtype=object)) == "pass").astype(float)

    # Air yards per attempt (for passes)
    if "air_yards" in pbp.columns:
        pbp["_air_yards"] = pbp["air_yards"].fillna(0)
    else:
        pbp["_air_yards"] = 0.0

    grp = pbp.groupby(["posteam", "season"], as_index=False)

    # Rates as percentages
    plays = grp.size().reset_index(name="n_plays")
    shotgun_rate = grp["_shotgun"].mean().reset_index().rename(columns={"_shotgun": "shotgun_rate"})
    no_huddle_rate = grp["_no_huddle"].mean().reset_index().rename(columns={"_no_huddle": "no_huddle_rate"})
    shotgun_rate["shotgun_rate"] *= 100
    no_huddle_rate["no_huddle_rate"] *= 100

    pass_attempts = grp["_pass_attempt"].sum().reset_index().rename(columns={"_pass_attempt": "pass_attempts"})
    pa_passes = grp["_play_action_pass"].sum().reset_index().rename(columns={"_play_action_pass": "play_action_passes"})
    merge_pa = pass_attempts.merge(pa_passes, on=["posteam", "season"])
    merge_pa["play_action_rate"] = (merge_pa["play_action_passes"] / merge_pa["pass_attempts"].replace(0, float("nan")) * 100).fillna(0)

    passes = pbp.loc[pbp["_pass_attempt"] == 1]
    air = (
        passes.groupby(["posteam", "season"], as_index=False)["_air_yards"]
        .mean()
        .rename(columns={"_air_yards": "air_yards_per_att"})
    )
    air["air_yards_per_att"] = air["air_yards_per_att"].fillna(0)

    # Motion: nflfastR has 'motion' or similar in some years
    if "pre_snap_motion" in pbp.columns or "motion" in pbp.columns:
        motion_col = "pre_snap_motion" if "pre_snap_motion" in pbp.columns else "motion"
        motion_rate = grp[motion_col].mean().reset_index().rename(columns={motion_col: "motion_rate"})
        motion_rate["motion_rate"] *= 100
    else:
        motion_rate = shotgun_rate[["posteam", "season"]].copy()
        motion_rate["motion_rate"] = float("nan")  # not available in older PBP

    out = plays.merge(shotgun_rate, on=["posteam", "season"])
    out = out.merge(no_huddle_rate, on=["posteam", "season"])
    out = out.merge(merge_pa[["posteam", "season", "play_action_rate"]], on=["posteam", "season"])
    out = out.merge(air, on=["posteam", "season"])
    out = out.merge(motion_rate, on=["posteam", "season"])
    out["under_center_rate"] = 100.0 - out["shotgun_rate"]
    out = out.rename(columns={"posteam": "team_abbr"})
    return out


def build_and_save_pbp_scheme(seasons: list[int] | None = None, output_path: str | Path | None = None) -> Path:
    """
    Load PBP for seasons (default 2015–current year), build scheme metrics, save CSV.
    """
    if seasons is None:
        from datetime import datetime
        current = datetime.now().year
        seasons = list(range(2015, current + 1))
    pbp = load_pbp(seasons)
    df = build_scheme_metrics_from_pbp(pbp)
    out_path = Path(output_path or SCHEME_DATA_DIR) / "scheme_from_pbp.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    import sys
    years = list(map(int, sys.argv[1:])) if len(sys.argv) > 1 else None
    path = build_and_save_pbp_scheme(seasons=years)
    print(f"Saved PBP scheme metrics to {path}")
