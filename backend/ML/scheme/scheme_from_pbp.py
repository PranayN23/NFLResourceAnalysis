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

# Directory for locally downloaded nflfastR pbp CSVs (e.g., pbp_2015.csv ... pbp_2025.csv)
RAW_PBP_DIR = Path(SCHEME_DATA_DIR).parent / "raw_pbp"
RAW_PBP_DIR.mkdir(parents=True, exist_ok=True)


def load_pbp(seasons: list[int]):
    """
    Load play-by-play for given seasons.

    Preference order:
    1. Local nflfastR-style CSVs in RAW_PBP_DIR named pbp_<year>.csv (these include play_action).
    2. Fallback to nfl_data_py or nflreadpy if local CSVs are missing (no reliable play_action/motion).

    This lets you get true play-action rates for seasons where you've downloaded the enriched pbp.
    """
    dfs: list[pd.DataFrame] = []
    missing_for_csv: list[int] = []

    for yr in seasons:
        local_path = RAW_PBP_DIR / f"pbp_{yr}.csv"
        if local_path.exists():
            df_local = pd.read_csv(local_path)
            dfs.append(df_local)
        else:
            missing_for_csv.append(yr)

    # Fallback for seasons without local CSVs
    if missing_for_csv:
        try:
            import nfl_data_py as nfl  # type: ignore
            df_fb = nfl.import_pbp_data(missing_for_csv)
            dfs.append(pd.DataFrame(df_fb) if hasattr(df_fb, "to_pandas") else df_fb)
        except ImportError:
            try:
                import nflreadpy  # type: ignore
                df_fb = nflreadpy.load_pbp(seasons=missing_for_csv)
                dfs.append(df_fb.to_pandas() if hasattr(df_fb, "to_pandas") else pd.DataFrame(df_fb))
            except ImportError:
                raise ImportError(
                    "No local pbp_YYYY.csv files found for seasons "
                    f"{missing_for_csv} and neither nfl_data_py nor nflreadpy are installed. "
                    "Download nflfastR PBP CSVs into scheme/raw_pbp or install one of those packages."
                )

    if not dfs:
        raise FileNotFoundError(
            f"No PBP data found for seasons {seasons}. "
            f"Expected local CSVs like {RAW_PBP_DIR}/pbp_YYYY.csv or a working nfl_data_py/nflreadpy installation."
        )

    return pd.concat(dfs, ignore_index=True)


def _safe_agg(series: pd.Series) -> float:
    """Mean of numeric series, or 0 if empty/invalid."""
    if series is None or series.empty:
        return 0.0
    return float(series.mean())


def build_scheme_metrics_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate PBP to team-season scheme metrics.
    pbp must have: posteam, season, and (if present) shotgun, no_huddle, play_type, desc, down, pass_attempt, rush_attempt.
    
    Computes:
    - Basic rates: motion_rate, play_action_rate, shotgun_rate, under_center_rate, no_huddle_rate, air_yards_per_att
    - Formation-specific rates: under_center_play_action_rate, under_center_pass_rate, under_center_run_rate,
      shotgun_play_action_rate, shotgun_pass_rate, shotgun_run_rate
    - Down-specific rates: down_1_pass_rate, down_2_pass_rate, down_3_pass_rate, down_1_run_rate, down_2_run_rate, down_3_run_rate
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

    # Play action: requires a dedicated 'play_action' column from nflfastR CSVs
    # Description-based detection is unreliable and inconsistent across teams
    if "pass_attempt" in pbp.columns and "play_action" in pbp.columns:
        # Reliable: nflfastR CSV with play_action column
        pbp["_play_action"] = pbp["play_action"].fillna(0).astype(float)
        pbp["_play_action_pass"] = ((pbp["pass_attempt"] == 1) & (pbp["play_action"] == 1)).astype(float)
    elif "play_action" in pbp.columns:
        pbp["_play_action"] = pbp["play_action"].fillna(0).astype(float)
        pbp["_play_action_pass"] = pbp["_play_action"].copy()
    else:
        # No reliable play_action column - set to 0 (description parsing is too unreliable)
        # To get accurate play action rates, download nflfastR PBP CSVs with play_action column
        pbp["_play_action"] = 0.0
        pbp["_play_action_pass"] = 0.0

    # Pass attempts (for play action rate = play action passes / pass attempts)
    if "pass_attempt" in pbp.columns:
        pbp["_pass_attempt"] = pbp["pass_attempt"].fillna(0).astype(float)
    else:
        pbp["_pass_attempt"] = (pbp.get("play_type", pd.Series(dtype=object)) == "pass").astype(float)

    # Rush attempts
    if "rush_attempt" in pbp.columns:
        pbp["_rush_attempt"] = pbp["rush_attempt"].fillna(0).astype(float)
    else:
        pbp["_rush_attempt"] = (pbp.get("play_type", pd.Series(dtype=object)) == "run").astype(float)

    # Down (for down-specific rates)
    if "down" in pbp.columns:
        pbp["_down"] = pbp["down"].fillna(0).astype(int)
    else:
        pbp["_down"] = 0

    # Air yards per attempt (for passes)
    if "air_yards" in pbp.columns:
        pbp["_air_yards"] = pbp["air_yards"].fillna(0)
    else:
        pbp["_air_yards"] = 0.0

    # Read option: designed QB runs detected via play description
    # Note: We can't reliably detect all QB rushing without position data (knowing the rusher is a QB),
    # so we only track read option which can be identified via description keywords
    desc = pbp.get("desc", pd.Series(dtype=object))
    pbp["_read_option"] = (
        (pbp["_rush_attempt"] == 1) &
        (desc.astype(str).str.lower().str.contains("read option|option", na=False, regex=True))
    ).astype(float)
    
    # Exclude scrambles if qb_scramble column exists
    if "qb_scramble" in pbp.columns:
        pbp["_read_option"] = (
            (pbp["_read_option"] == 1) & 
            (pbp["qb_scramble"].fillna(1) == 0)  # 0 = designed run, 1 = scramble
        ).astype(float)

    grp = pbp.groupby(["posteam", "season"], as_index=False)

    # Basic rates as percentages
    # Count plays per team-season (use count on a column to ensure DataFrame output)
    plays = grp["_shotgun"].count().reset_index().rename(columns={"_shotgun": "n_plays"})
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
        # No reliable motion field in this PBP; keep column but fill with 0
        motion_rate = shotgun_rate[["posteam", "season"]].copy()
        motion_rate["motion_rate"] = 0.0

    # ===== FORMATION-SPECIFIC RATES (Under Center vs Shotgun) =====
    # Under center plays (shotgun = 0)
    uc_plays = pbp.loc[pbp["_shotgun"] == 0]
    uc_grp = uc_plays.groupby(["posteam", "season"], as_index=False)
    uc_total = uc_grp["_shotgun"].count().reset_index().rename(columns={"_shotgun": "uc_n_plays"})
    
    # Under center play action passes
    uc_pa_passes = uc_grp["_play_action_pass"].sum().reset_index().rename(columns={"_play_action_pass": "uc_pa_passes"})
    uc_pa_rate = uc_total.merge(uc_pa_passes, on=["posteam", "season"], how="left")
    uc_pa_rate["uc_pa_passes"] = uc_pa_rate["uc_pa_passes"].fillna(0)
    uc_pa_rate["under_center_play_action_rate"] = (uc_pa_rate["uc_pa_passes"] / uc_pa_rate["uc_n_plays"].replace(0, float("nan")) * 100).fillna(0)
    
    # Under center pass rate (all passes: dropback + play action)
    # When play action data is missing/unreliable, this counts all passes
    uc_pass_mask = (uc_plays["_pass_attempt"] == 1)
    uc_pass_grp = uc_plays.loc[uc_pass_mask].groupby(["posteam", "season"], as_index=False)
    uc_pass = uc_pass_grp["_pass_attempt"].count().reset_index().rename(columns={"_pass_attempt": "uc_passes"})
    uc_pass_rate = uc_total.merge(uc_pass, on=["posteam", "season"], how="left")
    uc_pass_rate["uc_passes"] = uc_pass_rate["uc_passes"].fillna(0)
    uc_pass_rate["under_center_pass_rate"] = (uc_pass_rate["uc_passes"] / uc_pass_rate["uc_n_plays"].replace(0, float("nan")) * 100).fillna(0)
    
    # Under center run rate
    uc_runs = uc_grp["_rush_attempt"].sum().reset_index().rename(columns={"_rush_attempt": "uc_runs"})
    uc_run_rate = uc_total.merge(uc_runs, on=["posteam", "season"], how="left")
    uc_run_rate["uc_runs"] = uc_run_rate["uc_runs"].fillna(0)
    uc_run_rate["under_center_run_rate"] = (uc_run_rate["uc_runs"] / uc_run_rate["uc_n_plays"].replace(0, float("nan")) * 100).fillna(0)
    
    # Shotgun plays (shotgun = 1)
    sg_plays = pbp.loc[pbp["_shotgun"] == 1]
    sg_grp = sg_plays.groupby(["posteam", "season"], as_index=False)
    sg_total = sg_grp["_shotgun"].count().reset_index().rename(columns={"_shotgun": "sg_n_plays"})
    
    # Shotgun play action passes
    sg_pa_passes = sg_grp["_play_action_pass"].sum().reset_index().rename(columns={"_play_action_pass": "sg_pa_passes"})
    sg_pa_rate = sg_total.merge(sg_pa_passes, on=["posteam", "season"], how="left")
    sg_pa_rate["sg_pa_passes"] = sg_pa_rate["sg_pa_passes"].fillna(0)
    sg_pa_rate["shotgun_play_action_rate"] = (sg_pa_rate["sg_pa_passes"] / sg_pa_rate["sg_n_plays"].replace(0, float("nan")) * 100).fillna(0)
    
    # Shotgun pass rate (all passes: dropback + play action)
    # When play action data is missing/unreliable, this counts all passes
    sg_pass_mask = (sg_plays["_pass_attempt"] == 1)
    sg_pass_grp = sg_plays.loc[sg_pass_mask].groupby(["posteam", "season"], as_index=False)
    sg_pass = sg_pass_grp["_pass_attempt"].count().reset_index().rename(columns={"_pass_attempt": "sg_passes"})
    sg_pass_rate = sg_total.merge(sg_pass, on=["posteam", "season"], how="left")
    sg_pass_rate["sg_passes"] = sg_pass_rate["sg_passes"].fillna(0)
    sg_pass_rate["shotgun_pass_rate"] = (sg_pass_rate["sg_passes"] / sg_pass_rate["sg_n_plays"].replace(0, float("nan")) * 100).fillna(0)
    
    # Shotgun run rate
    sg_runs = sg_grp["_rush_attempt"].sum().reset_index().rename(columns={"_rush_attempt": "sg_runs"})
    sg_run_rate = sg_total.merge(sg_runs, on=["posteam", "season"], how="left")
    sg_run_rate["sg_runs"] = sg_run_rate["sg_runs"].fillna(0)
    sg_run_rate["shotgun_run_rate"] = (sg_run_rate["sg_runs"] / sg_run_rate["sg_n_plays"].replace(0, float("nan")) * 100).fillna(0)
    
    # ===== DOWN-SPECIFIC RATES =====
    # 1st down
    down1 = pbp.loc[pbp["_down"] == 1]
    if len(down1) > 0:
        d1_grp = down1.groupby(["posteam", "season"], as_index=False)
        d1_total = d1_grp["_down"].count().reset_index().rename(columns={"_down": "d1_n_plays"})
        d1_passes = d1_grp["_pass_attempt"].sum().reset_index().rename(columns={"_pass_attempt": "d1_passes"})
        d1_runs = d1_grp["_rush_attempt"].sum().reset_index().rename(columns={"_rush_attempt": "d1_runs"})
        d1_rates = d1_total.merge(d1_passes, on=["posteam", "season"], how="left").merge(d1_runs, on=["posteam", "season"], how="left")
        d1_rates["d1_passes"] = d1_rates["d1_passes"].fillna(0)
        d1_rates["d1_runs"] = d1_rates["d1_runs"].fillna(0)
        # Normalize to passes + runs only (so pass_rate + run_rate = 100%)
        d1_total_offensive = d1_rates["d1_passes"] + d1_rates["d1_runs"]
        d1_rates["down_1_pass_rate"] = (d1_rates["d1_passes"] / d1_total_offensive.replace(0, float("nan")) * 100).fillna(0)
        d1_rates["down_1_run_rate"] = (d1_rates["d1_runs"] / d1_total_offensive.replace(0, float("nan")) * 100).fillna(0)
    else:
        d1_rates = plays[["posteam", "season"]].copy()
        d1_rates["down_1_pass_rate"] = 0.0
        d1_rates["down_1_run_rate"] = 0.0
    
    # 2nd down
    down2 = pbp.loc[pbp["_down"] == 2]
    if len(down2) > 0:
        d2_grp = down2.groupby(["posteam", "season"], as_index=False)
        d2_total = d2_grp["_down"].count().reset_index().rename(columns={"_down": "d2_n_plays"})
        d2_passes = d2_grp["_pass_attempt"].sum().reset_index().rename(columns={"_pass_attempt": "d2_passes"})
        d2_runs = d2_grp["_rush_attempt"].sum().reset_index().rename(columns={"_rush_attempt": "d2_runs"})
        d2_rates = d2_total.merge(d2_passes, on=["posteam", "season"], how="left").merge(d2_runs, on=["posteam", "season"], how="left")
        d2_rates["d2_passes"] = d2_rates["d2_passes"].fillna(0)
        d2_rates["d2_runs"] = d2_rates["d2_runs"].fillna(0)
        # Normalize to passes + runs only (so pass_rate + run_rate = 100%)
        d2_total_offensive = d2_rates["d2_passes"] + d2_rates["d2_runs"]
        d2_rates["down_2_pass_rate"] = (d2_rates["d2_passes"] / d2_total_offensive.replace(0, float("nan")) * 100).fillna(0)
        d2_rates["down_2_run_rate"] = (d2_rates["d2_runs"] / d2_total_offensive.replace(0, float("nan")) * 100).fillna(0)
    else:
        d2_rates = plays[["posteam", "season"]].copy()
        d2_rates["down_2_pass_rate"] = 0.0
        d2_rates["down_2_run_rate"] = 0.0
    
    # 3rd down
    down3 = pbp.loc[pbp["_down"] == 3]
    if len(down3) > 0:
        d3_grp = down3.groupby(["posteam", "season"], as_index=False)
        d3_total = d3_grp["_down"].count().reset_index().rename(columns={"_down": "d3_n_plays"})
        d3_passes = d3_grp["_pass_attempt"].sum().reset_index().rename(columns={"_pass_attempt": "d3_passes"})
        d3_runs = d3_grp["_rush_attempt"].sum().reset_index().rename(columns={"_rush_attempt": "d3_runs"})
        d3_rates = d3_total.merge(d3_passes, on=["posteam", "season"], how="left").merge(d3_runs, on=["posteam", "season"], how="left")
        d3_rates["d3_passes"] = d3_rates["d3_passes"].fillna(0)
        d3_rates["d3_runs"] = d3_rates["d3_runs"].fillna(0)
        # Normalize to passes + runs only (so pass_rate + run_rate = 100%)
        d3_total_offensive = d3_rates["d3_passes"] + d3_rates["d3_runs"]
        d3_rates["down_3_pass_rate"] = (d3_rates["d3_passes"] / d3_total_offensive.replace(0, float("nan")) * 100).fillna(0)
        d3_rates["down_3_run_rate"] = (d3_rates["d3_runs"] / d3_total_offensive.replace(0, float("nan")) * 100).fillna(0)
    else:
        d3_rates = plays[["posteam", "season"]].copy()
        d3_rates["down_3_pass_rate"] = 0.0
        d3_rates["down_3_run_rate"] = 0.0

    # Merge all metrics
    out = plays.merge(shotgun_rate, on=["posteam", "season"])
    out = out.merge(no_huddle_rate, on=["posteam", "season"])
    out = out.merge(merge_pa[["posteam", "season", "play_action_rate"]], on=["posteam", "season"])
    out = out.merge(air, on=["posteam", "season"])
    out = out.merge(motion_rate, on=["posteam", "season"])
    out["under_center_rate"] = 100.0 - out["shotgun_rate"]
    
    # Merge formation-specific rates
    out = out.merge(uc_pa_rate[["posteam", "season", "under_center_play_action_rate"]], on=["posteam", "season"], how="left")
    out = out.merge(uc_pass_rate[["posteam", "season", "under_center_pass_rate"]], on=["posteam", "season"], how="left")
    out = out.merge(uc_run_rate[["posteam", "season", "under_center_run_rate"]], on=["posteam", "season"], how="left")
    out = out.merge(sg_pa_rate[["posteam", "season", "shotgun_play_action_rate"]], on=["posteam", "season"], how="left")
    out = out.merge(sg_pass_rate[["posteam", "season", "shotgun_pass_rate"]], on=["posteam", "season"], how="left")
    out = out.merge(sg_run_rate[["posteam", "season", "shotgun_run_rate"]], on=["posteam", "season"], how="left")
    
    # Merge down-specific rates
    out = out.merge(d1_rates[["posteam", "season", "down_1_pass_rate", "down_1_run_rate"]], on=["posteam", "season"], how="left")
    out = out.merge(d2_rates[["posteam", "season", "down_2_pass_rate", "down_2_run_rate"]], on=["posteam", "season"], how="left")
    out = out.merge(d3_rates[["posteam", "season", "down_3_pass_rate", "down_3_run_rate"]], on=["posteam", "season"], how="left")
    
    # Read option rate: % of all plays that are read option (designed QB runs)
    read_option_grp = grp["_read_option"].sum().reset_index().rename(columns={"_read_option": "read_option_plays"})
    read_option_rate = plays.merge(read_option_grp, on=["posteam", "season"], how="left")
    read_option_rate["read_option_plays"] = read_option_rate["read_option_plays"].fillna(0)
    read_option_rate["read_option_rate"] = (read_option_rate["read_option_plays"] / read_option_rate["n_plays"].replace(0, float("nan")) * 100).fillna(0)
    out = out.merge(read_option_rate[["posteam", "season", "read_option_rate"]], on=["posteam", "season"], how="left")
    
    # Fill NaN with 0 for any missing rates
    formation_cols = ["under_center_play_action_rate", "under_center_pass_rate", "under_center_run_rate",
                      "shotgun_play_action_rate", "shotgun_pass_rate", "shotgun_run_rate"]
    down_cols = ["down_1_pass_rate", "down_1_run_rate", "down_2_pass_rate", "down_2_run_rate", 
                 "down_3_pass_rate", "down_3_run_rate"]
    for col in formation_cols + down_cols:
        if col in out.columns:
            out[col] = out[col].fillna(0)
    
    out = out.rename(columns={"posteam": "team_abbr"})
    return out


def build_and_save_pbp_scheme(seasons: list[int] | None = None, output_path: str | Path | None = None) -> Path:
    """
    Load PBP for seasons (default 2015–2025), build scheme metrics, save CSV.
    Note: nflreadpy/nfl_data_py typically support up to the most recent completed season (2025).
    """
    if seasons is None:
        # Default to 2015-2025 (nflreadpy supports up to 2025 as of Feb 2026)
        from datetime import datetime
        current_year = datetime.now().year
        # Cap at 2025 since that's the latest supported season
        max_season = min(current_year - 1, 2025) if current_year > 2025 else current_year
        seasons = list(range(2015, max_season + 1))
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
