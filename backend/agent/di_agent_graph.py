"""
DI GM Agent — Defensive Interior (DT/NT)

LangGraph agent that evaluates a Defensive Interior free agent and returns a
graded SIGN / PASS recommendation. Run-stopping is the primary value driver:
the stats grade weights stop rate (50%), TFL rate (25%), and pressure rate (25%).
The composite grade blends 40% model PFF grade + 60% stats-based grade, then
projects year-by-year over the contract length accounting for the empirical DI
age curve, cap inflation, and time discounting.
"""

from typing import TypedDict, Dict, List
from langgraph.graph import StateGraph, END
from backend.agent.di_model_wrapper import DIModelInference
from backend.agent.team_context import (
    assess_team_fit as _assess_team_fit_logic,
    aav_to_cap_pcts,
    decision_fair_aav_with_replacement,
    cap_scale_for_year,
)
from backend.agent.grade_projection import (
    grade_to_tier_universal,
    player_recent_grade_yoy,
    apply_yearly_grade_step,
    projection_trend_multiplier,
)
from backend.agent.stat_projection_utils import (
    run_def_snap_load_17,
    inactivity_retirement_penalty,
    apply_inactivity_to_projection_list,
    apply_projection_plausibility_caps,
    shrink_model_grade_for_season_snap_volume,
    snap_value_reliability_factor,
)
import pandas as pd
import numpy as np
import os, datetime

from backend.agent.api_year_utils import resolve_player_age_for_evaluation

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DI_CSV_PATH    = os.path.join(_BASE, "ML", "DI.csv")
DI_TRANSFORMER = os.path.join(_BASE, "ML", "DI_Pranay_Transformers", "di_best_classifier.pth")
DI_SCALER      = os.path.join(_BASE, "ML", "DI_Pranay_Transformers", "di_player_scaler.joblib")
DI_XGB         = os.path.join(_BASE, "ML", "DI_Pranay_Transformers", "di_best_xgb.joblib")

di_engine = DIModelInference(DI_TRANSFORMER, scaler_path=DI_SCALER, xgb_path=None)


# ─────────────────────────────────────────────
# Grade → Market Value curve (2026 OTC calibrated for DI)
# Reference contracts: Chris Jones $34.5M, Q.Williams $27M,
# Buckner $21M, D.Payne $22M, J.Allen $20M
# ─────────────────────────────────────────────
_GRADE_ANCHORS = [45,   55,   60,   65,   70,   75,   80,   85,   88,   92,   96,  100]
_VALUE_ANCHORS = [1.23, 3.07, 7.36, 13.49, 18.39, 23.30, 28.20, 33.10, 34.94, 36.78, 40.46, 42.92]
MARKET_CALIBRATION_FACTOR = 0.88


def grade_to_market_value(grade: float) -> float:
    grade = max(45.0, min(100.0, float(grade)))
    return round(float(np.interp(grade, _GRADE_ANCHORS, _VALUE_ANCHORS)) * MARKET_CALIBRATION_FACTOR, 2)


# ─────────────────────────────────────────────
# Stats-based grade equivalent (run-first weighting)
#
# All stats use snap-rate denominators so missed games don't distort
# the grade. Empirical percentiles from 4,431 DI season records:
#
#   stop_rate (stops / defense_snaps × 100):
#     10th=1.87  25th=2.63  50th=3.45  75th=4.30  90th=5.22
#
#   tfl_rate (TFL / defense_snaps × 100):
#     25th=0.0   50th=0.41  75th=0.74  90th=1.07
#
#   pressure_rate (total_pressures / defense_snaps × 100):
#     10th=1.26  25th=2.06  50th=3.25  75th=4.78  90th=6.35
#
#   sack_rate (sacks / defense_snaps × 100):
#     50th=0.37  75th=0.71  90th=1.07
# ─────────────────────────────────────────────
_STOP_RATE_ANCHORS = [0.0,  1.87, 2.63, 3.45, 4.30, 6.50]
_TFL_RATE_ANCHORS  = [0.0,  0.10, 0.30, 0.60, 0.90, 1.50]
_PRES_RATE_ANCHORS = [0.0,  1.26, 2.50, 4.00, 5.50, 8.50]
_SACK_RATE_ANCHORS = [0.0,  0.05, 0.37, 0.71, 1.07, 1.80]
_STAT_GRD_SCALE    = [45.0, 55.0, 65.0, 75.0, 85.0, 99.0]


def _stats_grade(stop_rate: float, tfl_rate: float,
                 pressure_rate: float, sack_rate: float) -> float:
    """
    Map per-snap rates to a 45-99 grade-equivalent score.
    Weights: stop_rate 40%, tfl_rate 20%, pressure_rate 20%, sack_rate 20%.
    Run stopping remains the primary value driver; pass-rush stats included
    at lower weight to reward interior rushers.
    """
    sr  = float(np.interp(stop_rate,     _STOP_RATE_ANCHORS, _STAT_GRD_SCALE))
    tr  = float(np.interp(tfl_rate,      _TFL_RATE_ANCHORS,  _STAT_GRD_SCALE))
    pr  = float(np.interp(pressure_rate, _PRES_RATE_ANCHORS, _STAT_GRD_SCALE))
    skr = float(np.interp(sack_rate,     _SACK_RATE_ANCHORS, _STAT_GRD_SCALE))
    return round(0.40 * sr + 0.20 * tr + 0.20 * pr + 0.20 * skr, 2)


def _composite_grade(model_grade: float, stats_gr: float) -> float:
    """Blend model PFF grade (40%) with stats-based grade (60%)."""
    return round(0.40 * model_grade + 0.60 * stats_gr, 2)


def _grade_to_tier(grade: float) -> str:
    return grade_to_tier_universal(grade)


# ─────────────────────────────────────────────
# Age-based annual grade delta (DI-specific; from empirical DI season transitions).
# Interior peak/cliff pattern differs from EDGE and from offensive skill positions.
# ─────────────────────────────────────────────
_AGE_DELTAS = {
    21: +2.4, 22: +4.7, 23: -0.3, 24: +0.5,
    25: -1.2, 26: -2.0, 27:  0.0, 28: -2.4, 29: -1.0,
    30: -2.7, 31: +0.3, 32: -0.2, 33: -8.5, 34: -3.0, 35: -4.6,
}


def _annual_grade_delta(age: int) -> float:
    if age <= 21: return +2.4
    if age >= 36: return -5.0
    return _AGE_DELTAS.get(age, -3.0)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _safe_float(val, default=0.0) -> float:
    try:
        f = float(val)
        return default if np.isnan(f) else f
    except Exception:
        return default


def _has_valid_stats(row) -> bool:
    """True if the row has at least one key counting stat."""
    for col in ("stops", "tackles_for_loss", "total_pressures", "snap_counts_defense"):
        val = row.get(col)
        try:
            f = float(val)
            if not np.isnan(f):
                return True
        except Exception:
            pass
    return False


# ─────────────────────────────────────────────
# Career stats extraction
# ─────────────────────────────────────────────
def extract_career_stats(history: pd.DataFrame) -> List[dict]:
    """Return per-season stats for every season that has actual data."""
    seasons = []
    for _, row in history.sort_values("Year").iterrows():
        if not _has_valid_stats(row):
            continue
        year      = int(_safe_float(row.get("Year"), 2024))
        max_games = 17.0 if year >= 2021 else 16.0
        games     = max(1.0, min(max_games, _safe_float(row.get("player_game_count"), max_games)))

        snaps_d  = _safe_float(row.get("snap_counts_defense"), 1.0) or 1.0
        snaps_dl = _safe_float(row.get("snap_counts_dl"),      1.0) or 1.0
        stops    = _safe_float(row.get("stops"))
        tfl      = _safe_float(row.get("tackles_for_loss"))
        prs      = _safe_float(row.get("total_pressures"))
        sacks    = _safe_float(row.get("sacks"))
        ff       = _safe_float(row.get("forced_fumbles"))
        pr_grade = _safe_float(row.get("grades_pass_rush_defense"))
        rd_grade = _safe_float(row.get("grades_run_defense"))
        overall  = _safe_float(row.get("grades_defense"))

        stop_rate     = round(stops / snaps_d * 100, 1) if snaps_d > 0 else 0.0
        tfl_rate      = round(tfl   / snaps_d * 100, 1) if snaps_d > 0 else 0.0
        pressure_rate = round(prs   / snaps_d * 100, 1) if snaps_d > 0 else 0.0

        seasons.append({
            "season":        year,
            "games_played":  int(games),
            "max_games":     int(max_games),
            "stops":         round(stops, 1),
            "tfl":           round(tfl,   1),
            "pressures":     round(prs,   1),
            "sacks":         round(sacks, 1),
            "forced_fumbles":round(ff,    1),
            "stop_rate":     stop_rate,
            "tfl_rate":      tfl_rate,
            "pressure_rate": pressure_rate,
            "snaps_defense": round(snaps_d),
            "snaps_dl":      round(snaps_dl),
            "pass_rush_grade":round(pr_grade, 1),
            "run_def_grade":  round(rd_grade,  1),
            "overall_grade":  round(overall,   1),
        })
    return seasons


# ─────────────────────────────────────────────
# Last-season stats + career-weighted rates
# ─────────────────────────────────────────────
def extract_last_season_stats(history: pd.DataFrame) -> dict:
    """
    Display stats from the most recent season with valid data.
    Rate stats and 17g projections use career-weighted averages so that
    an injury-shortened season doesn't inflate projections.
    """
    sorted_hist = history.sort_values("Year")
    valid_rows  = sorted_hist[sorted_hist.apply(_has_valid_stats, axis=1)]
    if valid_rows.empty:
        valid_rows = sorted_hist

    row = valid_rows.iloc[-1]

    year      = int(_safe_float(row.get("Year"), 2024))
    max_games = 17.0 if year >= 2021 else 16.0
    games     = max(1.0, min(max_games, _safe_float(row.get("player_game_count"), max_games)))
    avail     = round(games / max_games, 3)

    stops    = _safe_float(row.get("stops"))
    tfl      = _safe_float(row.get("tackles_for_loss"))
    prs      = _safe_float(row.get("total_pressures"))
    sacks    = _safe_float(row.get("sacks"))
    ff       = _safe_float(row.get("forced_fumbles"))
    snaps_d  = _safe_float(row.get("snap_counts_defense"), 1.0) or 1.0
    snaps_dl = _safe_float(row.get("snap_counts_dl"),      1.0) or 1.0
    pr_grade = _safe_float(row.get("grades_pass_rush_defense"))
    rd_grade = _safe_float(row.get("grades_run_defense"))

    # ── Career-weighted rates (stable across small samples) ────────
    c_stops = c_tfl = c_prs = c_sacks = 0.0
    c_d_snaps = c_dl_snaps = c_games = 0.0

    for _, r in valid_rows.iterrows():
        yr_g   = 17.0 if int(_safe_float(r.get("Year"), 2024)) >= 2021 else 16.0
        g      = max(1.0, min(yr_g, _safe_float(r.get("player_game_count"), yr_g)))
        sd     = _safe_float(r.get("snap_counts_defense"), 0.0)
        sdl    = _safe_float(r.get("snap_counts_dl"),      0.0)
        c_stops    += _safe_float(r.get("stops"))
        c_tfl      += _safe_float(r.get("tackles_for_loss"))
        c_prs      += _safe_float(r.get("total_pressures"))
        c_sacks    += _safe_float(r.get("sacks"))
        c_d_snaps  += sd
        c_dl_snaps += sdl
        c_games    += g

    c_d_snaps  = max(c_d_snaps,  1.0)
    c_dl_snaps = max(c_dl_snaps, 1.0)
    c_games    = max(c_games,    1.0)

    career_stop_rate     = c_stops  / c_d_snaps * 100
    career_tfl_rate      = c_tfl    / c_d_snaps * 100
    career_pressure_rate = c_prs    / c_d_snaps * 100
    career_sack_rate     = c_sacks  / c_d_snaps * 100

    # Career average snaps per game → 17-game projections (starter-role floor)
    proj_snaps_d  = round(run_def_snap_load_17(c_d_snaps, c_games))
    proj_snaps_dl = round(run_def_snap_load_17(c_dl_snaps, c_games, floor_17=600.0))

    stops_17g = round(career_stop_rate     / 100 * proj_snaps_d,  1)
    tfl_17g   = round(career_tfl_rate      / 100 * proj_snaps_d,  1)
    prs_17g   = round(career_pressure_rate / 100 * proj_snaps_d,  1)
    sacks_17g = round((c_sacks / c_d_snaps) * proj_snaps_d,       1)
    ff_17g    = round((ff / games) * 17,                           1)

    return {
        "season":          year,
        "games_played":    int(games),
        "max_games":       int(max_games),
        "availability":    avail,
        # Actual last-season counts
        "stops":           round(stops, 1),
        "tfl":             round(tfl,   1),
        "pressures":       round(prs,   1),
        "sacks":           round(sacks, 1),
        "forced_fumbles":  round(ff,    1),
        "snaps_defense":   round(snaps_d),
        "snaps_dl":        round(snaps_dl),
        "pass_rush_grade": round(pr_grade, 1),
        "run_def_grade":   round(rd_grade, 1),
        # Career-weighted rates (used for grade)
        "stop_rate":       round(career_stop_rate,     2),
        "tfl_rate":        round(career_tfl_rate,      2),
        "pressure_rate":   round(career_pressure_rate, 2),
        "sack_rate":       round(career_sack_rate,     2),
        # 17g full-health projections
        "stops_17g":    stops_17g,
        "tfl_17g":      tfl_17g,
        "prs_17g":      prs_17g,
        "sacks_17g":    sacks_17g,
        "ff_17g":       ff_17g,
        "snaps_d_17g":  proj_snaps_d,
        "snaps_dl_17g": proj_snaps_dl,
    }


def _compute_health_factor(history: pd.DataFrame) -> tuple:
    """
    Weighted availability over last 3 seasons (50/30/20 recent-heavy).
    Returns (grade_adjustment_pts, weighted_avg_availability).
    Calibration: 100% avail → +2.5 pts, 75% → 0, 25% → -5 pts.
    """
    sorted_hist = history.sort_values("Year")
    valid_rows  = sorted_hist[sorted_hist.apply(_has_valid_stats, axis=1)]
    if valid_rows.empty:
        valid_rows = sorted_hist
    recent  = valid_rows.tail(3).reset_index(drop=True)
    weights = [0.20, 0.30, 0.50]
    n       = len(recent)

    avail_list = []
    for _, row in recent.iterrows():
        yr    = int(_safe_float(row.get("Year"), 2024))
        max_g = 17.0 if yr >= 2021 else 16.0
        g     = max(1.0, min(max_g, _safe_float(row.get("player_game_count"), max_g)))
        avail_list.append(g / max_g)

    w = weights[-n:]
    w = [x / sum(w) for x in w]
    avg_avail = sum(a * wt for a, wt in zip(avail_list, w))
    adj = max(-5.0, min(2.5, (avg_avail - 0.75) * 10.0))
    return round(adj, 2), round(avg_avail, 3)


# ─────────────────────────────────────────────
# Stat projection
# ─────────────────────────────────────────────
def project_stats(
    last_stats: dict,
    composite_gr: float,
    current_age: int,
    contract_years: int,
    history: pd.DataFrame = None,
    grade_col: str = "grades_defense",
) -> List[dict]:
    """
    Project stats forward year-by-year (17 healthy games assumed).
    Counts scaled by grade ratio; capped at historical maximums.
    """
    projections = []
    grade = composite_gr
    player_yoy = player_recent_grade_yoy(history, grade_col)

    for yr in range(1, contract_years + 1):
        age = current_age + yr - 1
        if yr > 1:
            grade = apply_yearly_grade_step(grade, age - 1, player_yoy, _annual_grade_delta)

        base_scale = max(0.25, min(1.5, grade / composite_gr)) if composite_gr > 0 else 1.0
        trend_mult = projection_trend_multiplier("DI", age, yr, player_yoy)
        scale = max(0.25, min(1.8, base_scale * trend_mult))

        projections.append({
            "year":             yr,
            "age":              age,
            "projected_grade":  round(grade, 1),
            "stops":            round(min(65.0, last_stats["stops_17g"]    * scale), 1),
            "tfl":              round(min(20.0, last_stats["tfl_17g"]      * scale), 1),
            "pressures":        round(min(80.0, last_stats["prs_17g"]      * scale), 1),
            "sacks":            round(min(15.0, last_stats["sacks_17g"]    * scale), 1),
            "stop_rate":        round(min(10.0, last_stats["stop_rate"]    * scale), 1),
            "pass_rush_grade":  round(min(99.0, last_stats["pass_rush_grade"] * scale), 1),
            "run_def_grade":    round(min(99.0, last_stats["run_def_grade"]   * scale), 1),
        })

    return projections


# ─────────────────────────────────────────────
# Contract-adjusted valuation
# ─────────────────────────────────────────────
DISCOUNT_RATE   = 0.08    # 8%/yr — roster uncertainty & cap risk
CAP_GROWTH_RATE = 0.065   # 6.5%/yr — historical NFL cap growth


def compute_contract_value(
    composite_gr,
    current_age,
    contract_years,
    salary_ask,
    history: pd.DataFrame = None,
    grade_col: str = "grades_defense",
    analysis_year: int = 2026,
):
    breakdown           = []
    total_disc_value    = 0.0
    total_disc_ask      = 0.0
    total_nominal_value = 0.0
    weighted_fair_num   = 0.0
    weighted_burden_num = 0.0
    weight_den          = 0.0
    grade               = float(composite_gr)
    player_yoy = player_recent_grade_yoy(history, grade_col)
    snap_rel, _ = snap_value_reliability_factor(history)
    cap_scale = cap_scale_for_year(analysis_year)

    for yr in range(1, contract_years + 1):
        age = current_age + yr - 1
        if yr > 1:
            grade = apply_yearly_grade_step(grade, age - 1, player_yoy, _annual_grade_delta)

        cap_factor    = (1.0 + CAP_GROWTH_RATE) ** (yr - 1)
        time_discount = 1.0 / ((1.0 + DISCOUNT_RATE) ** (yr - 1))

        base_value    = grade_to_market_value(grade) * snap_rel * cap_scale
        nominal_value = base_value * cap_factor
        disc_value    = nominal_value * time_discount

        cap_adj_ask   = salary_ask / cap_factor
        disc_ask      = cap_adj_ask * time_discount
        front_weight  = 1.0 / float(yr)

        total_nominal_value += nominal_value
        total_disc_value    += disc_value
        total_disc_ask      += disc_ask
        weighted_fair_num   += nominal_value * front_weight
        weighted_burden_num += cap_adj_ask * front_weight
        weight_den          += front_weight

        breakdown.append({
            "year":             yr,
            "age":              age,
            "projected_grade":  round(grade, 1),
            "market_value":     base_value,
            "nominal_value":    round(nominal_value, 2),
            "cap_adj_ask":      round(cap_adj_ask, 2),
            "discounted_value": round(disc_value, 2),
            "year_surplus":     round(base_value - cap_adj_ask, 2),
        })

    fair_aav             = round(weighted_fair_num / max(weight_den, 1e-6), 2)
    effective_cap_burden = round(weighted_burden_num / max(weight_den, 1e-6), 2)
    return fair_aav, effective_cap_burden, round(total_nominal_value, 2), breakdown


# ─────────────────────────────────────────────
# Agent State
# ─────────────────────────────────────────────
class DIAgentState(TypedDict):
    player_name:    str
    salary_ask:     float
    contract_years: int
    player_history:      pd.DataFrame
    player_history_full: pd.DataFrame
    analysis_year:       int

    predicted_tier:    str
    confidence:        Dict[str, float]
    current_age:       int
    last_season_stats: dict
    career_stats:      List[dict]
    stats_score:       float   # stats-based grade
    composite_grade:   float   # 40% model + 60% stats

    valuation:            float
    effective_cap_burden: float
    total_nominal_value:  float
    year_breakdown:       List[dict]
    projected_stats:      List[dict]

    # Optional team context (empty when not in team mode)
    team_name:              str
    team_cap_available_pct: float
    positional_need:        float
    need_label:             str
    current_roster:         List[dict]
    signing_cap_pcts:       List[float]
    team_fit_summary:       str

    decision:  str
    reasoning: str


# ─────────────────────────────────────────────
# Node 1: Predict Performance
# ─────────────────────────────────────────────
def predict_performance(state: DIAgentState):
    print(f"[DI Agent] Predicting performance for {state['player_name']}...")

    tier, details = di_engine.get_prediction(state["player_history"])
    raw_mg = float(details.get("predicted_grade", 60.0))

    history      = state["player_history"]
    current_year = int(state.get("analysis_year") or datetime.date.today().year)
    resolved_age = resolve_player_age_for_evaluation(state.get("player_history_full"), history, analysis_year=current_year)
    if resolved_age is not None:
        current_age = resolved_age
    elif "age" in history.columns and "Year" in history.columns:
        last_row           = history.sort_values("Year").iloc[-1]
        age_at_last_season = int(float(last_row["age"]))
        last_season_year   = int(float(last_row["Year"]))
        current_age        = age_at_last_season + (current_year - last_season_year)
    else:
        current_age = 28

    last_stats   = extract_last_season_stats(history)
    career_stats = extract_career_stats(history)

    health_adj, avg_avail = _compute_health_factor(history)
    inactivity_adj, _ = inactivity_retirement_penalty(history, current_year=current_year)

    model_grade, snap_m = shrink_model_grade_for_season_snap_volume(
        raw_mg,
        history,
        grade_col="grades_defense",
        snap_profile=[
            ("snap_counts_defense", 820.0),
            ("total_snaps", 800.0),
        ],
    )

    sg = _stats_grade(
        last_stats["stop_rate"],
        last_stats["tfl_rate"],
        last_stats["pressure_rate"],
        last_stats["sack_rate"],
    )

    # Composite: 40% model, 60% stats, then nudge by health history
    raw_cg = _composite_grade(model_grade, sg)
    cg     = round(max(45.0, min(99.0, raw_cg + health_adj + inactivity_adj)), 2)

    return {
        "predicted_tier":    _grade_to_tier(cg),
        "current_age":       current_age,
        "last_season_stats": last_stats,
        "career_stats":      career_stats,
        "stats_score":       sg,
        "composite_grade":   cg,
        "confidence": {
            "model_grade":       round(model_grade, 2),
            "model_grade_pre_snap_volume": round(raw_mg, 2),
            "stats_grade":       sg,
            "composite_grade":   cg,
            "health_factor":     health_adj,
            "inactivity_penalty": inactivity_adj,
            "avg_availability":  avg_avail,
            "snap_volume_stress": snap_m.get("snap_volume_stress", 1.0),
            "prior_full_snap_season": snap_m.get("prior_full_snap_season", False),
            "xgb_grade":         details.get("xgb_grade"),
            "transformer_grade": details.get("transformer_grade"),
            "age_adjustment":    details.get("age_adjustment"),
        },
    }


# ─────────────────────────────────────────────
# Node 2: Evaluate Value
# ─────────────────────────────────────────────
def evaluate_value(state: DIAgentState):
    cg             = state["composite_grade"]
    current_age    = state["current_age"]
    contract_years = state["contract_years"]
    salary_ask     = state["salary_ask"]

    hist = state.get("player_history")
    ay = int(state.get("analysis_year") or 2026)
    fair_aav, eff_burden, total_nom, breakdown = compute_contract_value(
        cg, current_age, contract_years, salary_ask, history=hist, grade_col="grades_defense",
        analysis_year=ay,
    )
    stat_proj = project_stats(
        state["last_season_stats"], cg, current_age, contract_years,
        history=hist, grade_col="grades_defense",
    )
    inact_pen = float((state.get("confidence") or {}).get("inactivity_penalty", 0.0))
    stat_proj = apply_inactivity_to_projection_list(stat_proj, inact_pen)
    stat_proj = apply_projection_plausibility_caps(stat_proj, state.get("career_stats") or [])

    return {
        "valuation":            fair_aav,
        "effective_cap_burden": eff_burden,
        "total_nominal_value":  total_nom,
        "year_breakdown":       breakdown,
        "projected_stats":      stat_proj,
    }


# ─────────────────────────────────────────────
# Node 2b: Assess Team Fit (optional — no-op without team)
# ─────────────────────────────────────────────
def assess_team_fit(state: DIAgentState):
    team = state.get("team_name", "")
    if not team:
        return {}

    cap_pcts = aav_to_cap_pcts(
        state["salary_ask"],
        state["contract_years"],
        int(state.get("analysis_year") or 2025),
    )
    return {"signing_cap_pcts": cap_pcts}


# ─────────────────────────────────────────────
# Node 3: Make Decision
# ─────────────────────────────────────────────
def make_decision(state: DIAgentState):
    ask    = state["salary_ask"]
    val    = state["valuation"]
    burden = state["effective_cap_burden"]
    tier   = state["predicted_tier"]
    cg     = state["composite_grade"]
    mg     = state["confidence"].get("model_grade", cg)
    sg     = state["stats_score"]
    age    = state["current_age"]
    years  = state["contract_years"]
    total  = state["total_nominal_value"]
    adj    = state["confidence"].get("age_adjustment")

    health_adj = state["confidence"].get("health_factor", 0)
    avg_avail  = state["confidence"].get("avg_availability", 1.0)
    health_str = (
        f" Health factor: {'+' if health_adj >= 0 else ''}{health_adj} pts "
        f"(avg availability {round(avg_avail*100)}% over recent seasons)."
    )
    adj_str   = f", age penalty: -{adj} pts" if adj else ""
    total_ask = round(ask * years, 2)
    cap_note  = (
        f"With ~{int(CAP_GROWTH_RATE*100)}%/yr cap growth the fixed ${ask}M AAV "
        f"costs effectively ${burden}M/yr in present-value cap terms."
    )

    if age <= 25:   trajectory = "still developing toward his prime"
    elif age <= 29: trajectory = "in his prime run-stopping years"
    elif age <= 32: trajectory = "entering post-prime decline"
    else:           trajectory = "in steep age-related decline"

    team_nm = state.get("team_name", "")
    roster = state.get("current_roster") or []
    val_dec = val
    rep_note = ""
    if team_nm and roster:
        _scale = cap_scale_for_year(int(state.get("analysis_year") or 2026))
        val_dec, rep_note = decision_fair_aav_with_replacement(
            val, lambda g: grade_to_market_value(g) * _scale, cg, roster, "DI",
        )

    surplus = round(val - burden, 2)
    surplus_pct = (val - burden) / max(val, 0.01) * 100

    if surplus_pct >= 20:
        decision    = "Exceptional Value"
        verdict_str = f"surplus of ${surplus}M/yr ({surplus_pct:.0f}% below fair value) — an exceptional deal."
        rec         = "Strongly recommend signing immediately."
    elif surplus_pct >= 5:
        decision    = "Good Signing"
        verdict_str = f"surplus of ${surplus}M/yr — good value at this price."
        rec         = "Recommend signing."
    elif surplus_pct >= -5:
        decision    = "Fair Deal"
        verdict_str = f"${abs(surplus)}M/yr {'surplus' if surplus >= 0 else 'overpay'} — roughly market rate."
        rec         = "Acceptable signing at or near fair value."
    elif surplus_pct >= -15:
        decision    = "Slight Overpay"
        verdict_str = f"overpay of ${abs(surplus)}M/yr ({abs(surplus_pct):.0f}% above fair value) — modest premium."
        rec         = "Manageable overpay; proceed with caution."
    elif surplus_pct >= -30:
        decision    = "Overpay"
        verdict_str = f"overpay of ${abs(surplus)}M/yr ({abs(surplus_pct):.0f}% above fair value) — significant premium."
        rec         = "Recommend passing unless positional need is critical."
    else:
        decision    = "Poor Signing"
        verdict_str = f"overpay of ${abs(surplus)}M/yr ({abs(surplus_pct):.0f}% above fair value) — severely overpriced."
        rec         = "Strongly recommend passing."

    reason = (
        f"{state['player_name']} (age {age}) projects as a {tier} defensive interior lineman. "
        f"Model grade: {mg:.1f} · Stats grade (stop rate 40% / TFL 20% / pressure 20% / sacks 20%, career-weighted): {sg:.1f} → "
        f"Composite: {cg:.1f}{adj_str}.{health_str} "
        f"He is {trajectory}. Over a {years}-yr contract (17 healthy games assumed per year) "
        f"the composite projects a cap-inflation-adjusted fair value of ${val}M/yr vs. "
        f"an effective cap burden of ${burden}M/yr — {verdict_str} "
        f"{cap_note} Total nominal player value: ${total}M vs. total ask: ${total_ask}M. "
        f"{rec}"
    )
    if rep_note:
        reason = reason + rep_note

    # Team-mode adjustment
    team = state.get("team_name", "")
    fit_summary = ""
    if team:
        need_score = state.get("positional_need", 50)
        need_lbl = state.get("need_label", "Average")
        cap_pcts = state.get("signing_cap_pcts", [])
        avail_pct = state.get("team_cap_available_pct", 100)
        roster = state.get("current_roster", [])

        adjusted_decision, fit_summary, team_reason = _assess_team_fit_logic(
            base_decision=decision,
            surplus_pct=surplus_pct,
            need_score=need_score,
            need_label=need_lbl,
            signing_cap_pcts=cap_pcts,
            available_cap_pct=avail_pct,
            roster=roster,
            player_name=state["player_name"],
        )
        decision = adjusted_decision
        reason = reason + " " + team_reason

    return {"decision": decision, "reasoning": reason, "team_fit_summary": fit_summary}


# ─────────────────────────────────────────────
# Graph Construction
# ─────────────────────────────────────────────
_workflow = StateGraph(DIAgentState)
_workflow.add_node("predict_performance", predict_performance)
_workflow.add_node("evaluate_value",      evaluate_value)
_workflow.add_node("assess_team_fit",     assess_team_fit)
_workflow.add_node("make_decision",       make_decision)
_workflow.set_entry_point("predict_performance")
_workflow.add_edge("predict_performance", "evaluate_value")
_workflow.add_edge("evaluate_value",      "assess_team_fit")
_workflow.add_edge("assess_team_fit",     "make_decision")
_workflow.add_edge("make_decision",       END)
di_gm_agent = _workflow.compile()
