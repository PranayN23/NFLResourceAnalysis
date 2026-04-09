"""
ED GM Agent — Edge Defender (EDGE/DE)

LangGraph agent that evaluates an Edge Defender free agent and returns a
SIGN / PASS recommendation. Uses a composite grade that blends the ML model's
predicted PFF grade with an objective stats-based grade (pressure rate, sack
rate, stops), then projects that composite grade — and the associated stats —
forward year-by-year over the contract, accounting for the empirical age curve,
cap inflation, and time discounting.
"""

from typing import TypedDict, Dict, List
from langgraph.graph import StateGraph, END
from backend.agent.ed_model_wrapper import EDModelInference
from backend.agent.team_context import (
    assess_team_fit as _assess_team_fit_logic,
    aav_to_cap_pcts,
    decision_fair_aav_with_replacement,
)
from backend.agent.grade_projection import (
    grade_to_tier_universal,
    player_recent_grade_yoy,
    apply_yearly_grade_step,
)
from backend.agent.stat_projection_utils import pass_rush_snap_load_17, run_def_snap_load_17, inactivity_retirement_penalty, apply_inactivity_to_projection_list, apply_projection_plausibility_caps
import pandas as pd
import numpy as np
import os, datetime

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

ED_CSV_PATH    = os.path.join(_BASE, "ML", "ED.csv")
ED_TRANSFORMER = os.path.join(_BASE, "ML", "ED_Transformers", "ed_best_classifier.pth")
ED_SCALER      = os.path.join(_BASE, "ML", "ED_Transformers", "ed_player_scaler.joblib")
ED_XGB         = os.path.join(_BASE, "ML", "ED_Transformers", "ed_best_xgb.joblib")

ed_engine = EDModelInference(ED_TRANSFORMER, scaler_path=ED_SCALER, xgb_path=None)


# ─────────────────────────────────────────────
# Grade → Market Value curve (2026 OTC calibrated)
# ─────────────────────────────────────────────
_GRADE_ANCHORS = [45,   55,   60,   65,   70,   75,   80,   85,   88,   92,   96,   100]
_VALUE_ANCHORS = [0.75, 2.00, 4.50, 9.50, 15.0, 22.0, 40.0, 44.0, 47.0, 50.0, 53.0, 56.0]
MARKET_CALIBRATION_FACTOR = 0.88


def grade_to_market_value(grade: float) -> float:
    grade = max(45.0, min(100.0, float(grade)))
    return round(float(np.interp(grade, _GRADE_ANCHORS, _VALUE_ANCHORS)) * MARKET_CALIBRATION_FACTOR, 2)


# ─────────────────────────────────────────────
# Stats-based grade equivalent
#
# All stats are snap-rate or 17-game normalised so that players who
# missed games are not unfairly penalised in the grade calculation.
# Injury history is handled separately via the health factor.
#
# Anchors (empirical medians, 17-game full-season basis):
#   pressure_pct  (total_pressures / pass_rush_snaps × 100):
#     Reserve 7.81  Rotation 9.75  Starter 11.55  Elite 14.19
#   sack_rate (sacks / pass_rush_snaps × 100):
#     Reserve 1.07  Rotation 1.45  Starter 1.92   Elite 2.24
#   stops_17g (stops projected to 17 healthy games):
#     Reserve 12.0  Rotation 18.9  Starter 25.5   Elite 34.9
# ─────────────────────────────────────────────
_PR_PCT_ANCHORS = [0.0,  7.81,  9.75, 11.55, 14.19, 22.0]
_SR_ANCHORS     = [0.0,  1.07,  1.45,  1.92,  2.24,  4.5]
_STOP17_ANCHORS = [0.0, 12.0,  18.9,  25.5,  34.9,  60.0]
_STAT_GRD_SCALE = [45.0, 55.0,  65.0,  75.0,  85.0,  99.0]


def _stats_grade(pressure_pct: float, sack_rate: float, stops_17g: float) -> float:
    """Map snap-normalised / 17g stats to a 45-99 grade-equivalent score."""
    pr = float(np.interp(pressure_pct, _PR_PCT_ANCHORS, _STAT_GRD_SCALE))
    sr = float(np.interp(sack_rate,    _SR_ANCHORS,     _STAT_GRD_SCALE))
    st = float(np.interp(stops_17g,    _STOP17_ANCHORS, _STAT_GRD_SCALE))
    return round(0.50 * pr + 0.35 * sr + 0.15 * st, 2)


def _composite_grade(model_grade: float, stats_gr: float) -> float:
    """Blend model PFF grade (40%) with stats-based grade (60%)."""
    return round(0.40 * model_grade + 0.60 * stats_gr, 2)


def _grade_to_tier(grade: float) -> str:
    return grade_to_tier_universal(grade)


# ─────────────────────────────────────────────
# Age-based annual grade delta (EDGE-specific; from empirical ED season transitions).
# Do not reuse for QB/WR/RB — those agents use position-tailored tables.
# ─────────────────────────────────────────────
_AGE_DELTAS = {
    20: +3.0, 21: +3.4, 22: +3.5, 23: +2.9, 24: +1.2,
    25: -2.1, 26: -0.8, 27: -2.7, 28: -0.1, 29: -0.9,
    30: -4.1, 31: -0.5, 32: -8.8, 33: -1.2, 34: -5.0, 35: -5.0,
}


def _annual_grade_delta(age: int) -> float:
    if age <= 20: return +3.0
    if age >= 36: return -5.0
    return _AGE_DELTAS.get(age, -3.0)


# ─────────────────────────────────────────────
# Last-season stats extraction
# ─────────────────────────────────────────────
def _safe_float(val, default=0.0) -> float:
    try:
        f = float(val)
        return default if np.isnan(f) else f
    except Exception:
        return default


def _has_valid_stats(row) -> bool:
    """Return True if a row has at least one key counting stat."""
    for col in ("sacks", "total_pressures", "stops", "snap_counts_pass_rush"):
        val = row.get(col)
        try:
            f = float(val)
            if not np.isnan(f):
                return True
        except Exception:
            pass
    return False


def extract_career_stats(history: pd.DataFrame) -> List[dict]:
    """Return per-season stats for every recorded season that has actual data."""
    seasons = []
    for _, row in history.sort_values("Year").iterrows():
        if not _has_valid_stats(row):
            continue  # skip seasons with no stat data
        year      = int(_safe_float(row.get("Year"), 2024))
        max_games = 17.0 if year >= 2021 else 16.0
        games     = max(1.0, min(max_games, _safe_float(row.get("player_game_count"), max_games)))

        snaps_pr = _safe_float(row.get("snap_counts_pass_rush"), 1.0) or 1.0
        snaps_d  = _safe_float(row.get("snap_counts_defense"),   1.0) or 1.0
        sacks    = _safe_float(row.get("sacks"))
        total_pr = _safe_float(row.get("total_pressures"))
        stops    = _safe_float(row.get("stops"))
        tfl      = _safe_float(row.get("tackles_for_loss"))
        ff       = _safe_float(row.get("forced_fumbles"))
        pr_grade = _safe_float(row.get("grades_pass_rush_defense"))
        rd_grade = _safe_float(row.get("grades_run_defense"))
        overall  = _safe_float(row.get("grades_defense"))

        pressure_pct = round(total_pr / snaps_pr * 100, 1) if snaps_pr > 0 else 0.0
        sack_rate    = round(sacks    / snaps_pr * 100, 1) if snaps_pr > 0 else 0.0

        seasons.append({
            "season":        year,
            "games_played":  int(games),
            "max_games":     int(max_games),
            "sacks":         round(sacks,    1),
            "pressures":     round(total_pr, 1),
            "pressure_pct":  pressure_pct,
            "sack_rate":     sack_rate,
            "stops":         round(stops,    1),
            "tfl":           round(tfl,      1),
            "forced_fumbles":round(ff,       1),
            "snaps_pass_rush":round(snaps_pr),
            "snaps_defense": round(snaps_d),
            "pass_rush_grade":round(pr_grade, 1),
            "run_def_grade": round(rd_grade,  1),
            "overall_grade": round(overall,   1),
        })
    return seasons


def extract_last_season_stats(history: pd.DataFrame) -> dict:
    """
    Pull key stats for grade calculation and projection.

    Display stats come from the most recent season WITH actual data.
    17-game projections use career-weighted rates (total career counting
    stats / total career snaps) so that a 5-game sample doesn't produce
    wildly inflated numbers.
    """
    sorted_hist = history.sort_values("Year")

    # Most recent season with valid stats (for display & availability)
    valid_rows = sorted_hist[sorted_hist.apply(_has_valid_stats, axis=1)]
    if valid_rows.empty:
        valid_rows = sorted_hist  # fall back to any row

    row = valid_rows.iloc[-1]

    year      = int(_safe_float(row.get("Year"), 2024))
    max_games = 17.0 if year >= 2021 else 16.0
    games     = max(1.0, min(max_games, _safe_float(row.get("player_game_count"), max_games)))
    avail     = round(games / max_games, 3)

    sacks    = _safe_float(row.get("sacks"))
    total_pr = _safe_float(row.get("total_pressures"))
    stops    = _safe_float(row.get("stops"))
    tfl      = _safe_float(row.get("tackles_for_loss"))
    ff       = _safe_float(row.get("forced_fumbles"))
    snaps_pr = _safe_float(row.get("snap_counts_pass_rush"), 1.0) or 1.0
    snaps_d  = _safe_float(row.get("snap_counts_defense"),   1.0) or 1.0
    pr_grade = _safe_float(row.get("grades_pass_rush_defense"))
    rd_grade = _safe_float(row.get("grades_run_defense"))

    # ── Career-weighted rates (stable, sample-size-resistant) ──────────
    c_sacks = c_prs = c_stops = c_pr_snaps = c_d_snaps = c_games = 0.0
    for _, r in valid_rows.iterrows():
        yr_g  = 17.0 if int(_safe_float(r.get("Year"), 2024)) >= 2021 else 16.0
        g     = max(1.0, min(yr_g, _safe_float(r.get("player_game_count"), yr_g)))
        sp    = _safe_float(r.get("snap_counts_pass_rush"), 0.0)
        sd    = _safe_float(r.get("snap_counts_defense"),   0.0)
        c_sacks    += _safe_float(r.get("sacks"))
        c_prs      += _safe_float(r.get("total_pressures"))
        c_stops    += _safe_float(r.get("stops"))
        c_pr_snaps += sp
        c_d_snaps  += sd
        c_games    += g

    c_pr_snaps = max(c_pr_snaps, 1.0)
    c_d_snaps  = max(c_d_snaps,  1.0)
    c_games    = max(c_games,    1.0)

    career_sack_rate    = c_sacks / c_pr_snaps * 100      # % per PR snap
    career_pressure_pct = c_prs   / c_pr_snaps * 100
    career_stop_rate    = c_stops  / c_d_snaps             # stops per D snap

    # Career average snaps per game → scale to 17 healthy games (starter-role floor)
    proj_snaps_pr = round(pass_rush_snap_load_17(c_pr_snaps, c_games))
    proj_snaps_d  = round(run_def_snap_load_17(c_d_snaps, c_games))

    sacks_17g     = round(career_sack_rate    / 100 * proj_snaps_pr, 1)
    pressures_17g = round(career_pressure_pct / 100 * proj_snaps_pr, 1)
    stops_17g     = round(career_stop_rate         * proj_snaps_d,   1)
    tfl_17g       = round((tfl  / games) * 17,                       1)
    ff_17g        = round((ff   / games) * 17,                       1)

    return {
        "season":          year,
        "games_played":    int(games),
        "max_games":       int(max_games),
        "availability":    avail,
        # Actual last-season counts (for display)
        "sacks":           round(sacks,    1),
        "pressures":       round(total_pr, 1),
        "pressure_pct":    round(career_pressure_pct, 2),  # career rate
        "sack_rate":       round(career_sack_rate,    2),  # career rate
        "stops":           round(stops,    1),
        "tfl":             round(tfl,      1),
        "forced_fumbles":  round(ff,       1),
        "snaps_pass_rush": round(snaps_pr),
        "snaps_defense":   round(snaps_d),
        "pass_rush_grade": round(pr_grade, 1),
        "run_def_grade":   round(rd_grade, 1),
        # 17-game full-health projections using career-weighted rates
        "sacks_17g":       sacks_17g,
        "pressures_17g":   pressures_17g,
        "stops_17g":       stops_17g,
        "tfl_17g":         tfl_17g,
        "ff_17g":          ff_17g,
        "snaps_pr_17g":    proj_snaps_pr,
        "snaps_d_17g":     proj_snaps_d,
    }


def _compute_health_factor(history: pd.DataFrame) -> tuple:
    """
    Weighted availability over the last 3 seasons (50 / 30 / 20).
    Returns (grade_adjustment_pts, weighted_avg_availability).

    Adjustment calibration:
      100% available → +2.5 pts  (stays healthy, earns bonus)
       75% available →  0 pts    (neutral baseline)
       50% available → -2.5 pts
       25% available → -5.0 pts  (chronic injury risk)
    """
    recent  = history.sort_values("Year").tail(3).reset_index(drop=True)
    weights = [0.20, 0.30, 0.50]  # oldest to newest
    n       = len(recent)

    avail_list = []
    for i, (_, row) in enumerate(recent.iterrows()):
        yr      = int(_safe_float(row.get("Year"), 2024))
        max_g   = 17.0 if yr >= 2021 else 16.0
        games   = max(1.0, min(max_g, _safe_float(row.get("player_game_count"), max_g)))
        avail_list.append(games / max_g)

    # Align weights to however many seasons are available
    w = weights[-n:]
    w = [x / sum(w) for x in w]  # renormalise
    avg_avail = sum(a * wt for a, wt in zip(avail_list, w))

    adj = (avg_avail - 0.75) * 10.0   # ±2.5 at ±25% from baseline
    adj = max(-5.0, min(2.5, adj))
    return round(adj, 2), round(avg_avail, 3)


# ─────────────────────────────────────────────
# Stat projection forward over contract
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
    Project stats forward year-by-year assuming full 17-game healthy seasons.
    Base = 17g-projected stats from last season; each year scaled by the
    grade ratio (projected_grade / composite_grade).
    """
    projections = []
    grade = composite_gr
    player_yoy = player_recent_grade_yoy(history, grade_col)

    for yr in range(1, contract_years + 1):
        age = current_age + yr - 1
        if yr > 1:
            grade = apply_yearly_grade_step(grade, age - 1, player_yoy, _annual_grade_delta)

        scale = max(0.25, min(1.5, grade / composite_gr)) if composite_gr > 0 else 1.0

        projections.append({
            "year":             yr,
            "age":              age,
            "projected_grade":  round(grade, 1),
            # All projected stats assume 17 healthy games
            "sacks":            round(min(22.0, last_stats["sacks_17g"]     * scale), 1),
            "pressures":        round(min(120.0,last_stats["pressures_17g"] * scale), 1),
            "pressure_pct":     round(min(25.0, last_stats["pressure_pct"]  * scale), 1),
            "stops":            round(min(65.0, last_stats["stops_17g"]     * scale), 1),
            "pass_rush_grade":  round(min(99.0, last_stats["pass_rush_grade"] * scale), 1),
            "run_def_grade":    round(min(99.0, last_stats["run_def_grade"]   * scale), 1),
        })

    return projections


# ─────────────────────────────────────────────
# Contract-adjusted valuation
# ─────────────────────────────────────────────
DISCOUNT_RATE   = 0.08   # 8%/yr — roster uncertainty & cap risk
CAP_GROWTH_RATE = 0.065  # 6.5%/yr — historical NFL cap growth


def compute_contract_value(
    composite_gr: float,
    current_age: int,
    contract_years: int,
    salary_ask: float,
    history: pd.DataFrame = None,
    grade_col: str = "grades_defense",
) -> tuple:
    """
    Returns:
        fair_aav             – PV of cap-inflated player value / yr
        effective_cap_burden – PV of cap-adjusted ask / yr
        total_nominal_value  – sum of nominal player values
        breakdown            – per-year dicts
    """
    breakdown           = []
    total_disc_value    = 0.0
    total_disc_ask      = 0.0
    total_nominal_value = 0.0
    grade               = float(composite_gr)
    player_yoy = player_recent_grade_yoy(history, grade_col)

    for yr in range(1, contract_years + 1):
        age = current_age + yr - 1
        if yr > 1:
            grade = apply_yearly_grade_step(grade, age - 1, player_yoy, _annual_grade_delta)

        cap_factor    = (1.0 + CAP_GROWTH_RATE) ** (yr - 1)
        time_discount = 1.0 / ((1.0 + DISCOUNT_RATE)   ** (yr - 1))

        base_value    = grade_to_market_value(grade)
        nominal_value = base_value * cap_factor
        disc_value    = nominal_value * time_discount

        cap_adj_ask   = salary_ask / cap_factor
        disc_ask      = cap_adj_ask * time_discount
        year_surplus  = round(base_value - cap_adj_ask, 2)

        total_nominal_value += nominal_value
        total_disc_value    += disc_value
        total_disc_ask      += disc_ask

        breakdown.append({
            "year":             yr,
            "age":              age,
            "projected_grade":  round(grade, 1),
            "market_value":     base_value,
            "nominal_value":    round(nominal_value, 2),
            "cap_adj_ask":      round(cap_adj_ask, 2),
            "discounted_value": round(disc_value, 2),
            "year_surplus":     year_surplus,
        })

    fair_aav             = round(total_disc_value / contract_years, 2)
    effective_cap_burden = round(total_disc_ask   / contract_years, 2)
    return fair_aav, effective_cap_burden, round(total_nominal_value, 2), breakdown


# ─────────────────────────────────────────────
# Agent State
# ─────────────────────────────────────────────
class EDAgentState(TypedDict):
    # Inputs
    player_name:     str
    salary_ask:      float
    contract_years:  int
    player_history:  pd.DataFrame

    # Populated by predict_performance
    predicted_tier:    str
    confidence:        Dict[str, float]
    current_age:       int
    last_season_stats: dict
    career_stats:      List[dict]
    stats_score:       float   # stats-based grade equivalent
    composite_grade:   float   # 40% model + 60% stats

    # Populated by evaluate_value
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

    # Populated by make_decision
    decision:  str
    reasoning: str


# ─────────────────────────────────────────────
# Node 1: Predict Performance
# ─────────────────────────────────────────────
def predict_performance(state: EDAgentState):
    print(f"[ED Agent] Predicting performance for {state['player_name']}...")

    tier, details = ed_engine.get_prediction(state["player_history"])
    model_grade   = details.get("predicted_grade", 60.0)

    # Age from last recorded season + current year offset
    history      = state["player_history"]
    current_year = datetime.date.today().year
    if "age" in history.columns and "Year" in history.columns:
        last_row           = history.sort_values("Year").iloc[-1]
        age_at_last_season = int(float(last_row["age"]))
        last_season_year   = int(float(last_row["Year"]))
        current_age        = age_at_last_season + (current_year - last_season_year)
    else:
        current_age = 28

    # Extract last season's stats (includes 17g projections)
    last_stats   = extract_last_season_stats(history)
    career_stats = extract_career_stats(history)

    # Health factor from last 3 seasons' availability (weighted recent-heavy)
    health_adj, avg_avail = _compute_health_factor(history)
    inactivity_adj, _ = inactivity_retirement_penalty(history, current_year=current_year)

    # Stats grade uses snap-rate metrics + 17g-projected stops (health-independent)
    sg = _stats_grade(
        last_stats["pressure_pct"],   # rate: unaffected by games played
        last_stats["sack_rate"],      # rate: unaffected by games played
        last_stats["stops_17g"],      # projected to full healthy season
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
            "stats_grade":       sg,
            "composite_grade":   cg,
            "health_factor":     health_adj,
            "inactivity_penalty": inactivity_adj,
            "avg_availability":  avg_avail,
            "xgb_grade":         details.get("xgb_grade"),
            "transformer_grade": details.get("transformer_grade"),
            "age_adjustment":    details.get("age_adjustment"),
        },
    }


# ─────────────────────────────────────────────
# Node 2: Evaluate Value
# ─────────────────────────────────────────────
def evaluate_value(state: EDAgentState):
    cg             = state["composite_grade"]
    current_age    = state["current_age"]
    contract_years = state["contract_years"]
    salary_ask     = state["salary_ask"]

    hist = state.get("player_history")
    fair_aav, eff_burden, total_nom, breakdown = compute_contract_value(
        cg, current_age, contract_years, salary_ask, history=hist, grade_col="grades_defense",
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
def assess_team_fit(state: EDAgentState):
    team = state.get("team_name", "")
    if not team:
        return {}

    cap_pcts = aav_to_cap_pcts(state["salary_ask"], state["contract_years"])
    return {"signing_cap_pcts": cap_pcts}


# ─────────────────────────────────────────────
# Node 3: Make Decision
# ─────────────────────────────────────────────
def make_decision(state: EDAgentState):
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

    health_adj  = state["confidence"].get("health_factor", 0)
    avg_avail   = state["confidence"].get("avg_availability", 1.0)
    health_str  = (
        f" Health factor: {'+' if health_adj >= 0 else ''}{health_adj} pts "
        f"(avg availability {round(avg_avail*100)}% over recent seasons)."
    )
    adj_str = f", age penalty: -{adj} pts" if adj else ""

    if age <= 26:   trajectory = "still developing toward his prime"
    elif age <= 29: trajectory = "in his prime years"
    elif age <= 32: trajectory = "entering post-prime decline"
    else:           trajectory = "in steep age-related decline"

    total_ask = round(ask * years, 2)
    cap_note  = (
        f"With ~{int(CAP_GROWTH_RATE*100)}%/yr cap growth the fixed ${ask}M AAV "
        f"costs effectively ${burden}M/yr in present-value cap terms."
    )

    team_nm = state.get("team_name", "")
    roster = state.get("current_roster") or []
    val_dec = val
    rep_note = ""
    if team_nm and roster:
        val_dec, rep_note = decision_fair_aav_with_replacement(
            val, grade_to_market_value, cg, roster, "ED",
        )

    surplus = round(val_dec - burden, 2)
    surplus_pct = (val_dec - burden) / max(val_dec, 0.01) * 100

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
        f"{state['player_name']} (age {age}) projects as a {tier} edge defender. "
        f"Model grade: {mg:.1f} · Stats grade (17g basis): {sg:.1f} → "
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
_workflow = StateGraph(EDAgentState)
_workflow.add_node("predict_performance", predict_performance)
_workflow.add_node("evaluate_value",      evaluate_value)
_workflow.add_node("assess_team_fit",     assess_team_fit)
_workflow.add_node("make_decision",       make_decision)
_workflow.set_entry_point("predict_performance")
_workflow.add_edge("predict_performance", "evaluate_value")
_workflow.add_edge("evaluate_value",      "assess_team_fit")
_workflow.add_edge("assess_team_fit",     "make_decision")
_workflow.add_edge("make_decision",       END)
ed_gm_agent = _workflow.compile()
