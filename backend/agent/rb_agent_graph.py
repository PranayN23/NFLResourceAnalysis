"""
RB GM Agent — Siddarth Nachannagari

LangGraph agent that evaluates a Running Back free agent and returns a
tiered signing recommendation. Uses the RB model's PFF grade blended with
an objective stats-based grade (yards/touch, elusive rating, explosive
run rate, scoring rate), then projects that composite grade — and the
associated stats — forward year-by-year over the contract, accounting
for the empirical RB age curve, cap inflation, and time discounting.

RBs are the most age-sensitive position in football — the decline starts
at 27 and accelerates hard after 29. The market values them accordingly.
"""

from typing import TypedDict, Dict, List
from langgraph.graph import StateGraph, END
from backend.agent.rb_model_wrapper import RBModelInference
import pandas as pd
import numpy as np
import os, datetime

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

RB_CSV_PATH    = os.path.join(_BASE, "ML", "HB.csv")
RB_TRANSFORMER = os.path.join(_BASE, "ML", "RB_Pranay_Transformers", "rb_best_classifier.pth")
RB_SCALER      = os.path.join(_BASE, "ML", "RB_Pranay_Transformers", "rb_player_scaler.joblib")
RB_XGB         = os.path.join(_BASE, "ML", "RB_Pranay_Transformers", "rb_best_xgb.joblib")

rb_engine = RBModelInference(RB_TRANSFORMER, scaler_path=RB_SCALER, xgb_path=RB_XGB)


# ─────────────────────────────────────────────
# Grade → Market Value curve (2026 OTC calibrated for RBs)
#
# RBs are the most undervalued position. Top RB contracts currently
# max out around $18–21M AAV (CMC, Barkley, Henry tier). Starters sit
# $6–12M, rotational backs $2–4M, depth at veteran minimum.
# ─────────────────────────────────────────────
_GRADE_ANCHORS = [45,   55,   60,   65,   70,   75,   80,   85,   90,   95]
_VALUE_ANCHORS = [0.80, 1.50, 2.50, 4.00, 6.50, 10.0, 14.0, 17.0, 19.5, 22.0]


def grade_to_market_value(grade: float) -> float:
    grade = max(45.0, min(99.0, float(grade)))
    return round(float(np.interp(grade, _GRADE_ANCHORS, _VALUE_ANCHORS)), 2)


# ─────────────────────────────────────────────
# Stats-based grade equivalent
#
# RB production is best captured by efficiency + explosiveness, not
# raw volume (volume is scheme-dependent). Anchors are empirical
# medians across the HB.csv.
#
# yards_per_touch   — 3.5 reserve, 4.3 rotation, 5.0 starter, 5.9 elite
# yco_per_attempt   — 1.6, 2.1, 2.7, 3.4
# elusive_rating    — 30, 50, 75, 110
# breakaway_percent — 15, 28, 40, 55
# ─────────────────────────────────────────────
_YPT_ANCHORS       = [0.0,  3.5,  4.3,  5.0,  5.9,  8.0]
_YCO_ATT_ANCHORS   = [0.0,  1.6,  2.1,  2.7,  3.4,  5.0]
_ELUSIVE_ANCHORS   = [0.0, 30.0, 50.0, 75.0, 110.0, 180.0]
_BREAKAWAY_ANCHORS = [0.0, 15.0, 28.0, 40.0, 55.0, 80.0]
_STAT_GRD_SCALE    = [45.0, 55.0, 65.0, 75.0, 85.0, 99.0]


def _stats_grade(ypt: float, yco_att: float, elusive: float, breakaway: float) -> float:
    """Map efficiency stats to a 45-99 grade-equivalent score."""
    a = float(np.interp(ypt,       _YPT_ANCHORS,       _STAT_GRD_SCALE))
    b = float(np.interp(yco_att,   _YCO_ATT_ANCHORS,   _STAT_GRD_SCALE))
    c = float(np.interp(elusive,   _ELUSIVE_ANCHORS,   _STAT_GRD_SCALE))
    d = float(np.interp(breakaway, _BREAKAWAY_ANCHORS, _STAT_GRD_SCALE))
    return round(0.30 * a + 0.25 * b + 0.25 * c + 0.20 * d, 2)


def _composite_grade(model_grade: float, stats_gr: float) -> float:
    """Blend model PFF grade (40%) with stats-based grade (60%)."""
    return round(0.40 * model_grade + 0.60 * stats_gr, 2)


def _grade_to_tier(grade: float) -> str:
    if grade >= 80: return "Elite"
    if grade >= 70: return "Starter"
    if grade >= 60: return "Rotation"
    return "Reserve/Poor"


# ─────────────────────────────────────────────
# Age-based annual grade delta for RBs
#
# RBs age the fastest of any position. Peak is 24-26, then decline
# is rapid and non-linear. Calibrated from empirical RB season
# transitions.
# ─────────────────────────────────────────────
_AGE_DELTAS = {
    20: +3.5, 21: +3.0, 22: +2.5, 23: +1.5, 24: +0.5,
    25: -0.3, 26: -1.5, 27: -2.8, 28: -4.0, 29: -5.5,
    30: -7.0, 31: -8.5, 32: -10.0, 33: -10.0, 34: -10.0,
}


def _annual_grade_delta(age: int) -> float:
    if age <= 20: return +3.5
    if age >= 35: return -10.0
    return _AGE_DELTAS.get(age, -5.0)


# ─────────────────────────────────────────────
# Stats extraction helpers
# ─────────────────────────────────────────────
def _safe_float(val, default=0.0) -> float:
    try:
        f = float(val)
        return default if np.isnan(f) else f
    except Exception:
        return default


def _has_valid_stats(row) -> bool:
    """Return True if a row has at least one key counting stat."""
    for col in ("total_touches", "yards", "attempts", "receptions"):
        val = row.get(col)
        try:
            f = float(val)
            if not np.isnan(f) and f > 0:
                return True
        except Exception:
            pass
    return False


def _season_stats_row(row) -> dict:
    """Pull display stats for a single season row."""
    year     = int(_safe_float(row.get("Year"), 2024))
    attempts = _safe_float(row.get("attempts"))
    yards    = _safe_float(row.get("yards"))
    yco      = _safe_float(row.get("yards_after_contact"))
    yco_att  = _safe_float(row.get("yco_attempt"))
    touches  = _safe_float(row.get("total_touches"))
    tds      = _safe_float(row.get("touchdowns"))
    receptions = _safe_float(row.get("receptions"))
    rec_yards  = _safe_float(row.get("rec_yards"))
    targets    = _safe_float(row.get("targets"))
    elusive    = _safe_float(row.get("elusive_rating"))
    breakaway  = _safe_float(row.get("breakaway_percent"))
    explosive  = _safe_float(row.get("explosive"))
    first_downs = _safe_float(row.get("first_downs"))
    fumbles    = _safe_float(row.get("fumbles"))
    ypa        = _safe_float(row.get("ypa"))
    yprr       = _safe_float(row.get("yprr"))
    overall    = _safe_float(row.get("grades_offense"))
    run_gr     = _safe_float(row.get("grades_run"))
    pass_gr    = _safe_float(row.get("grades_pass_route"))

    ypt = round(yards / touches, 2) if touches > 0 else 0.0

    return {
        "season":          year,
        "attempts":        int(attempts),
        "yards":           int(yards),
        "yards_after_contact": int(yco),
        "yco_per_attempt": round(yco_att, 2),
        "ypa":             round(ypa, 2),
        "total_touches":   int(touches),
        "touchdowns":      int(tds),
        "receptions":      int(receptions),
        "rec_yards":       int(rec_yards),
        "targets":         int(targets),
        "yards_per_touch": ypt,
        "elusive_rating":  round(elusive, 1),
        "breakaway_pct":   round(breakaway, 1),
        "explosive":       int(explosive),
        "first_downs":     int(first_downs),
        "fumbles":         int(fumbles),
        "yprr":            round(yprr, 2),
        "run_grade":       round(run_gr, 1),
        "pass_grade":      round(pass_gr, 1),
        "overall_grade":   round(overall, 1),
    }


def extract_career_stats(history: pd.DataFrame) -> List[dict]:
    """Return per-season stats for every recorded season that has data."""
    seasons = []
    for _, row in history.sort_values("Year").iterrows():
        if not _has_valid_stats(row):
            continue
        seasons.append(_season_stats_row(row))
    return seasons


def extract_last_season_stats(history: pd.DataFrame) -> dict:
    """Pull the most recent season with actual data."""
    sorted_hist = history.sort_values("Year")
    valid_rows = sorted_hist[sorted_hist.apply(_has_valid_stats, axis=1)]
    if valid_rows.empty:
        valid_rows = sorted_hist

    row = valid_rows.iloc[-1]
    stats = _season_stats_row(row)

    # Career-weighted efficiency rates (sample-size-resistant)
    c_yards = c_touches = c_yco = c_attempts = c_tds = 0.0
    c_elusive = c_breakaway = 0.0
    n = 0
    for _, r in valid_rows.iterrows():
        att = _safe_float(r.get("attempts"))
        c_yards    += _safe_float(r.get("yards"))
        c_touches  += _safe_float(r.get("total_touches"))
        c_yco      += _safe_float(r.get("yards_after_contact"))
        c_attempts += att
        c_tds      += _safe_float(r.get("touchdowns"))
        c_elusive   += _safe_float(r.get("elusive_rating"))
        c_breakaway += _safe_float(r.get("breakaway_percent"))
        n += 1

    career_ypt       = c_yards / c_touches if c_touches > 0 else 0.0
    career_yco_att   = c_yco   / c_attempts if c_attempts > 0 else 0.0
    career_elusive   = c_elusive   / n if n > 0 else 0.0
    career_breakaway = c_breakaway / n if n > 0 else 0.0

    stats["career_ypt"]       = round(career_ypt, 2)
    stats["career_yco_att"]   = round(career_yco_att, 2)
    stats["career_elusive"]   = round(career_elusive, 1)
    stats["career_breakaway"] = round(career_breakaway, 1)

    return stats


# ─────────────────────────────────────────────
# Stat projection forward over contract
# ─────────────────────────────────────────────
def project_stats(last_stats: dict, composite_gr: float,
                  current_age: int, contract_years: int) -> List[dict]:
    """
    Project RB stats forward year-by-year. Each year's projected grade
    scales the previous season's key counting stats.
    """
    projections = []
    grade = composite_gr

    for yr in range(1, contract_years + 1):
        age = current_age + yr - 1
        if yr > 1:
            grade = max(45.0, min(99.0, grade + _annual_grade_delta(age - 1)))

        scale = max(0.25, min(1.4, grade / composite_gr)) if composite_gr > 0 else 1.0

        projections.append({
            "year":             yr,
            "age":              age,
            "projected_grade":  round(grade, 1),
            "attempts":         int(round(last_stats["attempts"]      * scale)),
            "yards":            int(round(last_stats["yards"]         * scale)),
            "yards_per_touch":  round(min(8.0,  last_stats["yards_per_touch"] * ((scale + 1) / 2)), 2),
            "yco_per_attempt":  round(min(5.0,  last_stats["yco_per_attempt"] * ((scale + 1) / 2)), 2),
            "touchdowns":       int(round(last_stats["touchdowns"]    * scale)),
            "receptions":       int(round(last_stats["receptions"]    * scale)),
            "rec_yards":        int(round(last_stats["rec_yards"]     * scale)),
            "elusive_rating":   round(min(180.0, last_stats["elusive_rating"] * ((scale + 1) / 2)), 1),
            "breakaway_pct":    round(min(80.0,  last_stats["breakaway_pct"]  * ((scale + 1) / 2)), 1),
            "run_grade":        round(min(99.0, last_stats["run_grade"]  * scale), 1),
            "overall_grade":    round(grade, 1),
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

    for yr in range(1, contract_years + 1):
        age = current_age + yr - 1
        if yr > 1:
            grade = max(45.0, min(99.0, grade + _annual_grade_delta(age - 1)))

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
class RBAgentState(TypedDict):
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
    stats_score:       float
    composite_grade:   float

    # Populated by evaluate_value
    valuation:            float
    effective_cap_burden: float
    total_nominal_value:  float
    year_breakdown:       List[dict]
    projected_stats:      List[dict]

    # Populated by make_decision
    decision:  str
    reasoning: str


# ─────────────────────────────────────────────
# Node 1: Predict Performance
# ─────────────────────────────────────────────
def predict_performance(state: RBAgentState):
    print(f"[RB Agent] Predicting performance for {state['player_name']}...")

    tier, details = rb_engine.get_prediction(state["player_history"], apply_calibration=True)
    model_grade   = float(details.get("predicted_grade", 60.0))

    history      = state["player_history"]
    current_year = datetime.date.today().year
    if "age" in history.columns and "Year" in history.columns:
        last_row           = history.sort_values("Year").iloc[-1]
        age_at_last_season = int(_safe_float(last_row["age"], 26))
        last_season_year   = int(_safe_float(last_row["Year"], current_year - 1))
        current_age        = age_at_last_season + (current_year - last_season_year)
    else:
        current_age = 26

    last_stats   = extract_last_season_stats(history)
    career_stats = extract_career_stats(history)

    # Stats grade uses career-weighted efficiency metrics
    sg = _stats_grade(
        last_stats.get("career_ypt",       last_stats["yards_per_touch"]),
        last_stats.get("career_yco_att",   last_stats["yco_per_attempt"]),
        last_stats.get("career_elusive",   last_stats["elusive_rating"]),
        last_stats.get("career_breakaway", last_stats["breakaway_pct"]),
    )

    cg = _composite_grade(model_grade, sg)
    cg = round(max(45.0, min(99.0, cg)), 2)

    return {
        "predicted_tier":    _grade_to_tier(cg),
        "current_age":       current_age,
        "last_season_stats": last_stats,
        "career_stats":      career_stats,
        "stats_score":       sg,
        "composite_grade":   cg,
        "confidence": {
            "model_grade":       float(round(model_grade, 2)),
            "stats_grade":       float(sg),
            "composite_grade":   float(cg),
            "xgb_grade":         float(details["xgb_grade"]) if details.get("xgb_grade") is not None else None,
            "transformer_grade": float(details["transformer_grade"]) if details.get("transformer_grade") is not None else None,
            "age_adjustment":    float(details["age_adjustment"]) if details.get("age_adjustment") is not None else None,
            "volatility_index":  float(details["volatility_index"]) if details.get("volatility_index") is not None else None,
        },
    }


# ─────────────────────────────────────────────
# Node 2: Evaluate Value
# ─────────────────────────────────────────────
def evaluate_value(state: RBAgentState):
    cg             = state["composite_grade"]
    current_age    = state["current_age"]
    contract_years = state["contract_years"]
    salary_ask     = state["salary_ask"]

    fair_aav, eff_burden, total_nom, breakdown = compute_contract_value(
        cg, current_age, contract_years, salary_ask
    )

    stat_proj = project_stats(
        state["last_season_stats"], cg, current_age, contract_years
    )

    return {
        "valuation":            fair_aav,
        "effective_cap_burden": eff_burden,
        "total_nominal_value":  total_nom,
        "year_breakdown":       breakdown,
        "projected_stats":      stat_proj,
    }


# ─────────────────────────────────────────────
# Node 3: Make Decision
# ─────────────────────────────────────────────
def make_decision(state: RBAgentState):
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

    adj_str = f", age penalty: -{adj} pts" if adj else ""

    if age <= 24:   trajectory = "still ascending toward his peak"
    elif age <= 26: trajectory = "in his prime years"
    elif age <= 28: trajectory = "at the edge of his prime — decline is imminent"
    elif age <= 30: trajectory = "in post-prime decline — the cliff is near"
    else:           trajectory = "past the RB cliff and in steep decline"

    total_ask = round(ask * years, 2)
    cap_note  = (
        f"With ~{int(CAP_GROWTH_RATE*100)}%/yr cap growth the fixed ${ask}M AAV "
        f"costs effectively ${burden}M/yr in present-value cap terms."
    )

    surplus     = round(val - burden, 2)
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
        f"{state['player_name']} (age {age}) projects as a {tier} running back. "
        f"Model grade: {mg:.1f} · Stats grade (efficiency basis): {sg:.1f} → "
        f"Composite: {cg:.1f}{adj_str}. He is {trajectory}. "
        f"Over a {years}-yr contract the composite projects a cap-inflation-adjusted "
        f"fair value of ${val}M/yr vs. an effective cap burden of ${burden}M/yr — "
        f"{verdict_str} {cap_note} Total nominal player value: ${total}M vs. total ask: "
        f"${total_ask}M. {rec}"
    )

    return {"decision": decision, "reasoning": reason}


# ─────────────────────────────────────────────
# Graph Construction
# ─────────────────────────────────────────────
_workflow = StateGraph(RBAgentState)
_workflow.add_node("predict_performance", predict_performance)
_workflow.add_node("evaluate_value",      evaluate_value)
_workflow.add_node("make_decision",       make_decision)
_workflow.set_entry_point("predict_performance")
_workflow.add_edge("predict_performance", "evaluate_value")
_workflow.add_edge("evaluate_value",      "make_decision")
_workflow.add_edge("make_decision",       END)
rb_gm_agent = _workflow.compile()
