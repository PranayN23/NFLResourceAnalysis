"""
ED GM Agent — Edge Defender (EDGE/DE)

LangGraph agent that evaluates an Edge Defender free agent and returns a
SIGN / PASS recommendation based on the predicted grade tier vs. salary ask,
accounting for contract length, age-based performance decay, and time discounting.
"""

from typing import TypedDict, Dict, List
from langgraph.graph import StateGraph, END
from backend.agent.ed_model_wrapper import EDModelInference
import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

ED_CSV_PATH    = os.path.join(_BASE, "ML", "ED.csv")
ED_TRANSFORMER = os.path.join(_BASE, "ML", "ED_Transformers", "ed_best_classifier.pth")
ED_SCALER      = os.path.join(_BASE, "ML", "ED_Transformers", "ed_player_scaler.joblib")
ED_XGB         = os.path.join(_BASE, "ML", "ED_Transformers", "ed_best_xgb.joblib")

# Load inference engine once at import time
ed_engine = EDModelInference(ED_TRANSFORMER, scaler_path=ED_SCALER, xgb_path=ED_XGB)

# ─────────────────────────────────────────────
# Grade → Market Value curve
#
# Calibrated to the 2026 NFL edge rusher market (OTC data).
# Reference contracts:
#   Parsons $46.5M, Hutchinson $45M, Watt $41M,
#   Garrett $40M  → grades ~90-95 map to $40-47M
#   Crosby $35.5M, Bosa $34M       → grades ~85-88 map to $32-36M
#   Burns $28.2M, Oweh $24M        → grades ~80-84 map to $24-30M
#   Highsmith $17M, Greenard $19M  → grades ~73-77 map to $16-21M
#   Cooper $13.5M, Landry $14.5M   → grades ~68-72 map to $12-16M
#   Ossai $11.5M, Armstrong $11M   → grades ~63-67 map to $9-13M
#   Walker $9.3M, Enagbare $9M     → grades ~60-63 map to $7-10M
#   Depth / practice squad         → grades <58 map to $1-4M
# ─────────────────────────────────────────────
_GRADE_ANCHORS = [45,   55,   60,   65,   70,   75,   80,   85,   88,   92,   96,   100]
_VALUE_ANCHORS = [0.75, 2.00, 4.00, 8.50, 13.5, 19.0, 26.0, 33.0, 37.0, 42.0, 46.0, 50.0]


def grade_to_market_value(grade: float) -> float:
    """Return a player-specific single-year AAV estimate ($M) from a PFF grade."""
    grade = max(45.0, min(100.0, float(grade)))
    return round(float(np.interp(grade, _GRADE_ANCHORS, _VALUE_ANCHORS)), 2)


# ─────────────────────────────────────────────
# Age-based annual grade delta — derived from ED.csv (n=1,433 transitions)
#
# Empirical median year-over-year grade change by age:
#   Age 21 → +3.4   Age 22 → +3.5   Age 23 → +2.9
#   Age 24 → +1.2   Age 25 → -2.1   Age 26 → -0.8
#   Age 27 → -2.7   Age 28 → -0.1   Age 29 → -0.9
#   Age 30 → -4.1   Age 31 → -0.5   Age 32 → -8.8
#   Age 33 → -1.2   Age 34+ → -5.0  (small sample, use conservative)
#
# Age 36+ has <5 observations; capped at -5.0/yr.
# ─────────────────────────────────────────────
_AGE_DELTAS = {
    20: +3.0,
    21: +3.4,
    22: +3.5,
    23: +2.9,
    24: +1.2,
    25: -2.1,
    26: -0.8,
    27: -2.7,
    28: -0.1,
    29: -0.9,
    30: -4.1,
    31: -0.5,
    32: -8.8,
    33: -1.2,
    34: -5.0,
    35: -5.0,
}


def _annual_grade_delta(age: int) -> float:
    """Return empirically-derived annual grade delta for a given age."""
    if age <= 20:
        return +3.0
    if age >= 36:
        return -5.0
    return _AGE_DELTAS.get(age, -3.0)


# ─────────────────────────────────────────────
# Contract-adjusted valuation
#
# Two adjustments are applied per contract year:
#
# 1. CAP INFLATION — the NFL salary cap grows ~6.5%/yr.
#    A player's market value grows proportionally with the cap,
#    so their nominal worth in year N is:
#        base_value × (1 + CAP_GROWTH) ^ (N-1)
#    Conversely, a fixed AAV becomes a smaller % of the cap each
#    year, so the real cost of the ask shrinks:
#        cap_adj_ask = AAV / (1 + CAP_GROWTH) ^ (N-1)
#
# 2. TIME DISCOUNT — 8%/yr for roster uncertainty and cap-hit risk.
#    Applied to the cap-inflated player value to get its PV.
#    Net effective rate ≈ 8% - 6.5% = ~1.5%/yr, meaning future
#    years are barely discounted once cap growth is factored in.
#
# Decision comparison:
#   fair_aav (PV of cap-adjusted player value)
#   vs effective_cap_burden (PV of cap-adjusted ask)
# ─────────────────────────────────────────────
DISCOUNT_RATE   = 0.08   # 8%/yr — roster uncertainty & cap risk
CAP_GROWTH_RATE = 0.065  # 6.5%/yr — historical NFL cap growth average


def compute_contract_value(
    grade: float,
    current_age: int,
    contract_years: int,
    salary_ask: float,
) -> tuple[float, float, float, List[dict]]:
    """
    Returns:
        fair_aav             – PV of cap-inflated player value / yr ($M)
        effective_cap_burden – PV of cap-adjusted ask / yr ($M, decreases over time)
        total_nominal_value  – sum of nominal (cap-inflated) player values ($M)
        breakdown            – list of per-year dicts for display
    """
    breakdown               = []
    total_disc_value        = 0.0
    total_disc_ask          = 0.0
    total_nominal_value     = 0.0
    current_grade           = float(grade)

    for yr in range(1, contract_years + 1):
        age_this_year = current_age + yr - 1

        # Age-curve delta applied from year 2 onward
        if yr > 1:
            current_grade = max(45.0, min(99.0, current_grade + _annual_grade_delta(age_this_year - 1)))

        cap_factor    = (1.0 + CAP_GROWTH_RATE) ** (yr - 1)
        time_discount = 1.0 / ((1.0 + DISCOUNT_RATE) ** (yr - 1))

        base_value    = grade_to_market_value(current_grade)   # today's cap dollars
        nominal_value = base_value * cap_factor                 # inflated to future cap
        disc_value    = nominal_value * time_discount           # present value of inflated value

        cap_adj_ask   = salary_ask / cap_factor                 # ask in today's cap % terms
        disc_ask      = cap_adj_ask * time_discount             # PV of cap-adjusted ask

        total_nominal_value  += nominal_value
        total_disc_value     += disc_value
        total_disc_ask       += disc_ask

        breakdown.append({
            "year":             yr,
            "age":              age_this_year,
            "projected_grade":  round(current_grade, 1),
            "market_value":     base_value,
            "nominal_value":    round(nominal_value, 2),
            "cap_adj_ask":      round(cap_adj_ask, 2),
            "discounted_value": round(disc_value, 2),
        })

    fair_aav             = round(total_disc_value    / contract_years, 2)
    effective_cap_burden = round(total_disc_ask      / contract_years, 2)
    return fair_aav, effective_cap_burden, round(total_nominal_value, 2), breakdown


# ─────────────────────────────────────────────
# Agent State
# ─────────────────────────────────────────────
class EDAgentState(TypedDict):
    # Inputs
    player_name:     str
    salary_ask:      float          # AAV in $M
    contract_years:  int            # length of contract (1-5)
    player_history:  pd.DataFrame   # raw historical rows from ED.csv

    # Outputs (populated by graph nodes)
    predicted_tier:       str
    confidence:           Dict[str, float]
    current_age:          int
    valuation:            float   # fair AAV — PV of cap-inflated player value
    effective_cap_burden: float   # PV of cap-adjusted ask (real cost of contract)
    total_nominal_value:  float   # sum of nominal (cap-inflated) yearly player values
    year_breakdown:       List[dict]
    decision:             str     # "SIGN" or "PASS"
    reasoning:            str


# ─────────────────────────────────────────────
# Node 1: Predict Performance
# ─────────────────────────────────────────────
def predict_performance(state: EDAgentState):
    """Run the ED ensemble model to predict tier and confidence details."""
    print(f"[ED Agent] Predicting performance for {state['player_name']}...")

    tier, details = ed_engine.get_prediction(state["player_history"])

    # Extract current age from the most recent season row, adjusted to today's year
    history = state["player_history"]
    import datetime
    current_year = datetime.date.today().year
    if "age" in history.columns and "Year" in history.columns:
        last_row = history.sort_values("Year").iloc[-1]
        age_at_last_season = int(float(last_row["age"]))
        last_season_year   = int(float(last_row["Year"]))
        current_age = age_at_last_season + (current_year - last_season_year)
    else:
        current_age = 28

    return {
        "predicted_tier": tier,
        "current_age": current_age,
        "confidence": {
            "predicted_grade":   details.get("predicted_grade"),
            "xgb_grade":         details.get("xgb_grade"),
            "transformer_grade": details.get("transformer_grade"),
            "age_adjustment":    details.get("age_adjustment"),
        },
    }


# ─────────────────────────────────────────────
# Node 2: Evaluate Value
# ─────────────────────────────────────────────
def evaluate_value(state: EDAgentState):
    """Project value over the contract length with cap inflation and time discounting."""
    grade          = state["confidence"].get("predicted_grade", 55.0)
    current_age    = state.get("current_age", 28)
    contract_years = state.get("contract_years", 1)
    salary_ask     = state["salary_ask"]

    fair_aav, effective_cap_burden, total_nominal_value, breakdown = compute_contract_value(
        grade, current_age, contract_years, salary_ask
    )

    return {
        "valuation":            fair_aav,
        "effective_cap_burden": effective_cap_burden,
        "total_nominal_value":  total_nominal_value,
        "year_breakdown":       breakdown,
    }


# ─────────────────────────────────────────────
# Node 3: Make Decision
# ─────────────────────────────────────────────
def make_decision(state: EDAgentState):
    """
    SIGN if fair_aav (cap-inflated player value) >= effective_cap_burden (cap-adjusted ask).
    Both figures are in present-value terms so they're directly comparable.
    """
    ask      = state["salary_ask"]
    val      = state["valuation"]            # PV of cap-inflated player value
    burden   = state["effective_cap_burden"] # PV of cap-adjusted ask
    tier     = state["predicted_tier"]
    grade    = state["confidence"].get("predicted_grade", "N/A")
    age      = state.get("current_age", "?")
    years    = state.get("contract_years", 1)
    total_nv = state.get("total_nominal_value", 0.0)
    adj      = state["confidence"].get("age_adjustment", None)

    adj_str   = f", age penalty applied: -{adj} pts" if adj else ""
    total_ask = round(ask * years, 2)

    if age <= 26:
        trajectory = "still developing and approaching his prime"
    elif age <= 29:
        trajectory = "in his prime years"
    elif age <= 32:
        trajectory = "entering post-prime decline"
    else:
        trajectory = "in steep age-related decline"

    cap_note = (
        f"With the cap growing at ~{int(CAP_GROWTH_RATE*100)}%/yr, the fixed "
        f"AAV of ${ask}M becomes progressively cheaper as a cap percentage — "
        f"the cap-adjusted real cost averages ${burden}M/yr over the deal."
    )

    if val >= burden:
        decision = "SIGN"
        surplus  = round(val - burden, 2)
        reason   = (
            f"{state['player_name']} (age {age}) projects as a {tier} edge defender "
            f"(predicted grade: {grade}{adj_str}). He is {trajectory}. "
            f"Over a {years}-year contract his cap-inflation-adjusted fair value averages "
            f"${val}M/yr vs. an effective cap burden of ${burden}M/yr — "
            f"a surplus of ${surplus}M/yr in real cap terms. "
            f"{cap_note} "
            f"Total nominal player value over the deal: ${total_nv}M vs. total ask: ${total_ask}M. "
            f"Recommend signing."
        )
    else:
        decision = "PASS"
        overpay  = round(burden - val, 2)
        reason   = (
            f"{state['player_name']} (age {age}) projects as a {tier} edge defender "
            f"(predicted grade: {grade}{adj_str}). He is {trajectory}. "
            f"Over a {years}-year contract his cap-inflation-adjusted fair value averages "
            f"${val}M/yr vs. an effective cap burden of ${burden}M/yr — "
            f"an overpay of ${overpay}M/yr even after accounting for cap growth. "
            f"{cap_note} "
            f"Total nominal player value over the deal: ${total_nv}M vs. total ask: ${total_ask}M. "
            f"Recommend passing."
        )

    return {"decision": decision, "reasoning": reason}


# ─────────────────────────────────────────────
# Graph Construction
# ─────────────────────────────────────────────
_workflow = StateGraph(EDAgentState)

_workflow.add_node("predict_performance", predict_performance)
_workflow.add_node("evaluate_value",      evaluate_value)
_workflow.add_node("make_decision",       make_decision)

_workflow.set_entry_point("predict_performance")

_workflow.add_edge("predict_performance", "evaluate_value")
_workflow.add_edge("evaluate_value",      "make_decision")
_workflow.add_edge("make_decision",       END)

ed_gm_agent = _workflow.compile()
