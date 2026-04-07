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
# Continuous interpolation from PFF grade to AAV ($M).
# Calibrated to the 2022-2025 NFL edge rusher market.
# ─────────────────────────────────────────────
_GRADE_ANCHORS = [45,   55,   60,   65,   70,   75,   80,   85,   90,   95,   100]
_VALUE_ANCHORS = [0.75, 1.50, 2.50, 4.50, 7.50, 12.0, 17.0, 22.5, 28.0, 32.0, 35.0]


def grade_to_market_value(grade: float) -> float:
    """Return a player-specific single-year AAV estimate ($M) from a PFF grade."""
    grade = max(45.0, min(100.0, float(grade)))
    return round(float(np.interp(grade, _GRADE_ANCHORS, _VALUE_ANCHORS)), 2)


# ─────────────────────────────────────────────
# Age-based annual grade delta (growth + decline)
#
# Edge defenders develop through their mid-20s and peak ~27-29.
# Positive = improvement, negative = decline.
#
#   ≤ 22  → +2.5 pts/yr  (rookie development)
#   23-24 → +1.5 pts/yr  (sophomore breakout)
#   25-26 → +0.75 pts/yr (approaching prime)
#   27-29 →  0.0 pts/yr  (prime — stable peak)
#   30-31 → -1.0 pts/yr  (early decline)
#   32-33 → -1.5 pts/yr  (post-prime)
#   34+   → -2.5 pts/yr  (steep decline)
# ─────────────────────────────────────────────
def _annual_grade_delta(age: int) -> float:
    if age <= 22:
        return +2.5
    elif age <= 24:
        return +1.5
    elif age <= 26:
        return +0.75
    elif age <= 29:
        return  0.0
    elif age <= 31:
        return -1.0
    elif age <= 33:
        return -1.5
    else:
        return -2.5


# ─────────────────────────────────────────────
# Contract-adjusted valuation
#
# Projects grade forward year-by-year, converts each year's
# grade to a market value, then applies a time-discount factor
# (8% / yr) to reflect roster uncertainty and cap risk.
# Returns the effective fair AAV and a per-year breakdown.
# ─────────────────────────────────────────────
DISCOUNT_RATE = 0.08  # 8% annual discount


def compute_contract_value(
    grade: float,
    current_age: int,
    contract_years: int,
) -> tuple[float, float, List[dict]]:
    """
    Returns:
        fair_aav        – effective fair AAV accounting for decline & discounting ($M)
        total_fair_val  – sum of undiscounted yearly market values ($M)
        breakdown       – list of per-year dicts for display
    """
    breakdown = []
    total_discounted = 0.0
    total_fair_val   = 0.0
    current_grade    = float(grade)

    for yr in range(1, contract_years + 1):
        age_this_year = current_age + yr - 1

        # Apply age-curve delta from year 2 onward (year 1 = current model projection)
        if yr > 1:
            current_grade = max(45.0, min(99.0, current_grade + _annual_grade_delta(age_this_year - 1)))

        year_value  = grade_to_market_value(current_grade)
        discount    = 1.0 / ((1.0 + DISCOUNT_RATE) ** (yr - 1))
        disc_value  = round(year_value * discount, 2)

        total_fair_val   += year_value
        total_discounted += disc_value

        breakdown.append({
            "year":             yr,
            "age":              age_this_year,
            "projected_grade":  round(current_grade, 1),
            "market_value":     year_value,
            "discounted_value": disc_value,
        })

    # Effective fair AAV = total discounted value / years
    fair_aav = round(total_discounted / contract_years, 2)
    return fair_aav, round(total_fair_val, 2), breakdown


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
    predicted_tier:  str
    confidence:      Dict[str, float]
    current_age:     int
    valuation:       float          # effective fair AAV (contract-adjusted)
    total_fair_val:  float          # sum of undiscounted yearly values
    year_breakdown:  List[dict]
    decision:        str            # "SIGN" or "PASS"
    reasoning:       str


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
    """Project value over the contract length, discounting future years."""
    grade         = state["confidence"].get("predicted_grade", 55.0)
    current_age   = state.get("current_age", 28)
    contract_years = state.get("contract_years", 1)

    fair_aav, total_fair_val, breakdown = compute_contract_value(
        grade, current_age, contract_years
    )

    return {
        "valuation":      fair_aav,
        "total_fair_val": total_fair_val,
        "year_breakdown": breakdown,
    }


# ─────────────────────────────────────────────
# Node 3: Make Decision
# ─────────────────────────────────────────────
def make_decision(state: EDAgentState):
    """Return SIGN or PASS based on effective fair AAV vs. salary ask."""
    ask    = state["salary_ask"]
    val    = state["valuation"]          # effective fair AAV
    tier   = state["predicted_tier"]
    grade  = state["confidence"].get("predicted_grade", "N/A")
    age    = state.get("current_age", "?")
    years  = state.get("contract_years", 1)
    total  = state.get("total_fair_val", val * years)
    adj    = state["confidence"].get("age_adjustment", None)

    adj_str   = f", age penalty applied: -{adj} pts" if adj else ""
    total_ask = round(ask * years, 2)

    # Describe trajectory based on age
    if age <= 26:
        trajectory = "still developing and approaching his prime"
    elif age <= 29:
        trajectory = "in his prime years"
    elif age <= 32:
        trajectory = "entering post-prime decline"
    else:
        trajectory = "in steep age-related decline"

    if ask <= val:
        decision = "SIGN"
        surplus  = round(val - ask, 2)
        reason   = (
            f"{state['player_name']} (age {age}) projects as a {tier} edge defender "
            f"(predicted grade: {grade}{adj_str}). He is {trajectory}. "
            f"Over a {years}-year contract, the year-by-year grade trajectory and "
            f"a {int(DISCOUNT_RATE*100)}% annual time discount yield an effective fair AAV "
            f"of ${val}M/yr (total undiscounted fair value: ${total}M vs. total ask: ${total_ask}M). "
            f"At ${ask}M/yr this is a surplus of ${surplus}M/yr. Recommend signing."
        )
    else:
        decision = "PASS"
        overpay  = round(ask - val, 2)
        reason   = (
            f"{state['player_name']} (age {age}) projects as a {tier} edge defender "
            f"(predicted grade: {grade}{adj_str}). He is {trajectory}. "
            f"Over a {years}-year contract, the year-by-year grade trajectory and "
            f"a {int(DISCOUNT_RATE*100)}% annual time discount yield an effective fair AAV "
            f"of ${val}M/yr (total undiscounted fair value: ${total}M vs. total ask: ${total_ask}M). "
            f"At ${ask}M/yr this is an overpay of ${overpay}M/yr. Recommend passing."
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
