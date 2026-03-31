"""
RB GM Agent — Siddarth Nachannagari

LangGraph agent that evaluates a Running Back free agent and returns a
SIGN / PASS recommendation based on the predicted grade tier vs. salary ask.

Mirrors the pattern from agent_graph.py (QB agent) but uses RBModelInference
and RB-specific salary valuation anchors.

Usage:
    from backend.agent.rb_agent_graph import rb_gm_agent, RB_CSV_PATH
    result = rb_gm_agent.invoke({
        "player_name": "Saquon Barkley",
        "salary_ask": 14.0,
        "player_history": <DataFrame of player's historical seasons>,
        "predicted_tier": "", "confidence": {}, "valuation": 0.0,
        "decision": "", "reasoning": ""
    })
"""

from typing import TypedDict, Dict
from langgraph.graph import StateGraph, END
from backend.agent.rb_model_wrapper import RBModelInference
import pandas as pd
import os

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

RB_CSV_PATH       = os.path.join(_BASE, "ML", "HB.csv")
RB_TRANSFORMER    = os.path.join(_BASE, "ML", "RB_Pranay_Transformers", "rb_best_classifier.pth")
RB_SCALER         = os.path.join(_BASE, "ML", "RB_Pranay_Transformers", "rb_player_scaler.joblib")
RB_XGB            = os.path.join(_BASE, "ML", "RB_Pranay_Transformers", "rb_best_xgb.joblib")

# Load inference engine once at import time
rb_engine = RBModelInference(RB_TRANSFORMER, scaler_path=RB_SCALER, xgb_path=RB_XGB)

# ─────────────────────────────────────────────
# RB Salary Valuation Anchors (current NFL market)
#
# RBs are the most undervalued position in the NFL.
# These anchors reflect realistic AAV for free-agent contracts
# based on the 2022-2025 market.
#
# FUTURE WORK: Replace with a data-driven market value model
# trained on historical contract AAV vs. grade outcomes.
# ─────────────────────────────────────────────
RB_FAIR_VALUE = {
    "Elite":         18.0,   # Franchise RB — top 10 (e.g., CMC, Barkley level)
    "Starter":       10.0,   # Reliable starter — above average
    "Rotation":       4.0,   # Committee back / role player
    "Reserve/Poor":   1.0,   # Depth / practice squad
}


# ─────────────────────────────────────────────
# Agent State
# ─────────────────────────────────────────────
class RBAgentState(TypedDict):
    # Inputs
    player_name:    str
    salary_ask:     float           # in millions (AAV)
    player_history: pd.DataFrame   # raw historical seasons from HB.csv

    # Outputs (populated by graph nodes)
    predicted_tier: str
    confidence:     Dict[str, float]
    valuation:      float           # estimated fair market value (millions)
    decision:       str             # "SIGN" or "PASS"
    reasoning:      str


# ─────────────────────────────────────────────
# Node 1: Predict Performance
# ─────────────────────────────────────────────
def predict_performance(state: RBAgentState):
    """Run the RB ensemble model to predict tier and confidence details."""
    print(f"[RB Agent] Predicting performance for {state['player_name']}...")

    tier, details = rb_engine.get_prediction(state["player_history"], apply_calibration=True)

    return {
        "predicted_tier": tier,
        "confidence": {
            "predicted_grade":    details.get("predicted_grade"),
            "xgb_grade":          details.get("xgb_grade"),
            "transformer_grade":  details.get("transformer_grade"),
            "age_adjustment":     details.get("age_adjustment"),
            "volatility_index":   details.get("volatility_index"),
            "conf_lower":         details.get("confidence_interval", (None, None))[0],
            "conf_upper":         details.get("confidence_interval", (None, None))[1],
        }
    }


# ─────────────────────────────────────────────
# Node 2: Evaluate Value
# ─────────────────────────────────────────────
def evaluate_value(state: RBAgentState):
    """Map predicted tier to an estimated fair market value."""
    tier = state["predicted_tier"]
    fair_value = RB_FAIR_VALUE.get(tier, 1.0)
    return {"valuation": fair_value}


# ─────────────────────────────────────────────
# Node 3: Make Decision
# ─────────────────────────────────────────────
def make_decision(state: RBAgentState):
    """Return SIGN or PASS based on salary ask vs. fair value."""
    ask   = state["salary_ask"]
    val   = state["valuation"]
    tier  = state["predicted_tier"]
    grade = state["confidence"].get("predicted_grade", "N/A")
    vol   = state["confidence"].get("volatility_index", None)
    lower = state["confidence"].get("conf_lower")
    upper = state["confidence"].get("conf_upper")

    conf_str = f" (grade range: {lower}–{upper})" if lower is not None else ""
    vol_str  = f", volatility index: {vol:.2f}" if vol is not None else ""

    if ask <= val:
        decision = "SIGN"
        surplus  = round(val - ask, 1)
        reason   = (
            f"{state['player_name']} projects as a {tier} RB "
            f"(predicted grade: {grade}{conf_str}{vol_str}). "
            f"Estimated fair value is ${val}M/yr — asking ${ask}M/yr is a "
            f"surplus value of ${surplus}M. Recommend signing."
        )
    else:
        decision = "PASS"
        overpay  = round(ask - val, 1)
        reason   = (
            f"{state['player_name']} projects as a {tier} RB "
            f"(predicted grade: {grade}{conf_str}{vol_str}). "
            f"Estimated fair value is ${val}M/yr — asking ${ask}M/yr is an "
            f"overpay of ${overpay}M. Recommend passing."
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
