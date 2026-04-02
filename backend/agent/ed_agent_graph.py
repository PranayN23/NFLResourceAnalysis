"""
ED GM Agent — Edge Defender (EDGE/DE)

LangGraph agent that evaluates an Edge Defender free agent and returns a
SIGN / PASS recommendation based on the predicted grade tier vs. salary ask.

Mirrors the pattern from rb_agent_graph.py (RB agent) but uses EDModelInference
and ED-specific salary valuation anchors.

Usage:
    from backend.agent.ed_agent_graph import ed_gm_agent, ED_CSV_PATH
    result = ed_gm_agent.invoke({
        "player_name": "Myles Garrett",
        "salary_ask": 25.0,
        "player_history": <DataFrame of player's historical seasons>,
        "predicted_tier": "", "confidence": {}, "valuation": 0.0,
        "decision": "", "reasoning": ""
    })
"""

from typing import TypedDict, Dict
from langgraph.graph import StateGraph, END
from backend.agent.ed_model_wrapper import EDModelInference
import pandas as pd
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
# ED Salary Valuation Anchors (current NFL market)
#
# Edge rushers are among the highest-paid non-QB positions.
# These anchors reflect realistic AAV for free-agent contracts
# based on the 2022-2025 market.
#
# FUTURE WORK: Replace with a data-driven market value model
# trained on historical contract AAV vs. grade outcomes.
# ─────────────────────────────────────────────
ED_FAIR_VALUE = {
    "Elite":         28.0,   # Top pass rusher — Garrett/Bosa/Micah level
    "Starter":       16.0,   # Reliable starter / above-average edge
    "Rotation":       6.0,   # Situational pass rusher / rotational player
    "Reserve/Poor":   1.5,   # Depth / camp body
}


# ─────────────────────────────────────────────
# Agent State
# ─────────────────────────────────────────────
class EDAgentState(TypedDict):
    # Inputs
    player_name:    str
    salary_ask:     float           # in millions (AAV)
    player_history: pd.DataFrame   # raw historical seasons from ED.csv

    # Outputs (populated by graph nodes)
    predicted_tier: str
    confidence:     Dict[str, float]
    valuation:      float           # estimated fair market value (millions)
    decision:       str             # "SIGN" or "PASS"
    reasoning:      str


# ─────────────────────────────────────────────
# Node 1: Predict Performance
# ─────────────────────────────────────────────
def predict_performance(state: EDAgentState):
    """Run the ED ensemble model to predict tier and confidence details."""
    print(f"[ED Agent] Predicting performance for {state['player_name']}...")

    tier, details = ed_engine.get_prediction(state["player_history"])

    return {
        "predicted_tier": tier,
        "confidence": {
            "predicted_grade":   details.get("predicted_grade"),
            "xgb_grade":         details.get("xgb_grade"),
            "transformer_grade": details.get("transformer_grade"),
            "age_adjustment":    details.get("age_adjustment"),
        }
    }


# ─────────────────────────────────────────────
# Node 2: Evaluate Value
# ─────────────────────────────────────────────
def evaluate_value(state: EDAgentState):
    """Map predicted tier to an estimated fair market value."""
    tier = state["predicted_tier"]
    fair_value = ED_FAIR_VALUE.get(tier, 1.5)
    return {"valuation": fair_value}


# ─────────────────────────────────────────────
# Node 3: Make Decision
# ─────────────────────────────────────────────
def make_decision(state: EDAgentState):
    """Return SIGN or PASS based on salary ask vs. fair value."""
    ask   = state["salary_ask"]
    val   = state["valuation"]
    tier  = state["predicted_tier"]
    grade = state["confidence"].get("predicted_grade", "N/A")
    adj   = state["confidence"].get("age_adjustment", None)

    adj_str = f", age adjustment: -{adj} pts" if adj is not None else ""

    if ask <= val:
        decision = "SIGN"
        surplus  = round(val - ask, 1)
        reason   = (
            f"{state['player_name']} projects as a {tier} edge defender "
            f"(predicted grade: {grade}{adj_str}). "
            f"Estimated fair value is ${val}M/yr — asking ${ask}M/yr is a "
            f"surplus value of ${surplus}M. Recommend signing."
        )
    else:
        decision = "PASS"
        overpay  = round(ask - val, 1)
        reason   = (
            f"{state['player_name']} projects as a {tier} edge defender "
            f"(predicted grade: {grade}{adj_str}). "
            f"Estimated fair value is ${val}M/yr — asking ${ask}M/yr is an "
            f"overpay of ${overpay}M. Recommend passing."
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
