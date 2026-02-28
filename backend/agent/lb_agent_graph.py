
from typing import TypedDict, Literal, Dict
from langgraph.graph import StateGraph, END
from backend.agent.lb_model_wrapper import LBModelInference
import pandas as pd
import os

# Initialize Model Wrapper
# Using relative paths from project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_PATH = os.path.join(BASE_DIR, "backend/ML/LB_Transformers/best_lb_classifier.pth")
CSV_PATH = os.path.join(BASE_DIR, "backend/ML/LB.csv")
SCALER_PATH = os.path.join(BASE_DIR, "backend/ML/LB_Transformers/lb_scaler.joblib")

# Global inference instance (loaded once)
inference_engine = LBModelInference(MODEL_PATH, scaler_path=SCALER_PATH)

class AgentState(TypedDict):
    player_name: str
    salary_ask: float # in Millions
    player_history: pd.DataFrame # Raw data for the player

    # Outputs
    predicted_tier: str
    confidence: Dict[str, float]
    valuation: float # Estimated fair value
    decision: str # "SIGN" or "PASS"
    reasoning: str

def predict_performance(state: AgentState):
    """Run the PyTorch Model to get the Tier"""
    print(f"Agent: Predicting performance for {state['player_name']}...")

    tier, conf = inference_engine.predict(state['player_history'])

    return {
        "predicted_tier": tier,
        "confidence": conf
    }

def evaluate_value(state: AgentState):
    """Compare Predicted Tier to Salary Ask"""
    tier = state['predicted_tier']

    # LB-specific valuation (LBs are paid less than QBs)
    # Elite LBs: ~$20M/year (e.g., Micah Parsons, Fred Warner)
    # Starter LBs: ~$10M/year
    # Reserve LBs: ~$3M/year
    fair_value = 0.0
    if tier == "Elite":
        fair_value = 20.0
    elif tier == "Starter":
        fair_value = 10.0
    else:
        fair_value = 3.0

    return {"valuation": fair_value}

def make_decision(state: AgentState):
    """Final Sign/Pass Decision"""
    ask = state['salary_ask']
    val = state['valuation']
    tier = state['predicted_tier']

    decision = "PASS"
    reason = ""

    if ask <= val:
        decision = "SIGN"
        reason = f"Player is projected as {tier} (Value ${val}M) and asking for ${ask}M. This is a surplus value of ${val-ask}M."
    else:
        decision = "PASS"
        reason = f"Player is projected as {tier} (Value ${val}M) but is asking for ${ask}M. Overpay of ${ask-val}M."

    return {"decision": decision, "reasoning": reason}

# --- GRAPH CONSTRUCTION ---
workflow = StateGraph(AgentState)

workflow.add_node("predict_performance", predict_performance)
workflow.add_node("evaluate_value", evaluate_value)
workflow.add_node("make_decision", make_decision)

workflow.set_entry_point("predict_performance")

workflow.add_edge("predict_performance", "evaluate_value")
workflow.add_edge("evaluate_value", "make_decision")
workflow.add_edge("make_decision", END)

# Compile
lb_gm_agent = workflow.compile()
