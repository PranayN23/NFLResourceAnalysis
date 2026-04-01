
from typing import TypedDict, Dict
from langgraph.graph import StateGraph, END
from backend.agent.cb_model_wrapper import CBModelInference
import pandas as pd
import os

# Initialize Model Wrapper
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_PATH = os.path.join(BASE_DIR, "backend/ML/CB_Transformers/best_cb_classifier.pth")
CSV_PATH = os.path.join(BASE_DIR, "backend/ML/CB.csv")
SCALER_PATH = os.path.join(BASE_DIR, "backend/ML/CB_Transformers/cb_scaler.joblib")

# Global inference instance (loaded once)
inference_engine = CBModelInference(MODEL_PATH, scaler_path=SCALER_PATH)

class AgentState(TypedDict):
    player_name: str
    salary_ask: float  # in Millions
    player_history: pd.DataFrame  # Raw data for the player

    # Outputs
    predicted_tier: str
    confidence: Dict[str, float]
    valuation: float  # Estimated fair value
    decision: str     # "SIGN" or "PASS"
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

    # CB-specific valuation
    # Elite CBs: ~$18M/year (e.g., Jalen Ramsey, Sauce Gardner)
    # Starter CBs: ~$8M/year
    # Reserve CBs: ~$2M/year
    fair_value = 0.0
    if tier == "Elite":
        fair_value = 18.0
    elif tier == "Starter":
        fair_value = 8.0
    else:
        fair_value = 2.0

    return {"valuation": fair_value}


def make_decision(state: AgentState):
    """Final Sign/Pass Decision"""
    ask = state['salary_ask']
    val = state['valuation']
    tier = state['predicted_tier']

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
cb_gm_agent = workflow.compile()
