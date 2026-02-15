
from typing import TypedDict, Literal, Dict
from langgraph.graph import StateGraph, END
from backend.agent.model_wrapper import PlayerModelInference
import pandas as pd
import os

# Initialize Model Wrapper
# Using absolute paths to avoid CWD issues
MODEL_PATH = "/Users/pranaynandkeolyar/Documents/NFLSalaryCap/backend/ML/transformers/best_classifier.pth"
CSV_PATH = "/Users/pranaynandkeolyar/Documents/NFLSalaryCap/backend/ML/QB.csv"
SCALER_PATH = "/Users/pranaynandkeolyar/Documents/NFLSalaryCap/backend/ML/transformers/player_scaler.joblib"

# Global inference instance (loaded once)
# Now uses atomic scaler load instead of refitting on startup
inference_engine = PlayerModelInference(MODEL_PATH, scaler_path=SCALER_PATH)

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
    ask = state['salary_ask']
    
    # Simple Valuation Logic (Can be enhanced later)
    # Elite/High Quality -> Worth $40M+
    # Starter/Average -> Worth $20M - $35M
    # Reserve/Poor -> Worth < $10M
    
    # TEAM REVIEW NOTE:
    # The valuation logic below is currently a placeholder (Hardcoded Rules).
    # FUTURE WORK: Replace this with a "Market Value Model" trained on Salary Cap data.
    # Currently: Elite=$45M, Starter=$25M, Reserve=$5M.
    fair_value = 0.0
    if tier == "Elite/High Quality":
        fair_value = 45.0
    elif tier == "Starter/Average":
        fair_value = 25.0
    else:
        fair_value = 5.0
        
    return {"valuation": fair_value}

def make_decision(state: AgentState):
    """Final Sign/Pass Decision"""
    # TEAM REVIEW NOTE (STAGE 4 - AGENT LOGIC):
    # This logic is binary (Strict ROI).
    # REVIEW: Should we allow a "Soft Overpay" (e.g., within 10%) for Elite players?
    # Currently, if Ask > Value ($1), it's a hard PASS.
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
gm_agent = workflow.compile()
