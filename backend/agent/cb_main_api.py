import os
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.agent.cb_agent_graph import cb_gm_agent

app = FastAPI(title="CB GM Agent API")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CSV_PATH = os.path.join(BASE_DIR, "backend/ML/CB.csv")

# Load CB data once at startup
df_cb = pd.read_csv(CSV_PATH)
numeric_cols = [
    'grades_defense', 'grades_coverage_defense', 'grades_tackle',
    'qb_rating_against', 'pass_break_ups', 'interceptions',
    'targets', 'snap_counts_corner', 'snap_counts_coverage',
    'snap_counts_slot', 'snap_counts_defense', 'Cap_Space',
    'tackles', 'stops', 'missed_tackle_rate'
]
df_cb['adjusted_value'] = pd.to_numeric(df_cb['adjusted_value'], errors='coerce').fillna(0)
for col in numeric_cols:
    if col in df_cb.columns:
        df_cb[col] = pd.to_numeric(df_cb[col], errors='coerce')


class EvaluateCBRequest(BaseModel):
    player_name: str
    salary_ask: float  # in millions


class EvaluateCBResponse(BaseModel):
    player_name: str
    predicted_tier: str
    predicted_grade: float
    transformer_grade: float
    age_adjustment: float
    volatility_index: float
    confidence_interval: list
    fair_value: float
    decision: str
    reasoning: str


@app.post("/evaluate-cb", response_model=EvaluateCBResponse)
def evaluate_cb(request: EvaluateCBRequest):
    player_name = request.player_name
    salary_ask = request.salary_ask

    history = df_cb[df_cb['player'] == player_name].copy()
    if history.empty:
        raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found in CB database.")

    history = history[history['snap_counts_defense'] >= 200].copy()
    if history.empty:
        raise HTTPException(
            status_code=422,
            detail=f"Player '{player_name}' has no seasons with >= 200 defensive snaps."
        )

    initial_state = {
        "player_name": player_name,
        "salary_ask": salary_ask,
        "player_history": history,
        "predicted_tier": "",
        "confidence": {},
        "valuation": 0.0,
        "decision": "",
        "reasoning": ""
    }

    result = cb_gm_agent.invoke(initial_state)

    conf = result["confidence"]
    return EvaluateCBResponse(
        player_name=player_name,
        predicted_tier=result["predicted_tier"],
        predicted_grade=conf.get("predicted_grade", 0.0),
        transformer_grade=conf.get("transformer_grade", 0.0),
        age_adjustment=conf.get("age_adjustment", 0.0),
        volatility_index=conf.get("volatility_index", 0.0),
        confidence_interval=list(conf.get("confidence_interval", (0.0, 0.0))),
        fair_value=result["valuation"],
        decision=result["decision"],
        reasoning=result["reasoning"]
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
