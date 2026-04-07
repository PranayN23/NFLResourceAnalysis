"""
ED GM Agent API

FastAPI chatbot endpoint for evaluating Edge Defender free agents.
Mirrors the pattern from main_api.py (QB) and rb_main_api.py (RB).

Usage:
    uvicorn backend.agent.ed_main_api:app --host 0.0.0.0 --port 8002
    POST /evaluate  {"player_name": "Myles Garrett", "salary_ask": 25.0}
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.agent.ed_agent_graph import ed_gm_agent, ED_CSV_PATH
import pandas as pd
import uvicorn
import os

app = FastAPI(title="NFL ED GM Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load player data once at startup
if os.path.exists(ED_CSV_PATH):
    df_players = pd.read_csv(ED_CSV_PATH)
    # Normalise player name column
    if "player" not in df_players.columns:
        for candidate in ["Player", "Name", "name"]:
            if candidate in df_players.columns:
                df_players.rename(columns={candidate: "player"}, inplace=True)
                break
    print(f"[ED API] Loaded player database: {len(df_players)} rows")
else:
    print(f"WARNING: ED player CSV not found at {ED_CSV_PATH}. Agent will fail to retrieve history.")
    df_players = pd.DataFrame()


@app.get("/ed-players")
async def get_ed_players():
    """Return sorted list of all ED player names available for evaluation."""
    if df_players.empty:
        raise HTTPException(status_code=503, detail="Player database not loaded.")
    names = sorted(df_players["player"].dropna().unique().tolist())
    return {"players": names}


class EvaluationRequest(BaseModel):
    player_name: str
    salary_ask: float  # in millions (AAV)


@app.post("/evaluate")
async def evaluate_player(req: EvaluationRequest):
    """
    Evaluate an Edge Defender free agent.

    Looks up the player's historical stats and runs the ED GM agent workflow,
    returning a SIGN / PASS recommendation with reasoning.
    """
    # 1. Retrieve player history
    player_data = df_players[df_players["player"] == req.player_name].copy()

    if len(player_data) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Player '{req.player_name}' not found in database.",
        )

    # 2. Initialise graph state
    initial_state = {
        "player_name":    req.player_name,
        "salary_ask":     req.salary_ask,
        "player_history": player_data,
        # Output fields — initialised to defaults
        "predicted_tier": "",
        "confidence":     {},
        "valuation":      0.0,
        "decision":       "",
        "reasoning":      "",
    }

    # 3. Run the LangGraph agent
    final_state = ed_gm_agent.invoke(initial_state)

    # 4. Return structured response
    return {
        "player":   req.player_name,
        "decision": final_state["decision"],
        "reasoning": final_state["reasoning"],
        "data": {
            "predicted_tier":    final_state["predicted_tier"],
            "valuation_estimate": final_state["valuation"],
            "confidence":        final_state["confidence"],
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
