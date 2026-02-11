
import os

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.agent.agent_graph import EDGE_CSV_PATH, QB_CSV_PATH, gm_agent

app = FastAPI(title="NFL GM Agent API")

# Load Data once for retrieval (QB + EDGE)
def _load_player_df(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        print(f"WARNING: Player CSV not found at {csv_path}.")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    if "player" not in df.columns:
        possible = ["Player", "Name", "name"]
        for p in possible:
            if p in df.columns:
                df.rename(columns={p: "player"}, inplace=True)
                break
    print(f"Loaded Player Database from {csv_path}: {len(df)} rows")
    return df


df_qb_players = _load_player_df(QB_CSV_PATH)
df_edge_players = _load_player_df(EDGE_CSV_PATH)


class EvaluationRequest(BaseModel):
    player_name: str
    salary_ask: float  # in Millions
    position: str = "QB"  # "QB" or "EDGE"


@app.post("/evaluate")
async def evaluate_player(req: EvaluationRequest):
    """
    Evaluates a Free Agent by looking up their stats and running the GM Agent Workflow.
    """
    pos = req.position.upper()
    if pos == "EDGE":
        df_players = df_edge_players
    else:
        df_players = df_qb_players
        pos = "QB"

    # 1. Retrieve Player History
    player_data = df_players[df_players["player"] == req.player_name].copy()

    if len(player_data) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Player '{req.player_name}' not found in {pos} database.",
        )

    # 2. Initialize State
    initial_state = {
        "player_name": req.player_name,
        "salary_ask": req.salary_ask,
        "position": pos,
        "player_history": player_data,
        # Outputs initialized to defaults
        "predicted_tier": "",
        "confidence": {},
        "valuation": 0.0,
        "decision": "",
        "reasoning": "",
    }

    # 3. Run Graph
    final_state = gm_agent.invoke(initial_state)

    # 4. Return Result
    return {
        "player": req.player_name,
        "position": pos,
        "decision": final_state["decision"],
        "reasoning": final_state["reasoning"],
        "data": {
            "predicted_tier": final_state["predicted_tier"],
            "valuation_estimate": final_state["valuation"],
            "confidence": final_state["confidence"],
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
