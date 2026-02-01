
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.agent.agent_graph import gm_agent, CSV_PATH
import pandas as pd
import uvicorn
import os

app = FastAPI(title="NFL GM Agent API")

# Load Data once for retrieval
if os.path.exists(CSV_PATH):
    df_players = pd.read_csv(CSV_PATH)
    # Ensure necessary columns
    if 'player' not in df_players.columns:
         # Try to find it again like in training
         possible = ['Player', 'Name', 'name']
         for p in possible:
             if p in df_players.columns:
                 df_players.rename(columns={p: 'player'}, inplace=True)
                 break
    print(f"Loaded Player Database: {len(df_players)} rows")
else:
    print("WARNING: Player CSV not found. Agent will fail to retrieve history.")
    df_players = pd.DataFrame()

class EvaluationRequest(BaseModel):
    player_name: str
    salary_ask: float # in Millions

@app.post("/evaluate")
async def evaluate_player(req: EvaluationRequest):
    """
    Evaluates a Free Agent by looking up their stats and running the GM Agent Workflow.
    """
    # 1. Retrieve Player History
    player_data = df_players[df_players['player'] == req.player_name].copy()
    
    if len(player_data) == 0:
        raise HTTPException(status_code=404, detail=f"Player '{req.player_name}' not found in database.")
    
    # 2. Initialize State
    initial_state = {
        "player_name": req.player_name,
        "salary_ask": req.salary_ask,
        "player_history": player_data,
        # Outputs initialized to defaults
        "predicted_tier": "",
        "confidence": {},
        "valuation": 0.0,
        "decision": "",
        "reasoning": ""
    }
    
    # 3. Run Graph
    final_state = gm_agent.invoke(initial_state)
    
    # 4. Return Result
    return {
        "player": req.player_name,
        "decision": final_state['decision'],
        "reasoning": final_state['reasoning'],
        "data": {
            "predicted_tier": final_state['predicted_tier'],
            "valuation_estimate": final_state['valuation'],
            "confidence": final_state['confidence']
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
