
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from backend.agent.agent_graph import gm_agent, CSV_PATH
from backend.agent.team_agent_graph import run_team_evaluation
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


class PlayerProjection(BaseModel):
    name: str
    position: str
    projected_grade: float

class RosterEvaluationRequest(BaseModel):
    team: str                                  # e.g. "KC", "BAL"
    year: int                                  # base year to read context from
    position_overrides: Optional[dict] = None  # e.g. {"lag_qb_grade": 55.0}
    players: Optional[list[PlayerProjection]] = None # List of players to aggregate into overrides


@app.post("/evaluate-roster")
async def evaluate_roster(req: RosterEvaluationRequest):
    """
    Evaluates a team's roster and predicts next-season Net EPA and Win %.
    Optionally supports what-if analysis via position_overrides.

    Example:
        POST /evaluate-roster
        {"team": "KC", "year": 2023}

    What-if example:
        {"team": "KC", "year": 2023, "position_overrides": {"lag_qb_grade": 55.0}}
    """
    try:
        if req.players:
            # Aggregate players manually if team_agent_graph supports it or just directly call team model wrapper
            from backend.agent.team_model_wrapper import get_team_model
            model = get_team_model()
            result = model.project_roster_performance(
                team=req.team,
                year=req.year,
                players=[p.dict() for p in req.players]
            )
            return {
                "team": req.team.upper(),
                "base_year": req.year,
                "predicted_year": req.year + 1,
                "prediction": result,
                "position_overrides_applied": "from_players",
            }
        
        final_state = run_team_evaluation(
            team=req.team,
            year=req.year,
            position_overrides=req.position_overrides,
        )

        if final_state.get("error"):
            raise HTTPException(status_code=404, detail=final_state["error"])

        return {
            "team": req.team.upper(),
            "base_year": req.year,
            "predicted_year": req.year + 1,
            "prediction": final_state.get("prediction", {}),
            "key_positions": final_state.get("key_positions", []),
            "verdict": final_state.get("verdict", ""),
            "position_overrides_applied": req.position_overrides or {},
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
