"""
RB GM Agent API

FastAPI endpoint for evaluating Running Back free agents.
Accepts player name, salary ask (AAV), and contract length in years.

Usage:
    uvicorn backend.agent.rb_main_api:app --host 0.0.0.0 --port 8004
    POST /evaluate  {"player_name": "Saquon Barkley", "salary_ask": 14.0, "contract_years": 3}
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

_thread_pool = ThreadPoolExecutor(max_workers=2)
from backend.agent.rb_agent_graph import rb_gm_agent, RB_CSV_PATH
import pandas as pd
import uvicorn
import os

app = FastAPI(title="NFL RB GM Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load player data once at startup
if os.path.exists(RB_CSV_PATH):
    df_players = pd.read_csv(RB_CSV_PATH)
    if "player" not in df_players.columns:
        for candidate in ["Player", "Name", "name"]:
            if candidate in df_players.columns:
                df_players.rename(columns={candidate: "player"}, inplace=True)
                break
    print(f"[RB API] Loaded player database: {len(df_players)} rows")
else:
    print(f"WARNING: RB player CSV not found at {RB_CSV_PATH}.")
    df_players = pd.DataFrame()


@app.get("/rb-players")
async def get_rb_players():
    """Return sorted list of all RB player names available for evaluation."""
    if df_players.empty:
        raise HTTPException(status_code=503, detail="Player database not loaded.")
    names = sorted(df_players["player"].dropna().unique().tolist())
    return {"players": names}


class EvaluationRequest(BaseModel):
    player_name:    str
    salary_ask:     float           # AAV in $M
    contract_years: int = Field(default=1, ge=1, le=10)


@app.post("/evaluate")
async def evaluate_player(req: EvaluationRequest):
    """
    Evaluate a Running Back free agent.

    Runs the RB GM agent workflow accounting for contract length,
    RB-specific age-based performance decay, cap growth, and time
    discounting. Returns a tiered recommendation with per-year
    breakdown and projected stats.
    """
    player_data = df_players[df_players["player"] == req.player_name].copy()

    if len(player_data) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Player '{req.player_name}' not found in database.",
        )

    initial_state = {
        "player_name":    req.player_name,
        "salary_ask":     req.salary_ask,
        "contract_years": req.contract_years,
        "player_history": player_data,
        "predicted_tier":    "",
        "confidence":        {},
        "current_age":       26,
        "last_season_stats": {},
        "career_stats":      [],
        "stats_score":       0.0,
        "composite_grade":   0.0,
        "valuation":            0.0,
        "effective_cap_burden": 0.0,
        "total_nominal_value":  0.0,
        "year_breakdown":    [],
        "projected_stats":   [],
        "decision":          "",
        "reasoning":         "",
    }

    loop = asyncio.get_event_loop()
    final_state = await loop.run_in_executor(
        _thread_pool, rb_gm_agent.invoke, initial_state
    )

    return {
        "player":    req.player_name,
        "decision":  final_state["decision"],
        "reasoning": final_state["reasoning"],
        "data": {
            "predicted_tier":       final_state["predicted_tier"],
            "current_age":          final_state["current_age"],
            "contract_years":       req.contract_years,
            "effective_fair_aav":   final_state["valuation"],
            "effective_cap_burden": final_state["effective_cap_burden"],
            "total_nominal_value":  final_state["total_nominal_value"],
            "total_ask":            round(req.salary_ask * req.contract_years, 2),
            "confidence":           final_state["confidence"],
            "year_breakdown":       final_state["year_breakdown"],
            "last_season_stats":    final_state["last_season_stats"],
            "projected_stats":      final_state["projected_stats"],
            "career_stats":         final_state["career_stats"],
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)
