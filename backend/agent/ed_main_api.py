"""
ED GM Agent API

FastAPI endpoint for evaluating Edge Defender free agents.
Accepts player name, salary ask (AAV), and contract length in years.

Usage:
    uvicorn backend.agent.ed_main_api:app --host 0.0.0.0 --port 8002
    POST /evaluate  {"player_name": "Myles Garrett", "salary_ask": 25.0, "contract_years": 3}
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

_thread_pool = ThreadPoolExecutor(max_workers=2)
from backend.agent.ed_agent_graph import ed_gm_agent, ED_CSV_PATH
from backend.agent.api_year_utils import clamp_analysis_year, history_as_of_year
from backend.agent.team_context import (
    get_team_roster, compute_positional_need, get_team_cap,
    get_all_teams, aav_to_cap_pcts, is_player_on_team,
    get_roster_without_player,
)
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
    if "player" not in df_players.columns:
        for candidate in ["Player", "Name", "name"]:
            if candidate in df_players.columns:
                df_players.rename(columns={candidate: "player"}, inplace=True)
                break
    print(f"[ED API] Loaded player database: {len(df_players)} rows")
else:
    print(f"WARNING: ED player CSV not found at {ED_CSV_PATH}.")
    df_players = pd.DataFrame()


@app.get("/ed-players")
async def get_ed_players():
    """Return sorted list of all ED player names available for evaluation."""
    if df_players.empty:
        raise HTTPException(status_code=503, detail="Player database not loaded.")
    names = sorted(df_players["player"].dropna().unique().tolist())
    return {"players": names}


@app.get("/teams")
async def get_teams(analysis_year: int = Query(2025)):
    """Return sorted list of all 32 team names from cap data."""
    teams = get_all_teams(reference_year=analysis_year)
    if not teams:
        teams = sorted(df_players["Team"].dropna().unique().tolist()) if not df_players.empty else []
    return {"teams": teams}


@app.get("/team-roster")
async def team_roster(team: str = Query(..., description="Team name"), analysis_year: int = Query(2025)):
    """Return the team's ED players, cap summary, and positional need."""
    if df_players.empty:
        raise HTTPException(status_code=503, detail="Player database not loaded.")

    roster = get_team_roster(team, df_players, reference_year=analysis_year)
    need_score, need_label = compute_positional_need(roster, position_df=df_players, team=team, reference_year=analysis_year)
    allocated_pct, available_pct = get_team_cap(team, reference_year=analysis_year)

    return {
        "team": team,
        "roster": roster,
        "positional_need": need_score,
        "need_label": need_label,
        "allocated_cap_pct": allocated_pct,
        "available_cap_pct": available_pct,
    }


class EvaluationRequest(BaseModel):
    player_name:    str
    salary_ask:     float           # AAV in $M
    contract_years: int = Field(default=1, ge=1, le=10)
    team:              str   = ""
    cap_available_pct: float = 0.0
    analysis_year:    int = Field(default=2025, ge=1900, le=2025)


@app.post("/evaluate")
async def evaluate_player(req: EvaluationRequest):
    """
    Evaluate an Edge Defender free agent.

    Runs the ED GM agent workflow accounting for contract length,
    age-based performance decay, and time discounting.
    Returns a SIGN / PASS recommendation with per-year breakdown.
    """
    analysis_year = clamp_analysis_year(req.analysis_year)
    player_data = history_as_of_year(df_players[df_players["player"] == req.player_name].copy(), analysis_year)

    if len(player_data) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Player '{req.player_name}' not found in database.",
        )

    # Build team context if team mode is active
    team_ctx = {}
    team_state_fields = {
        "team_name": "",
        "team_cap_available_pct": 0.0,
        "positional_need": 0.0,
        "need_label": "",
        "current_roster": [],
        "signing_cap_pcts": [],
        "team_fit_summary": "",
    }

    if req.team:
        roster = get_team_roster(req.team, df_players, reference_year=analysis_year)
        re_signing = is_player_on_team(req.player_name, req.team, df_players)

        if re_signing:
            roster_without = get_roster_without_player(roster, req.player_name)
            need_score, need_label = compute_positional_need(
                roster_without, position_df=df_players, team=req.team,
                exclude_player=req.player_name,
            )
            player_cap = next(
                (p["cap_pct"] for p in roster if p["player"].strip().lower() == req.player_name.strip().lower()),
                0.0,
            )
        else:
            roster_without = roster
            need_score, need_label = compute_positional_need(
                roster, position_df=df_players, team=req.team,
            )
            player_cap = 0.0

        allocated_pct, available_pct = get_team_cap(req.team, reference_year=analysis_year)
        cap_avail = req.cap_available_pct if req.cap_available_pct > 0 else available_pct
        if re_signing:
            cap_avail = cap_avail + player_cap
        signing_pcts = aav_to_cap_pcts(req.salary_ask, req.contract_years)

        team_state_fields = {
            "team_name": req.team,
            "team_cap_available_pct": cap_avail,
            "positional_need": need_score,
            "need_label": need_label,
            "current_roster": roster_without if re_signing else roster,
            "signing_cap_pcts": signing_pcts,
            "team_fit_summary": "",
        }
        team_ctx = {
            "team": req.team,
            "allocated_cap_pct": allocated_pct,
            "available_cap_pct": cap_avail,
            "signing_cap_pcts": signing_pcts,
            "positional_need": need_score,
            "need_label": need_label,
            "current_roster": roster_without if re_signing else roster,
            "is_re_signing": re_signing,
            "freed_cap_pct": player_cap,
        }

    initial_state = {
        "player_name":    req.player_name,
        "salary_ask":     req.salary_ask,
        "contract_years": req.contract_years,
            "analysis_year": analysis_year,
        "analysis_year": analysis_year,
        "player_history": player_data,
        "predicted_tier":    "",
        "confidence":        {},
        "current_age":       28,
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
        **team_state_fields,
    }

    loop = asyncio.get_event_loop()
    final_state = await loop.run_in_executor(
        _thread_pool, ed_gm_agent.invoke, initial_state
    )

    if req.team:
        team_ctx["fit_summary"] = final_state.get("team_fit_summary", "")

    response = {
        "player":         req.player_name,
        "decision":       final_state["decision"],
        "reasoning":      final_state["reasoning"],
        "data": {
            "predicted_tier":       final_state["predicted_tier"],
            "current_age":          final_state["current_age"],
            "contract_years":       req.contract_years,
            "analysis_year":       analysis_year,
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
    if team_ctx:
        response["team_context"] = team_ctx

    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
