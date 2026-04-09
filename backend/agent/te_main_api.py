"""TE GM Agent API — port 8007"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd, uvicorn, os

_thread_pool = ThreadPoolExecutor(max_workers=2)
from backend.agent.te_agent_graph import te_gm_agent, TE_CSV_PATH
from backend.agent.api_year_utils import clamp_analysis_year, history_as_of_year
from backend.agent.team_summary import build_team_year_summary, build_team_position_rankings, build_player_directory
from backend.agent.scheme_personnel import compact_scheme_personnel_for_api
from backend.agent.team_context import (
    get_team_roster, compute_positional_need, get_team_cap,
    get_all_teams, aav_to_cap_pcts, is_player_on_team, get_roster_without_player,
)

app = FastAPI(title="NFL TE GM Agent API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

TE_GRADE_COL  = "grades_offense"
TE_SNAP_COL   = "routes"
TE_PROD_STATS = ["yards", "touchdowns", "targets"]

df_players = pd.DataFrame()
if os.path.exists(TE_CSV_PATH):
    df_players = pd.read_csv(TE_CSV_PATH)
    print(f"[TE API] Loaded {len(df_players)} rows")


@app.get("/te-players")
async def get_te_players():
    if df_players.empty:
        raise HTTPException(status_code=503, detail="Player database not loaded.")
    years = pd.to_numeric(df_players.get("Year"), errors="coerce").dropna() if "Year" in df_players.columns else pd.Series(dtype=float)
    min_year = int(years.min()) if not years.empty else 2025
    max_data_year = int(years.max()) if not years.empty else 2025
    return {"players": sorted(df_players["player"].dropna().unique().tolist()), "analysis_year_min": min_year, "analysis_year_max": 2025, "max_data_year": max_data_year}


@app.get("/teams")
async def get_teams(analysis_year: int = Query(2025)):
    teams = get_all_teams(reference_year=analysis_year)
    if not teams and not df_players.empty:
        teams = sorted(df_players["Team"].dropna().unique().tolist())
    return {"teams": teams}




@app.get("/team-summary")
async def team_summary(team: str = Query(...), analysis_year: int = Query(2025)):
    return build_team_year_summary(team, analysis_year)




@app.get("/player-directory")
async def player_directory(analysis_year: int = Query(2025)):
    return build_player_directory(analysis_year)
@app.get("/team-rankings")
async def team_rankings(team: str = Query(...), analysis_year: int = Query(2025)):
    return build_team_position_rankings(team, analysis_year)
@app.get("/team-roster")
async def team_roster(team: str = Query(...), analysis_year: int = Query(2025)):
    if df_players.empty:
        raise HTTPException(status_code=503, detail="Player database not loaded.")
    roster = get_team_roster(team, df_players, grade_col=TE_GRADE_COL, snap_col=TE_SNAP_COL, reference_year=analysis_year)
    need_score, need_label = compute_positional_need(roster, position_df=df_players, team=team,
                                                      grade_col=TE_GRADE_COL, snap_col=TE_SNAP_COL,
                                                      prod_stat_cols=TE_PROD_STATS, reference_year=analysis_year,
                                                      position_key="TE")
    allocated_pct, available_pct = get_team_cap(team, reference_year=analysis_year)
    return {
        "team": team,
        "roster": roster,
        "positional_need": need_score,
        "need_label": need_label,
        "allocated_cap_pct": allocated_pct,
        "available_cap_pct": available_pct,
        "scheme_personnel": compact_scheme_personnel_for_api(team, reference_year=analysis_year),
    }


class EvaluationRequest(BaseModel):
    player_name:    str
    salary_ask:     float
    contract_years: int = Field(default=1, ge=1, le=7)
    team:              str   = ""
    cap_available_pct: float = 0.0
    analysis_year:    int = Field(default=2025, ge=1900, le=2025)


@app.post("/evaluate")
async def evaluate_player(req: EvaluationRequest):
    analysis_year = clamp_analysis_year(req.analysis_year)
    player_data = history_as_of_year(df_players[df_players["player"] == req.player_name].copy(), analysis_year)
    if len(player_data) == 0:
        raise HTTPException(status_code=404, detail=f"Player '{req.player_name}' not found.")

    team_state_fields = {
        "team_name": "", "team_cap_available_pct": 0.0, "positional_need": 0.0,
        "need_label": "", "current_roster": [], "signing_cap_pcts": [], "team_fit_summary": "",
    }
    team_ctx = {}

    if req.team:
        roster = get_team_roster(req.team, df_players, grade_col=TE_GRADE_COL, snap_col=TE_SNAP_COL, reference_year=analysis_year)
        re_signing = is_player_on_team(req.player_name, req.team, df_players, reference_year=analysis_year)
        if re_signing:
            roster_without = get_roster_without_player(roster, req.player_name)
            need_score, need_label = compute_positional_need(roster_without, position_df=df_players,
                team=req.team, exclude_player=req.player_name, grade_col=TE_GRADE_COL,
                snap_col=TE_SNAP_COL, prod_stat_cols=TE_PROD_STATS, reference_year=analysis_year,
                position_key="TE")
            player_cap = next((p["cap_pct"] for p in roster if p["player"].strip().lower() == req.player_name.strip().lower()), 0.0)
        else:
            roster_without = roster
            need_score, need_label = compute_positional_need(roster, position_df=df_players,
                team=req.team, grade_col=TE_GRADE_COL, snap_col=TE_SNAP_COL, prod_stat_cols=TE_PROD_STATS, reference_year=analysis_year,
                position_key="TE")
            player_cap = 0.0
        allocated_pct, available_pct = get_team_cap(req.team, reference_year=analysis_year)
        cap_avail = req.cap_available_pct if req.cap_available_pct > 0 else available_pct
        if re_signing: cap_avail += player_cap
        signing_pcts = aav_to_cap_pcts(req.salary_ask, req.contract_years)
        team_state_fields = {
            "team_name": req.team, "team_cap_available_pct": cap_avail,
            "positional_need": need_score, "need_label": need_label,
            "current_roster": roster_without if re_signing else roster,
            "signing_cap_pcts": signing_pcts, "team_fit_summary": "",
        }
        team_ctx = {
            "team": req.team, "allocated_cap_pct": allocated_pct, "available_cap_pct": cap_avail,
            "signing_cap_pcts": signing_pcts, "positional_need": need_score, "need_label": need_label,
            "current_roster": roster_without if re_signing else roster,
            "is_re_signing": re_signing, "freed_cap_pct": player_cap,
        }

    initial_state = {
        "player_name": req.player_name, "salary_ask": req.salary_ask,
        "contract_years": req.contract_years,
            "analysis_year": analysis_year, "player_history": player_data,
        "predicted_tier": "", "confidence": {}, "current_age": 26,
        "last_season_stats": {}, "career_stats": [], "stats_score": 0.0,
        "composite_grade": 0.0, "valuation": 0.0, "effective_cap_burden": 0.0,
        "total_nominal_value": 0.0, "year_breakdown": [], "projected_stats": [],
        "decision": "", "reasoning": "", **team_state_fields,
    }

    loop = asyncio.get_event_loop()
    final_state = await loop.run_in_executor(_thread_pool, te_gm_agent.invoke, initial_state)
    if req.team:
        team_ctx["fit_summary"] = final_state.get("team_fit_summary", "")

    response = {
        "player": req.player_name, "decision": final_state["decision"],
        "reasoning": final_state["reasoning"],
        "data": {
            "predicted_tier": final_state["predicted_tier"], "current_age": final_state["current_age"],
            "contract_years": req.contract_years,
            "analysis_year": analysis_year, "effective_fair_aav": final_state["valuation"],
            "effective_cap_burden": final_state["effective_cap_burden"],
            "total_nominal_value": final_state["total_nominal_value"],
            "total_ask": round(req.salary_ask * req.contract_years, 2),
            "confidence": final_state["confidence"], "year_breakdown": final_state["year_breakdown"],
            "last_season_stats": final_state["last_season_stats"],
            "projected_stats": final_state["projected_stats"], "career_stats": final_state["career_stats"],
        },
    }
    if team_ctx: response["team_context"] = team_ctx
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8007)
