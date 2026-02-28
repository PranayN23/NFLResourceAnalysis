"""
team_agent_graph.py
LangGraph workflow for team roster evaluation.

Nodes:
  1. build_roster_features  — load team-year context from dataset
  2. predict_team_epa       — run TeamModelInference
  3. identify_key_positions — rank positions by impact score
  4. generate_roster_verdict — produce tier + narrative
"""

from __future__ import annotations
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from backend.agent.team_model_wrapper import get_team_model


class TeamEvalState(TypedDict):
    team: str
    year: int
    position_overrides: Optional[dict]
    context: dict
    prediction: dict
    key_positions: list[dict]
    verdict: str
    error: Optional[str]


# ── Node 1: Build roster features ──────────────────────────────────────────────
def build_roster_features(state: TeamEvalState) -> TeamEvalState:
    import pandas as pd
    import os

    dataset_path = os.path.join(
        os.path.dirname(__file__), "..", "ML", "team_model", "team_dataset.csv"
    )
    try:
        df = pd.read_csv(dataset_path)
        row = df[(df["abbr"] == state["team"].upper()) & (df["year"] == int(state["year"]))]
        if row.empty:
            return {**state, "error": f"No data for team={state['team']} year={state['year']}"}
        return {**state, "context": row.iloc[0].to_dict(), "error": None}
    except Exception as e:
        return {**state, "error": str(e)}


# ── Node 2: Predict team EPA ───────────────────────────────────────────────────
def predict_team_epa(state: TeamEvalState) -> TeamEvalState:
    if state.get("error"):
        return state
    try:
        model = get_team_model()
        result = model.predict(
            state["context"],
            position_overrides=state.get("position_overrides"),
        )
        return {**state, "prediction": result}
    except Exception as e:
        return {**state, "error": str(e)}


# ── Node 3: Identify key positions ────────────────────────────────────────────
def identify_key_positions(state: TeamEvalState) -> TeamEvalState:
    if state.get("error"):
        return state
    impact = state["prediction"].get("position_impact", {})
    ranked = [
        {"position": pos, "importance": round(score, 4)}
        for pos, score in sorted(impact.items(), key=lambda x: x[1], reverse=True)
        if score > 0
    ]
    return {**state, "key_positions": ranked}


# ── Node 4: Generate roster verdict ──────────────────────────────────────────
def generate_roster_verdict(state: TeamEvalState) -> TeamEvalState:
    if state.get("error"):
        return {**state, "verdict": f"Error: {state['error']}"}

    pred = state["prediction"]
    team = state["team"].upper()
    year = state["year"]
    tier = pred["tier"]
    epa = pred["predicted_net_epa"]
    win_pct = pred["predicted_win_pct"]
    wins = pred["predicted_wins"]
    key_pos = state.get("key_positions", [])

    # Top 3 impactful positions
    top_positions = ", ".join(p["position"] for p in key_pos[:3]) if key_pos else "N/A"

    # Narrative
    narrative = (
        f"{team} Roster Evaluation (Base Year: {year} → Predicting {year + 1})\n"
        f"{'='*60}\n"
        f"Tier        : {tier}\n"
        f"Net EPA     : {epa:+.4f}\n"
        f"Win %       : {win_pct:.1%}  (~{wins:.1f} wins)\n"
        f"Key Positions: {top_positions}\n"
    )

    if state.get("position_overrides"):
        narrative += f"\n[What-If] Overrides applied: {state['position_overrides']}\n"

    return {**state, "verdict": narrative}


# ── Route: check for errors ───────────────────────────────────────────────────
def should_continue(state: TeamEvalState) -> str:
    return "error" if state.get("error") else "continue"


# ── Build graph ───────────────────────────────────────────────────────────────
def build_team_eval_graph():
    graph = StateGraph(TeamEvalState)

    graph.add_node("build_roster_features", build_roster_features)
    graph.add_node("predict_team_epa", predict_team_epa)
    graph.add_node("identify_key_positions", identify_key_positions)
    graph.add_node("generate_roster_verdict", generate_roster_verdict)

    graph.set_entry_point("build_roster_features")
    graph.add_edge("build_roster_features", "predict_team_epa")
    graph.add_edge("predict_team_epa", "identify_key_positions")
    graph.add_edge("identify_key_positions", "generate_roster_verdict")
    graph.add_edge("generate_roster_verdict", END)

    return graph.compile()


# Module-level compiled graph
_team_graph = None

def get_team_graph():
    global _team_graph
    if _team_graph is None:
        _team_graph = build_team_eval_graph()
    return _team_graph


def run_team_evaluation(
    team: str,
    year: int,
    position_overrides: dict | None = None,
) -> dict:
    """
    Convenience function to run the full team evaluation pipeline.
    Returns the final state dict.
    """
    graph = get_team_graph()
    initial_state: TeamEvalState = {
        "team": team,
        "year": year,
        "position_overrides": position_overrides or {},
        "context": {},
        "prediction": {},
        "key_positions": [],
        "verdict": "",
        "error": None,
    }
    final_state = graph.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    # Quick test
    result = run_team_evaluation("KC", 2023)
    print(result["verdict"])
    if result.get("prediction"):
        print("Prediction:", result["prediction"])

    print("\n--- What-If: KC 2023 with backup QB ---")
    result2 = run_team_evaluation("KC", 2023, position_overrides={"lag_qb_grade": 55.0})
    print(result2["verdict"])
