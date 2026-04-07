"""
Verification script for the ED GM Agent.
Mirrors verify_agent.py (QB) in structure.
Run from project root: python -m backend.agent.verify_ed_agent
"""

import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.agent.ed_agent_graph import ed_gm_agent, ED_CSV_PATH


def _run(name, ask, history):
    state = {
        "player_name":    name,
        "salary_ask":     ask,
        "player_history": history,
        "predicted_tier": "",
        "confidence":     {},
        "valuation":      0.0,
        "decision":       "",
        "reasoning":      "",
    }
    return ed_gm_agent.invoke(state)


# ─────────────────────────────────────────────────────────
# Test 1 — Mock Elite edge rusher
# ─────────────────────────────────────────────────────────
def test_mock_elite():
    print("\n--- Test 1: Mock Elite Edge Rusher ---")
    mock = pd.DataFrame([
        {
            "Year": 2021, "player": "Mock Elite", "age": 24.0,
            "grades_defense": 85.0, "grades_pass_rush_defense": 86.0, "grades_run_defense": 82.0,
            "total_pressures": 55, "sacks": 10, "hits": 12, "hurries": 33, "stops": 30,
            "tackles_for_loss": 8, "penalties": 1,
            "snap_counts_defense": 700, "snap_counts_dl": 700,
            "snap_counts_dl_outside_t": 600, "snap_counts_dl_over_t": 100,
            "adjusted_value": 90.0, "Cap_Space": 15.0, "Net EPA": 40.0,
        },
        {
            "Year": 2022, "player": "Mock Elite", "age": 25.0,
            "grades_defense": 87.0, "grades_pass_rush_defense": 88.0, "grades_run_defense": 84.0,
            "total_pressures": 60, "sacks": 13, "hits": 14, "hurries": 33, "stops": 33,
            "tackles_for_loss": 10, "penalties": 1,
            "snap_counts_defense": 750, "snap_counts_dl": 750,
            "snap_counts_dl_outside_t": 640, "snap_counts_dl_over_t": 110,
            "adjusted_value": 95.0, "Cap_Space": 14.0, "Net EPA": 45.0,
        },
        {
            "Year": 2023, "player": "Mock Elite", "age": 26.0,
            "grades_defense": 90.0, "grades_pass_rush_defense": 91.0, "grades_run_defense": 86.0,
            "total_pressures": 65, "sacks": 15, "hits": 16, "hurries": 34, "stops": 36,
            "tackles_for_loss": 12, "penalties": 0,
            "snap_counts_defense": 800, "snap_counts_dl": 800,
            "snap_counts_dl_outside_t": 680, "snap_counts_dl_over_t": 120,
            "adjusted_value": 100.0, "Cap_Space": 13.0, "Net EPA": 50.0,
        },
    ])

    result = _run("Mock Elite", 20.0, mock)
    print(f"  Tier:      {result['predicted_tier']}")
    print(f"  Grade:     {result['confidence'].get('predicted_grade')}")
    print(f"  Valuation: ${result['valuation']}M")
    print(f"  Decision:  {result['decision']}")
    print(f"  Reasoning: {result['reasoning']}")


# ─────────────────────────────────────────────────────────
# Test 2 — Real players from ED.csv
# ─────────────────────────────────────────────────────────
def test_real_players():
    print("\n--- Test 2: Real ED Players ---")
    if not os.path.exists(ED_CSV_PATH):
        print(f"  ERROR: CSV not found at {ED_CSV_PATH}")
        return

    df = pd.read_csv(ED_CSV_PATH)

    test_cases = [
        ("T.J. Watt",          25.0),
        ("Myles Garrett",      28.0),
        ("Brian Burns",        18.0),
        ("Jonathan Greenard",  16.0),
    ]

    for name, ask in test_cases:
        history = df[df["player"] == name].copy()
        if history.empty:
            print(f"  {name}: not found in CSV, skipping.")
            continue

        result = _run(name, ask, history)
        surplus_or_overpay = round(result["valuation"] - ask, 1)
        tag = f"+${surplus_or_overpay}M surplus" if surplus_or_overpay >= 0 else f"-${abs(surplus_or_overpay)}M overpay"
        print(
            f"  {name:<22} | tier: {result['predicted_tier']:<12} | "
            f"grade: {result['confidence'].get('predicted_grade'):<6} | "
            f"ask: ${ask}M | val: ${result['valuation']}M ({tag}) | {result['decision']}"
        )


# ─────────────────────────────────────────────────────────
# Test 3 — Full 2024 league sweep (top 20 by snaps)
# ─────────────────────────────────────────────────────────
def test_full_league():
    print("\n--- Test 3: 2024 Full League Sweep (Top 20 by snaps) ---")
    if not os.path.exists(ED_CSV_PATH):
        print(f"  ERROR: CSV not found at {ED_CSV_PATH}")
        return

    df = pd.read_csv(ED_CSV_PATH)
    top20 = (
        df[df["Year"] == 2024]
        .sort_values("snap_counts_defense", ascending=False)
        .head(20)["player"]
        .tolist()
    )

    rows = []
    for name in top20:
        history = df[df["player"] == name].copy()
        result = _run(name, 16.0, history)   # test vs. starter-level ask
        rows.append({
            "Player":    name,
            "Tier":      result["predicted_tier"],
            "Grade":     result["confidence"].get("predicted_grade"),
            "Value":     f"${result['valuation']}M",
            "Decision":  result["decision"],
        })

    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    test_mock_elite()
    test_real_players()
    test_full_league()
