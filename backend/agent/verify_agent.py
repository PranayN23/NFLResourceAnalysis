
import pandas as pd
import torch
import os
import sys

# Add the project root to sys.path so we can import from backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.agent.agent_graph import gm_agent, CSV_PATH
from backend.agent.model_wrapper import PlayerModelInference

def test_graph_with_mock_data():
    print("\n--- Testing Graph with Mock Data ---")
    # Mock history for a hypothetical "Elite" player (including new engineering features)
    mock_history = pd.DataFrame([{
        'Year': 2021, 'grades_pass': 85.0, 'grades_offense': 88.0, 'qb_rating': 105.0,
        'adjusted_value': 100.0, 'Cap_Space': 10.0, 'ypa': 8.5, 'twp_rate': 1.5,
        'btt_rate': 6.0, 'completion_percent': 68.0, 'Net EPA': 50.0, 'player': 'Mock Elite'
    }, {
        'Year': 2022, 'grades_pass': 87.0, 'grades_offense': 90.0, 'qb_rating': 110.0,
        'adjusted_value': 105.0, 'Cap_Space': 12.0, 'ypa': 8.8, 'twp_rate': 1.2,
        'btt_rate': 6.5, 'completion_percent': 70.0, 'Net EPA': 55.0, 'player': 'Mock Elite'
    }, {
        'Year': 2023, 'grades_pass': 89.0, 'grades_offense': 92.0, 'qb_rating': 115.0,
        'adjusted_value': 110.0, 'Cap_Space': 15.0, 'ypa': 9.0, 'twp_rate': 1.0,
        'btt_rate': 7.0, 'completion_percent': 72.0, 'Net EPA': 60.0, 'player': 'Mock Elite'
    }])

    initial_state = {
        "player_name": "Mock Elite",
        "salary_ask": 40.0,
        "player_history": mock_history,
        "predicted_tier": "", "confidence": {}, "valuation": 0.0, "decision": "", "reasoning": ""
    }

    result = gm_agent.invoke(initial_state)
    print(f"Player: {result['player_name']}")
    print(f"Tier: {result['predicted_tier']}")
    print(f"Valuation: ${result['valuation']}M")
    print(f"Decision: {result['decision']}")
    print(f"Reasoning: {result['reasoning']}")

def test_graph_with_real_players():
    print("\n--- Testing Graph with Real Players ---")
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    # Standardize column name just in case
    if 'Player' in df.columns:
        df.rename(columns={'Player': 'player'}, inplace=True)
    
    # Let's pick a few known QBs
    # Looking at the codebase, Patrick Mahomes and Dak Prescott are often used in NFL contexts.
    # We'll try to find them or pick 2 from the top.
    
    test_players = ["Patrick Mahomes", "Dak Prescott", "Josh Allen", "Jayden Daniels"]
    
    for name in test_players:
        player_data = df[df['player'] == name].copy()
        if len(player_data) < 3:
            # If not found, pick the first player with enough history
            continue
            
        print(f"\nEvaluating {name}...")
        initial_state = {
            "player_name": name,
            "salary_ask": 45.0, # Test with a high ask
            "player_history": player_data,
            "predicted_tier": "", "confidence": {}, "valuation": 0.0, "decision": "", "reasoning": ""
        }
        
        result = gm_agent.invoke(initial_state)
        print(f"Tier: {result['predicted_tier']}")
        print(f"Valuation: ${result['valuation']}M")
        print(f"Decision: {result['decision']}")

def full_2024_league_test():
    print("\n--- 2024 Full League Verification (32 QBs) ---")
    df = pd.read_csv(CSV_PATH)
    
    # Get top 32 by dropbacks in 2024
    df_2024_top = df[df['Year'] == 2024].sort_values('dropbacks', ascending=False).head(32)
    qb_list = df_2024_top['player'].tolist()
    
    results = []
    
    for name in qb_list:
        player_data = df[df['player'] == name].copy()
        
        # Test with a market average ask for a starter ($40M)
        initial_state = {
            "player_name": name,
            "salary_ask": 40.0,
            "player_history": player_data,
            "predicted_tier": "", "confidence": {}, "valuation": 0.0, "decision": "", "reasoning": ""
        }
        
        res = gm_agent.invoke(initial_state)
        results.append({
            "Player": name,
            "Predicted Tier": res['predicted_tier'],
            "Valuation": f"${res['valuation']}M",
            "Decision": res['decision']
        })
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    test_graph_with_mock_data()
    test_graph_with_real_players()
    full_2024_league_test()
