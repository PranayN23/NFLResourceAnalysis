"""
test_team_model.py
A comprehensive testing script to verify the accuracy and behavior of the Team EPA/Win Prediction Model.
It tests the latest season's predictions against the actual model outputs and demonstrates the 'what-if' projection scenarios.
"""

import os
import sys
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

# Add backend to path to import wrapper correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from backend.agent.team_model_wrapper import get_team_model

def run_historical_verification():
    print("\n" + "="*50)
    print("1. Historical Accuracy Verification (2023 predicting 2024)")
    print("="*50)
    
    model_wrapper = get_team_model()
    team_dataset_path = os.path.join(os.path.dirname(__file__), "team_dataset.csv")
    
    if not os.path.exists(team_dataset_path):
        print("Error: team_dataset.csv not found.")
        return
        
    df = pd.read_csv(team_dataset_path)
    
    # We want to predict 2024 (which implies using 2023 as base year context)
    # The dataset has next_net_epa and next_win_pct as the ground truth targets for the base_year.
    test_year = 2023 
    
    test_data = df[df['year'] == test_year].copy()
    
    if test_data.empty:
        print(f"No data available for base year {test_year}.")
        return
        
    actuals_epa = []
    preds_epa = []
    actuals_win = []
    preds_win = []
    actual_tiers = []
    pred_tiers = []
    
    print(f"\nEvaluating {len(test_data)} teams for the {test_year+1} season...\n")
    print(f"{'Team':<5} | {'Actual EPA':<12} | {'Pred EPA':<12} | {'Act Tier':<20} | {'Pred Tier':<20} | {'Tier Match?'}")
    print("-" * 105)
    
    from backend.agent.team_model_wrapper import _epa_to_tier
    
    for _, row in test_data.iterrows():
        team = row['abbr']
        
        # Ground truth for year+1
        actual_epa = row['next_net_epa']
        actual_win_pct = row['next_win_pct']
        
        # If actuals aren't available, skip (for instance if 2024 data isn't fully in nfl_epa.csv)
        if pd.isna(actual_epa) or pd.isna(actual_win_pct):
            continue
            
        try:
            # Predict using baseline context
            result = model_wrapper.evaluate_roster(team, test_year, dataset_path=team_dataset_path)
            
            pred_epa = result['predicted_net_epa']
            pred_win = result['predicted_win_pct']
            tier = result['tier']
            act_tier = _epa_to_tier(actual_epa)
            
            actuals_epa.append(actual_epa)
            preds_epa.append(pred_epa)
            actuals_win.append(actual_win_pct)
            preds_win.append(pred_win)
            actual_tiers.append(act_tier)
            pred_tiers.append(tier)
            match = "✅" if act_tier == tier else "❌"
            
            print(f"{team:<5} | {actual_epa:>12.4f} | {pred_epa:>12.4f} | {act_tier:<20} | {tier:<20} | {match}")
            
        except Exception as e:
            print(f"Error predicting {team}: {e}")
            
    if actuals_epa:
        # Tier accuracy
        matches = sum(1 for a, p in zip(actual_tiers, pred_tiers) if a == p)
        tier_acc = matches / len(actual_tiers)
        print("\n--- Model Accuracy Metrics (Test Set) ---")
        print(f"Net EPA MAE  : {mean_absolute_error(actuals_epa, preds_epa):.4f}")
        print(f"Net EPA R²   : {r2_score(actuals_epa, preds_epa):.4f}")
        print(f"Win %   MAE  : {mean_absolute_error(actuals_win, preds_win):.4f}")
        print(f"Win %   R²   : {r2_score(actuals_win, preds_win):.4f}")
        print(f"Tier Accuracy: {tier_acc:.1%} ({matches}/{len(actual_tiers)} correct boundaries)")

def run_what_if_scenarios():
    print("\n" + "="*50)
    print("2. 'What-If' Roster Projection Scenarios")
    print("="*50)
    
    model_wrapper = get_team_model()
    
    team = "KC"
    year = 2023
    
    print(f"\nScenario: Modifying the {team} {year+1} Roster")
    
    # 1. Baseline
    baseline = model_wrapper.evaluate_roster(team, year)
    print(f"\n[Baseline] {team} ({year+1} Projection):")
    print(f"  Predicted Net EPA : {baseline['predicted_net_epa']:.4f}")
    print(f"  Predicted Win %   : {baseline['predicted_win_pct']:.3f} ({baseline['predicted_wins']} wins)")
    print(f"  Tier              : {baseline['tier']}")
    
    # 2. Elite QB (Grade: 95.0)
    elite_qb_players = [{"name": "Elite QB", "position": "QB", "projected_grade": 95.0}]
    elite_qb = model_wrapper.project_roster_performance(team, year, elite_qb_players)
    print(f"\n[Scenario A] Substitute with 95.0 Grade QB:")
    print(f"  Predicted Net EPA : {elite_qb['predicted_net_epa']:.4f} (Change: {elite_qb['predicted_net_epa'] - baseline['predicted_net_epa']:+.4f})")
    print(f"  Predicted Win %   : {elite_qb['predicted_win_pct']:.3f} (Change: {elite_qb['predicted_win_pct'] - baseline['predicted_win_pct']:+.3f})")
    
    # 3. Backup QB (Grade: 55.0)
    backup_qb_players = [{"name": "Backup QB", "position": "QB", "projected_grade": 55.0}]
    backup_qb = model_wrapper.project_roster_performance(team, year, backup_qb_players)
    print(f"\n[Scenario B] Substitute with 55.0 Grade QB:")
    print(f"  Predicted Net EPA : {backup_qb['predicted_net_epa']:.4f} (Change: {backup_qb['predicted_net_epa'] - baseline['predicted_net_epa']:+.4f})")
    print(f"  Predicted Win %   : {backup_qb['predicted_win_pct']:.3f} (Change: {backup_qb['predicted_win_pct'] - baseline['predicted_win_pct']:+.3f})")
    
    # 4. Entire Elite O-Line (Assuming 5 players, avg 85.0 grade)
    elite_ol_players = [
        {"name": "LT", "position": "OL", "projected_grade": 85.0},
        {"name": "LG", "position": "OL", "projected_grade": 85.0},
        {"name": "C", "position": "OL", "projected_grade": 85.0},
        {"name": "RG", "position": "OL", "projected_grade": 85.0},
        {"name": "RT", "position": "OL", "projected_grade": 85.0},
    ]
    elite_ol = model_wrapper.project_roster_performance(team, year, elite_ol_players)
    print(f"\n[Scenario C] Substitute with Elite 85.0 Grade O-Line:")
    print(f"  Predicted Net EPA : {elite_ol['predicted_net_epa']:.4f} (Change: {elite_ol['predicted_net_epa'] - baseline['predicted_net_epa']:+.4f})")
    print(f"  Predicted Win %   : {elite_ol['predicted_win_pct']:.3f} (Change: {elite_ol['predicted_win_pct'] - baseline['predicted_win_pct']:+.3f})")

if __name__ == "__main__":
    run_historical_verification()
    run_what_if_scenarios()
    print("\nVerification Complete.")
