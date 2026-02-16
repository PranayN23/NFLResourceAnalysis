#!/usr/bin/env python
"""
Diagnostic script to analyze prediction errors and identify issues
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from backend.agent.rb_model_wrapper import RBModelInference

# Load test data
csv_path = 'backend/ML/HB.csv'
df = pd.read_csv(csv_path)

# Initialize model
base_dir = os.path.dirname(__file__)
TRANSFORMER_PATH = os.path.join(base_dir, "rb_best_classifier.pth")
XGB_PATH = os.path.join(base_dir, "rb_best_xgb.joblib")
SCALER_PATH = os.path.join(base_dir, "rb_player_scaler.joblib")

engine = RBModelInference(TRANSFORMER_PATH, scaler_path=SCALER_PATH, xgb_path=XGB_PATH)

print("="*80)
print("PREDICTION DIAGNOSTICS")
print("="*80)
print(f"Transformer features: {len(engine.transformer_features)}")
print(f"XGBoost features: {len(engine.xgb_features)}")

# Test on 2024 data
test_year = 2024
df_2024 = df[(df['Year'] == test_year) & (df['total_touches'] >= 50)].copy()
df_2024 = df_2024.sort_values('total_touches', ascending=False).head(32)

results = []

for _, row in df_2024.iterrows():
    name = row['player']
    history = df[(df['player'] == name) & (df['Year'] < test_year)].copy()
    
    if len(history) == 0:
        continue
    
    actual = row['grades_offense']
    actual_tier = engine.get_tier(actual)
    
    # Get individual predictions
    tier_t, details_t = engine.get_prediction(history, mode="transformer")
    tier_x, details_x = engine.get_prediction(history, mode="xgb")
    tier_e, details_e = engine.get_prediction(history, mode="ensemble")
    
    results.append({
        'player': name,
        'actual_grade': actual,
        'actual_tier': actual_tier,
        'transformer_pred': details_t['predicted_grade'],
        'xgb_pred': details_x['predicted_grade'],
        'ensemble_pred': details_e['predicted_grade'],
        'age': row['age'],
        'age_adjustment': details_e['age_adjustment'],
        'transformer_error': actual - details_t['predicted_grade'],
        'xgb_error': actual - details_x['predicted_grade'],
        'ensemble_error': actual - details_e['predicted_grade'],
    })

results_df = pd.DataFrame(results)

print("\n[1] Individual Model Performance:")
print(f"Transformer MAE: {results_df['transformer_error'].abs().mean():.2f}")
print(f"XGBoost MAE: {results_df['xgb_error'].abs().mean():.2f}")
print(f"Ensemble MAE: {results_df['ensemble_error'].abs().mean():.2f}")

print("\n[2] Errors by Actual Tier:")
for tier in ['Elite', 'Starter', 'Rotation']:
    tier_data = results_df[results_df['actual_tier'] == tier]
    if len(tier_data) > 0:
        print(f"\n{tier} (n={len(tier_data)}):")
        print(f"  Transformer MAE: {tier_data['transformer_error'].abs().mean():.2f}")
        print(f"  XGBoost MAE: {tier_data['xgb_error'].abs().mean():.2f}")
        print(f"  Ensemble MAE: {tier_data['ensemble_error'].abs().mean():.2f}")
        print(f"  Mean Error (bias): {tier_data['ensemble_error'].mean():.2f}")

print("\n[3] Errors by Age:")
for age_group in [(0, 25), (26, 27), (28, 29), (30, 100)]:
    age_data = results_df[results_df['age'].between(age_group[0], age_group[1])]
    if len(age_data) > 0:
        print(f"\nAge {age_group[0]}-{age_group[1]} (n={len(age_data)}):")
        print(f"  Mean Error: {age_data['ensemble_error'].mean():.2f}")
        print(f"  Mean Age Adjustment: {age_data['age_adjustment'].mean():.2f}")

print("\n[4] Worst Underpredictions (Elite players):")
elite_players = results_df[results_df['actual_tier'] == 'Elite'].nlargest(10, 'ensemble_error')
print(elite_players[['player', 'actual_grade', 'transformer_pred', 'xgb_pred', 
                     'ensemble_pred', 'age', 'age_adjustment', 'ensemble_error']].to_string(index=False))

print("\n[5] Model Comparison:")
print("Which model performs better for elite players?")
elite = results_df[results_df['actual_tier'] == 'Elite']
if len(elite) > 0:
    trans_mae_elite = elite['transformer_error'].abs().mean()
    xgb_mae_elite = elite['xgb_error'].abs().mean()
    print(f"  Transformer MAE (Elite): {trans_mae_elite:.2f}")
    print(f"  XGBoost MAE (Elite): {xgb_mae_elite:.2f}")
    if trans_mae_elite < xgb_mae_elite:
        print("  → Transformer performs better for elite players")
    else:
        print("  → XGBoost performs better for elite players")

print("\n" + "="*80)
