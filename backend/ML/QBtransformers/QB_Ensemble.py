import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Import classes and wrapper
from backend.ML.QBtransformers.Player_Model_QB import PlayerTransformerRegressor, Time2Vec
from backend.agent.model_wrapper import PlayerModelInference

# ==========================================
# 1. SETUP & CONFIG
# ==========================================
DATA_FILE = os.path.join(os.path.dirname(__file__), "../QB.csv")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "best_classifier.pth")
SCALER_OUT = os.path.join(os.path.dirname(__file__), "player_scaler.joblib")
XGB_MODEL_OUT = os.path.join(os.path.dirname(__file__), "best_xgb.joblib")

# MODE = "VALIDATION"  # Predict 2024 (Train < 2024)
MODE = "DREAM"       # Predict 2025 (Train <= 2024)

print(f"==== STARTING QB ENSEMBLE MODELING (Mode: {MODE}) ====")

TRANSFORMER_FEATURES = [
    'grades_pass', 'grades_offense', 'qb_rating', 'adjusted_value',
    'Cap_Space', 'ypa', 'twp_rate', 'btt_rate', 'completion_percent',
    'years_in_league', 'delta_grade', 'delta_epa', 'delta_btt',
    'team_performance_proxy'
]

XGB_FEATURES = [
    'lag_grades_offense', 'lag_Net_EPA', 'lag_btt_rate', 'lag_twp_rate',
    'lag_qb_rating', 'lag_ypa', 'adjusted_value', 'years_in_league',
    'delta_grade_lag', 'team_performance_proxy_lag'
]

# ==========================================
# 2. DATA PREPARATION (RE-TRAINING XGB)
# ==========================================
def prepare_data():
    df = pd.read_csv(DATA_FILE)
    df['adjusted_value'] = pd.to_numeric(df['adjusted_value'], errors='coerce').fillna(0)
    df = df[df['dropbacks'] >= 100].copy()
    df = df.sort_values(['player', 'Year'])
    groups = df.groupby('player')
    
    df["years_in_league"] = groups.cumcount()
    df["delta_grade"] = groups["grades_offense"].diff().fillna(0)
    df["delta_epa"]   = groups["Net EPA"].diff().fillna(0)
    df["delta_btt"]   = groups["btt_rate"].diff().fillna(0)
    df['team_performance_proxy'] = df.groupby(['Team', 'Year'])['Net EPA'].transform('mean')
    
    df['lag_grades_offense'] = groups['grades_offense'].shift(1)
    df['lag_Net_EPA'] = groups['Net EPA'].shift(1)
    df['lag_btt_rate'] = groups['btt_rate'].shift(1)
    df['lag_twp_rate'] = groups['twp_rate'].shift(1)
    df['lag_qb_rating'] = groups['qb_rating'].shift(1)
    df['lag_ypa'] = groups['ypa'].shift(1)
    df['delta_grade_lag'] = groups['lag_grades_offense'].diff().fillna(0)
    df['team_performance_proxy_lag'] = groups['team_performance_proxy'].shift(1)
    
    target_col = 'grades_offense'
    df_clean = df.dropna(subset=XGB_FEATURES + [target_col]).copy()
    
    return df_clean, df, target_col

df_clean, df_all, target_col = prepare_data()

# ==========================================
# 3. COMPONENT INITIALIZATION
# ==========================================
# Train XGBoost on all data <= 2024 for Dream Mode
print("\n[1/3] Refreshing XGBoost Component (Training <= 2024)...")
xgb_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
xgb_model.fit(df_clean[XGB_FEATURES], df_clean[target_col])
joblib.dump(xgb_model, XGB_MODEL_OUT)

# Initialize production wrapper for ensembling & calibration
engine = PlayerModelInference(MODEL_OUT, scaler_path=SCALER_OUT, xgb_path=XGB_MODEL_OUT)

# ==========================================
# 4. EXECUTION
# ==========================================
if MODE == "VALIDATION":
    test_df = df_clean[df_clean["Year"] == 2024].copy()
    results = []
    for _, row in test_df.iterrows():
        name = row['player']
        history = df_all[(df_all['player'] == name) & (df_all['Year'] < 2024)].copy()
        tier, details = engine.get_prediction(history, apply_calibration=True)
        
        results.append({
            "player": name,
            "Team": row["Team"],
            "Year": 2024,
            "grades_offense": row["grades_offense"],
            "Pred_XGB": details["xgb_grade"],
            "Pred_Transformer": details["transformer_grade"],
            "Ensemble_Pred": details["predicted_grade"]
        })
    
    final_df = pd.DataFrame(results)
    out_path = os.path.join(os.path.dirname(__file__), "QB_2024_Validation_Results.csv")
    final_df.to_csv(out_path, index=False)
    print(f"Saved Validation Report to {out_path}")

elif MODE == "DREAM":
    # Predict 2025
    active_2024 = df_all[df_all["Year"] == 2024].copy()
    rows_2025 = []
    
    print("\n[2/3] Generating 2025 'Dream' Projections with Calibration...")
    
    for _, row in active_2024.iterrows():
        player = row['player']
        if player == "Derek Carr": continue # Retired
        
        history = df_all[df_all["player"] == player].sort_values("Year").tail(5)
        
        # The wrapper handles the 2025 projection logic internally when we pass the 2024-ending history
        tier, details = engine.get_prediction(history, apply_calibration=True)
        
        rows_2025.append({
            "player": player,
            "Team": row["Team"],
            "Tier": tier,
            "Pred_XGB": details["xgb_grade"],
            "Pred_Transformer": details["transformer_grade"],
            "Ensemble_Pred": details["predicted_grade"],
            "Conf_Lower": details["confidence_interval"][0],
            "Conf_Upper": details["confidence_interval"][1],
            "Vol_Index": details["volatility_index"],
            "Calibration_Penalty": details["calibration_penalty"]
        })
        
    final_2025 = pd.DataFrame(rows_2025).sort_values("Ensemble_Pred", ascending=False)
    final_out = os.path.join(os.path.dirname(__file__), "QB_2025_Final_Rankings.csv")
    final_2025.to_csv(final_out, index=False)
    
    print("\n==== 🏆 TOP 10 2025 QB RANKINGS (CHAOS-AWARE) ====")
    display_cols = ["player", "Team", "Tier", "Ensemble_Pred", "Conf_Lower", "Conf_Upper", "Vol_Index"]
    print(final_2025[display_cols].head(10).to_string(index=False))
    print(f"\n[3/3] Full rankings saved to {final_out}")

if __name__ == "__main__":
    pass
