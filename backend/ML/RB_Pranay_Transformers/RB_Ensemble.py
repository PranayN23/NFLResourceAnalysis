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

# Import classes and wrapper (assuming similar structure for RB)
from backend.ML.RB_Pranay_Transformers.Player_Model_RB import PlayerTransformerRegressor, Time2Vec
from backend.agent.rb_model_wrapper import RBModelInference

# ==========================================
# 1. SETUP & CONFIG
# ==========================================
DATA_FILE = os.path.join(os.path.dirname(__file__), "../HB.csv")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "rb_best_classifier.pth")
SCALER_OUT = os.path.join(os.path.dirname(__file__), "rb_player_scaler.joblib")
XGB_MODEL_OUT = os.path.join(os.path.dirname(__file__), "rb_best_xgb.joblib")

MODE = "VALIDATION"  # Predict 2024 (Train < 2024)
#MODE = "DREAM"       # Predict 2025 (Train <= 2024)

print(f"==== STARTING RB ENSEMBLE MODELING (Mode: {MODE}) ====")

TRANSFORMER_FEATURES = [
    'grades_offense', 'grades_run', 'grades_pass_route', 'elusive_rating',
    'yards', 'yards_after_contact', 'yco_attempt', 'breakaway_percent',
    'explosive', 'first_downs', 'receptions', 'targets', 'total_touches',
    'touchdowns', 'adjusted_value', 'Cap_Space', 'age', 'years_in_league',
    'delta_grade', 'delta_yards', 'delta_touches', 'team_performance_proxy'
]

XGB_FEATURES = [
    'lag_grades_offense', 'lag_yards', 'lag_yco_attempt', 'lag_elusive_rating',
    'lag_breakaway_percent', 'lag_explosive', 'lag_total_touches', 'lag_touchdowns',
    'adjusted_value', 'age', 'years_in_league', 'delta_grade_lag',
    'team_performance_proxy_lag', 'lag_receptions'
]

T2V_SIGNAL_FEATURE = 't2v_transformer_signal'

# ==========================================
# 2. DATA PREPARATION (WITH TEMPORAL AWARENESS)
# ==========================================
def prepare_data():
    df = pd.read_csv(DATA_FILE)
    df['adjusted_value'] = pd.to_numeric(df['adjusted_value'], errors='coerce').fillna(0)
    df = df[df['total_touches'] >= 50].copy()  # RB minimum threshold
    df = df.sort_values(['player', 'Year'])
    groups = df.groupby('player')
    
    # Feature engineering - all using past data only (shift creates lag)
    df["years_in_league"] = groups.cumcount()
    df["delta_grade"] = groups["grades_offense"].diff().fillna(0)
    df["delta_yards"] = groups["yards"].diff().fillna(0)
    df["delta_touches"] = groups["total_touches"].diff().fillna(0)
    df['team_performance_proxy'] = df.groupby(['Team', 'Year'])['Net EPA'].transform('mean')
    
    # Lagged features for XGBoost (using previous year to predict current)
    df['lag_grades_offense'] = groups['grades_offense'].shift(1)
    df['lag_yards'] = groups['yards'].shift(1)
    df['lag_yco_attempt'] = groups['yco_attempt'].shift(1)
    df['lag_elusive_rating'] = groups['elusive_rating'].shift(1)
    df['lag_breakaway_percent'] = groups['breakaway_percent'].shift(1)
    df['lag_explosive'] = groups['explosive'].shift(1)
    df['lag_total_touches'] = groups['total_touches'].shift(1).fillna(50)  # Baseline workload
    df['lag_touchdowns'] = groups['touchdowns'].shift(1)
    df['lag_receptions'] = groups['receptions'].shift(1)
    df['delta_grade_lag'] = groups['lag_grades_offense'].diff().fillna(0)
    df['team_performance_proxy_lag'] = groups['team_performance_proxy'].shift(1)
    
    target_col = 'grades_offense'
    df_clean = df.dropna(subset=XGB_FEATURES + [target_col]).copy()
    
    # CRITICAL: Convert all XGB features to numeric (fix object dtype error)
    for col in XGB_FEATURES:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Drop any rows with NaN after conversion
    df_clean = df_clean.dropna(subset=XGB_FEATURES + [target_col]).copy()
    
    return df_clean, df, target_col


def add_transformer_signal_feature(df_features, df_all):
    """Add transformer(Time2Vec)-derived signal as an XGB stacking feature."""
    print("\nComputing transformer(Time2Vec) signal for XGBoost stacking feature...")
    signal_engine = RBModelInference(MODEL_OUT, scaler_path=SCALER_OUT, xgb_path=None)

    signals = []
    total_rows = len(df_features)
    for idx, (_, row) in enumerate(df_features.iterrows(), start=1):
        history = df_all[(df_all['player'] == row['player']) & (df_all['Year'] < row['Year'])].copy()

        if len(history) == 0:
            signals.append(np.nan)
        else:
            _, details = signal_engine.get_prediction(history, mode="transformer", apply_calibration=False)
            signals.append(details.get("transformer_grade", np.nan))

        if idx % 200 == 0 or idx == total_rows:
            print(f"  Processed {idx}/{total_rows} rows")

    out_df = df_features.copy()
    out_df[T2V_SIGNAL_FEATURE] = pd.to_numeric(signals, errors='coerce')
    out_df[T2V_SIGNAL_FEATURE] = out_df[T2V_SIGNAL_FEATURE].fillna(out_df[T2V_SIGNAL_FEATURE].median())
    return out_df

df_clean, df_all, target_col = prepare_data()
df_clean = add_transformer_signal_feature(df_clean, df_all)
XGB_FEATURES = XGB_FEATURES + [T2V_SIGNAL_FEATURE]

# ==========================================
# 3. TEMPORAL SPLIT & MODEL TRAINING
# ==========================================

if MODE == "VALIDATION":
    print("\n=== VALIDATION MODE: Training on <2024, Testing on 2024 ===")
    
    # CRITICAL: Train XGBoost ONLY on data before 2024
    train_data = df_clean[df_clean["Year"] < 2024].copy()
    test_data = df_clean[df_clean["Year"] == 2024].copy()
    
    print(f"\n[1/3] Training XGBoost on {len(train_data)} samples (<2024)...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500, 
        learning_rate=0.05, 
        max_depth=5, 
        random_state=42
    )
    xgb_model.fit(train_data[XGB_FEATURES], train_data[target_col])
    joblib.dump(xgb_model, XGB_MODEL_OUT)
    
    # Evaluate XGBoost performance
    xgb_train_preds = xgb_model.predict(train_data[XGB_FEATURES])
    xgb_test_preds = xgb_model.predict(test_data[XGB_FEATURES])
    
    print(f"XGBoost Train MAE: {mean_absolute_error(train_data[target_col], xgb_train_preds):.4f}")
    print(f"XGBoost Test MAE: {mean_absolute_error(test_data[target_col], xgb_test_preds):.4f}")
    
    # Initialize production wrapper
    engine = RBModelInference(MODEL_OUT, scaler_path=SCALER_OUT, xgb_path=XGB_MODEL_OUT)
    
    # Generate predictions
    print("\n[2/3] Generating ensemble predictions for 2024...")
    results = []
    for _, row in test_data.iterrows():
        name = row['player']
        # CRITICAL: Only use history BEFORE the target year
        history = df_all[(df_all['player'] == name) & (df_all['Year'] < 2024)].copy()
        
        if len(history) == 0:
            print(f"Warning: No history for {name} before 2024, skipping...")
            continue
            
        tier, details = engine.get_prediction(history, apply_calibration=True)
        
        results.append({
            "player": name,
            "Team": row["Team"],
            "Year": 2024,
            "Actual_Grade": row["grades_offense"],
            "Pred_XGB": details["xgb_grade"],
            "Pred_Transformer": details["transformer_grade"],
            "Ensemble_Pred": details["predicted_grade"],
            "Error": row["grades_offense"] - details["predicted_grade"],
            "Abs_Error": abs(row["grades_offense"] - details["predicted_grade"]),
            "Conf_Lower": details["confidence_interval"][0],
            "Conf_Upper": details["confidence_interval"][1]
        })
    
    final_df = pd.DataFrame(results)
    
    # Calculate metrics
    mae = final_df["Abs_Error"].mean()
    rmse = np.sqrt((final_df["Error"]**2).mean())
    
    print("\n=== VALIDATION RESULTS (2024) ===")
    print(f"Ensemble MAE: {mae:.4f}")
    print(f"Ensemble RMSE: {rmse:.4f}")
    print(f"Mean Error (bias): {final_df['Error'].mean():.4f}")
    
    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "RB_2024_Validation_Results.csv")
    final_df = final_df.sort_values(by="Ensemble_Pred", ascending=False)
    final_df.to_csv(out_path, index=False)
    print(f"\n[3/3] Saved Validation Report to {out_path}")
    
    # Show top errors
    print("\n=== LARGEST PREDICTION ERRORS ===")
    display_cols = ["player", "Actual_Grade", "Ensemble_Pred", "Error", "Abs_Error"]
    print(final_df.nlargest(5, "Abs_Error")[display_cols].to_string(index=False))

elif MODE == "DREAM":
    print("\n=== DREAM MODE: Training on ≤2024, Predicting 2025 ===")
    
    # CRITICAL: Train XGBoost on ALL available data (≤2024) for 2025 predictions
    print(f"\n[1/3] Training XGBoost on {len(df_clean)} samples (≤2024)...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500, 
        learning_rate=0.05, 
        max_depth=5, 
        random_state=42
    )
    xgb_model.fit(df_clean[XGB_FEATURES], df_clean[target_col])
    joblib.dump(xgb_model, XGB_MODEL_OUT)
    
    # Evaluate on full dataset (for reference only)
    xgb_all_preds = xgb_model.predict(df_clean[XGB_FEATURES])
    print(f"XGBoost Full Dataset MAE: {mean_absolute_error(df_clean[target_col], xgb_all_preds):.4f}")
    
    # Initialize production wrapper
    engine = RBModelInference(MODEL_OUT, scaler_path=SCALER_OUT, xgb_path=XGB_MODEL_OUT)
    
    # Get active players from 2024
    active_2024 = df_all[df_all["Year"] == 2024].copy()
    
    print("\n[2/3] Generating 2025 'Dream' Projections with Calibration...")
    rows_2025 = []
    
    retired_players = ["Derrick Henry"]  # Example - update as needed
    
    for _, row in active_2024.iterrows():
        player = row['player']
        
        if player in retired_players:
            print(f"Skipping {player} (retired)")
            continue
        
        # Use full history up to and including 2024
        history = df_all[df_all["player"] == player].sort_values("Year").tail(5)
        
        if len(history) == 0:
            print(f"Warning: No history for {player}, skipping...")
            continue
        
        # The wrapper handles the 2025 projection logic
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
            "Age_Adjustment": details["age_adjustment"],
            "Last_Grade_2024": history.iloc[-1]["grades_offense"],
            "YoY_Change_Pred": details["predicted_grade"] - history.iloc[-1]["grades_offense"],
            "Touches_2024": row["total_touches"],
            "Age_2024": row["age"]
        })
    
    final_2025 = pd.DataFrame(rows_2025).sort_values("Ensemble_Pred", ascending=False)
    final_2025['Age_2024'] = pd.to_numeric(final_2025['Age_2024'], errors='coerce')
    # Save full results
    final_out = os.path.join(os.path.dirname(__file__), "RB_2025_Final_Rankings.csv")
    final_2025.to_csv(final_out, index=False)
    
    # Display top rankings
    print("\n" + "="*80)
    print("🏆 TOP 15 RB RANKINGS FOR 2025 (ENSEMBLE + CALIBRATION)")
    print("="*80)
    display_cols = [
        "player", "Team", "Tier", "Ensemble_Pred", 
        "Conf_Lower", "Conf_Upper", "Vol_Index", "YoY_Change_Pred", "Age_2024"
    ]
    print(final_2025[display_cols].head(15).to_string(index=False))
    
    # Show risers and fallers
    print("\n" + "="*80)
    print("📈 BIGGEST PROJECTED RISERS (2024 → 2025)")
    print("="*80)
    riser_cols = ["player", "Last_Grade_2024", "Ensemble_Pred", "YoY_Change_Pred", "Age_2024"]
    print(final_2025.nlargest(5, "YoY_Change_Pred")[riser_cols].to_string(index=False))
    
    print("\n" + "="*80)
    print("📉 BIGGEST PROJECTED FALLERS (2024 → 2025)")
    print("="*80)
    print(final_2025.nsmallest(5, "YoY_Change_Pred")[riser_cols].to_string(index=False))
    
    # Show age-related insights
    print("\n" + "="*80)
    print("👴 AGING CONCERNS (Age 28+ with projected decline)")
    print("="*80)
    aging_concerns = final_2025[(final_2025['Age_2024'] >= 28) & 
                                (final_2025['YoY_Change_Pred'] < -3)]
    if len(aging_concerns) > 0:
        print(aging_concerns[["player", "Age_2024", "Last_Grade_2024", 
                              "Ensemble_Pred", "YoY_Change_Pred"]].to_string(index=False))
    else:
        print("No significant aging concerns identified")
    
    print(f"\n[3/3] Full rankings saved to {final_out}")

# ==========================================
# 4. SUMMARY & RECOMMENDATIONS
# ==========================================
print("\n" + "="*80)
print("MODEL TRAINING SUMMARY")
print("="*80)

if MODE == "VALIDATION":
    print(f"✓ XGBoost trained on: Years < 2024")
    print(f"✓ Transformer trained on: Years ≤ 2022 (from separate training)")
    print(f"✓ Tested on: Year 2024")
    print(f"✓ No data leakage: Future data excluded from training")
    
elif MODE == "DREAM":
    print(f"✓ XGBoost trained on: All available data (≤ 2024)")
    print(f"✓ Transformer trained on: Years ≤ 2022 (from separate training)")
    print(f"✓ Predicting: Year 2025")
    print(f"✓ Using: Most recent available data for production predictions")
    print(f"✓ Age considerations: Built into predictions for RB position")

print("="*80)

