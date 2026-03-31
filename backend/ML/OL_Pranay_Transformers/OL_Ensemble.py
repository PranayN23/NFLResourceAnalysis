import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import joblib

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Import OL Transformer and wrapper
from backend.ML.OL_Pranay_Transformers.Player_Model_OL import PlayerTransformerRegressor, Time2Vec
from backend.agent.ol_model_wrapper import OLModelInference  # Assuming you made OL version like RB wrapper

# ==========================================
# 1. SETUP & CONFIG
# ==========================================
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#MODE = "VALIDATION"  # Predict 2024
MODE = "DREAM"      # Predict 2025

MODEL_OUT = os.path.join(os.path.dirname(__file__), "ol_best_classifier.pth")
SCALER_OUT = os.path.join(os.path.dirname(__file__), "ol_player_scaler.joblib")
XGB_MODEL_OUT = os.path.join(os.path.dirname(__file__), "ol_best_xgb.joblib")

print(f"==== STARTING OL ENSEMBLE MODELING (Mode: {MODE}) ====")

# Transformer & XGBoost features
TRANSFORMER_FEATURES = [
    'adjusted_value', 'Cap_Space', 'age', 'years_in_league',
    'delta_grade', 'delta_run_block', 'delta_pass_block',
    'team_performance_proxy', 'sacks_allowed_rate', 'hits_allowed_rate',
    'hurries_allowed_rate', 'pressures_allowed_rate', 'penalties_rate',
    'pass_block_efficiency', 'snap_counts_block_share', 'snap_counts_run_block_share',
    'snap_counts_pass_block_share', 'pos_T', 'pos_G', 'pos_C'
]

XGB_FEATURES = [
    'lag_grades_offense', 'lag_grades_run_block', 'lag_grades_pass_block',
    'adjusted_value', 'age', 'years_in_league',
    'delta_grade_lag', 'team_performance_proxy_lag',
    'sacks_allowed_rate', 'hits_allowed_rate', 'hurries_allowed_rate'
]

T2V_SIGNAL_FEATURE = 't2v_transformer_signal'


# ==========================================
# 2. DATA LOADING & PREPARATION
# ==========================================
# Load all 3 OL files
df_guard = pd.read_csv(os.path.join(parent_dir, "G.csv"))
df_center = pd.read_csv(os.path.join(parent_dir, "C.csv"))
df_tackle = pd.read_csv(os.path.join(parent_dir, "T.csv"))

# Combine into a single DataFrame
df = pd.concat([df_guard, df_center, df_tackle], axis=0, ignore_index=True)

# One-hot encode positions
df = pd.get_dummies(df, columns=['position'], prefix='pos')

# Ensure numeric columns
grade_cols = ['grades_offense', 'grades_run_block', 'grades_pass_block']
for col in grade_cols + ['adjusted_value', 'Cap_Space', 'age', 'snap_counts_offense',
                         'snap_counts_run_block', 'snap_counts_pass_block', 'snap_counts_block',
                         'sacks_allowed', 'hits_allowed', 'hurries_allowed', 'pressures_allowed', 'penalties',
                         'pbe', 'Net EPA']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Filter OL players with enough snaps
df = df[df['snap_counts_offense'] >= 100].copy()

# Sort by player and year
df.sort_values(by=['player', 'Year'], inplace=True)

# =========================
# Feature Engineering
# =========================
df["years_in_league"] = df.groupby("player").cumcount()
df["delta_grade"] = df.groupby("player")["grades_offense"].diff().fillna(0)
df["delta_run_block"] = df.groupby("player")["grades_run_block"].diff().fillna(0)
df["delta_pass_block"] = df.groupby("player")["grades_pass_block"].diff().fillna(0)
df['team_performance_proxy'] = df.groupby(['Team', 'Year'])['Net EPA'].transform('mean')

# Efficiency / rate metrics
df['hits_allowed_rate'] = df['hits_allowed'] / df['snap_counts_pass_block']
df['hurries_allowed_rate'] = df['hurries_allowed'] / df['snap_counts_pass_block']
df['pressures_allowed_rate'] = df['pressures_allowed'] / df['snap_counts_pass_block']
df['penalties_rate'] = df['penalties'] / df['snap_counts_offense']
df['pass_block_efficiency'] = df['pbe']

# Snap share per OL position
df['snap_counts_block_share'] = df['snap_counts_block'] / df['snap_counts_offense']
df['snap_counts_run_block_share'] = df['snap_counts_run_block'] / df['snap_counts_offense']
df['snap_counts_pass_block_share'] = df['snap_counts_pass_block'] / df['snap_counts_offense']

# Lagged features for XGB
groups = df.groupby('player')
df['lag_grades_offense'] = groups['grades_offense'].shift(1)
df['lag_grades_run_block'] = groups['grades_run_block'].shift(1)
df['lag_grades_pass_block'] = groups['grades_pass_block'].shift(1)
df['delta_grade_lag'] = groups['grades_offense'].diff().shift(1)
df['team_performance_proxy_lag'] = groups['team_performance_proxy'].shift(1)
df['sacks_allowed_rate'] = df['sacks_allowed'] / df['snap_counts_pass_block']

target_col = 'grades_offense'

# Drop rows with missing values
df_clean = df.dropna(subset=TRANSFORMER_FEATURES + XGB_FEATURES + [target_col]).copy()


def add_transformer_signal_feature(df_features, df_all):
    """Add transformer(Time2Vec)-derived signal as an XGB stacking feature."""
    print("\nComputing transformer(Time2Vec) signal for OL XGBoost stacking feature...")
    signal_engine = OLModelInference(MODEL_OUT, scaler_path=SCALER_OUT, xgb_path=None)

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


df_clean = add_transformer_signal_feature(df_clean, df)
XGB_FEATURES = XGB_FEATURES + [T2V_SIGNAL_FEATURE]

# =========================
# Normalize features
# =========================
scaler = StandardScaler()
scaler.fit(df_clean[TRANSFORMER_FEATURES])
df_clean_scaled = df_clean.copy()
df_clean_scaled[TRANSFORMER_FEATURES] = scaler.transform(df_clean[TRANSFORMER_FEATURES])
joblib.dump(scaler, SCALER_OUT)
print(f"Scaler saved to {SCALER_OUT}")

# ==========================================
# 3. TEMPORAL SPLIT & MODELING
# ==========================================
if MODE == "VALIDATION":
    print("=== VALIDATION: Training <2024, Testing 2024 ===")
    train_data = df_clean[df_clean["Year"] < 2024]
    test_data = df_clean[df_clean["Year"] == 2024]

    # Train XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
    xgb_model.fit(train_data[XGB_FEATURES], train_data[target_col])
    joblib.dump(xgb_model, XGB_MODEL_OUT)

    # Initialize OL wrapper
    engine = OLModelInference(MODEL_OUT, SCALER_OUT, XGB_MODEL_OUT)

    # Generate predictions
    results = []
    for _, row in test_data.iterrows():
        name = row['player']
        history = df[df['player'] == name]
        history = history[history['Year'] < 2024]  # Only past years
        if len(history) == 0:
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
    final_df = final_df.sort_values(by="Ensemble_Pred", ascending=False)
    mae = final_df["Abs_Error"].mean()
    rmse = np.sqrt((final_df["Error"]**2).mean())

    print(f"\nEnsemble MAE: {mae:.4f}")
    print(f"Ensemble RMSE: {rmse:.4f}")

    out_path = os.path.join(os.path.dirname(__file__), "OL_2024_Validation_Results.csv")
    final_df.to_csv(out_path, index=False)
    print(f"Validation results saved to {out_path}")

elif MODE == "DREAM":
    print("=== DREAM: Predicting 2025 ===")
    xgb_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
    xgb_model.fit(df_clean[XGB_FEATURES], df_clean[target_col])
    joblib.dump(xgb_model, XGB_MODEL_OUT)
    engine = OLModelInference(MODEL_OUT, SCALER_OUT, XGB_MODEL_OUT)

    # Predict 2025 for active 2024 players
    active_2024 = df[df["Year"] == 2024]
    rows_2025 = []
    for _, row in active_2024.iterrows():
        player = row['player']
        history = df[df['player'] == player].sort_values('Year').tail(5)
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
            "Touches_2024": row["snap_counts_offense"],
            "Age_2024": row["age"]
        })
    final_2025 = pd.DataFrame(rows_2025).sort_values("Ensemble_Pred", ascending=False)
    final_out = os.path.join(os.path.dirname(__file__), "OL_2025_Final_Rankings.csv")
    final_2025.to_csv(final_out, index=False)
    print(f"2025 projections saved to {final_out}")
else:
    print("Invalid MODE selected. Choose 'VALIDATION' or 'DREAM'.")