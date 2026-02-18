"""
Run this once after training to compute the LB calibration constants.
Prints the exact values to paste into lb_model_wrapper.py.

Usage:
    python compute_lb_calibration.py
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error

from backend.ML.LB_Pranay_Transformers.Player_Model_LB import (
    PlayerTransformerRegressor,
    create_sequences_temporal,
    SEQ_LEN,
)

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
DATA_PATH   = "backend/ML/LB.csv"
MODEL_PATH  = "backend/ML/LB_Pranay_Transformers/lb_best_classifier.pth"
SCALER_PATH = "backend/ML/LB_Pranay_Transformers/lb_player_scaler.joblib"

TRAIN_END_YEAR = 2022
VAL_YEAR       = 2023
DEVICE         = torch.device("cpu")

# ─────────────────────────────────────────────────────────────
# FEATURES (MUST MATCH TRAINING EXACTLY)
# ─────────────────────────────────────────────────────────────
features = [
    "grades_defense",
    "pressure_rate",
    "sack_rate",
    "hit_rate",
    "hurry_rate",
    "stop_rate",
    "tfl_rate",
    "penalty_rate",
    "missed_tackle_rate",
    "target_rate",
    "int_rate",
    "pbu_rate",
    "box_share",
    "offball_share",
    "age",
    "years_in_league",
    "adjusted_value",
    "Cap_Space",
    "team_performance_proxy",
    "delta_grade",
]

target_col = "grades_defense"

# ─────────────────────────────────────────────────────────────
# LOAD + REBUILD DATA EXACTLY LIKE TRAINING
# ─────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df = df[df["position"] == "LB"].copy()
df.sort_values(by=["player", "Year"], inplace=True)

numeric_cols = [
    "grades_defense", "grades_coverage_defense", "grades_run_defense",
    "grades_pass_rush_defense", "grades_tackle",
    "sacks", "hits", "hurries", "total_pressures", "stops",
    "tackles", "tackles_for_loss", "assists", "missed_tackles",
    "targets", "interceptions", "pass_break_ups", "penalties",
    "snap_counts_defense", "snap_counts_box", "snap_counts_offball",
    "snap_counts_pass_rush", "snap_counts_run_defense",
    "age", "adjusted_value", "Cap_Space",
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["years_in_league"] = df.groupby("player").cumcount()
df["delta_grade"] = df.groupby("player")["grades_defense"].diff().fillna(0)
df["team_performance_proxy"] = df.groupby(["Team", "Year"])["Net EPA"].transform("mean")

def safe_div(a, b):
    return np.where(b == 0, 0, a / b)

snap = df["snap_counts_defense"]

df["pressure_rate"]      = safe_div(df["total_pressures"], snap)
df["sack_rate"]          = safe_div(df["sacks"], snap)
df["hit_rate"]           = safe_div(df["hits"], snap)
df["hurry_rate"]         = safe_div(df["hurries"], snap)
df["stop_rate"]          = safe_div(df["stops"], snap)
df["tfl_rate"]           = safe_div(df["tackles_for_loss"], snap)
df["penalty_rate"]       = safe_div(df["penalties"], snap)
df["missed_tackle_rate"] = safe_div(df["missed_tackles"], snap)
df["target_rate"]        = safe_div(df["targets"], snap)
df["int_rate"]           = safe_div(df["interceptions"], snap)
df["pbu_rate"]           = safe_div(df["pass_break_ups"], snap)
df["box_share"]          = safe_div(df["snap_counts_box"], snap)
df["offball_share"]      = safe_div(df["snap_counts_offball"], snap)

# Ensure all features exist
missing = [col for col in features if col not in df.columns]
if missing:
    raise ValueError(f"Missing feature columns: {missing}")

df = df.dropna(subset=features + [target_col]).copy()

# ─────────────────────────────────────────────────────────────
# SCALE USING TRAINED SCALER
# ─────────────────────────────────────────────────────────────
scaler = joblib.load(SCALER_PATH)
df_scaled = df.copy()
df_scaled[features] = scaler.transform(df[features])

# ─────────────────────────────────────────────────────────────
# CREATE TEMPORAL SEQUENCES
# ─────────────────────────────────────────────────────────────
X_all, y_all, masks_all, years_all, players_all = create_sequences_temporal(
    df_scaled, df, SEQ_LEN, features, target_col
)

val_idx = years_all == VAL_YEAR
X_val = X_all[val_idx]
y_val = y_all[val_idx]
m_val = masks_all[val_idx]

if len(y_val) == 0:
    raise ValueError("No validation samples found for calibration.")

print(f"Validation samples (year={VAL_YEAR}): {len(y_val)}")

# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
model = PlayerTransformerRegressor(
    input_dim=len(features),
    seq_len=SEQ_LEN,
    num_layers=2,     # MUST MATCH TRAINING
    ff_dim=128,       # MUST MATCH TRAINING
    dropout=0.2,      # MUST MATCH TRAINING
).to(DEVICE).float()

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ─────────────────────────────────────────────────────────────
# RUN INFERENCE
# ─────────────────────────────────────────────────────────────
with torch.no_grad():
    x = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    m = torch.tensor(m_val, dtype=torch.bool).to(DEVICE)
    raw_preds = model(x, mask=m).squeeze(-1).cpu().numpy()

# ─────────────────────────────────────────────────────────────
# COMPUTE CALIBRATION CONSTANTS
# ─────────────────────────────────────────────────────────────
pred_mean   = float(np.mean(raw_preds))
actual_mean = float(np.mean(y_val))
pred_std    = float(np.std(raw_preds))
actual_std  = float(np.std(y_val))

std_ratio = actual_std / pred_std if pred_std > 0 else 1.0

print("\n" + "=" * 60)
print("CALIBRATION CONSTANTS — paste into lb_model_wrapper.py")
print("=" * 60)
print(f"CALIB_PRED_MEAN   = {pred_mean:.6f}")
print(f"CALIB_ACTUAL_MEAN = {actual_mean:.6f}")
print(f"CALIB_STD_RATIO   = {std_ratio:.6f}")
print("=" * 60)

# Diagnostics
calibrated = (raw_preds - pred_mean) * std_ratio + actual_mean

print("\nDiagnostics:")
print(f"MAE before calibration: {mean_absolute_error(y_val, raw_preds):.4f}")
print(f"MAE after  calibration: {mean_absolute_error(y_val, calibrated):.4f}")
print(f"Raw pred mean/std: {pred_mean:.2f} / {pred_std:.2f}")
print(f"Actual   mean/std: {actual_mean:.2f} / {actual_std:.2f}")
