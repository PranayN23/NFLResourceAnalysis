"""
Run this once after training to compute the ED calibration constants.
Prints the exact values to paste into ED_inference.py.

Usage:
    python compute_ed_calibration.py
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from backend.ML.ED_Pranay_Transformers.Player_Model_ED import (
    PlayerTransformerRegressor,
    create_sequences_temporal,
    SEQ_LEN,
)

# ── Paths (adjust if yours differ) ──────────────────────────────────────────
DATA_PATH      = "backend/ML/ED.csv"
MODEL_PATH     = "backend/ML/ED_Transformers/ed_best_classifier.pth"
SCALER_PATH    = "backend/ML/ED_Transformers/ed_player_scaler.joblib"

TRAIN_END_YEAR = 2022
VAL_YEAR       = 2023   # calibrate on val set (unseen during training)
DEVICE         = torch.device("cpu")

# ── Feature list — must match Player_Model_ED.py exactly ────────────────────
features = [
    "grades_defense",
    "grades_run_defense",
    "grades_pass_rush_defense",
    "pressure_rate",
    "sack_rate",
    "hit_rate",
    "hurry_rate",
    "stop_rate",
    "tfl_rate",
    "penalty_rate",
    "outside_t_share",
    "over_t_share",
    "age",
    "years_in_league",
    "adjusted_value",
    "Cap_Space",
    "team_performance_proxy",
    "delta_grade",
    "delta_pass_rush",
    "delta_run_def",
]
target_col = "grades_defense"

# ── Rebuild the same cleaned/engineered df as training ──────────────────────
df = pd.read_csv(DATA_PATH)
df = df[df["position"] == "ED"].copy()
df.sort_values(by=["player", "Year"], inplace=True)

numeric_cols = [
    "grades_pass_rush_defense", "grades_run_defense", "grades_defense",
    "pressures", "sacks", "tackles", "assists", "missed_tackles",
    "age", "snap_counts_defense", "hits", "hurries", "stops",
    "tackles_for_loss", "total_pressures", "penalties",
    "snap_counts_pass_rush", "snap_counts_run_defense",
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["years_in_league"] = df.groupby("player").cumcount()
df["delta_grade"]     = df.groupby("player")["grades_defense"].diff().fillna(0)
df["delta_pass_rush"] = df.groupby("player")["grades_pass_rush_defense"].diff().fillna(0)
df["delta_run_def"]   = df.groupby("player")["grades_run_defense"].diff().fillna(0)
df["team_performance_proxy"] = df.groupby(["Team", "Year"])["Net EPA"].transform("mean")

def safe_div(a, b):
    return np.where(b == 0, 0, a / b)

snap    = df["snap_counts_defense"]
snap_dl = df["snap_counts_dl"]

df["pressure_rate"]   = safe_div(df["total_pressures"].values,              snap)
df["sack_rate"]       = safe_div(df["sacks"].values,                        snap)
df["hit_rate"]        = safe_div(df["hits"].values,                         snap)
df["hurry_rate"]      = safe_div(df["hurries"].values,                      snap)
df["stop_rate"]       = safe_div(df["stops"].values,                        snap)
df["tfl_rate"]        = safe_div(df["tackles_for_loss"].values,             snap)
df["penalty_rate"]    = safe_div(df["penalties"].values,                    snap)
df["outside_t_share"] = safe_div(df["snap_counts_dl_outside_t"].values,     snap_dl)
df["over_t_share"]    = safe_div(df["snap_counts_dl_over_t"].values,        snap_dl)

all_model_cols = features + [target_col]
for col in all_model_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df_clean = df.dropna(subset=all_model_cols).copy()

# ── Scale using saved scaler ─────────────────────────────────────────────────
scaler = joblib.load(SCALER_PATH)
df_clean_scaled           = df_clean.copy()
df_clean_scaled[features] = scaler.transform(df_clean[features])

# ── Build sequences & isolate val year ──────────────────────────────────────
X_all, y_all, masks_all, years_all = create_sequences_temporal(
    df_clean_scaled, df_clean, SEQ_LEN, features, target_col
)

val_idx = years_all == VAL_YEAR
X_val, y_val, m_val = X_all[val_idx], y_all[val_idx], masks_all[val_idx]

print(f"Val samples (year={VAL_YEAR}): {len(y_val)}")

# ── Load model & get raw predictions ────────────────────────────────────────
model = PlayerTransformerRegressor(
    input_dim=len(features),
    seq_len=SEQ_LEN,
    num_layers=1,
    dropout=0.3,
).to(DEVICE).float()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

with torch.no_grad():
    x = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    m = torch.tensor(m_val, dtype=torch.bool).to(DEVICE)
    raw_preds = model(x, mask=m).squeeze(-1).cpu().numpy()

# ── Compute calibration constants ────────────────────────────────────────────
pred_mean   = float(np.mean(raw_preds))
actual_mean = float(np.mean(y_val))
pred_std    = float(np.std(raw_preds))
actual_std  = float(np.std(y_val))
std_ratio   = actual_std / pred_std if pred_std > 0 else 1.0

print("\n" + "=" * 55)
print("  CALIBRATION CONSTANTS — paste into ED_inference.py")
print("=" * 55)
print(f"  CALIB_PRED_MEAN   = {pred_mean:.4f}")
print(f"  CALIB_ACTUAL_MEAN = {actual_mean:.4f}")
print(f"  CALIB_STD_RATIO   = {std_ratio:.4f}")
print("=" * 55)

print(f"\nDiagnostics:")
print(f"  Raw pred  — mean: {pred_mean:.2f}, std: {pred_std:.2f}, "
      f"min: {raw_preds.min():.2f}, max: {raw_preds.max():.2f}")
print(f"  Actual    — mean: {actual_mean:.2f}, std: {actual_std:.2f}, "
      f"min: {y_val.min():.2f}, max: {y_val.max():.2f}")

# Verify calibration looks right
calibrated = (raw_preds - pred_mean) * std_ratio + actual_mean
from sklearn.metrics import mean_absolute_error
print(f"\nMAE before calibration: {mean_absolute_error(y_val, raw_preds):.4f}")
print(f"MAE after  calibration: {mean_absolute_error(y_val, calibrated):.4f}")
print("(MAE won't change much — calibration fixes the spread, not the MAE)")