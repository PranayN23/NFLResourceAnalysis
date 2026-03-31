"""
DIAGNOSTIC SCRIPT: Trace NaN Root Cause in Model Predictions
Goal: Identify where nan originates without applying fixes
"""
import pandas as pd
import numpy as np
import torch
import sys
from pathlib import Path

print("="*80)
print("DIAGNOSTIC: NaN ROOT CAUSE ANALYSIS")
print("="*80)

# Load test players
rb_df = pd.read_csv("backend/ML/HB.csv")

# Player with known NaN issue
nan_player = "Mike Boone"
nan_candidate = rb_df[rb_df['player'].astype(str).str.lower() == nan_player.lower()].copy()

# Find a healthy RB for comparison
healthy_candidates = rb_df[rb_df['player'].astype(str).str.lower() != nan_player.lower()]
healthy_player_name = healthy_candidates['player'].iloc[0]
healthy_candidate = healthy_candidates[healthy_candidates['player'] == healthy_player_name].copy().head(len(nan_candidate))

print(f"\nPLAYERS TO COMPARE:")
print(f"  NaN Player: {nan_player} ({len(nan_candidate)} rows)")
print(f"  Healthy Player: {healthy_player_name} ({len(healthy_candidate)} rows)")

# ==============================================================================
# STEP 1: CHECK RAW DATA
# ==============================================================================
print(f"\n{'='*80}")
print(f"STEP 1: RAW CSV DATA INSPECTION")
print(f"{'='*80}")

print(f"\n{nan_player} - Raw data (all columns):")
print(nan_candidate[['player', 'Year', 'grades_offense', 'yards', 'total_touches', 'receptions', 'touchdowns']].to_string())

print(f"\nNull value counts in {nan_player} data:")
print(nan_candidate.isnull().sum())

print(f"\n{healthy_player_name} - Raw data (sample):")
print(healthy_candidate[['player', 'Year', 'grades_offense', 'yards', 'total_touches', 'receptions', 'touchdowns']].head().to_string())

# ==============================================================================
# STEP 2: FEATURE ENGINEERING - TRACE EACH STEP
# ==============================================================================
print(f"\n{'='*80}")
print(f"STEP 2: FEATURE ENGINEERING TRACE")
print(f"{'='*80}")

# Initialize wrapper to access feature prep logic
from backend.agent.rb_model_wrapper import RBModelInference

transformer_path = "backend/ML/RB_Pranay_Transformers/rb_best_classifier.pth"
scaler_path = "backend/ML/RB_Pranay_Transformers/rb_player_scaler.joblib"

wrapper = RBModelInference(transformer_path, scaler_path)

# Prepare features for NaN player
print(f"\nPreparing features for {nan_player}...")
df_nan_prepared, df_nan_xgb = wrapper._prepare_features(nan_candidate)

# Prepare features for healthy player
print(f"Preparing features for {healthy_player_name}...")
df_healthy_prepared, df_healthy_xgb = wrapper._prepare_features(healthy_candidate)

# Show engineered features
print(f"\n{nan_player} - Engineered features (last row):")
for col in wrapper.transformer_features[:5]:
    val = df_nan_prepared[col].iloc[-1]
    print(f"  {col}: {val} (type: {type(val).__name__})")
print(f"  ... ({len(wrapper.transformer_features)} total features)")

print(f"\n{healthy_player_name} - Engineered features (last row):")
for col in wrapper.transformer_features[:5]:
    val = df_healthy_prepared[col].iloc[-1]
    print(f"  {col}: {val} (type: {type(val).__name__})")
print(f"  ... ({len(wrapper.transformer_features)} total features)")

# Check for NaN in engineered features
nan_cols_nan_player = []
nan_cols_healthy_player = []

for col in wrapper.transformer_features:
    if df_nan_prepared[col].isna().any():
        count = df_nan_prepared[col].isna().sum()
        nan_cols_nan_player.append((col, count))
    
    if df_healthy_prepared[col].isna().any():
        count = df_healthy_prepared[col].isna().sum()
        nan_cols_healthy_player.append((col, count))

if nan_cols_nan_player:
    print(f"\n⚠️  NaN values in {nan_player} engineered features:")
    for col, count in nan_cols_nan_player:
        print(f"  {col}: {count} NaN values")
else:
    print(f"\n✓ No NaN in {nan_player} engineered features")

if nan_cols_healthy_player:
    print(f"\n⚠️  NaN values in {healthy_player_name} engineered features:")
    for col, count in nan_cols_healthy_player:
        print(f"  {col}: {count} NaN values")
else:
    print(f"\n✓ No NaN in {healthy_player_name} engineered features")

# ==============================================================================
# STEP 3: SCALER TRANSFORMATION
# ==============================================================================
print(f"\n{'='*80}")
print(f"STEP 3: SCALER TRANSFORMATION")
print(f"{'='*80}")

p_history_nan = df_nan_prepared.tail(wrapper.max_seq_len)
p_history_healthy = df_healthy_prepared.tail(wrapper.max_seq_len)

print(f"\n{nan_player} - Before scaling (last row of sequence):")
last_row_nan = p_history_nan[wrapper.transformer_features].iloc[-1]
for i, col in enumerate(wrapper.transformer_features[:3]):
    print(f"  {col}: {last_row_nan[col]}")
print(f"  ... ({len(wrapper.transformer_features)} total)")

print(f"\n{healthy_player_name} - Before scaling (last row of sequence):")
last_row_healthy = p_history_healthy[wrapper.transformer_features].iloc[-1]
for i, col in enumerate(wrapper.transformer_features[:3]):
    print(f"  {col}: {last_row_healthy[col]}")
print(f"  ... ({len(wrapper.transformer_features)} total)")

# Scale
try:
    history_vals_nan = wrapper.scaler.transform(p_history_nan[wrapper.transformer_features])
    print(f"\n{nan_player} - After scaling:")
    print(f"  Shape: {history_vals_nan.shape}")
    print(f"  Has NaN: {np.isnan(history_vals_nan).any()}")
    if np.isnan(history_vals_nan).any():
        nan_positions = np.argwhere(np.isnan(history_vals_nan))
        print(f"  NaN positions: {nan_positions[:5].tolist()} (showing first 5)")
        for pos in nan_positions[:3]:
            r, c = pos
            col_name = wrapper.transformer_features[c]
            orig_val = p_history_nan[wrapper.transformer_features].iloc[r, c]
            print(f"    Row {r}, Col {col_name}: original={orig_val}, scaled=NaN")
except Exception as e:
    print(f"\n✗ Scaling failed for {nan_player}: {e}")

try:
    history_vals_healthy = wrapper.scaler.transform(p_history_healthy[wrapper.transformer_features])
    print(f"\n{healthy_player_name} - After scaling:")
    print(f"  Shape: {history_vals_healthy.shape}")
    print(f"  Has NaN: {np.isnan(history_vals_healthy).any()}")
except Exception as e:
    print(f"\n✗ Scaling failed for {healthy_player_name}: {e}")

# ==============================================================================
# STEP 4: TENSOR CONVERSION & PADDING
# ==============================================================================
print(f"\n{'='*80}")
print(f"STEP 4: TENSOR CONVERSION & PADDING")
print(f"{'='*80}")

# NaN player
actual_len_nan = len(history_vals_nan)
pad_nan = np.zeros((wrapper.max_seq_len - actual_len_nan, len(wrapper.transformer_features)))
padded_x_nan = np.vstack([pad_nan, history_vals_nan])
mask_nan = [True] * (wrapper.max_seq_len - actual_len_nan) + [False] * actual_len_nan

print(f"\n{nan_player} - Tensor preparation:")
print(f"  Actual sequence length: {actual_len_nan}")
print(f"  Max sequence length: {wrapper.max_seq_len}")
print(f"  Padding added: {wrapper.max_seq_len - actual_len_nan} rows")
print(f"  Padded array shape: {padded_x_nan.shape}")
print(f"  Has NaN in padded: {np.isnan(padded_x_nan).any()}")
if np.isnan(padded_x_nan).any():
    print(f"  NaN comes from: {'scaled features' if np.isnan(history_vals_nan).any() else 'UNKNOWN'}")

x_tensor_nan = torch.tensor(padded_x_nan, dtype=torch.float32).unsqueeze(0)
m_tensor_nan = torch.tensor(mask_nan, dtype=torch.bool).unsqueeze(0)

print(f"  PyTorch tensor shape: {x_tensor_nan.shape}")
print(f"  Has NaN in tensor: {torch.isnan(x_tensor_nan).any().item()}")

# Healthy player
actual_len_healthy = len(history_vals_healthy)
pad_healthy = np.zeros((wrapper.max_seq_len - actual_len_healthy, len(wrapper.transformer_features)))
padded_x_healthy = np.vstack([pad_healthy, history_vals_healthy])
mask_healthy = [True] * (wrapper.max_seq_len - actual_len_healthy) + [False] * actual_len_healthy

print(f"\n{healthy_player_name} - Tensor preparation:")
print(f"  Actual sequence length: {actual_len_healthy}")
print(f"  Padded array shape: {padded_x_healthy.shape}")
print(f"  Has NaN in padded: {np.isnan(padded_x_healthy).any()}")

x_tensor_healthy = torch.tensor(padded_x_healthy, dtype=torch.float32).unsqueeze(0)
m_tensor_healthy = torch.tensor(mask_healthy, dtype=torch.bool).unsqueeze(0)

print(f"  PyTorch tensor shape: {x_tensor_healthy.shape}")
print(f"  Has NaN in tensor: {torch.isnan(x_tensor_healthy).any().item()}")

# ==============================================================================
# STEP 5: MODEL FORWARD PASS
# ==============================================================================
print(f"\n{'='*80}")
print(f"STEP 5: MODEL FORWARD PASS")
print(f"{'='*80}")

print(f"\nInputting to transformer model...")
print(f"{nan_player}:")
print(f"  Input tensor has NaN: {torch.isnan(x_tensor_nan).any().item()}")
print(f"  Input tensor min/max: {x_tensor_nan[~torch.isnan(x_tensor_nan)].min().item():.4f} / {x_tensor_nan[~torch.isnan(x_tensor_nan)].max().item():.4f}")

with torch.no_grad():
    transformer_grade_nan = wrapper.model(x_tensor_nan, mask=m_tensor_nan).item()

print(f"  Output grade: {transformer_grade_nan}")
print(f"  Is NaN: {np.isnan(transformer_grade_nan)}")

print(f"\n{healthy_player_name}:")
print(f"  Input tensor has NaN: {torch.isnan(x_tensor_healthy).any().item()}")
print(f"  Input tensor min/max: {x_tensor_healthy[~torch.isnan(x_tensor_healthy)].min().item():.4f} / {x_tensor_healthy[~torch.isnan(x_tensor_healthy)].max().item():.4f}")

with torch.no_grad():
    transformer_grade_healthy = wrapper.model(x_tensor_healthy, mask=m_tensor_healthy).item()

print(f"  Output grade: {transformer_grade_healthy}")
print(f"  Is NaN: {np.isnan(transformer_grade_healthy)}")

# ==============================================================================
# STEP 6: ENSEMBLE & AGE DECAY
# ==============================================================================
print(f"\n{'='*80}")
print(f"STEP 6: ENSEMBLE & AGE DECAY")
print(f"{'='*80}")

# NaN player ensemble
xgb_grade_nan = 0.0
if wrapper.xgb_model:
    try:
        xgb_grade_nan = wrapper.xgb_model.predict(df_nan_xgb[wrapper.xgb_features])[0]
    except Exception as e:
        print(f"XGB prediction failed for {nan_player}: {e}")

print(f"\n{nan_player}:")
print(f"  Transformer grade: {transformer_grade_nan} (NaN: {np.isnan(transformer_grade_nan)})")
print(f"  XGB grade: {xgb_grade_nan} (NaN: {np.isnan(xgb_grade_nan)})")

trans_weight = 0.50
xgb_weight = 0.50
ensemble_grade_nan = (transformer_grade_nan * trans_weight) + (xgb_grade_nan * xgb_weight)
print(f"  Ensemble (50/50): {ensemble_grade_nan} (NaN: {np.isnan(ensemble_grade_nan)})")

# Age decay
current_age_nan = float(df_nan_prepared.iloc[-1]['age'])
age_adjustment_nan = wrapper.get_age_decay_factor(current_age_nan)
print(f"  Current age: {current_age_nan}")
print(f"  Age adjustment: {age_adjustment_nan}")

final_grade_nan = ensemble_grade_nan - age_adjustment_nan
print(f"  Final grade (ensemble - age_adj): {final_grade_nan} (NaN: {np.isnan(final_grade_nan)})")

# Healthy player
xgb_grade_healthy = 0.0
if wrapper.xgb_model:
    try:
        xgb_grade_healthy = wrapper.xgb_model.predict(df_healthy_xgb[wrapper.xgb_features])[0]
    except Exception as e:
        print(f"\nXGB prediction failed for {healthy_player_name}: {e}")

print(f"\n{healthy_player_name}:")
print(f"  Transformer grade: {transformer_grade_healthy} (NaN: {np.isnan(transformer_grade_healthy)})")
print(f"  XGB grade: {xgb_grade_healthy} (NaN: {np.isnan(xgb_grade_healthy)})")

ensemble_grade_healthy = (transformer_grade_healthy * trans_weight) + (xgb_grade_healthy * xgb_weight)
print(f"  Ensemble (50/50): {ensemble_grade_healthy} (NaN: {np.isnan(ensemble_grade_healthy)})")

current_age_healthy = float(df_healthy_prepared.iloc[-1]['age'])
age_adjustment_healthy = wrapper.get_age_decay_factor(current_age_healthy)
print(f"  Current age: {current_age_healthy}")
print(f"  Age adjustment: {age_adjustment_healthy}")

final_grade_healthy = ensemble_grade_healthy - age_adjustment_healthy
print(f"  Final grade (ensemble - age_adj): {final_grade_healthy} (NaN: {np.isnan(final_grade_healthy)})")

# ==============================================================================
# SUMMARY: WHERE DOES NaN ORIGINATE?
# ==============================================================================
print(f"\n{'='*80}")
print(f"DIAGNOSIS SUMMARY")
print(f"{'='*80}")

print(f"\n{nan_player}:")
print(f"  Raw data: Has NaN={nan_candidate.isnull().any().any()}")
print(f"  Engineered features: Has NaN={len(nan_cols_nan_player) > 0}")
print(f"  After scaling: Has NaN={np.isnan(history_vals_nan).any()}")
print(f"  Transformer input: Has NaN={torch.isnan(x_tensor_nan).any().item()}")
print(f"  Transformer output: {transformer_grade_nan} (NaN={np.isnan(transformer_grade_nan)})")
print(f"  Final grade: {final_grade_nan} (NaN={np.isnan(final_grade_nan)})")

if np.isnan(final_grade_nan):
    if np.isnan(transformer_grade_nan):
        print(f"\n❌ ROOT CAUSE: Transformer forward pass produces NaN")
        if np.isnan(history_vals_nan).any():
            print(f"   └─ Input contains NaN from scaler transformation")
        else:
            print(f"   └─ Input is valid but model output is NaN (possible gradient explosion)")
    else:
        print(f"\n❌ ROOT CAUSE: Age decay produces NaN (age_adjustment={age_adjustment_nan})")

print(f"\n✓ COMPARISON: {healthy_player_name} final grade: {final_grade_healthy} (NaN={np.isnan(final_grade_healthy)})")
