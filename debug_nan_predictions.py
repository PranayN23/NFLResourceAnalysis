"""
Debug script to understand why NaN grades are being predicted by model wrappers.
"""
import pandas as pd
import numpy as np
import torch
from backend.agent.rb_model_wrapper import RBModelInference

print("="*70)
print("DEBUGGING NaN PREDICTIONS IN RB MODEL")
print("="*70)

# Load test player with known NaN issue
rb_df = pd.read_csv("backend/ML/HB.csv")
boone = rb_df[rb_df['player'].astype(str).str.lower() == 'mike boone'].copy()

print(f"\nPlayer: Mike Boone")
print(f"Years available: {sorted(boone['Year'].tolist())}")
print(f"\nRaw data (last 5 rows):")
cols_to_show = ['player', 'Year', 'grades_offense', 'yards', 'yco_attempt', 'receptions', 'total_touches', 'adjusted_value']
print(boone[cols_to_show].tail(5).to_string())

# Initialize wrapper
transformer_path = "backend/ML/RB_Pranay_Transformers/rb_best_classifier.pth"
scaler_path = "backend/ML/RB_Pranay_Transformers/rb_player_scaler.joblib"

print(f"\nLoading models:")
print(f"  Transformer: {transformer_path}")
print(f"  Scaler: {scaler_path}")

wrapper = RBModelInference(transformer_path, scaler_path)

# Step 1: Prepare features
print(f"\n1. FEATURE PREPARATION")
print(f"   Expected transformer_features: {len(wrapper.transformer_features)}")
print(f"   Features: {', '.join(wrapper.transformer_features[:5])}...")

df_prepared, df_xgb = wrapper._prepare_features(boone)

print(f"\n2. CHECK FOR NaN IN PREPARED FEATURES")
print(f"   Data shape after preparation: {df_prepared.shape}")

# Show all NaN locations
nan_features = {}
for col in wrapper.transformer_features:
    nan_mask = df_prepared[col].isna()
    if nan_mask.any():
        nan_features[col] = (nan_mask.sum(), df_prepared[nan_mask][['Year', col]].to_dict('records'))

if nan_features:
    print(f"\n   ⚠️  FOUND NaN VALUES IN:")
    for col, (count, examples) in nan_features.items():
        print(f"     - {col}: {count} NaN values")
        for ex in examples[:2]:
            print(f"       Year {ex['Year']}: {ex[col]}")
else:
    print(f"   ✓ No NaN in transformer features")

# Step 2: Check scaler and scaling
print(f"\n3. SCALING STEP")
p_history_tail = df_prepared.tail(wrapper.max_seq_len)
print(f"   Tail shape for scaling: {p_history_tail.shape}")

try:
    history_vals = wrapper.scaler.transform(p_history_tail[wrapper.transformer_features])
    print(f"   Scaled shape: {history_vals.shape}")
    print(f"   Any NaN after scaling: {np.isnan(history_vals).any()}")
    if np.isnan(history_vals).any():
        nan_positions = np.where(np.isnan(history_vals))
        print(f"   NaN positions: {nan_positions}")
        for i in range(min(3, len(nan_positions[0]))):
            r, c = nan_positions[0][i], nan_positions[1][i]
            col_name = wrapper.transformer_features[c]
            print(f"   - Row {r}, Col {col_name}: {history_vals[r, c]}")
except Exception as e:
    print(f"   ✗ Error during scaling: {e}")

# Step 3: Tensor conversion
print(f"\n4. TENSOR CONVERSION & MODEL INFERENCE")
try:
    actual_len = len(history_vals)
    pad = np.zeros((wrapper.max_seq_len - actual_len, len(wrapper.transformer_features)))
    padded_x = np.vstack([pad, history_vals])
    
    print(f"   Padded shape: {padded_x.shape}")
    print(f"   Any NaN in padded: {np.isnan(padded_x).any()}")
    
    x_tensor = torch.tensor(padded_x, dtype=torch.float32).unsqueeze(0)
    mask = [True] * (wrapper.max_seq_len - actual_len) + [False] * actual_len
    m_tensor = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
    
    print(f"   Tensor shape: {x_tensor.shape}")
    print(f"   Mask: {mask}")
    
    # Run model
    with torch.no_grad():
        transformer_grade = wrapper.model(x_tensor, mask=m_tensor).item()
    
    print(f"   Transformer output: {transformer_grade}")
    print(f"   Is NaN: {np.isnan(transformer_grade)}")
    
except Exception as e:
    print(f"   ✗ Error during inference: {e}")
    import traceback
    traceback.print_exc()

# Full prediction test
print(f"\n5. FULL PREDICTION")
try:
    tier, details = wrapper.predict(boone, mode="ensemble")
    print(f"   Tier: {tier}")
    print(f"   Predicted grade: {details.get('predicted_grade')}")
    print(f"   Transformer grade: {details.get('transformer_grade')}")
    print(f"   XGB grade: {details.get('xgb_grade')}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
