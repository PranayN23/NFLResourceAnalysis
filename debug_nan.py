import pandas as pd
import numpy as np
from backend.agent.rb_model_wrapper import RBModelInference

# Load raw data
rb_df = pd.read_csv("backend/ML/HB.csv")
boone = rb_df[rb_df['player'].astype(str).str.lower() == 'mike boone']

print("Mike Boone raw data:")
print(boone[['player', 'Year', 'grades_offense', 'yards', 'total_touches']].tail(3).to_string())

# Try to prepare features
wrapper = RBModelInference(
    "backend/ML/RB_Pranay_Transformers/rb_best_classifier.pth",
    "backend/ML/RB_Pranay_Transformers/rb_player_scaler.joblib"
)

df_history, df_xgb = wrapper._prepare_features(boone)
print("\n\nPrepared features (last row):")
print(df_history[wrapper.transformer_features].tail(1).to_string())

# Check for NaN in transformer features
print("\n\nChecking for NaN in transformer features:")
for col in wrapper.transformer_features:
    last_val = df_history[col].iloc[-1]
    is_nan = pd.isna(last_val)
    print(f"  {col}: {last_val} (NaN: {is_nan})")

# Check scaled version
p_history_tail = df_history.tail(wrapper.max_seq_len)
history_vals = wrapper.scaler.transform(p_history_tail[wrapper.transformer_features])
print(f"\nScaled history shape: {history_vals.shape}")
print(f"Scaled history (last row): {history_vals[-1]}")
print(f"Any NaN in scaled: {np.isnan(history_vals).any()}")
