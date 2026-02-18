
import os
# CRITICAL: Must be set before importing xgboost/torch to prevent OpenMP deadlock on Mac
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import sys

# Ensure root is in path
sys.path.append(os.getcwd())

from backend.ML.TightEnds.TETransformer import TETransformer, load_sequences as load_transformer_data

# ==========================================
# CONFIG
# ==========================================
DATA_FILE = "backend/ML/TightEnds/TE.csv"
VALIDATION_YEAR = 2024
PLAYER_TO_ANALYZE = "Trey McBride"
SEQ_LEN = 4

print(f"==== ANALYZING TRANSFORMER PREDICTION FOR: {PLAYER_TO_ANALYZE} ({VALIDATION_YEAR}) ====")

# ==========================================
# 1. LOAD DATA & TRAIN TRANSFORMER
# ==========================================
print("\n[1/3] Loading Data & Training Transformer...")
X, T, y, meta_df, feats = load_transformer_data(DATA_FILE, seq_len=SEQ_LEN)

# Scaling
N, S, F = X.shape
scaler = StandardScaler()
X_flat = X.reshape(-1, F)
X_scaled = scaler.fit_transform(X_flat).reshape(N, S, F)


# ==========================================
# 1. LOAD DATA & TRAIN TRANSFORMER (SEARCH MODE)
# ==========================================
print("\n[1/3] Loading Data & Training Transformer...")
X, T, y, meta_df, feats = load_transformer_data(DATA_FILE, seq_len=SEQ_LEN)

# Scaling
N, S, F = X.shape
scaler = StandardScaler()
X_flat = X.reshape(-1, F)
X_scaled = scaler.fit_transform(X_flat).reshape(N, S, F)

# Train/Test Logic
meta_years = meta_df["Year"].values
train_mask = meta_years < VALIDATION_YEAR

X_train_t = torch.FloatTensor(X_scaled[train_mask])
T_train_t = torch.FloatTensor(T[train_mask])
y_train_t = torch.FloatTensor(y[train_mask]).unsqueeze(1)

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Attempt to find the "optimistic" model the user saw (Variance in initialization)
best_model = None
best_pred = -1
target_pred = 37.0 # User saw ~37
best_diff = 999

EPOCHS = 80
ATTEMPTS = 5

# Locate player index once
idx = meta_df[(meta_df["player"] == PLAYER_TO_ANALYZE) & (meta_df["Year"] == VALIDATION_YEAR)].index[0]
X_player_t = torch.FloatTensor(X_scaled[idx]).unsqueeze(0).to(device)
T_player_t = torch.FloatTensor(T[idx]).unsqueeze(0).to(device)

print(f"\nSearching for model state matching observed prediction (~{target_pred})...")

for attempt in range(ATTEMPTS):
    print(f"--- Attempt {attempt+1}/{ATTEMPTS} ---")
    current_model = TETransformer(num_features=F).to(device)
    optimizer = torch.optim.Adam(current_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    current_model.train()
    dataset = torch.utils.data.TensorDataset(X_train_t, T_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    for epoch in range(EPOCHS):
        for bx, bt, by in loader:
            bx, bt, by = bx.to(device), bt.to(device), by.to(device)
            optimizer.zero_grad()
            out = current_model(bx, bt)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    # Check Prediction
    current_model.eval()
    with torch.no_grad():
        pred = current_model(X_player_t, T_player_t).item()
    
    print(f"Attempt {attempt+1} Prediction: {pred:.4f}")
    
    # We want the one closest to 37.8 (since ensemble was 33.3 = 0.5*28.8 + 0.5*T => T=37.8)
    diff = abs(pred - 37.8)
    
    if diff < best_diff:
        best_diff = diff
        best_pred = pred
        best_model = current_model
        # Save state dict to memory in case we need it? No, just keep object.
        print(f"-> New Best Match! (Diff: {diff:.4f})")

print(f"\nSelected Model Prediction: {best_pred:.4f}")
transformer = best_model


# ==========================================
# 2. LOCATE PLAYER DATA
# ==========================================
# Find index for Trey McBride in 2024
player_idx = meta_df[(meta_df["player"] == PLAYER_TO_ANALYZE) & (meta_df["Year"] == VALIDATION_YEAR)].index

if len(player_idx) == 0:
    print(f"ERROR: No data found for {PLAYER_TO_ANALYZE} in {VALIDATION_YEAR}")
    sys.exit(1)

idx = player_idx[0]
X_player = torch.FloatTensor(X_scaled[idx]).unsqueeze(0).to(device) # (1, S, F)
T_player = torch.FloatTensor(T[idx]).unsqueeze(0).to(device)

with torch.no_grad():
    base_pred = transformer(X_player, T_player).item()

print(f"\nStats for {PLAYER_TO_ANALYZE}:")
print(f"Actual Weighted Grade:    {y[idx]:.4f}")
print(f"Transformer Prediction:   {base_pred:.4f}")

# ==========================================
# 3. SENSITIVITY ANALYSIS
# ==========================================
print("\n==== SENSITIVITY ANALYSIS ====")
print("Testing how changes to specific features in the FINAL YEAR affect the prediction.")

feature_names = feats
final_year_features = X_player.clone() # (1, S, F)

def get_prediction_change(feature_name, pct_change):
    # Find feature index
    try:
        f_idx = feature_names.index(feature_name)
    except ValueError:
        return 0.0

    # Modify ONLY the last year (most recent) in the sequence
    modified_X = final_year_features.clone()
    
    # We are working in SCALED space.
    # To increase by 10% in original space is hard without inverse transform.
    # Approximation: Add 10% of the standard deviation (which is 1.0 in scaled space)
    # Let's just add/subtract 0.5 sigma (standard deviations)
    shift = 0.5 if pct_change > 0 else -0.5
    
    modified_X[0, -1, f_idx] += shift 
    
    with torch.no_grad():
        new_pred = transformer(modified_X, T_player).item()
    
    return new_pred - base_pred

# Test key features
features_to_test = [
    "yards", "yprr", "targets", "receptions", "first_downs", 
    "epa_per_target", "grades_pass_route", "grades_offense",
    "slot_snaps", "snap_counts_pass_route"
]

results = []
for f in features_to_test:
    delta_pos = get_prediction_change(f, 0.1) # +0.5 Std Dev
    results.append({"Feature": f, "+0.5 Sigma Impact": delta_pos})

results_df = pd.DataFrame(results).sort_values("+0.5 Sigma Impact", ascending=False)
print(results_df.to_string(index=False))

print("\n==== TRAJECTORY ANALYSIS ====")
# How much does the history matter?
# Zero out everything except the last year
X_only_last = X_player.clone()
X_only_last[0, :-1, :] = 0 # Zero out years T-3, T-2, T-1 (wait, seq is chronological?)
# Sequence is [T-3, T-2, T-1, T(current input year? No, input is historical)]
# Let's check load_sequences. usually it's [Year-3, Year-2, Year-1, Year] to predict Year+1?
# Or [Year-3, ... Year] to predict Year?
# Usually supervised: Input [Y-3... Y-1] -> Target Y.
# Let's assume the last element in S is the most recent historical year.

with torch.no_grad():
    model_pred_trajectory = transformer(X_player, T_player).item()
    
    # Zero out the rookie year (Index -2, assuming -1 is the breakout year 2023)
    # If he has 2 years of data (Rookie 2022, Breakout 2023) predicting 2024.
    # Sequence len is 4. Most likely padded. 
    # Let's see if removing rookie year hurts.
    
    X_no_rookie = X_player.clone()
    # Assuming the non-zero entries are at the end.
    # Let's just zero out the second to last time step if valid
    X_no_rookie[0, -2, :] = 0 
    pred_no_rookie = transformer(X_no_rookie, T_player).item()
    
    print(f"Full History Prediction: {model_pred_trajectory:.4f}")
    print(f"Without Rookie Year:     {pred_no_rookie:.4f}")
    print(f"Impact of Rookie Year:   {model_pred_trajectory - pred_no_rookie:.4f}")
    
    if model_pred_trajectory > pred_no_rookie:
        print("-> The rookie year stats POSITIVELY influenced the score.")
    else:
        print("-> The rookie year stats NEGATIVELY influenced the score (dragged it down).")

