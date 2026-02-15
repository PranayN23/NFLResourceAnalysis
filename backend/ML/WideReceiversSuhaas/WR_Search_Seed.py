
import os
# CRITICAL: Must be set before importing xgboost/torch to prevent OpenMP deadlock on Mac
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import sys
import random

# Ensure root is in path
sys.path.append(os.getcwd())

from backend.ML.WideReceivers.WRTransformer import WRTransformer, load_sequences as load_transformer_data

# ==========================================
# CONFIG
# ==========================================
DATA_FILE = "backend/ML/WR.csv"
VALIDATION_YEAR = 2024
PLAYER_TO_ANALYZE = "Tyreek Hill" # Let's pick a high variance elite WR to maximize upside
SEQ_LEN = 4
SEEDS_TO_TEST = 50

print(f"==== SEARCHING FOR AGGRESSIVE SEED FOR: {PLAYER_TO_ANALYZE} ====")

# ==========================================
# LOAD DATA ONCE
# ==========================================
print("Loading Data...")
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

# Locate player index
idx_list = meta_df[(meta_df["player"] == PLAYER_TO_ANALYZE) & (meta_df["Year"] == VALIDATION_YEAR)].index
if len(idx_list) == 0:
    print("Player not found in validation year.")
    sys.exit(1)
idx = idx_list[0]
X_player_t = torch.FloatTensor(X_scaled[idx]).unsqueeze(0).to(device)
T_player_t = torch.FloatTensor(T[idx]).unsqueeze(0).to(device)

results = []
EPOCHS = 60 

print(f"\nTesting {SEEDS_TO_TEST} seeds...")

for seed in range(SEEDS_TO_TEST):
    # SET SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Init Model
    model = WRTransformer(num_features=F).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Train
    model.train()
    dataset = torch.utils.data.TensorDataset(X_train_t, T_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    for epoch in range(EPOCHS):
        for bx, bt, by in loader:
            bx, bt, by = bx.to(device), bt.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx, bt)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    # Predict
    model.eval()
    with torch.no_grad():
        pred = model(X_player_t, T_player_t).item()
        
    print(f"Seed {seed}: Prediction = {pred:.4f}")
    results.append({"Seed": seed, "Prediction": pred})

# Sort by Prediction Descending (Most Aggressive/Optimistic first)
results_df = pd.DataFrame(results).sort_values("Prediction", ascending=False)

print("\n==== TOP 10 AGGRESSIVE SEEDS ====")
print(results_df.head(10).to_string(index=False))

best_seed = results_df.iloc[0]["Seed"]
best_pred = results_df.iloc[0]["Prediction"]
print(f"\nRECOMMENDATION: Use Seed {best_seed} (Pred: {best_pred:.4f})")
