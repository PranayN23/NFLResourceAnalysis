
import os
# CRITICAL: Must be set before importing xgboost/torch to prevent OpenMP deadlock on Mac
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append(os.getcwd()) # Ensure root is in path

from backend.ML.TightEnds.TETime2Vec import load_and_engineer_features as load_xgb_data
from backend.ML.TightEnds.TETransformer import TETransformer, load_sequences as load_transformer_data

# ==========================================
# 1. SETUP & CONFIG
# ==========================================
DATA_FILE = "backend/ML/TightEnds/TE.csv"

# ==========================================
# 0. MODE SELECTION
# ==========================================
# MODE = "VALIDATION"  # Predict 2024 (Train < 2024) and compare with Actuals
MODE = "DREAM"       # Predict 2025 (Train <= 2024)

print(f"==== STARTING ENSEMBLE MODELING (Mode: {MODE}) ====")

# ==========================================
# 2. RUN XGBOOST MODEL (Weight: 60%)
# ==========================================
print("\n[1/3] Training XGBoost (Hybrid Time2Vec)...")

# Load Data
data, full_df = load_xgb_data(DATA_FILE)

# Features
predictors = [c for c in data.columns if "_prev" in c or "_trend" in c or "_rolling" in c]
predictors += [
    "age","age_sq","Cap_Space",
    "time_linear","time_sin_1","time_cos_1",
    "time_sin_2","time_cos_2",
    "time_sin_3","time_cos_3",
    "career_year","career_year_sq",
    "is_prime", "is_decline",
    "Growth_Potential", "Team_TE_EPA_History"
]
predictors = [c for c in predictors if c in data.columns]
target = "weighted_grade"
target_col = target

# TRAIN/TEST SPLIT BASED ON MODE
if MODE == "VALIDATION":
    train_data = data[data["Year"] < 2024]
    test_data = data[data["Year"] == 2024]
    print(f"Validation Mode: Training on {len(train_data)} samples (<2024), Validating on {len(test_data)} samples (2024)")
else:
    train_data = data[data["Year"] <= 2024]
    test_data = None # No test set for Dream mode
    print(f"Dream Mode: Training on {len(train_data)} samples (All Data)")

X_train = train_data[predictors]
y_train = train_data[target]

xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.7,
    colsample_bytree=0.7,
    n_jobs=-1,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# --- Predict (XGB) ---
if MODE == "VALIDATION":
    print("Generating XGBoost 2024 Validation Predictions...")
    X_val = test_data[predictors]
    # Prediction DataFrame
    pred_df_xgb = test_data[["player_id", "player", "Team", "Year", target, "weighted_grade_prev"]].copy()
    pred_df_xgb["Pred_XGB"] = xgb_model.predict(X_val)
    
else:
    print("Generating XGBoost 2025 Predictions...")
    players_2024 = full_df[full_df["Year"] == 2024]["player_id"].unique()
    pred_rows = []

    for pid in players_2024:
        p_data = full_df[full_df["player_id"] == pid].sort_values("Year")
        row_2024 = p_data[p_data["Year"] == 2024].iloc[-1]
        row_2023 = p_data[p_data["Year"] == 2023].iloc[-1] if not p_data[p_data["Year"] == 2023].empty else None

        # Construct Feature Vector 2025
        feat_dict = {"player_id": pid, "player": row_2024["player"], "Team": row_2024["Team"]}
        
        # Static & Time
        feat_dict["age"] = row_2024["age"] + 1
        feat_dict["age_sq"] = (row_2024["age"] + 1) ** 2
        feat_dict["Cap_Space"] = row_2024["Cap_Space"]
        t_future = 2025
        feat_dict["time_linear"] = t_future
        feat_dict["time_sin_1"] = np.sin(2 * np.pi * t_future / 2)
        feat_dict["time_cos_1"] = np.cos(2 * np.pi * t_future / 2)
        feat_dict["time_sin_2"] = np.sin(2 * np.pi * t_future / 4)
        feat_dict["time_cos_2"] = np.cos(2 * np.pi * t_future / 4)
        feat_dict["time_sin_3"] = np.sin(2 * np.pi * t_future / 8)
        feat_dict["time_cos_3"] = np.cos(2 * np.pi * t_future / 8)
        feat_dict["career_year"] = row_2024["career_year"] + 1
        feat_dict["career_year_sq"] = (feat_dict["career_year"]) ** 2
        
        # Age Features
        age_next = feat_dict["age"]
        feat_dict["is_prime"] = 1.0 if 23 <= age_next <= 26 else 0.0
        feat_dict["is_decline"] = 1.0 if age_next >= 30 else 0.0
        
        # Growth Potential (Rookie Scaling)
        cy = feat_dict["career_year"]
        if cy == 1: feat_dict["Growth_Potential"] = 1.3726
        elif cy == 2: feat_dict["Growth_Potential"] = 1.1393
        elif cy == 3: feat_dict["Growth_Potential"] = 0.9838
        else: feat_dict["Growth_Potential"] = 1.0
        
        # Team TE EPA History (2022-2024 Average for 2025 Prediction)
        team_history_df = full_df[full_df["Team"] == feat_dict["Team"]]
        epa_history = team_history_df[team_history_df["Year"].isin([2022, 2023, 2024])]["Net EPA"].mean()
        feat_dict["Team_TE_EPA_History"] = epa_history if not pd.isna(epa_history) else 0.0

        # Lags
        lag_cols_base = [
            "weighted_grade", "Net EPA", "grades_offense", "yards", "touchdowns", 
            "first_downs", "yprr", "receptions", "targets",
            "grades_pass_route", "wide_snaps", "slot_snaps",
            "epa_per_target", "targets_per_snap"
        ]

        # Lags
        lag_cols_base = [
            "weighted_grade", "Net EPA", "grades_offense", "yards", "touchdowns", 
            "first_downs", "yprr", "receptions", "targets",
            "grades_pass_route", "wide_snaps", "slot_snaps",
            "epa_per_target", "targets_per_snap"
        ]

        for col in lag_cols_base:
            v1 = row_2024[col] if col in row_2024 else np.nan
            v2 = row_2023[col] if row_2023 is not None and col in row_2023 else np.nan
            
            # ROOKIE/MISSING YEAR FIX: Treat missing prior year as 0 for Trend, but don't average it for Rolling
            if pd.notnull(v1) and pd.isna(v2):
                v2_val = 0.0
                feat_dict[f"{col}_prev"] = v1
                feat_dict[f"{col}_prev2"] = 0.0 # Impute 0
                feat_dict[f"{col}_trend"] = v1 - v2_val # Trend is the full value (0 -> v1)
                feat_dict[f"{col}_rolling2"] = v1 # Keep rolling as just the current year (don't average with 0)
            else:
                feat_dict[f"{col}_prev"] = v1
                feat_dict[f"{col}_prev2"] = v2
                feat_dict[f"{col}_trend"] = v1 - v2 if pd.notnull(v1) and pd.notnull(v2) else np.nan
                feat_dict[f"{col}_rolling2"] = (v1 + v2)/2 if pd.notnull(v1) and pd.notnull(v2) else v1

        pred_rows.append(feat_dict)

    pred_df_xgb = pd.DataFrame(pred_rows)
    X_pred_2025 = pred_df_xgb[predictors]
    pred_df_xgb["Pred_XGB"] = xgb_model.predict(X_pred_2025)

# ==========================================
# 3. RUN TRANSFORMER MODEL (Weight: 40%)
# ==========================================
print("\n[2/3] Training Transformer...")

SEQ_LEN = 4
X, T, y, meta_df, feats = load_transformer_data(DATA_FILE, seq_len=SEQ_LEN)

# Scaling
N, S, F = X.shape
scaler = StandardScaler()
X_flat = X.reshape(-1, F)
X_scaled = scaler.fit_transform(X_flat).reshape(N, S, F)

# Train split logic
meta_year = meta_df["Year"].values

if MODE == "VALIDATION":
    train_mask = meta_year < 2024
    test_mask = meta_year == 2024
else:
    train_mask = meta_year <= 2024 # All available labels
    test_mask = None

X_train_t = torch.FloatTensor(X_scaled[train_mask])
T_train_t = torch.FloatTensor(T[train_mask])
y_train_t = torch.FloatTensor(y[train_mask]).unsqueeze(1)

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# SET FIXED SEED FOR AGGRESSIVE BEHAVIOR (Seed 12 found via TE_Search_Seed.py)
torch.manual_seed(12)
np.random.seed(12)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(12)

transformer = TETransformer(num_features=F).to(device)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Train Loop
transformer.train()
EPOCHS = 60
dataset = torch.utils.data.TensorDataset(X_train_t, T_train_t, y_train_t)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

print("Training Transformer...")
for epoch in range(EPOCHS):
    epoch_loss = 0
    for bx, bt, by in loader:
        bx, bt, by = bx.to(device), bt.to(device), by.to(device)
        optimizer.zero_grad()
        out = transformer(bx, bt)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {epoch_loss/len(loader):.4f}")

# --- Predict (Transformer) ---
print(f"Generating Transformer Predictions (Mode: {MODE})...")
transformer.eval()
transformer_rows = []

if MODE == "VALIDATION":
    # Just run on the held-out test set
    X_val_t = torch.FloatTensor(X_scaled[test_mask]).to(device)
    T_val_t = torch.FloatTensor(T[test_mask]).to(device)
    
    with torch.no_grad():
        val_preds = transformer(X_val_t, T_val_t).cpu().flatten().numpy()
        
    # Get player IDs for merge. meta_df has "player" and "Team" but not ID?
    # load_sequences in TETransformer needs to return player_id if we want proper merge.
    # Hack: We will assume order is preserved (it is) and meta_df corresponds to X/y rows.
    
    # Actually, meta_df stores player/Team. Let's rely on that for merge keys?
    # Or better, add player_id to meta_df in TETransformer.py.
    # For now, let's look at what we have.
    val_meta = meta_df[test_mask].reset_index(drop=True)
    val_meta["Pred_Transformer"] = val_preds
    pred_df_trans = val_meta # Has player, Team, Pred_Transformer

else:
    # 2025 Prediction Construction (Same as before)
    player_groups = full_df.groupby("player_id")
    # ... (Logic to construct 2025 input sequence)
    # Re-using previous logic block for 2025
    
    for pid in players_2024: # We calculate for all active 
        # ... (Same logic as previous step)
        if pid not in player_groups.groups: continue
        group = player_groups.get_group(pid).sort_values("Year")
        last_years = group[group["Year"] <= 2024].tail(SEQ_LEN)
        
        if 2024 not in last_years["Year"].values: continue

        # Manual Feature Engineering (Must match TETransformer exactly)
        local_df = last_years.copy()
        local_df["weighted_grade"] = local_df["grades_offense"].fillna(0) * local_df["total_snaps"].fillna(0) / 1000.0
        local_df["efficiency"] = local_df["grades_offense"] / 100.0
        local_df["volume"] = local_df["total_snaps"] / 1000.0
        local_df["yprr_trend"] = local_df["yprr"].diff().fillna(0)
        local_df["efficiency_per_snap"] = local_df["weighted_grade"] / local_df["total_snaps"].replace(0, np.nan)
        local_df["efficiency_per_snap"] = local_df["efficiency_per_snap"].fillna(0)
        local_df["is_prime"] = ((local_df["age"] >= 23) & (local_df["age"] <= 26)).astype(float)
        local_df["is_decline"] = (local_df["age"] >= 30).astype(float)
        local_df = local_df.fillna(0)
        
        raw_seq = local_df[feats].values
        if len(raw_seq) < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - len(raw_seq), len(feats)))
            raw_seq = np.vstack([pad, raw_seq])
            
        scaled_seq = scaler.transform(raw_seq) 
        
        min_year_global = full_df["Year"].min()
        times = (local_df["Year"].values - min_year_global).reshape(-1, 1).astype(float)
        
        if len(times) < SEQ_LEN:
             pad_t = np.full((SEQ_LEN - len(times), 1), -1.0)
             times = np.vstack([pad_t, times])
             
        with torch.no_grad():
            t_in = torch.FloatTensor(scaled_seq).unsqueeze(0).to(device)
            time_in = torch.FloatTensor(times).unsqueeze(0).to(device)
            pred = transformer(t_in, time_in).cpu().item()
            
        transformer_rows.append({"player_id": pid, "player": row_2024["player"], "Pred_Transformer": pred})
        
    pred_df_trans = pd.DataFrame(transformer_rows)

# ==========================================
# 4. MERGE & ENSEMBLE
# ==========================================
print("\n[3/3] Ensembling...")

# Merge logic depends on cols available
if MODE == "VALIDATION":
    # Merge on Player + Team since we might lack ID in meta_df
    final_df = pd.merge(pred_df_xgb, pred_df_trans[["player", "Team", "Pred_Transformer"]], on=["player", "Team"], how="left")
else:
    final_df = pd.merge(pred_df_xgb, pred_df_trans[["player_id", "Pred_Transformer"]], on="player_id", how="left")

# Fill NaNs
final_df["Pred_Transformer"] = final_df["Pred_Transformer"].fillna(final_df["Pred_XGB"])

# WEIGHTS (Optimized: 0.75 XGB / 0.25 Transformer)
W_XGB = 0.75
W_TRANS = 0.25

# Calculate Ensemble
if MODE == "VALIDATION":
    final_df["Ensemble_Pred"] = (final_df["Pred_XGB"] * W_XGB) + (final_df["Pred_Transformer"] * W_TRANS)
    final_df["Predicted_Jump"] = final_df["Ensemble_Pred"] - final_df["weighted_grade_prev"]
    final_df["Error"] = final_df[target] - final_df["Ensemble_Pred"]
    final_df["Abs_Error"] = final_df["Error"].abs()
    
    # Calculate Metrics
    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(final_df[target], final_df["Ensemble_Pred"])
    r2 = r2_score(final_df[target], final_df["Ensemble_Pred"])
    
    print(f"\n==== ENSEMBLE VALIDATION RESULTS (2024) ====")
    print(f"R-Squared: {r2:.4f}")
    print(f"MAE:       {mae:.4f}")
    
    cols = ["player", "Team", "weighted_grade_prev", target, "Pred_XGB", "Pred_Transformer", "Ensemble_Pred", "Predicted_Jump", "Error"]
    print("\n==== TOP PREDICTIONS (Ensemble) ====")
    print(final_df.sort_values("Ensemble_Pred", ascending=False)[cols].head(15).to_string(index=False))
    
    print("\n==== BIGGEST MISSES ====")
    print(final_df.sort_values("Abs_Error", ascending=False)[cols].head(10).to_string(index=False))

else:
    final_df["TE_Value_Score_2025"] = (final_df["Pred_XGB"] * W_XGB) + (final_df["Pred_Transformer"] * W_TRANS)
    final_df["Predicted_Jump"] = final_df["TE_Value_Score_2025"] - final_df["weighted_grade_prev"]
    final_df = final_df.sort_values("TE_Value_Score_2025", ascending=False)
    
    cols = ["player", "Team", "weighted_grade_prev", "Pred_XGB", "Pred_Transformer", "TE_Value_Score_2025", "Predicted_Jump"]
    print("\n==== FINAL 2025 CONSENSUS RANKINGS ====")
    print(final_df[cols].head(25).to_string(index=False))
    
    final_df[cols].to_csv("backend/ML/TightEnds/TE_2025_Final_Rankings.csv", index=False)
    print("\nSaved Final Rankings to backend/ML/TightEnds/TE_2025_Final_Rankings.csv")
