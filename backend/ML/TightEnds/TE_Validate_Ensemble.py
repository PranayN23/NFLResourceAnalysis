
import os
import sys
# CRITICAL: Must be set before importing xgboost/torch to prevent OpenMP deadlock on Mac
os.environ["OMP_NUM_THREADS"] = "1"
import warnings

# Suppress Warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import pandas as pd
import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Ensure root is in path
sys.path.append(os.getcwd())

from backend.ML.TightEnds.TETime2Vec import load_and_engineer_features as load_xgb_data
from backend.ML.TightEnds.TETransformer import TETransformer, load_sequences as load_transformer_data

DATA_FILE = "backend/ML/TightEnds/TE.csv"
TARGET = "weighted_grade"

def validate_year(validation_year, verbose=False):
    """
    Validates the ensemble model for a specific year using STRICT walk-forward validation.
    Models are trained ONLY on data from years PRIOR to validation_year.
    """
    if verbose:
        print(f"==== STARTING ENSEMBLE VALIDATION (Predicting {validation_year}) ====")

    # ==========================================
    # 1. XGBOOST VALIDATION (75%)
    # ==========================================
    if verbose: print("\n[1/3] Training XGBoost...")
    
    # Load Data (Standard Engineering)
    data, _ = load_xgb_data(DATA_FILE)
    
    # Features
    predictors = [c for c in data.columns if "_prev" in c or "_trend" in c or "_rolling" in c]
    predictors += [
        "age", "age_sq", "Cap_Space",
        "time_linear", "time_sin_1", "time_cos_1", 
        "time_sin_2", "time_cos_2", "time_sin_3", "time_cos_3",
        "career_year", "career_year_sq",
        "is_prime", "is_decline",
        "Growth_Potential", "Team_TE_EPA_History"
    ]
    predictors = [c for c in predictors if c in data.columns]
    
    # STRICT SPLIT: Train < Year, Test == Year
    train_data = data[data["Year"] < validation_year]
    test_data = data[data["Year"] == validation_year]
    
    # Check sample size
    if len(train_data) == 0 or len(test_data) == 0:
        if verbose: print(f"Skipping {validation_year}: Insufficient data.")
        return None

    train_start = train_data["Year"].min()
    train_end = train_data["Year"].max()
    train_n = len(train_data)

    if verbose: print(f"XGB Train Samples: {len(train_data)} ({train_start}-{train_end}) | Test Samples: {len(test_data)}")
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.6,
        colsample_bytree=0.8,
        min_child_weight=1,
        n_jobs=1, # Fix for deadlock
        random_state=42,
        verbosity=0 # Silent
    )
    xgb_model.fit(train_data[predictors], train_data[TARGET])
    
    # Predict
    preds_xgb = xgb_model.predict(test_data[predictors])
    df_xgb = test_data[["player", "Team", "player_id", TARGET, "weighted_grade_prev"]].copy()
    df_xgb["Pred_XGB"] = preds_xgb
    
    # ==========================================
    # 2. TRANSFORMER VALIDATION (25%)
    # ==========================================
    if verbose: print("\n[2/3] Training Transformer...")
    
    SEQ_LEN = 4
    X, T, y, meta_df, feats = load_transformer_data(DATA_FILE, seq_len=SEQ_LEN)
    
    # Scaling
    N, S, F = X.shape
    scaler = StandardScaler()
    X_flat = X.reshape(-1, F)
    X_scaled = scaler.fit_transform(X_flat).reshape(N, S, F)
    
    # STRICT SPLIT
    meta_years = meta_df["Year"].values
    train_mask = meta_years < validation_year
    test_mask = meta_years == validation_year
    
    X_train_t = torch.FloatTensor(X_scaled[train_mask])
    T_train_t = torch.FloatTensor(T[train_mask])
    y_train_t = torch.FloatTensor(y[train_mask]).unsqueeze(1)
    
    X_test_t = torch.FloatTensor(X_scaled[test_mask])
    T_test_t = torch.FloatTensor(T[test_mask])
    
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    
    # Seed
    torch.manual_seed(12)
    np.random.seed(12)
    
    transformer = TETransformer(num_features=F).to(device)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Train
    transformer.train()
    EPOCHS = 60
    dataset = torch.utils.data.TensorDataset(X_train_t, T_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    for epoch in range(EPOCHS):
        for bx, bt, by in loader:
            bx, bt, by = bx.to(device), bt.to(device), by.to(device)
            optimizer.zero_grad()
            out = transformer(bx, bt)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    # Predict
    transformer.eval()
    with torch.no_grad():
        X_test_t, T_test_t = X_test_t.to(device), T_test_t.to(device)
        preds_trans = transformer(X_test_t, T_test_t).cpu().flatten().numpy()
        
    # Get Metadata for Merge
    df_trans = meta_df.iloc[test_mask].reset_index(drop=True)
    df_trans["Pred_Transformer"] = preds_trans
    
    # ==========================================
    # 3. ENSEMBLE
    # ==========================================
    # Use 'inner' to ensure we only evaluate players present in both models
    final_df = pd.merge(df_xgb, df_trans[["player", "Team", "Pred_Transformer"]], on=["player", "Team"], how="inner")
    
    # Weights (Optimized: 0.75 / 0.25)
    W_XGB = 0.75
    W_TRANS = 0.25
    
    final_df["Ensemble_Pred"] = (final_df["Pred_XGB"] * W_XGB) + (final_df["Pred_Transformer"] * W_TRANS)
    
    # Metrics
    r2 = r2_score(final_df[TARGET], final_df["Ensemble_Pred"])
    mae = mean_absolute_error(final_df[TARGET], final_df["Ensemble_Pred"])
    rmse = np.sqrt(mean_squared_error(final_df[TARGET], final_df["Ensemble_Pred"]))
    
    return {
        "Year": validation_year,
        "Train_Range": f"{int(train_start)}-{int(train_end)}",
        "Train_N": train_n,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "Samples": len(final_df)
    }

def analyze_specific_year(target_year):
    """
    Generates a detailed prediction report for a specific year.
    Prints Top 20 predictions and saves the full list to CSV.
    """
    print(f"\n==== GENERATING DETAILED REPORT FOR {target_year} ====")
    
    # Re-use the validation logic but keep the dataframe
    # 1. XGBOOST
    data, _ = load_xgb_data(DATA_FILE)
    
    predictors = [c for c in data.columns if "_prev" in c or "_trend" in c or "_rolling" in c]
    predictors += [
        "age", "age_sq", "Cap_Space",
        "time_linear", "time_sin_1", "time_cos_1", 
        "time_sin_2", "time_cos_2", "time_sin_3", "time_cos_3",
        "career_year", "career_year_sq",
        "is_prime", "is_decline",
        "Growth_Potential", "Team_TE_EPA_History"
    ]
    predictors = [c for c in predictors if c in data.columns]
    
    train_data = data[data["Year"] < target_year]
    test_data = data[data["Year"] == target_year]
    
    if len(train_data) == 0 or len(test_data) == 0:
        print(f"Error: Insufficient data for {target_year}")
        return

    print(f"Training XGBoost on {len(train_data)} samples (<{target_year})...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.6, colsample_bytree=0.8, min_child_weight=1,
        n_jobs=1, random_state=42, verbosity=0
    )
    xgb_model.fit(train_data[predictors], train_data[TARGET])
    
    # Predict XGB
    preds_xgb = xgb_model.predict(test_data[predictors])
    df_xgb = test_data[["player", "Team", "player_id", TARGET, "weighted_grade_prev"]].copy()
    df_xgb["Pred_XGB"] = preds_xgb
    
    # 2. TRANSFORMER
    SEQ_LEN = 4
    X, T, y, meta_df, feats = load_transformer_data(DATA_FILE, seq_len=SEQ_LEN)
    
    N, S, F = X.shape
    scaler = StandardScaler()
    X_flat = X.reshape(-1, F)
    X_scaled = scaler.fit_transform(X_flat).reshape(N, S, F)
    
    meta_years = meta_df["Year"].values
    train_mask = meta_years < target_year
    test_mask = meta_years == target_year
    
    X_train_t = torch.FloatTensor(X_scaled[train_mask])
    T_train_t = torch.FloatTensor(T[train_mask])
    y_train_t = torch.FloatTensor(y[train_mask]).unsqueeze(1)
    
    X_test_t = torch.FloatTensor(X_scaled[test_mask])
    T_test_t = torch.FloatTensor(T[test_mask])
    
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    torch.manual_seed(12)
    np.random.seed(12)
    
    print("Training Transformer...")
    transformer = TETransformer(num_features=F).to(device)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    transformer.train()
    dataset = torch.utils.data.TensorDataset(X_train_t, T_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    for epoch in range(60):
        for bx, bt, by in loader:
            bx, bt, by = bx.to(device), bt.to(device), by.to(device)
            optimizer.zero_grad()
            out = transformer(bx, bt)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    transformer.eval()
    with torch.no_grad():
        X_test_t, T_test_t = X_test_t.to(device), T_test_t.to(device)
        preds_trans = transformer(X_test_t, T_test_t).cpu().flatten().numpy()
        
    df_trans = meta_df.iloc[test_mask].reset_index(drop=True)
    df_trans["Pred_Transformer"] = preds_trans
    
    # 3. ENSEMBLE
    final_df = pd.merge(df_xgb, df_trans[["player", "Team", "Pred_Transformer"]], on=["player", "Team"], how="inner")
    
    W_XGB = 0.75
    W_TRANS = 0.25
    final_df["Ensemble_Pred"] = (final_df["Pred_XGB"] * W_XGB) + (final_df["Pred_Transformer"] * W_TRANS)
    final_df["Error"] = final_df[TARGET] - final_df["Ensemble_Pred"]

    # Metrics
    r2 = r2_score(final_df[TARGET], final_df["Ensemble_Pred"])
    print(f"\nR-Squared for {target_year}: {r2:.4f}")
    
    # Output to Console
    cols = ["player", "Team", "weighted_grade_prev", TARGET, "Pred_XGB", "Pred_Transformer", "Ensemble_Pred", "Error"]
    final_df = final_df.sort_values("Ensemble_Pred", ascending=False)
    
    print(f"\n==== TOP 25 PREDICTIONS FOR {target_year} ====")
    print(final_df[cols].head(25).to_string(index=False))
    
    filename = f"backend/ML/TightEnds/TE_{target_year}_Detailed_Report.csv"
    final_df.to_csv(filename, index=False)
    print(f"\nSaved detailed report to {filename}")

if __name__ == "__main__":
    # print("\n==========================================================================")
    # print("   HISTORICAL BACKTEST (Strict Walk-Forward: Training on Past Data Only)")
    # print("==========================================================================")
    # print(f"{'Year':<6} | {'Trained On':<12} | {'Train N':<8} | {'Test N':<8} | {'R2':<8} | {'MAE':<8} | {'RMSE':<8}")
    # print("-" * 88)
    
    # # Test from 2018 to 2024
    # years_to_test = range(2008, 2025)
    
    # results = []
    
    # for year in years_to_test:
    #     # Suppress stdout/stderr inside the function only if needed, 
    #     # but verbose=False usually handles it.
    #     metrics = validate_year(year, verbose=False)
    #     if metrics:
    #         print(f"{year:<6} | {metrics['Train_Range']:<12} | {metrics['Train_N']:<8} | {metrics['Samples']:<8} | {metrics['R2']:.4f}   | {metrics['MAE']:.4f}   | {metrics['RMSE']:.4f}")
    #         results.append(metrics)
    #     else:
    #         print(f"{year:<6} | INSUFFICIENT DATA")
            
    # print("-" * 88)

    analyze_specific_year(2024)
