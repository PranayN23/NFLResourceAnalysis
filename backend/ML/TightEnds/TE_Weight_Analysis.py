
import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Prevent OpenMP deadlock
os.environ["OMP_NUM_THREADS"] = "1"

# Ensure root is in path
sys.path.append(os.getcwd())

from backend.ML.TightEnds.TETime2Vec import load_and_engineer_features as load_xgb_data
from backend.ML.TightEnds.TETransformer import TETransformer, load_sequences as load_transformer_data

DATA_FILE = "backend/ML/TightEnds/TE.csv"

def run_weight_analysis(validation_year):
    print(f"\n\n==================================================")
    print(f"       RUNNING WEIGHT ANALYSIS FOR YEAR: {validation_year}")
    print(f"==================================================")
    
    # ==========================
    # 1. XGBOOST
    # ==========================
    print("Training XGBoost...")
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
    
    train_xgb = data[data["Year"] < validation_year]
    test_xgb = data[data["Year"] == validation_year]
    
    X_train_xgb = train_xgb[predictors]
    y_train_xgb = train_xgb["weighted_grade"]
    X_test_xgb = test_xgb[predictors]
    y_test_xgb = test_xgb["weighted_grade"]
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.01, max_depth=4,
        subsample=0.7, colsample_bytree=0.7, n_jobs=1, random_state=42
    )
    xgb_model.fit(X_train_xgb, y_train_xgb)
    xgb_preds = xgb_model.predict(X_test_xgb)
    
    # ==========================
    # 2. TRANSFORMER
    # ==========================
    print("Training Transformer...")
    # Seed for aggressive behavior
    torch.manual_seed(12)
    np.random.seed(12)
    
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    
    # Correct Unpacking: 5 values
    X_raw, T_raw, y_raw, meta_df, feats = load_transformer_data(DATA_FILE)
    
    # Scale Features (StandardScaler)
    N, S, F = X_raw.shape
    scaler = StandardScaler()
    X_flat = X_raw.reshape(-1, F)
    X_scaled = scaler.fit_transform(X_flat).reshape(N, S, F)
    
    # Split using metadata
    train_mask = meta_df["Year"] < validation_year
    test_mask = meta_df["Year"] == validation_year
    
    # Create Tensors
    X_train_t = torch.FloatTensor(X_scaled[train_mask]).to(device)
    T_train_t = torch.FloatTensor(T_raw[train_mask]).to(device)
    y_train_t = torch.FloatTensor(y_raw[train_mask]).unsqueeze(1).to(device)
    
    X_test_t = torch.FloatTensor(X_scaled[test_mask]).to(device)
    T_test_t = torch.FloatTensor(T_raw[test_mask]).to(device)
    # y_test_t not needed for prediction, but good for validation check
    
    # Model Init
    transformer = TETransformer(num_features=F, time_emb_dim=16, d_model=64, nhead=4, num_layers=3).to(device)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Train
    transformer.train()
    EPOCHS = 60
    batch_size = 32
    
    dataset = torch.utils.data.TensorDataset(X_train_t, T_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(EPOCHS):
        for bx, bt, by in loader:
            optimizer.zero_grad()
            out = transformer(bx, bt)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    # Predict Transformer
    transformer.eval()
    with torch.no_grad():
        trans_preds = transformer(X_test_t, T_test_t).cpu().numpy().flatten()
        
    # ==========================
    # 3. ENSEMBLE
    # ==========================
    # Create DF for Transformer results from the test metadata
    trans_results = meta_df[test_mask].copy()
    trans_results["Pred_Trans"] = trans_preds
    
    # Merge with XGB Results
    xgb_results = test_xgb.copy()
    xgb_results["Pred_XGB"] = xgb_preds
    
    # Merge on player_id and Year
    ensemble_df = pd.merge(xgb_results, trans_results[["player_id", "Year", "Pred_Trans"]], on=["player_id", "Year"], how="inner")
    
    actuals = ensemble_df["weighted_grade"].values
    pred_xgb = ensemble_df["Pred_XGB"].values
    pred_trans = ensemble_df["Pred_Trans"].values
    
    # Analysis Configuration
    weights = [
        (1.0, 0.0), # Pure XGB
        (0.75, 0.25),
        (0.50, 0.50), # Current
        (0.25, 0.75),
        (0.0, 1.0)  # Pure Transformer
    ]
    
    print(f"\nRESULTS FOR {validation_year}:")
    print(f"{'Weight (XGB/Trans)':<20} | {'R2':<8} | {'MAE':<8} | {'RMSE':<8}")
    print("-" * 60)
    
    best_r2 = -999
    best_w = None
    
    for w_xgb, w_trans in weights:
        pred_combined = (pred_xgb * w_xgb) + (pred_trans * w_trans)
        
        r2 = r2_score(actuals, pred_combined)
        mae = mean_absolute_error(actuals, pred_combined)
        rmse = np.sqrt(mean_squared_error(actuals, pred_combined))
        
        print(f"{w_xgb:.2f} / {w_trans:.2f}           | {r2:.4f}   | {mae:.4f}   | {rmse:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_w = (w_xgb, w_trans)
            
    print(f"\nBest Configuration for {validation_year}: XGB={best_w[0]}, Trans={best_w[1]} (R2={best_r2:.4f})")

if __name__ == "__main__":
    # Test 2024
    run_weight_analysis(2024)
    # Test 2023
    run_weight_analysis(2023)
