
import pandas as pd
import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import sys
import os

# Ensure root is in path
sys.path.append(os.getcwd())

from backend.ML.TightEnds.TETime2Vec import load_and_engineer_features as load_xgb_data
from backend.ML.TightEnds.TETransformer import TETransformer, load_sequences as load_transformer_data

# ==========================================
# VALIDATION CONFIG
# ==========================================
DATA_FILE = "backend/ML/TightEnds/TE.csv"
VALIDATION_YEAR = 2024
TARGET = "weighted_grade"

print(f"==== STARTING ENSEMBLE VALIDATION (Predicting {VALIDATION_YEAR}) ====")

# ==========================================
# 1. XGBOOST VALIDATION (50%)
# ==========================================
print("\n[1/3] Training XGBoost (Hybrid Time2Vec)...")
data, _ = load_xgb_data(DATA_FILE)

# Features
predictors = [c for c in data.columns if "_prev" in c or "_trend" in c or "_rolling" in c]
predictors += [
    "age", "age_sq", "Cap_Space",
    "time_linear", "time_sin_1", "time_cos_1", 
    "time_sin_2", "time_cos_2", "time_sin_3", "time_cos_3",
    "career_year", "career_year_sq",
    "is_prime", "is_decline"
]
predictors = [c for c in predictors if c in data.columns]

# Train: Year < 2024
train_data = data[data["Year"] < VALIDATION_YEAR]
test_data = data[data["Year"] == VALIDATION_YEAR]

print(f"XGB Train Samples: {len(train_data)} | Test Samples: {len(test_data)}")

xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.7,
    colsample_bytree=0.7,
    n_jobs=-1,
    random_state=42
)
xgb_model.fit(train_data[predictors], train_data[TARGET])

# Predict 2024
preds_xgb = xgb_model.predict(test_data[predictors])
df_xgb = test_data[["player", "Team", "player_id", TARGET, "weighted_grade_prev"]].copy()
df_xgb["Pred_XGB"] = preds_xgb

# ==========================================
# 2. TRANSFORMER VALIDATION (50%)
# ==========================================
print("\n[2/3] Training Transformer (Upside Model)...")
SEQ_LEN = 4
X, T, y, meta_df, feats = load_transformer_data(DATA_FILE, seq_len=SEQ_LEN)

# Scaling
N, S, F = X.shape
scaler = StandardScaler()
X_flat = X.reshape(-1, F)
X_scaled = scaler.fit_transform(X_flat).reshape(N, S, F)

# Train/Test Mask
meta_years = meta_df["Year"].values
train_mask = meta_years < VALIDATION_YEAR
test_mask = meta_years == VALIDATION_YEAR

X_train_t = torch.FloatTensor(X_scaled[train_mask])
T_train_t = torch.FloatTensor(T[train_mask])
y_train_t = torch.FloatTensor(y[train_mask]).unsqueeze(1)

X_test_t = torch.FloatTensor(X_scaled[test_mask])
T_test_t = torch.FloatTensor(T[test_mask])

print(f"Transformer Train Samples: {len(X_train_t)} | Test Samples: {len(X_test_t)}")

device = torch.device("cpu")
transformer = TETransformer(num_features=F).to(device)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Train
transformer.train()
EPOCHS = 80
dataset = torch.utils.data.TensorDataset(X_train_t, T_train_t, y_train_t)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(EPOCHS):
    for bx, bt, by in loader:
        optimizer.zero_grad()
        out = transformer(bx, bt)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()

# Predict 2024
transformer.eval()
with torch.no_grad():
    preds_trans = transformer(X_test_t, T_test_t).flatten().numpy()

# Get Metadata for Merge
df_trans = meta_df.iloc[test_mask].reset_index(drop=True)
df_trans["Pred_Transformer"] = preds_trans

# ==========================================
# 3. ENSEMBLE & REPORT
# ==========================================
print("\n[3/3] Ensembling & Validating...")

# Merge
final_df = pd.merge(df_xgb, df_trans[["player", "Team", "Pred_Transformer"]], on=["player", "Team"], how="inner")

# 50/50 Balance
W_XGB = 0.50
W_TRANS = 0.50

final_df["Ensemble_Pred"] = (final_df["Pred_XGB"] * W_XGB) + (final_df["Pred_Transformer"] * W_TRANS)
final_df["Predicted_Jump"] = final_df["Ensemble_Pred"] - final_df["weighted_grade_prev"]
final_df["Actual_Jump"] = final_df[TARGET] - final_df["weighted_grade_prev"]
final_df["Error"] = final_df[TARGET] - final_df["Ensemble_Pred"]
final_df["Abs_Error"] = final_df["Error"].abs()

# Metrics
r2 = r2_score(final_df[TARGET], final_df["Ensemble_Pred"])
mae = mean_absolute_error(final_df[TARGET], final_df["Ensemble_Pred"])
rmse = np.sqrt(mean_squared_error(final_df[TARGET], final_df["Ensemble_Pred"]))

print("\n==============================================")
print(f"   ENSEMBLE VALIDATION SCORECARD ({VALIDATION_YEAR})")
print("==============================================")
print(f"R-Squared (Accuracy): {r2:.4f}")
print(f"MAE (Avg Error):      {mae:.4f}")
print(f"RMSE:                 {rmse:.4f}")

# Show Predictions with Jump context
print("\n==== TOP 15 PREDICTIONS (Showing Predicted Breakouts) ====")
cols = ["player", "Team", "weighted_grade_prev", TARGET, "Ensemble_Pred", "Predicted_Jump", "Error"]
print(final_df.sort_values("Ensemble_Pred", ascending=False)[cols].head(15).to_string(index=False))

print("\n==== BIGGEST PREDICTED JUMPS (Model Predicts Breakout) ====")
print(final_df.sort_values("Predicted_Jump", ascending=False)[cols].head(10).to_string(index=False))

print("\n==== BIGGEST PREDICTION ERRORS (2024) ====")
print(final_df.sort_values("Abs_Error", ascending=False)[cols].head(10).to_string(index=False))

# Save
final_df.to_csv("backend/ML/TightEnds/TE_2024_Ensemble_Validation_Report.csv", index=False)
print("\nSaved detailed report to backend/ML/TightEnds/TE_2024_Ensemble_Validation_Report.csv")
