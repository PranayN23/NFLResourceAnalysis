
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

def load_and_engineer_features(filepath):
    df = pd.read_csv(filepath)
    df = df.replace("MISSING", np.nan)
    
    # Metadata columns to exclude from numeric conversion
    metadata_cols = ["player_id", "player", "Team", "position_x", "position", "franchise_id"]
    
    # Convert numeric
    for col in df.columns:
        if col not in metadata_cols and col != "Year":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    
    # Sort
    df = df.sort_values(["player_id", "Year"])
    
    # ==========================================
    # 1. BASE FEATURE ENGINEERING
    # ==========================================
    # Create Age Squared (Age curves are usually parabolic)
    df["age_sq"] = df["age"] ** 2
    
    # Efficiency Metrics (avoid division by zero)
    # Target Share Proxy (Targets / Total Snaps - crude but effective if team stats missing)
    df["targets_per_snap"] = df["targets"] / df["total_snaps"].replace(0, np.nan)
    df["epa_per_target"] = df["Net EPA"] / df["targets"].replace(0, np.nan)
    
    # ==========================================
    # 2. LAG FEATURES (History)
    # ==========================================
    # We want to use T-1 and T-2 to predict T
    # Or in our training setup: Use T to predict T+1.
    
    # Let's create explicit lag features for key metrics
    # Key Base Features to Lag
    lag_features = [
        "TE_Value_Score", "Net EPA", "grades_offense", "yards", "touchdowns", 
        "first_downs", "yprr", "receptions", "targets",
        "grades_pass_route", "wide_snaps", "slot_snaps",
        "epa_per_target", "targets_per_snap"
    ]
    
    for col in lag_features:
        if col in df.columns:
            # Shift 1: Previous Year
            df[f"{col}_prev"] = df.groupby("player_id")[col].shift(1)
            # Shift 2: 2 Years Ago (helps capture trends)
            df[f"{col}_prev2"] = df.groupby("player_id")[col].shift(2)
            
            # Diff: Trend (Prev - Prev2)
            df[f"{col}_trend"] = df[f"{col}_prev"] - df[f"{col}_prev2"]
            
            # Rolling Avg (last 2 years)
            df[f"{col}_rolling2"] = (df[f"{col}_prev"] + df[f"{col}_prev2"]) / 2

    # ==========================================
    # 3. SETUP TARGET
    # ==========================================
    # Target: "TE_Value_Score"
    target_col = "TE_Value_Score"
    
    if target_col not in df.columns:
        raise ValueError("TE_Value_Score not found in CSV. Please run TEAddValueMetric.py first.")
    
    # Drop rows where we don't have the primary previous year data
    model_df = df.dropna(subset=["TE_Value_Score_prev"])
    
    return model_df, df  # Return full df for 2025 prediction construction

# ==========================================
# 4. TRAIN XGBOOST
# ==========================================
print("Processing data...")
# Load
data, full_df = load_and_engineer_features("backend/ML/TightEnds/TE.csv")

# Define Predictors: All lag columns + static/current context (Age)
predictors = [c for c in data.columns if "_prev" in c or "_trend" in c or "_rolling" in c]
predictors += ["age", "age_sq", "Cap_Space"]

# Filter only existing columns
predictors = [c for c in predictors if c in data.columns]

target = "TE_Value_Score"

# Split Train/Test (Time-based)
# Train: < 2024
# Test: 2024
train = data[data["Year"] < 2024]
test = data[data["Year"] == 2024]

print(f"Train samples: {len(train)}")
print(f"Test samples: {len(test)}")

X_train = train[predictors]
y_train = train[target]

X_test = test[predictors]
y_test = test[target]

# Initialize XGBRegressor
model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.7,
    colsample_bytree=0.7,
    n_jobs=-1,
    random_state=42,
    early_stopping_rounds=50 
)

print("Training XGBoost...")
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100
)

# Evaluate
# Evaluate
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)


print("\n==== 2024 MODEL ACCURACY METRICS ====")
print(f"R-Squared (R²): {r2:.4f}  <-- Accuracy Score (1.0=Perfect, 0.0=Baseline, <0=Poor)")
print(f"MAE:  {mae:.4f}   <-- Avg Error")
print(f"RMSE: {rmse:.4f}")

# Detailed Validation (Cross-Validate against Actuals)
# Detailed Validation (Cross-Validate against Actuals)
val_df = test.copy()
val_df["Predicted_weighted_grade"] = preds
val_df["Error"] = val_df[target] - val_df["Predicted_weighted_grade"]
val_df["Abs_Error"] = val_df["Error"].abs()

# Sort by Prediction High to Low (Who did we think would be good?)
print("\n==== TOP 2024 PREDICTIONS VS ACTUAL (Validation) ====")
cols_to_show = ["player", "Team", "weighted_grade", "Predicted_weighted_grade", "Error"]
print(val_df[cols_to_show].sort_values("Predicted_weighted_grade", ascending=False).head(20).to_string(index=False))

# Sort by Biggest Misses (Where did we fail?)
print("\n==== BIGGEST PREDICTION ERRORS (2024) ====")
# Sort first then select
print(val_df.sort_values("Abs_Error", ascending=False)[cols_to_show].head(10).to_string(index=False))

# Save Validation Report
val_output_file = "backend/ML/TightEnds/TE_2024_Validation_Results.csv"
val_df = val_df.sort_values("Predicted_weighted_grade", ascending=False)
val_df[cols_to_show].to_csv(val_output_file, index=False)
print(f"\nDetailed validation results saved to {val_output_file}")

# Feature Importance
print("\nTop 10 Important Features:")
importances = pd.DataFrame({
    'feature': predictors,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(importances.head(10))

# ==========================================
# 5. PREDICT 2025
# ==========================================
print("\nGenerating 2025 Predictions...")
# To predict 2025, we act as if 'Year' is 2025.
# The 'prev' features come from 2024.
# The 'prev2' features come from 2023.

# We need to construct the 2025 input row for each player who played in 2024.
players_2024 = full_df[full_df["Year"] == 2024]["player_id"].unique()

pred_rows = []
for pid in players_2024:
    p_data = full_df[full_df["player_id"] == pid].sort_values("Year")
    
    # Get 2024 row (T-1)
    row_2024 = p_data[p_data["Year"] == 2024].iloc[-1]
    
    # Get 2023 row (T-2) if exists
    row_2023 = p_data[p_data["Year"] == 2023]
    if not row_2023.empty:
        row_2023 = row_2023.iloc[-1]
    else:
        row_2023 = None # Will result in NaNs for prev2, which is fine
        
    # Build Feature Vector
    feat_dict = {"player_id": pid, "player": row_2024["player"], "Team": row_2024["Team"]}
    
    # Static info
    feat_dict["age"] = row_2024["age"] + 1
    feat_dict["age_sq"] = (row_2024["age"] + 1) ** 2
    feat_dict["Cap_Space"] = row_2024["Cap_Space"] # Assume similar or update if known
    
    # Lag Features
    lag_cols_base = [
        "TE_Value_Score", "Net EPA", "grades_offense", "yards", "touchdowns", 
        "first_downs", "yprr", "receptions", "targets",
        "grades_pass_route", "wide_snaps", "slot_snaps",
        "epa_per_target", "targets_per_snap"
    ]
    
    # We must replicate the logic used in "load_and_engineer_features"
    # But manually mapping:
    # _prev -> comes from 2024
    # _prev2 -> comes from 2023
    # _rolling -> avg(2024, 2023)
    # _trend -> 2024 - 2023
    
    # We need to calculate auxiliary cols for 2024/2023 first (targets_per_snap, etc)
    # Actually, they are already calculated in 'full_df' because we ran the function!
    # So we just pull the values.
    
    for col in lag_cols_base:
        val_t1 = row_2024[col] if col in row_2024 else np.nan
        val_t2 = row_2023[col] if row_2023 is not None and col in row_2023 else np.nan
        
        feat_dict[f"{col}_prev"] = val_t1
        feat_dict[f"{col}_prev2"] = val_t2
        feat_dict[f"{col}_trend"] = val_t1 - val_t2 if pd.notnull(val_t1) and pd.notnull(val_t2) else np.nan
        feat_dict[f"{col}_rolling2"] = (val_t1 + val_t2) / 2 if pd.notnull(val_t1) and pd.notnull(val_t2) else val_t1 # Fallback to just prev if prev2 missing
        
    pred_rows.append(feat_dict)

pred_df = pd.DataFrame(pred_rows)
# Align columns
X_pred_2025 = pred_df[predictors]

# Predict
pred_2025_values = model.predict(X_pred_2025)
pred_df["Predicted_weighted_grade"] = pred_2025_values

# Show Top/Bottom
result = pred_df[["player", "Team", "Predicted_weighted_grade"]].sort_values("Predicted_weighted_grade", ascending=False)
print("\n==== TOP 20 TEs 2025 (XGBoost - Weighted Grade) ====")
print(result.head(20).to_string(index=False))

result.to_csv("backend/ML/TightEnds/TEXGBOOSTPredictions.csv", index=False)
print("\nSaved to backend/ML/TightEnds/TEXGBOOSTPredictions.csv")
