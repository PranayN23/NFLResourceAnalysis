
import os
# CRITICAL: Must be set before importing xgboost/torch to prevent OpenMP deadlock on Mac
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import sys

# Ensure root is in path
sys.path.append(os.getcwd())

from backend.ML.TightEnds.TETime2Vec import load_and_engineer_features as load_xgb_data

# ==========================================
# CONFIG
# ==========================================
DATA_FILE = "backend/ML/TightEnds/TE.csv"
VALIDATION_YEAR = 2024
TARGET = "weighted_grade"
PLAYER_TO_ANALYZE = "Trey McBride"

print(f"==== ANALYZING PREDICTION FOR: {PLAYER_TO_ANALYZE} ({VALIDATION_YEAR}) ====")

# ==========================================
# 1. LOAD & TRAIN XGBOOST
# ==========================================
print("\n[1/3] Loading Data & Training XGBoost...")
data, _ = load_xgb_data(DATA_FILE)

# Features (Same as Validation Script)
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

# ==========================================
# 2. LOCATE PLAYER DATA
# ==========================================
player_row = test_data[test_data["player"] == PLAYER_TO_ANALYZE]

if player_row.empty:
    print(f"ERROR: No data found for {PLAYER_TO_ANALYZE} in {VALIDATION_YEAR}")
    sys.exit(1)

# Get the feature vector for this player
X_player = player_row[predictors]
actual_score = player_row[TARGET].values[0]
predicted_score = xgb_model.predict(X_player)[0]

print(f"\nStats for {PLAYER_TO_ANALYZE}:")
print(f"Actual Weighted Grade:    {actual_score:.4f}")
print(f"Predicted XGB Score:      {predicted_score:.4f}")
print(f"Error (Actual - Pred):    {actual_score - predicted_score:.4f}")

# ==========================================
# 3. SHAP ANALYSIS
# ==========================================
print("\n[2/3] Calculating SHAP Values...")

# Initialize Explainer
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer(X_player)

# Get Base Value (The average prediction if we knew nothing)
base_value = shap_values.base_values[0]

print(f"\n==== EXPLANATION ====")
print(f"Base Value (Avg TE Score): {base_value:.4f}")
print(f"Final Prediction:          {predicted_score:.4f}")
print(f"Difference to Explain:     {predicted_score - base_value:.4f}")

print("\n==== TOP FACTORS PUSHING SCORE UP (+) ====")
# Convert to DataFrame for sorting
shap_df = pd.DataFrame({
    "Feature": predictors,
    "Value": X_player.values[0],
    "SHAP_Impact": shap_values.values[0]
})

top_positive = shap_df[shap_df["SHAP_Impact"] > 0].sort_values("SHAP_Impact", ascending=False).head(10)
print(top_positive.to_string(index=False))

print("\n==== TOP FACTORS PULLING SCORE DOWN (-) ====")
top_negative = shap_df[shap_df["SHAP_Impact"] < 0].sort_values("SHAP_Impact", ascending=True).head(10)
print(top_negative.to_string(index=False))

print("\n==== ANALYSIS ====")
print("look at the 'Value' column to see the raw input data.")
print("The 'SHAP_Impact' column shows how much that feature changed the prediction from the Base Value.")
