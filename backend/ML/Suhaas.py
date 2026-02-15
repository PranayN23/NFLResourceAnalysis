import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ====================================================
# 1. LOAD & CLEAN DATA
# ====================================================
df = pd.read_csv("backend/ML/TE.csv")
df = df.replace("MISSING", np.nan)

# Convert numeric columns
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="ignore")

# Remove rows with no base-grade
df = df.dropna(subset=["weighted_grade"])   # <-- IMPORTANT
df = df.sort_values(["player_id", "Year"]).reset_index(drop=True)

# ====================================================
# 2. CREATE NEXT-YEAR LABEL
# ====================================================
df["weighted_grade_next"] = df.groupby("player_id")["weighted_grade"].shift(-1)

# Only keep rows that have next-year grade
df_trainable = df.dropna(subset=["weighted_grade_next"]).reset_index(drop=True)

# ====================================================
# 3. FEATURES
# ====================================================
base_features = [
    "age","Cap_Space","adjusted_value","Net EPA","Win %",
    "avg_depth_of_target","avoided_tackles","caught_percent",
    "contested_catch_rate","drop_rate","grades_offense",
    "grades_pass_block","grades_pass_route","inline_rate",
    "pass_block_rate","route_rate","slot_rate","wide_rate",
    "yards_after_catch_per_reception","yards_per_reception","yprr",
    "touchdowns","receptions","first_downs","targets",
    "snap_counts_pass_block","snap_counts_run_block","total_snaps"
]

# Lag features
lag_cols = [
    "weighted_grade","grades_offense","grades_pass_block",
    "grades_pass_route","yards","receptions","touchdowns","yprr"
]

for col in lag_cols:
    df_trainable[col + "_lag1"] = df_trainable.groupby("player_id")[col].shift(1)

df_trainable = df_trainable.dropna(subset=[c+"_lag1" for c in lag_cols])

features = base_features + [c+"_lag1" for c in lag_cols]

# ====================================================
# 4. AUTO-DETECT LAST YEAR WITH COMPLETE LABELS
# ====================================================
last_trainable_year = df_trainable["Year"].max()
validation_year = last_trainable_year  # validate on last fully-labeled year

print("Last year with complete labels:", validation_year)

train = df_trainable[df_trainable["Year"] < validation_year]
valid = df_trainable[df_trainable["Year"] == validation_year]

X_train = train[features]
y_train = train["weighted_grade_next"]

X_valid = valid[features]
y_valid = valid["weighted_grade_next"]

# ====================================================
# 5. TRAIN MODEL
# ====================================================
model = XGBRegressor(
    n_estimators=600,
    learning_rate=0.04,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    random_state=42
)

model.fit(X_train, y_train)

# ====================================================
# 6. VALIDATION
# ====================================================
pred_valid = model.predict(X_valid)

print("\n==== VALIDATION RESULTS ====")
print("Validating on year:", validation_year)
print("R²:", r2_score(y_valid, pred_valid))
print("RMSE:", mean_squared_error(y_valid, pred_valid, squared=False))
print("MAE:", mean_absolute_error(y_valid, pred_valid))

results = valid[["player_id","player","Team","Year","weighted_grade_next"]].copy()
results["predicted"] = pred_valid

print("\n==== SAMPLE OUTPUT ====")
print(results.head())

# ====================================================
# 7. OPTIONAL: Predict 2025 next
# ====================================================
df_2024 = df[df["Year"] == validation_year][features]
pred_2025 = model.predict(df_2024)

df_predict_2025 = df[df["Year"] == validation_year][["player_id","player","Team","Year"]].copy()
df_predict_2025["predicted_2025_grade"] = pred_2025

print("\n==== PREDICT 2025 ====")
print(df_predict_2025.head())
