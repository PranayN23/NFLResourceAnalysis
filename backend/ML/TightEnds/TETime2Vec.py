
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
    # WEIGHTED GRADE (New Target)
    # Formula: (Grade) * (Snaps) / 1000. 
    # Example: 80 grade * 1000 snaps = 80.0 score. 60 grade * 500 snaps = 30.0 score.
    df["weighted_grade"] = df["grades_offense"].fillna(0) * df["total_snaps"].fillna(0) / 1000.0
    
    df["age_sq"] = df["age"] ** 2
    
    df["targets_per_snap"] = df["targets"] / df["total_snaps"].replace(0, np.nan)
    df["epa_per_target"] = df["Net EPA"] / df["targets"].replace(0, np.nan)

    # ==========================================
    # TIME2VEC STYLE TEMPORAL ENCODING
    # ==========================================

    t = df["Year"]

    df["time_linear"] = t

    df["time_sin_1"] = np.sin(2 * np.pi * t / 2)
    df["time_cos_1"] = np.cos(2 * np.pi * t / 2)

    df["time_sin_2"] = np.sin(2 * np.pi * t / 4)
    df["time_cos_2"] = np.cos(2 * np.pi * t / 4)

    df["time_sin_3"] = np.sin(2 * np.pi * t / 8)
    df["time_cos_3"] = np.cos(2 * np.pi * t / 8)

    df["career_year"] = df.groupby("player_id")["Year"].transform(lambda x: x - x.min())
    df["career_year_sq"] = df["career_year"] ** 2

    # ==========================================
    # 2. LAG FEATURES (History)
    # ==========================================

    lag_features = [
        "weighted_grade", "grades_offense", "yards", "touchdowns", 
        "first_downs", "yprr", "receptions", "targets",
        "grades_pass_route", "wide_snaps", "slot_snaps",
        "epa_per_target", "targets_per_snap"
    ]
    
    for col in lag_features:
        if col in df.columns:
            # Shift
            df[f"{col}_prev"] = df.groupby("player_id")[col].shift(1)
            df[f"{col}_prev2"] = df.groupby("player_id")[col].shift(2)
            
            # Trend: Impute 0 for prev2 if it is missing (e.g. Sophomores), so Trend = prev - 0
            p1 = df[f"{col}_prev"]
            p2_imputed = df[f"{col}_prev2"].fillna(0)
            df[f"{col}_trend"] = p1 - p2_imputed
            
            # Rolling: If prev2 is missing, use prev1. Else average.
            # Note: np.where allows us to handle the condition element-wise
            df[f"{col}_rolling2"] = np.where(df[f"{col}_prev2"].isna(), p1, (p1 + df[f"{col}_prev2"]) / 2)

    # Fix Fragmentation Warning
    df = df.copy()

    # ==========================================
    # 2.5 BREAKOUT & AGE INDICATORS
    # ==========================================
    # Prime Years (23-26) vs Decline (30+)
    df["is_prime"] = ((df["age"] >= 23) & (df["age"] <= 26)).astype(float)
    df["is_decline"] = (df["age"] >= 30).astype(float)

    # Catching the "Trey McBride" effect: High efficiency jump + age prime
    df["yprr_trend"] = df["yprr"] - df.groupby("player_id")["yprr"].shift(1)
    df["efficiency_spike"] = (df["grades_offense"] > 75) & (df["yprr"] > 1.8)
    df["efficiency_spike"] = df["efficiency_spike"].astype(float)
    
    # Catching the "Jonnu Smith" effect: High efficiency on low volume
    df["efficiency_per_snap"] = df["weighted_grade"] / df["total_snaps"].replace(0, np.nan)
    
    # ==========================================
    # 2.6 GROWTH CONSTANTS & TEAM HISTORY (User Request)
    # ==========================================
    # Constants derived from TE_Analyze_Rookie_Progression.py
    # Y1->Y2: 1.3726, Y2->Y3: 1.1393, Y3->Y4: 0.9838
    conditions = [
        df["career_year"] == 1,
        df["career_year"] == 2,
        df["career_year"] == 3
    ]
    choices = [1.3726, 1.1393, 0.9838]
    df["Growth_Potential"] = np.select(conditions, choices, default=1.0)
    
    # Team TE EPA History (3-Year Rolling Avg)
    # 1. Get Avg Net EPA per Team-Year (ignoring NaNs)
    team_epa = df.groupby(["Team", "Year"])["Net EPA"].mean().reset_index()
    team_epa = team_epa.sort_values(["Team", "Year"])
    
    # 2. Calc Rolling Mean (Closed='left' to use only PAST years)
    # shifting by 1 to ensure we don't include current year
    team_epa["Team_TE_EPA_History"] = team_epa.groupby("Team")["Net EPA"].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    
    # 3. Merge back
    df = pd.merge(df, team_epa[["Team", "Year", "Team_TE_EPA_History"]], on=["Team", "Year"], how="left")
    
    # Defragment again after new cols
    df = df.copy()

    # ==========================================
    # 3. SETUP TARGET
    # ==========================================

    # Target: "weighted_grade"
    target_col = "weighted_grade"
    
    # Drop rows where we don't have the primary previous year data
    model_df = df.dropna(subset=["weighted_grade_prev"])
    
    return model_df, df

# ==========================================
# 4. TRAIN XGBOOST
# ==========================================

if __name__ == "__main__":
    print("Processing data...")
    data, full_df = load_and_engineer_features("backend/ML/TightEnds/TE.csv")

    predictors = [c for c in data.columns if "_prev" in c or "_trend" in c or "_rolling" in c]

    predictors += [
        "age", "age_sq", "Cap_Space",

        "time_linear",
        "time_sin_1", "time_cos_1",
        "time_sin_2", "time_cos_2",
        "time_sin_3", "time_cos_3",

        "career_year", "career_year_sq"
    ]

    predictors = [c for c in predictors if c in data.columns]

    target = "weighted_grade" # Override for testing this script directly

    train = data[data["Year"] < 2024]
    test = data[data["Year"] == 2024]

    print(f"Train samples: {len(train)}")
    print(f"Test samples: {len(test)}")

    X_train = train[predictors]
    y_train = train[target]

    X_test = test[predictors]
    y_test = test[target]

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

    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("\n==== 2024 MODEL ACCURACY METRICS ====")
    print(f"R-Squared (R²): {r2:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    val_df = test.copy()
    val_df["Predicted_weighted_grade"] = preds
    val_df["Error"] = val_df[target] - val_df["Predicted_weighted_grade"]
    val_df["Abs_Error"] = val_df["Error"].abs()

    cols_to_show = ["player", "Team",  "weighted_grade", "Predicted_weighted_grade", "Error"]

    print("\n==== TOP 2024 PREDICTIONS VS ACTUAL ====")
    print(val_df[cols_to_show].sort_values("Predicted_weighted_grade", ascending=False).head(20).to_string(index=False))

    print("\n==== BIGGEST ERRORS ====")
    print(val_df.sort_values("Abs_Error", ascending=False)[cols_to_show].head(10).to_string(index=False))

    val_df[cols_to_show].to_csv("backend/ML/TightEnds/TE_2024_Validation_Results.csv", index=False)

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

    players_2024 = full_df[full_df["Year"] == 2024]["player_id"].unique()

    pred_rows = []

    for pid in players_2024:
        p_data = full_df[full_df["player_id"] == pid].sort_values("Year")

        row_2024 = p_data[p_data["Year"] == 2024].iloc[-1]

        row_2023 = p_data[p_data["Year"] == 2023]
        row_2023 = row_2023.iloc[-1] if not row_2023.empty else None

        feat_dict = {
            "player_id": pid,
            "player": row_2024["player"],
            "Team": row_2024["Team"]
        }

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
        feat_dict["career_year_sq"] = (row_2024["career_year"] + 1) ** 2

        lag_cols_base = [
            "weighted_grade", "Net EPA", "grades_offense", "yards", "touchdowns", 
            "first_downs", "yprr", "receptions", "targets",
            "grades_pass_route", "wide_snaps", "slot_snaps",
            "epa_per_target", "targets_per_snap"
        ]

        for col in lag_cols_base:
            val_t1 = row_2024[col] if col in row_2024 else np.nan
            val_t2 = row_2023[col] if row_2023 is not None and col in row_2023 else np.nan

            feat_dict[f"{col}_prev"] = val_t1
            feat_dict[f"{col}_prev2"] = val_t2
            feat_dict[f"{col}_trend"] = val_t1 - val_t2 if pd.notnull(val_t1) and pd.notnull(val_t2) else np.nan
            feat_dict[f"{col}_rolling2"] = (val_t1 + val_t2) / 2 if pd.notnull(val_t1) and pd.notnull(val_t2) else val_t1

        pred_rows.append(feat_dict)

    pred_df = pd.DataFrame(pred_rows)

    X_pred_2025 = pred_df[predictors]

    pred_df["Predicted_weighted_grade"] = model.predict(X_pred_2025)

    result = pred_df[["player", "Team", "Predicted_weighted_grade"]].sort_values(
        "Predicted_weighted_grade", ascending=False
    )

    print("\n==== TOP 20 TEs 2025 ====")
    print(result.head(20).to_string(index=False))

    result.to_csv("backend/ML/TightEnds/TEXGBOOSTPredictions.csv", index=False)

    print("\nSaved to backend/ML/TightEnds/TEXGBOOSTPredictions.csv")
