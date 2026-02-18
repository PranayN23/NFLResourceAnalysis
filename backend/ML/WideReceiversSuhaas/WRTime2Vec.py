
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
    # WR.csv has weighted_grade as Grade * Snaps. So we divide by 1000.
    if "grades_offense" in df.columns and "total_snaps" in df.columns:
        df["weighted_grade"] = df["grades_offense"].fillna(0) * df["total_snaps"].fillna(0) / 1000.0
    
    df["age_sq"] = df["age"] ** 2
    
    if "targets" in df.columns and "total_snaps" in df.columns:
        df["targets_per_snap"] = df["targets"] / df["total_snaps"].replace(0, np.nan)
    else:
        df["targets_per_snap"] = 0
        
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
    
    # Ensure columns exist before lagging
    lag_features = [c for c in lag_features if c in df.columns]
    
    for col in lag_features:
        # Shift
        df[f"{col}_prev"] = df.groupby("player_id")[col].shift(1)
        df[f"{col}_prev2"] = df.groupby("player_id")[col].shift(2)
        
        # Trend: Impute prev2 with prev1 for rookies (assume flat trend instead of massive 0->X spike)
        p1 = df[f"{col}_prev"]
        p2_imputed = df[f"{col}_prev2"].fillna(p1)
        df[f"{col}_trend"] = p1 - p2_imputed
        
        # Rolling: If prev2 is missing, use prev1. Else average.
        df[f"{col}_rolling2"] = np.where(df[f"{col}_prev2"].isna(), p1, (p1 + df[f"{col}_prev2"]) / 2)

    df = df.copy()

    # ==========================================
    # 2.5 BREAKOUT & AGE INDICATORS
    # ==========================================
    # Prime Years (23-27 for WRs maybe? sticking to 23-26 for now)
    df["is_prime"] = ((df["age"] >= 23) & (df["age"] <= 27)).astype(float)
    df["is_decline"] = (df["age"] >= 30).astype(float)

    # Efficiency Spike (High grade + high YPRR)
    if "grades_offense" in df.columns and "yprr" in df.columns:
        df["yprr_trend"] = df["yprr"] - df.groupby("player_id")["yprr"].shift(1)
        df["efficiency_spike"] = (df["grades_offense"] > 75) & (df["yprr"] > 1.8)
        df["efficiency_spike"] = df["efficiency_spike"].astype(float)
    
    # Efficiency per Snap
    df["efficiency_per_snap"] = df["weighted_grade"] / df["total_snaps"].replace(0, np.nan)
    
    # ==========================================
    # 2.6 GROWTH CONSTANTS & TEAM HISTORY
    # ==========================================
    # Placeholder for Rookie Progression Coefficients
    # We will update these after running WR_Analyze_Rookie_Progression.py
    df["Growth_Potential"] = 1.0
    
    # Team WR EPA History (3-Year Rolling Avg)
    if "Net EPA" in df.columns:
        team_epa = df.groupby(["Team", "Year"])["Net EPA"].mean().reset_index()
        team_epa = team_epa.sort_values(["Team", "Year"])
        
        team_epa["Team_WR_EPA_History"] = team_epa.groupby("Team")["Net EPA"].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        )
        
        df = pd.merge(df, team_epa[["Team", "Year", "Team_WR_EPA_History"]], on=["Team", "Year"], how="left")
    
    df = df.copy()

    # ==========================================
    # 3. SETUP TARGET
    # ==========================================

    target_col = "weighted_grade"
    
    # Drop rows where we don't have the primary previous year data
    model_df = df.dropna(subset=["weighted_grade_prev"])
    
    return model_df, df

# ==========================================
# 4. TRAIN XGBOOST
# ==========================================

if __name__ == "__main__":
    print("Processing data...")
    # Use standard WR path
    data, full_df = load_and_engineer_features("backend/ML/WR.csv")

    predictors = [c for c in data.columns if "_prev" in c or "_trend" in c or "_rolling" in c]

    predictors += [
        "age", "age_sq", "Cap_Space",
        "time_linear",
        "time_sin_1", "time_cos_1",
        "time_sin_2", "time_cos_2",
        "time_sin_3", "time_cos_3",
        "career_year", "career_year_sq",
        "is_prime", "is_decline", "efficiency_spike", "efficiency_per_snap",
        "Growth_Potential", "Team_WR_EPA_History"
    ]

    predictors = [c for c in predictors if c in data.columns]

    target = "weighted_grade"

    # Standard Validation Split (2024 is Test)
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

    cols_to_show = ["player", "Team", "weighted_grade", "Predicted_weighted_grade", "Error"]

    print("\n==== TOP 2024 PREDICTIONS VS ACTUAL ====")
    print(val_df[cols_to_show].sort_values("Predicted_weighted_grade", ascending=False).head(20).to_string(index=False))

    print("\nTop 10 Important Features:")
    importances = pd.DataFrame({
        'feature': predictors,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(importances.head(10))
    
    # Save predictions
    val_df.to_csv("backend/ML/WideReceivers/WR_2024_XGBoost_Validation.csv", index=False)
