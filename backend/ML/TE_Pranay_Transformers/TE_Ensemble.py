import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import torch
from sklearn.metrics import mean_absolute_error
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from backend.ML.TE_Pranay_Transformers.Player_Model_TE import (
    PlayerTransformerRegressor,
    SEQ_LEN,
)
from backend.agent.te_model_wrapper import TEModelInference

# ==========================================
# CONFIG
# ==========================================
DATA_FILE     = os.path.join(os.path.dirname(__file__), "../TightEnds/TE.csv")
MODEL_OUT     = os.path.join(os.path.dirname(__file__), "te_best_transformer.pth")
SCALER_OUT    = os.path.join(os.path.dirname(__file__), "te_player_scaler.joblib")
XGB_MODEL_OUT = os.path.join(os.path.dirname(__file__), "te_best_xgb.joblib")

MODE = "VALIDATION"

print(f"==== STARTING TE ENSEMBLE MODELING (Mode: {MODE}) ====")

# ==========================================
# FEATURE LISTS
# ==========================================
TRANSFORMER_FEATURES = [
    "grades_offense", "grades_pass_route", "grades_pass_block",
    "grades_hands_drop",
    "avg_depth_of_target", "caught_percent", "drop_rate",
    "yards", "yards_after_catch", "yards_per_reception", "yprr",
    "receptions", "targets", "touchdowns", "first_downs",
    "targeted_qb_rating", "route_rate",
    "slot_share", "inline_share", "target_share",
    "contested_rate", "block_share",
    "age", "years_in_league", "adjusted_value", "Cap_Space",
    "team_performance_proxy",
    "delta_grade", "delta_yprr", "delta_adot",
    "career_mean_grade",
]

XGB_FEATURES = [
    "lag_grades_offense", "lag_grades_pass_route", "lag_grades_pass_block",
    "lag_yprr", "lag_avg_depth_of_target", "lag_caught_percent",
    "lag_drop_rate", "lag_yards", "lag_targets", "lag_touchdowns",
    "lag_yards_after_catch", "lag_targeted_qb_rating",
    "lag_slot_share", "lag_inline_share", "lag_block_share",
    "adjusted_value", "age", "years_in_league",
    "delta_grade_lag", "team_performance_proxy_lag",
]


# ==========================================
# DATA PREPARATION
# ==========================================
def prepare_data():
    df = pd.read_csv(DATA_FILE)
    df = df[df["position"] == "TE"].copy()

    numeric_cols = [
        "grades_offense", "grades_pass_route", "grades_pass_block",
        "grades_hands_drop", "grades_hands_fumble",
        "avg_depth_of_target", "caught_percent", "contested_catch_rate",
        "drop_rate", "receptions", "targets", "yards", "touchdowns",
        "first_downs", "yards_after_catch", "yards_after_catch_per_reception",
        "yards_per_reception", "yprr", "targeted_qb_rating", "route_rate",
        "routes", "avoided_tackles", "total_snaps", "slot_snaps", "wide_snaps",
        "inline_snaps", "age", "adjusted_value", "Cap_Space", "Net EPA",
        "slot_rate", "wide_rate", "inline_rate", "contested_receptions",
        "contested_targets", "drops", "penalties",
        "snap_counts_pass_block", "snap_counts_run_block",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["routes"] >= 100].copy()
    df = df.sort_values(["player", "Year"])
    groups = df.groupby("player")

    # Feature engineering
    df["years_in_league"] = groups.cumcount()
    df["delta_grade"]     = groups["grades_offense"].diff().fillna(0)
    df["delta_yprr"]      = groups["yprr"].diff().fillna(0)
    df["delta_adot"]      = groups["avg_depth_of_target"].diff().fillna(0)
    df["team_performance_proxy"] = df.groupby(["Team", "Year"])["Net EPA"].transform("mean")

    total = df["slot_snaps"].fillna(0) + df["wide_snaps"].fillna(0) + df["inline_snaps"].fillna(0)
    total = total.replace(0, np.nan)
    df["slot_share"]   = (df["slot_snaps"]   / total).fillna(0)
    df["inline_share"] = (df["inline_snaps"] / total).fillna(0)

    df["target_share"] = df["targets"] / df["routes"].replace(0, np.nan)
    df["target_share"] = df["target_share"].fillna(0)
    df["contested_rate"] = df["contested_targets"] / df["targets"].replace(0, np.nan)
    df["contested_rate"] = df["contested_rate"].fillna(0)

    block_total = df["snap_counts_pass_block"].fillna(0) + df["snap_counts_run_block"].fillna(0)
    df["block_share"] = block_total / df["total_snaps"].replace(0, np.nan)
    df["block_share"] = df["block_share"].fillna(0)

    df["career_mean_grade"] = groups["grades_offense"] \
        .transform(lambda x: x.shift(1).expanding().mean())

    # Lag features for XGBoost
    for col in ["grades_offense", "grades_pass_route", "grades_pass_block",
                "yprr", "avg_depth_of_target", "caught_percent", "drop_rate",
                "yards", "targets", "touchdowns", "yards_after_catch",
                "targeted_qb_rating", "slot_share", "inline_share", "block_share",
                "team_performance_proxy"]:
        df[f"lag_{col}"] = groups[col].shift(1)

    df["delta_grade_lag"]            = groups["lag_grades_offense"].diff().fillna(0)
    df["team_performance_proxy_lag"] = groups["team_performance_proxy"].shift(1)

    target_col = "grades_offense"
    df_clean   = df.dropna(subset=XGB_FEATURES + [target_col]).copy()

    for col in XGB_FEATURES:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    df_clean = df_clean.dropna(subset=XGB_FEATURES + [target_col]).copy()

    return df_clean, df, target_col


df_clean, df_all, target_col = prepare_data()


# ==========================================
# VALIDATION MODE
# ==========================================
if MODE == "VALIDATION":
    print("\n=== VALIDATION: Train <2024, Test 2024 ===")

    train_data = df_clean[df_clean["Year"] < 2024]
    test_data  = df_clean[df_clean["Year"] == 2024]

    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    xgb_model.fit(train_data[XGB_FEATURES], train_data[target_col])
    joblib.dump(xgb_model, XGB_MODEL_OUT)

    xgb_train_mae = mean_absolute_error(
        train_data[target_col], xgb_model.predict(train_data[XGB_FEATURES])
    )
    xgb_test_mae = mean_absolute_error(
        test_data[target_col], xgb_model.predict(test_data[XGB_FEATURES])
    )
    print(f"XGB Train MAE: {xgb_train_mae:.4f}")
    print(f"XGB Test MAE:  {xgb_test_mae:.4f}")

    engine  = TEModelInference(MODEL_OUT, scaler_path=SCALER_OUT, xgb_path=XGB_MODEL_OUT)
    results = []

    for _, row in test_data.iterrows():
        name = row["player"]
        history = df_all[(df_all["player"] == name) & (df_all["Year"] < 2024)].copy()

        if len(history) == 0:
            continue

        tier, details = engine.get_prediction(history)

        results.append({
            "player":           name,
            "Team":             row["Team"],
            "Actual_Grade":     row[target_col],
            "Pred_XGB":         details["xgb_grade"],
            "Pred_Transformer": details["transformer_grade"],
            "Ensemble_Pred":    details["predicted_grade"],
            "Error":            row[target_col] - details["predicted_grade"],
            "Abs_Error":        abs(row[target_col] - details["predicted_grade"]),
        })

    final_df = pd.DataFrame(results).sort_values(by="Ensemble_Pred", ascending=False)
    print("\n=== ENSEMBLE RESULTS (2024) ===")
    print(f"Ensemble MAE:  {final_df['Abs_Error'].mean():.4f}")
    print(f"Bias:          {final_df['Error'].mean():.4f}")

    out_path = os.path.join(os.path.dirname(__file__), "TE_2024_Validation_Results.csv")
    final_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


# ==========================================
# DREAM MODE
# ==========================================
elif MODE == "DREAM":
    print("\n=== DREAM: Train ≤2024, Predict 2025 ===")

    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    xgb_model.fit(df_clean[XGB_FEATURES], df_clean[target_col])
    joblib.dump(xgb_model, XGB_MODEL_OUT)

    engine = TEModelInference(MODEL_OUT, scaler_path=SCALER_OUT, xgb_path=XGB_MODEL_OUT)

    active_2024 = df_all[df_all["Year"] == 2024]
    rows = []

    for _, row in active_2024.iterrows():
        player  = row["player"]
        history = df_all[df_all["player"] == player].sort_values("Year").tail(SEQ_LEN)

        if len(history) == 0:
            continue

        tier, details = engine.get_prediction(history)

        rows.append({
            "player":        player,
            "Team":          row["Team"],
            "Tier":          tier,
            "Ensemble_Pred": details["predicted_grade"],
            "Conf_Lower":    details["confidence_interval"][0],
            "Conf_Upper":    details["confidence_interval"][1],
            "Vol_Index":     details["volatility_index"],
            "Age_2024":      row["age"],
            "Last_Grade":    history.iloc[-1]["grades_offense"],
        })

    final_2025 = pd.DataFrame(rows).sort_values("Ensemble_Pred", ascending=False)
    final_out = os.path.join(os.path.dirname(__file__), "TE_2025_Final_Rankings.csv")
    final_2025.to_csv(final_out, index=False)

    print("\nTop 15 TE for 2025:")
    print(final_2025.head(15).to_string(index=False))

print("\n=== TE MODEL PIPELINE COMPLETE ===")
