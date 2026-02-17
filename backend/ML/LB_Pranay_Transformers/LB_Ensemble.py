import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from backend.ML.LB_Pranay_Transformers.Player_Model_LB import (
    PlayerTransformerRegressor,
    SEQ_LEN,
)
from backend.agent.lb_model_wrapper import LBModelInference

# ==========================================
# CONFIG
# ==========================================
DATA_FILE     = os.path.join(os.path.dirname(__file__), "../LB.csv")
MODEL_OUT     = os.path.join(os.path.dirname(__file__), "lb_best_classifier.pth")
SCALER_OUT    = os.path.join(os.path.dirname(__file__), "lb_player_scaler.joblib")
XGB_MODEL_OUT = os.path.join(os.path.dirname(__file__), "lb_best_xgb.joblib")

MODE = "VALIDATION"   # "VALIDATION" | "DREAM"

print(f"==== STARTING LB ENSEMBLE MODELING (Mode: {MODE}) ====")

# ==========================================
# FEATURE LISTS
# ==========================================
TRANSFORMER_FEATURES = [
    "grades_defense",
    "pressure_rate",
    "sack_rate",
    "hit_rate",
    "hurry_rate",
    "stop_rate",
    "tfl_rate",
    "penalty_rate",
    "missed_tackle_rate",
    "target_rate",
    "int_rate",
    "pbu_rate",
    "box_share",
    "offball_share",
    "age",
    "years_in_league",
    "adjusted_value",
    "Cap_Space",
    "team_performance_proxy",
    "delta_grade",
]

XGB_FEATURES = [
    "lag_grades_defense",
    "lag_pressure_rate",
    "lag_sack_rate",
    "lag_stop_rate",
    "lag_target_rate",
    "lag_int_rate",
    "lag_pbu_rate",
    "lag_box_share",
    "lag_offball_share",
    "adjusted_value",
    "age",
    "years_in_league",
    "delta_grade_lag",
    "team_performance_proxy_lag",
]


# ==========================================
# DATA PREP
# ==========================================
def prepare_data():
    df = pd.read_csv(DATA_FILE)
    df = df[df["position"] == "LB"].copy()
    df = df.sort_values(["player", "Year"])
    groups = df.groupby("player")

    numeric_cols = [
        "grades_defense", "grades_run_defense",
        "sacks", "hits", "hurries", "total_pressures", "stops",
        "tackles", "tackles_for_loss", "assists", "missed_tackles",
        "targets", "interceptions", "pass_break_ups", "penalties",
        "snap_counts_defense", "snap_counts_box", "snap_counts_offball",
        "snap_counts_run_defense",
        "adjusted_value", "Net EPA", "age", "Cap_Space",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    def safe_div(a, b):
        return np.where(b == 0, 0, a / b)

    snap = df["snap_counts_defense"]

    df["years_in_league"]        = groups.cumcount()
    df["delta_grade"]            = groups["grades_defense"].diff().fillna(0)
    df["team_performance_proxy"] = df.groupby(["Team", "Year"])["Net EPA"].transform("mean")

    df["pressure_rate"]      = safe_div(df["total_pressures"],    snap)
    df["sack_rate"]          = safe_div(df["sacks"],              snap)
    df["hit_rate"]           = safe_div(df["hits"],               snap)
    df["hurry_rate"]         = safe_div(df["hurries"],            snap)
    df["stop_rate"]          = safe_div(df["stops"],              snap)
    df["tfl_rate"]           = safe_div(df["tackles_for_loss"],   snap)
    df["penalty_rate"]       = safe_div(df["penalties"],          snap)
    df["missed_tackle_rate"] = safe_div(df["missed_tackles"],     snap)
    df["target_rate"]        = safe_div(df["targets"],            snap)
    df["int_rate"]           = safe_div(df["interceptions"],      snap)
    df["pbu_rate"]           = safe_div(df["pass_break_ups"],     snap)
    df["box_share"]          = safe_div(df["snap_counts_box"],     snap)
    df["offball_share"]      = safe_div(df["snap_counts_offball"], snap)

    for col in ["grades_defense", "pressure_rate", "sack_rate", "stop_rate",
                "target_rate", "int_rate", "pbu_rate", "box_share", "offball_share",
                "team_performance_proxy"]:
        df[f"lag_{col}"] = groups[col].shift(1)

    df["delta_grade_lag"]            = groups["lag_grades_defense"].diff().fillna(0)
    df["team_performance_proxy_lag"] = groups["team_performance_proxy"].shift(1)

    target_col = "grades_defense"
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

    print("\n=== VALIDATION: Train ≤2023, Test 2024 ===")

    train_data = df_clean[df_clean["Year"] < 2024]
    test_data  = df_clean[df_clean["Year"] == 2024]

    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,        # shallower than DI/ED — LB is noisier, avoid overfit
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3, # require more data per leaf — reduces overfit on small LB samples
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

    engine  = LBModelInference(MODEL_OUT, scaler_path=SCALER_OUT, xgb_path=XGB_MODEL_OUT)
    results = []

    for _, row in test_data.iterrows():
        name = row["player"]

        history = df_all[
            (df_all["player"] == name) &
            (df_all["Year"] < 2024)
        ].copy()

        if len(history) == 0:
            continue

        # Run all three modes so we can compare in the output CSV
        _, details_xgb  = engine.get_prediction(history, mode="xgb")
        _, details_ens  = engine.get_prediction(history, mode="ensemble")
        _, details_tr   = engine.get_prediction(history, mode="transformer")

        actual = row[target_col]
        results.append({
            "player":              name,
            "Actual_Grade":        actual,
            "Pred_XGB":            details_xgb["predicted_grade"],
            "Pred_Transformer":    details_tr["predicted_grade"],
            "Pred_Ensemble_80XGB": details_ens["predicted_grade"],
            "Abs_Err_XGB":         abs(actual - details_xgb["predicted_grade"]),
            "Abs_Err_Transformer": abs(actual - details_tr["predicted_grade"]),
            "Abs_Err_Ensemble":    abs(actual - details_ens["predicted_grade"]),
        })

    final_df = pd.DataFrame(results).sort_values("Pred_XGB", ascending=False)

    print("\n=== RESULTS COMPARISON (2024) ===")
    print(f"XGB-only MAE:       {final_df['Abs_Err_XGB'].mean():.4f}")
    print(f"Transformer MAE:    {final_df['Abs_Err_Transformer'].mean():.4f}")
    print(f"Ensemble 80/20 MAE: {final_df['Abs_Err_Ensemble'].mean():.4f}")
    print(f"\nRecommended mode:   {'xgb' if final_df['Abs_Err_XGB'].mean() <= final_df['Abs_Err_Ensemble'].mean() else 'ensemble'}")

    out_path = os.path.join(os.path.dirname(__file__), "LB_2024_Validation_Results.csv")
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
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
    )
    xgb_model.fit(df_clean[XGB_FEATURES], df_clean[target_col])
    joblib.dump(xgb_model, XGB_MODEL_OUT)

    engine = LBModelInference(MODEL_OUT, scaler_path=SCALER_OUT, xgb_path=XGB_MODEL_OUT)

    active_2024 = df_all[df_all["Year"] == 2024]
    rows        = []

    for _, row in active_2024.iterrows():
        player  = row["player"]
        history = df_all[df_all["player"] == player].sort_values("Year").tail(SEQ_LEN)

        if len(history) == 0:
            continue

        tier, details = engine.get_prediction(history, mode="xgb")

        rows.append({
            "player":        player,
            "Tier":          tier,
            "Ensemble_Pred": details["predicted_grade"],
            "Age_2024":      row["age"],
        })

    final_2025 = pd.DataFrame(rows).sort_values("Ensemble_Pred", ascending=False)

    final_out = os.path.join(os.path.dirname(__file__), "LB_2025_Final_Rankings.csv")
    final_2025.to_csv(final_out, index=False)

    print("\nTop 15 LB for 2025:")
    print(final_2025.head(15).to_string(index=False))


print("\n=== LB MODEL PIPELINE COMPLETE ===")