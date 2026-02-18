import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import torch
from sklearn.metrics import mean_absolute_error
import joblib

# Add project root so imports resolve regardless of CWD
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from backend.ML.ED_Pranay_Transformers.Player_Model_ED import (
    PlayerTransformerRegressor,
    SEQ_LEN,
)
from backend.agent.ed_model_wrapper import EDModelInference

# ==========================================
# CONFIG
# ==========================================
DATA_FILE     = os.path.join(os.path.dirname(__file__), "../ED.csv")
MODEL_OUT     = os.path.join(os.path.dirname(__file__), "../ED_Transformers/ed_best_classifier.pth")
SCALER_OUT    = os.path.join(os.path.dirname(__file__), "../ED_Transformers/ed_player_scaler.joblib")
XGB_MODEL_OUT = os.path.join(os.path.dirname(__file__), "../ED_Transformers/ed_best_xgb.joblib")

MODE = "DREAM"   # "VALIDATION" | "DREAM"

print(f"==== STARTING ED ENSEMBLE MODELING (Mode: {MODE}) ====")

# ==========================================
# FEATURE LISTS
#
# TRANSFORMER_FEATURES must exactly match the `features` list used
# when training Player_Model_ED.py — order matters for the scaler.
#
# XGB_FEATURES use lag-based versions of the same stats so the XGB
# can operate on a single flat row (no sequence needed).
# ==========================================
TRANSFORMER_FEATURES = [
    "grades_defense",
    "grades_run_defense",
    "grades_pass_rush_defense",
    "pressure_rate",
    "sack_rate",
    "hit_rate",
    "hurry_rate",
    "stop_rate",
    "tfl_rate",
    "penalty_rate",
    "outside_t_share",
    "over_t_share",
    "age",
    "years_in_league",
    "adjusted_value",
    "Cap_Space",
    "team_performance_proxy",
    "delta_grade",
    "delta_pass_rush",
    "delta_run_def",
]

XGB_FEATURES = [
    "lag_grades_defense",
    "lag_grades_pass_rush_defense",
    "lag_grades_run_defense",
    "lag_pressure_rate",
    "lag_sack_rate",
    "lag_stop_rate",
    "lag_outside_t_share",
    "lag_over_t_share",
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
    df = df[df["position"] == "ED"].copy()
    df = df.sort_values(["player", "Year"])
    groups = df.groupby("player")

    # Coerce raw numeric columns
    numeric_cols = [
        "grades_defense", "grades_run_defense", "grades_pass_rush_defense",
        "pressures", "sacks", "hits", "hurries", "tackles", "missed_tackles",
        "snap_counts_defense", "adjusted_value", "Net EPA", "age",
        "total_pressures", "stops", "tackles_for_loss", "penalties",
        "snap_counts_dl", "snap_counts_dl_over_t", "snap_counts_dl_outside_t",
        "Cap_Space",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --------------------------------------------------
    # Feature engineering — mirrors Player_Model_ED.py exactly
    # --------------------------------------------------
    def safe_div(a, b):
        return np.where(b == 0, 0, a / b)

    snap    = df["snap_counts_defense"]
    snap_dl = df["snap_counts_dl"]

    df["years_in_league"]        = groups.cumcount()
    df["delta_grade"]            = groups["grades_defense"].diff().fillna(0)
    df["delta_pass_rush"]        = groups["grades_pass_rush_defense"].diff().fillna(0)
    df["delta_run_def"]          = groups["grades_run_defense"].diff().fillna(0)
    df["team_performance_proxy"] = df.groupby(["Team", "Year"])["Net EPA"].transform("mean")

    df["pressure_rate"]   = safe_div(df["total_pressures"],              snap)
    df["sack_rate"]       = safe_div(df["sacks"],                        snap)
    df["hit_rate"]        = safe_div(df["hits"],                         snap)
    df["hurry_rate"]      = safe_div(df["hurries"],                      snap)
    df["stop_rate"]       = safe_div(df["stops"],                        snap)
    df["tfl_rate"]        = safe_div(df["tackles_for_loss"],             snap)
    df["penalty_rate"]    = safe_div(df["penalties"],                    snap)
    df["outside_t_share"] = safe_div(df["snap_counts_dl_outside_t"],     snap_dl)
    df["over_t_share"]    = safe_div(df["snap_counts_dl_over_t"],        snap_dl)

    # Lag versions for XGB (single-row features derived from prior season)
    for col in ["grades_defense", "grades_pass_rush_defense", "grades_run_defense",
                "pressure_rate", "sack_rate", "stop_rate", "outside_t_share", "over_t_share",
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

    engine  = EDModelInference(MODEL_OUT, scaler_path=SCALER_OUT, xgb_path=XGB_MODEL_OUT)
    results = []

    for _, row in test_data.iterrows():
        name = row["player"]

        history = df_all[
            (df_all["player"] == name) &
            (df_all["Year"] < 2024)
        ].copy()

        if len(history) == 0:
            continue

        tier, details = engine.get_prediction(history)

        results.append({
            "player":           name,
            "Actual_Grade":     row[target_col],
            "Pred_XGB":         details["xgb_grade"],
            "Pred_Transformer": details["transformer_grade"],
            "Ensemble_Pred":    details["predicted_grade"],
            "Error":            row[target_col] - details["predicted_grade"],
            "Abs_Error":        abs(row[target_col] - details["predicted_grade"]),
        })

    final_df = pd.DataFrame(results)
    final_df = final_df.sort_values(by="Ensemble_Pred", ascending=False)
    print("\n=== ENSEMBLE RESULTS (2024) ===")
    print(f"Ensemble MAE:  {final_df['Abs_Error'].mean():.4f}")
    print(f"Bias:          {final_df['Error'].mean():.4f}")

    out_path = os.path.join(os.path.dirname(__file__), "ED_2024_Validation_Results.csv")
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

    engine = EDModelInference(MODEL_OUT, scaler_path=SCALER_OUT, xgb_path=XGB_MODEL_OUT)

    active_2024 = df_all[df_all["Year"] == 2024]
    rows        = []

    for _, row in active_2024.iterrows():
        player  = row["player"]
        history = df_all[df_all["player"] == player].sort_values("Year").tail(SEQ_LEN)

        if len(history) == 0:
            continue

        tier, details = engine.get_prediction(history)

        rows.append({
            "player":        player,
            "Tier":          tier,
            "Ensemble_Pred": details["predicted_grade"],
            "Age_2024":      row["age"],
        })

    final_2025 = pd.DataFrame(rows).sort_values("Ensemble_Pred", ascending=False)

    final_out = os.path.join(os.path.dirname(__file__), "ED_2025_Final_Rankings.csv")
    final_2025.to_csv(final_out, index=False)

    print("\nTop 15 ED for 2025:")
    print(final_2025.head(15).to_string(index=False))


print("\n=== ED MODEL PIPELINE COMPLETE ===")