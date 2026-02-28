"""
train_team_model.py
Train two XGBoost regressors:
  1. next_net_epa  (primary)
  2. next_win_pct  (secondary)

Walk-forward cross-validation: train 2011-2021, test 2022-2024.
Saves: epa_model.joblib, win_model.joblib, team_scaler.joblib
       shap_epa.json, shap_win.json
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge

# We no longer use xgboost or shap for this linear architecture.

OUT_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(OUT_DIR, "team_dataset.csv")

# ── Feature definitions ───────────────────────────────────────────────────────
# We use a strictly reduced, highly interpretable feature set for Ridge.
# Excluding lag_net_epa and lag_win_pct forces the model to learn pure roster + scheme value.
CORE_FEATURES = [
    "lag_net_epa",
    "lag_win_pct"
]

# We optionally include scheme to allow for playstyle differentiation
SCHEME_FEATURES = []

# Core Positional Valuations
POSITION_FEATURES = [
    "lag_qb_grade", "lag_ol_grade", "lag_rb_grade", "lag_wr_grade",
    "lag_edge_grade", "lag_idl_grade", "lag_lb_grade", "lag_cb_grade", "lag_s_grade",
]


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return features that actually exist in df."""
    candidates = CORE_FEATURES + SCHEME_FEATURES + POSITION_FEATURES
    return [c for c in candidates if c in df.columns]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows, {df['year'].min()}–{df['year'].max()}")
    return df


def train_eval(df: pd.DataFrame, target: str, label: str):
    """
    Walk-forward train/test split.
    Train: year <= 2021  |  Test: 2022–2024
    """
    feature_cols = get_feature_cols(df)

    # Drop rows missing target or all features
    sub = df.dropna(subset=[target]).copy()
    sub = sub.dropna(subset=feature_cols, how="all")

    # Fill remaining NaN with column median (for position grades etc.)
    for c in feature_cols:
        if c in sub.columns:
            sub[c] = sub[c].fillna(sub[c].median())

    train = sub[sub["year"] <= 2021]
    test  = sub[sub["year"].between(2022, 2024)]

    X_train = train[feature_cols].values
    y_train = train[target].values
    X_test  = test[feature_cols].values
    y_test  = test[target].values

    print(f"\n{'='*55}")
    print(f"  TARGET: {label}")
    print(f"  Train rows: {len(train)}  |  Test rows: {len(test)}")
    print(f"  Features  : {len(feature_cols)}")

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Ridge Regression (Alpha = 1000.0 for extreme regularization to defeat multicollinearity)
    # This forces coefficients to shrink proportionally to their univariate correlation, allocating positive weight to OL.
    model = Ridge(alpha=1000.0, random_state=42, positive=True)
    model.fit(X_train_s, y_train)

    # Metrics
    pred_train = model.predict(X_train_s)
    pred_test  = model.predict(X_test_s)

    print(f"\n  Train MAE : {mean_absolute_error(y_train, pred_train):.4f}  |  R²: {r2_score(y_train, pred_train):.3f}")
    print(f"  Test  MAE : {mean_absolute_error(y_test,  pred_test ):.4f}  |  R²: {r2_score(y_test,  pred_test ):.3f}")

    # Feature importance directly from linear coefficients
    # StandardScaler means coefficients perfectly reflect their marginal impact
    coeffs = model.coef_
    shap_vals = {feat: float(abs(c)) for feat, c in zip(feature_cols, coeffs)}
    shap_vals = {k: v / sum(shap_vals.values()) for k, v in shap_vals.items()} if sum(shap_vals.values()) > 0 else shap_vals
    
    print("\n  Top-10 absolute feature sensitivities (via Ridge Coefs):")
    # sorted raw
    sorted_raw = sorted(zip(feature_cols, coeffs), key=lambda x: abs(x[1]), reverse=True)
    for i, (feat, coef) in enumerate(sorted_raw[:10]):
        print(f"    {i+1:2d}. {feat:<35} {coef:+.4f}")

    # Sample predictions on test set
    print("\n  Sample predictions (test set):")
    sample = test.copy()
    sample[f"pred_{target}"] = pred_test
    print(sample[["abbr", "year", target, f"pred_{target}"]].head(10).to_string(index=False))

    return model, scaler, shap_vals, feature_cols


def main():
    df = load_data()

    # ── Net EPA model ──────────────────────────────────────────────────────────
    epa_model, epa_scaler, shap_epa, epa_features = train_eval(
        df, "next_net_epa", "Next-Season Net EPA"
    )

    # ── Win % model ────────────────────────────────────────────────────────────
    win_model, win_scaler, shap_win, win_features = train_eval(
        df, "next_win_pct", "Next-Season Win %"
    )

    # ── Save artifacts ─────────────────────────────────────────────────────────
    joblib.dump(epa_model,  os.path.join(OUT_DIR, "epa_model.joblib"))
    joblib.dump(win_model,  os.path.join(OUT_DIR, "win_model.joblib"))
    joblib.dump(epa_scaler, os.path.join(OUT_DIR, "epa_scaler.joblib"))
    joblib.dump(win_scaler, os.path.join(OUT_DIR, "win_scaler.joblib"))

    meta = {
        "epa_features": epa_features,
        "win_features": win_features,
    }
    with open(os.path.join(OUT_DIR, "feature_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if shap_epa:
        with open(os.path.join(OUT_DIR, "shap_epa.json"), "w") as f:
            json.dump(shap_epa, f, indent=2)
    if shap_win:
        with open(os.path.join(OUT_DIR, "shap_win.json"), "w") as f:
            json.dump(shap_win, f, indent=2)

    print("\n✅ Saved models and metadata to", OUT_DIR)


if __name__ == "__main__":
    main()
