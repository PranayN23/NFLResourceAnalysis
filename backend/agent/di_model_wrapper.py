import os
import sys
import torch
import pandas as pd
import numpy as np
import joblib

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.ML.DI_Pranay_Transformers.Player_Model_DI import (
    PlayerTransformerRegressor,
    SEQ_LEN,
)


class DIModelInference:
    """
    Inference wrapper for the DI ensemble (Transformer + XGBoost).

    Usage
    -----
    engine = DIModelInference(transformer_path, scaler_path, xgb_path)
    tier, details = engine.get_prediction(player_history_df)
    """

    # ============================================================
    # FEATURE LISTS — must stay in sync with Player_Model_DI.py
    # and di_ensemble.py. Any change to the training feature order
    # must be mirrored here.
    # ============================================================
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
        "a_gap_share",
        "b_gap_share",
        "over_t_share",
        "outside_t_share",
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
        "lag_a_gap_share",
        "lag_b_gap_share",
        "adjusted_value",
        "age",
        "years_in_league",
        "delta_grade_lag",
        "team_performance_proxy_lag",
    ]

    # ============================================================
    # POST-HOC VARIANCE CALIBRATION
    #
    # The transformer regresses toward the mean, compressing its output
    # to roughly 50–74 while actual PFF grades span 30–90. This stretch
    # calibration maps the raw output back to the true grade scale so
    # tier thresholds (80/70/60) remain semantically correct.
    #
    # Derived from 2024 validation set (n=170):
    #   pred  mean=58.4, std=6.1
    #   actual mean=57.1, std=13.0  →  ratio = 13.0/6.1 = 2.131
    #
    # Formula: calibrated = (raw - CALIB_PRED_MEAN) * CALIB_STD_RATIO + CALIB_ACTUAL_MEAN
    # ============================================================
    CALIB_PRED_MEAN   = 58.4
    CALIB_ACTUAL_MEAN = 57.1
    CALIB_STD_RATIO   = 2.131

    def __init__(self, transformer_path, scaler_path=None, xgb_path=None):
        self.device = torch.device("cpu")

        # Load Transformer — checkpoint must exist (run Player_Model_DI.py first)
        if not os.path.exists(transformer_path):
            raise FileNotFoundError(
                f"Transformer checkpoint not found: {transformer_path}\n"
                "Run Player_Model_DI.py first to generate the checkpoint."
            )
        self.model = PlayerTransformerRegressor(
            input_dim=len(self.TRANSFORMER_FEATURES),
            seq_len=SEQ_LEN,          # imported constant — always in sync
            num_layers=1,
            dropout=0.3,
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(transformer_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

        # Load scaler (required for Transformer inference)
        if scaler_path and not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Scaler not found: {scaler_path}\n"
                "Run Player_Model_DI.py first to generate the scaler."
            )
        self.scaler = joblib.load(scaler_path) if scaler_path else None

        # XGB is optional — graceful degradation to Transformer-only
        self.xgb_model = joblib.load(xgb_path) if xgb_path and os.path.exists(xgb_path) else None

    # ============================================================
    # INTERNAL: Feature Engineering
    # ============================================================
    def _prepare_features(self, player_history: pd.DataFrame):
        """
        Replicate the feature engineering from Player_Model_DI.py on a
        single player's history DataFrame, then build the XGB input row.

        Parameters
        ----------
        player_history : raw player history rows, sorted by Year ascending.

        Returns
        -------
        df       : engineered DataFrame ready for Transformer sequencing.
        df_xgb   : single-row DataFrame ready for XGB prediction.
        """
        df = player_history.copy()

        # Coerce all raw columns we'll derive from
        numeric_cols = [
            "grades_defense", "grades_run_defense", "grades_pass_rush_defense",
            "total_pressures", "sacks", "hits", "hurries", "stops",
            "tackles_for_loss", "penalties", "snap_counts_defense",
            "snap_counts_dl", "snap_counts_dl_a_gap", "snap_counts_dl_b_gap",
            "snap_counts_dl_over_t", "snap_counts_dl_outside_t",
            "adjusted_value", "Cap_Space", "Net EPA", "age",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        df = df.sort_values("Year").reset_index(drop=True)

        # Derived counters
        df["years_in_league"] = range(len(df))
        df["delta_grade"]     = df["grades_defense"].diff().fillna(0)
        df["delta_pass_rush"] = df["grades_pass_rush_defense"].diff().fillna(0)
        df["delta_run_def"]   = df["grades_run_defense"].diff().fillna(0)

        # Team proxy — may be a single-player slice so use mean of available rows
        if "Team" in df.columns and "Net EPA" in df.columns:
            df["team_performance_proxy"] = df["Net EPA"].mean()
        else:
            df["team_performance_proxy"] = 0.0

        # Per-snap rate features
        def safe_div(a, b):
            return np.where(b == 0, 0.0, a / b)

        snap    = df["snap_counts_defense"].values
        snap_dl = df["snap_counts_dl"].values

        df["pressure_rate"]   = safe_div(df["total_pressures"].values,          snap)
        df["sack_rate"]       = safe_div(df["sacks"].values,                    snap)
        df["hit_rate"]        = safe_div(df["hits"].values,                     snap)
        df["hurry_rate"]      = safe_div(df["hurries"].values,                  snap)
        df["stop_rate"]       = safe_div(df["stops"].values,                    snap)
        df["tfl_rate"]        = safe_div(df["tackles_for_loss"].values,         snap)
        df["penalty_rate"]    = safe_div(df["penalties"].values,                snap)
        df["a_gap_share"]     = safe_div(df["snap_counts_dl_a_gap"].values,     snap_dl)
        df["b_gap_share"]     = safe_div(df["snap_counts_dl_b_gap"].values,     snap_dl)
        df["over_t_share"]    = safe_div(df["snap_counts_dl_over_t"].values,    snap_dl)
        df["outside_t_share"] = safe_div(df["snap_counts_dl_outside_t"].values, snap_dl)

        # ------------------------------------------
        # XGB input: lag features from the last
        # available season, predicting the NEXT year
        # ------------------------------------------
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last

        xgb_input = {
            "lag_grades_defense":            float(last["grades_defense"]),
            "lag_grades_pass_rush_defense":  float(last["grades_pass_rush_defense"]),
            "lag_grades_run_defense":        float(last["grades_run_defense"]),
            "lag_pressure_rate":             float(last["pressure_rate"]),
            "lag_sack_rate":                 float(last["sack_rate"]),
            "lag_stop_rate":                 float(last["stop_rate"]),
            "lag_a_gap_share":               float(last["a_gap_share"]),
            "lag_b_gap_share":               float(last["b_gap_share"]),
            "adjusted_value":                float(last["adjusted_value"]),
            "age":                           float(last["age"]) + 1,   # predicting next season
            "years_in_league":               int(last["years_in_league"]) + 1,
            "delta_grade_lag":               float(last["grades_defense"]) - float(prev["grades_defense"]),
            "team_performance_proxy_lag":    float(last["team_performance_proxy"]),
        }

        return df, pd.DataFrame([xgb_input])

    # ============================================================
    # INTERNAL: Transformer inference on engineered history
    # ============================================================
    def _transformer_predict(self, df_engineered: pd.DataFrame) -> float:
        """Scale and pad the player's history, run the Transformer."""
        if self.scaler is None:
            raise RuntimeError("Scaler not loaded — cannot run Transformer inference.")

        history = df_engineered[self.TRANSFORMER_FEATURES].tail(SEQ_LEN)
        scaled  = self.scaler.transform(history)

        actual_len = len(scaled)
        pad_len    = SEQ_LEN - actual_len
        padded     = np.vstack([np.zeros((pad_len, len(self.TRANSFORMER_FEATURES))), scaled])
        mask       = [True] * pad_len + [False] * actual_len

        with torch.no_grad():
            x = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(self.device)
            m = torch.tensor(mask,   dtype=torch.bool).unsqueeze(0).to(self.device)
            return self.model(x, mask=m).item()

    # ============================================================
    # PUBLIC: get_prediction  (called by di_ensemble.py)
    # ============================================================
    def get_prediction(self, player_history: pd.DataFrame, mode: str = "ensemble"):
        """
        Main entry point. Returns (tier, details_dict).

        Parameters
        ----------
        player_history : DataFrame of the player's historical seasons (raw CSV rows).
        mode           : "ensemble" | "transformer" | "xgb"

        Returns
        -------
        tier    : str  — "Elite" | "Starter" | "Rotation" | "Reserve/Poor"
        details : dict — predicted_grade, transformer_grade, xgb_grade, age_adjustment
        """
        if player_history.empty:
            return "No Data", {"error": "History is empty"}

        df_eng, df_xgb = self._prepare_features(player_history)

        # Transformer grade (raw, compressed to ~50-74)
        transformer_grade = self._transformer_predict(df_eng)

        # Stretch-calibrate back to PFF grade scale (30-90)
        # Raw output regresses to mean — this restores the variance
        transformer_grade = (
            (transformer_grade - self.CALIB_PRED_MEAN)
            * self.CALIB_STD_RATIO
            + self.CALIB_ACTUAL_MEAN
        )

        # XGB grade
        xgb_grade = None
        if self.xgb_model is not None:
            xgb_grade = float(self.xgb_model.predict(df_xgb[self.XGB_FEATURES])[0])

        # Blend
        if mode == "transformer" or xgb_grade is None:
            final_grade = transformer_grade
        elif mode == "xgb":
            final_grade = xgb_grade
        else:
            final_grade = 0.5 * transformer_grade + 0.5 * xgb_grade

        # Age adjustment (DI ages slower than skill positions)
        current_age    = float(df_eng.iloc[-1]["age"])
        age_adjustment = self._age_decay(current_age)
        final_grade   -= age_adjustment

        tier = self._tier(final_grade)

        return tier, {
            "predicted_grade":   round(final_grade,       2),
            "transformer_grade": round(transformer_grade, 2),
            "xgb_grade":         round(xgb_grade, 2) if xgb_grade is not None else None,
            "age_adjustment":    round(age_adjustment,    2),
        }

    # ============================================================
    # Age Decay Curve (DI-specific)
    # ============================================================
    @staticmethod
    def _age_decay(age: float) -> float:
        """
        Conservative decay: DI linemen peak later and decline more
        gradually than skill positions.

          ≤27   : no penalty
          28-30 : +0.8 pts/year
          31-33 : +1.2 pts/year (cumulative from 27)
          >33   : +1.5 pts/year, capped at 12.0
        """
        if age <= 27:
            return 0.0
        elif age <= 30:
            return (age - 27) * 0.8
        elif age <= 33:
            return 2.4 + (age - 30) * 1.2
        else:
            return min(6.0 + (age - 33) * 1.5, 12.0)

    # ============================================================
    # Tier Classification
    # ============================================================
    @staticmethod
    def _tier(grade: float) -> str:
        if grade >= 80:
            return "Elite"
        elif grade >= 70:
            return "Starter"
        elif grade >= 60:
            return "Rotation"
        else:
            return "Reserve/Poor"

    # ============================================================
    # Legacy alias — keeps any existing callers working
    # ============================================================
    def predict(self, player_history: pd.DataFrame, mode: str = "ensemble", **kwargs):
        return self.get_prediction(player_history, mode=mode)