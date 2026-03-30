import os
import sys
import torch
import pandas as pd
import numpy as np
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.ML.CB_Pranay_Transformers.Player_Model_CB import PlayerTransformerRegressor, SEQ_LEN


class CBModelInference:
    def __init__(self, transformer_path, scaler_path=None, xgb_path=None):
        self.device = torch.device('cpu')

        self.transformer_features = [
            "grades_defense", "grades_coverage_defense", "grades_tackle", "grades_defense_penalty",
            "qb_rating_against", "targets_per_snap", "yards_per_target", "reception_rate", "coverage_success_rate",
            "int_rate", "pbu_rate", "stop_rate", "missed_tackle_rate", "tfl_rate", "penalty_rate",
            "slot_share", "corner_share", "age", "years_in_league", "adjusted_value", "Cap_Space",
            "team_performance_proxy", "delta_grade", "delta_coverage", "delta_epa", "career_mean_grade",
        ]

        self.max_seq_len = SEQ_LEN
        self.model = PlayerTransformerRegressor(
            input_dim=len(self.transformer_features), seq_len=self.max_seq_len
        ).to(self.device)
        self.model.load_state_dict(torch.load(transformer_path, map_location=self.device))
        self.model.eval()

        self.scaler = None
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

        self.xgb_model = None
        if xgb_path and os.path.exists(xgb_path):
            self.xgb_model = joblib.load(xgb_path)

    def _prepare_features(self, player_history):
        df = player_history.copy()

        numeric_cols = [
            "grades_defense", "grades_coverage_defense", "grades_tackle", "grades_defense_penalty",
            "qb_rating_against", "interceptions", "pass_break_ups", "receptions", "targets", "yards",
            "tackles", "assists", "missed_tackles", "missed_tackle_rate", "stops", "snap_counts_defense",
            "snap_counts_corner", "snap_counts_coverage", "snap_counts_slot", "tackles_for_loss", "penalties",
            "forced_fumbles", "fumble_recoveries", "age", "Net EPA", "adjusted_value", "Cap_Space",
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        if "Year" in df.columns:
            df = df.sort_values("Year").reset_index(drop=True)

        df["years_in_league"] = range(len(df))
        df["delta_grade"] = df["grades_defense"].diff().fillna(0)
        df["delta_coverage"] = df["grades_coverage_defense"].diff().fillna(0)
        df["delta_epa"] = df["Net EPA"].diff().fillna(0)

        if "Team" in df.columns and "Net EPA" in df.columns:
            df["team_performance_proxy"] = df.groupby(["Team", "Year"])["Net EPA"].transform("mean")
        else:
            df["team_performance_proxy"] = 0.0

        def safe_div(num, den):
            den_safe = den.copy().astype(float)
            den_safe[den_safe == 0] = np.nan
            return (num / den_safe).fillna(0)

        snap = df.get("snap_counts_defense", pd.Series(0, index=df.index)).astype(float)
        df["targets_per_snap"] = safe_div(df.get("targets", 0).astype(float), snap)
        df["yards_per_target"] = safe_div(df.get("yards", 0).astype(float), df.get("targets", 0).astype(float))
        df["reception_rate"] = safe_div(df.get("receptions", 0).astype(float), df.get("targets", 0).astype(float))
        df["coverage_success_rate"] = safe_div(df.get("interceptions", 0).astype(float) + df.get("pass_break_ups", 0).astype(float), df.get("targets", 0).astype(float))
        df["int_rate"] = safe_div(df.get("interceptions", 0).astype(float), snap)
        df["pbu_rate"] = safe_div(df.get("pass_break_ups", 0).astype(float), snap)
        df["stop_rate"] = safe_div(df.get("stops", 0).astype(float), snap)
        df["missed_tackle_rate"] = safe_div(df.get("missed_tackles", 0).astype(float), snap)
        df["tfl_rate"] = safe_div(df.get("tackles_for_loss", 0).astype(float), snap)
        df["penalty_rate"] = safe_div(df.get("penalties", 0).astype(float), snap)
        df["slot_share"] = safe_div(df.get("snap_counts_slot", 0).astype(float), snap)
        df["corner_share"] = safe_div(df.get("snap_counts_corner", 0).astype(float), snap)

        df["career_mean_grade"] = df["grades_defense"].shift(1).expanding().mean().fillna(df["grades_defense"].mean())

        return df, pd.DataFrame()

    def predict(self, player_history, mode="ensemble", apply_calibration=True):
        if player_history.empty:
            return "No Data", {"error": "History is empty"}

        df_history, _ = self._prepare_features(player_history)
        p_history_tail = df_history.tail(self.max_seq_len)

        if self.scaler is None:
            raise FileNotFoundError("Scaler is required for CBModelInference but was not loaded.")

        history_vals = self.scaler.transform(p_history_tail[self.transformer_features])
        actual_len = len(history_vals)
        pad = np.zeros((self.max_seq_len - actual_len, len(self.transformer_features)))
        padded_x = np.vstack([pad, history_vals])
        mask = [True] * (self.max_seq_len - actual_len) + [False] * actual_len

        with torch.no_grad():
            x_tensor = torch.tensor(padded_x, dtype=torch.float32).unsqueeze(0)
            m_tensor = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
            transformer_grade = self.model(x_tensor, mask=m_tensor).item()

        final_grade = transformer_grade

        age_adjustment = 0.0
        if "age" in df_history.columns:
            current_age = float(df_history.iloc[-1]["age"])
            age_adjustment = self.get_age_decay_factor(current_age)
            final_grade -= age_adjustment

        tier = self.get_tier(final_grade)
        vol_score = self.get_volatility_score(df_history)

        return tier, {
            "predicted_grade": round(final_grade, 2),
            "transformer_grade": round(transformer_grade, 2),
            "xgb_grade": None,
            "age_adjustment": round(age_adjustment, 2),
            "volatility_index": round(vol_score, 3),
            "confidence_interval": self.get_confidence_interval(final_grade, vol_score),
        }

    def get_prediction(self, player_history, mode="ensemble", apply_calibration=True):
        return self.predict(player_history, mode=mode, apply_calibration=apply_calibration)

    def get_volatility_score(self, df_history):
        if len(df_history) < 2:
            return 0.5
        grade_std = df_history["grades_defense"].std()
        return min(1.0, grade_std / 15.0)

    def get_confidence_interval(self, grade, vol_score):
        base_mae = 8.0
        bound = base_mae * (1.0 + vol_score)
        return (round(grade - bound, 2), round(grade + bound, 2))

    def get_age_decay_factor(self, age):
        if age <= 28:
            return 0.0
        if age <= 33:
            return (age - 28) * 1.2
        return min((age - 33) * 2.0 + 6.0, 14.0)

    def get_tier(self, grade):
        if grade >= 80.0:
            return "Elite"
        elif grade >= 65.0:
            return "Starter"
        elif grade >= 55.0:
            return "Rotation"
        else:
            return "Reserve/Poor"
