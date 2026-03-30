import os
import torch
import pandas as pd
import numpy as np
import joblib
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.ML.WR_Pranay_Transformers.Player_Model_WR import PlayerTransformerRegressor


class WRModelInference:
    def __init__(self, transformer_path, scaler_path=None, xgb_path=None):
        self.device = torch.device('cpu')

        # WR Transformer features — must match Player_Model_WR.py exactly
        self.transformer_features = [
            "grades_offense", "grades_pass_route", "grades_hands_drop",
            "avg_depth_of_target", "caught_percent", "drop_rate",
            "yards", "yards_after_catch", "yards_per_reception", "yprr",
            "receptions", "targets", "touchdowns", "first_downs",
            "targeted_qb_rating", "route_rate",
            "slot_share", "wide_share", "target_share", "contested_rate",
            "age", "years_in_league", "adjusted_value", "Cap_Space",
            "team_performance_proxy",
            "delta_grade", "delta_yprr", "delta_adot",
            "career_mean_grade",
        ]

        # XGBoost features
        self.xgb_features = [
            "lag_grades_offense", "lag_grades_pass_route", "lag_yprr",
            "lag_avg_depth_of_target", "lag_caught_percent", "lag_drop_rate",
            "lag_yards", "lag_targets", "lag_touchdowns", "lag_yards_after_catch",
            "lag_targeted_qb_rating", "lag_slot_share", "lag_wide_share",
            "adjusted_value", "age", "years_in_league",
            "delta_grade_lag", "team_performance_proxy_lag",
        ]

        from backend.ML.WR_Pranay_Transformers.Player_Model_WR import SEQ_LEN
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
        """Prepare both transformer sequence and XGBoost lag features."""
        df = player_history.copy()

        numeric_cols = [
            "grades_offense", "grades_pass_route", "grades_hands_drop",
            "avg_depth_of_target", "caught_percent", "drop_rate",
            "yards", "yards_after_catch", "yards_per_reception", "yprr",
            "receptions", "targets", "touchdowns", "first_downs",
            "targeted_qb_rating", "route_rate", "routes",
            "slot_snaps", "wide_snaps", "inline_snaps", "total_snaps",
            "contested_targets", "contested_receptions",
            "age", "adjusted_value", "Cap_Space", "Net EPA",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.sort_values("Year")

        # Engineering
        df["years_in_league"] = range(len(df))
        df["delta_grade"]     = df["grades_offense"].diff().fillna(0)
        df["delta_yprr"]      = df["yprr"].diff().fillna(0)
        df["delta_adot"]      = df["avg_depth_of_target"].diff().fillna(0)
        df["team_performance_proxy"] = df.groupby(["Team", "Year"])["Net EPA"].transform("mean")

        total = df["slot_snaps"].fillna(0) + df["wide_snaps"].fillna(0) + df["inline_snaps"].fillna(0)
        total = total.replace(0, np.nan)
        df["slot_share"] = (df["slot_snaps"] / total).fillna(0)
        df["wide_share"] = (df["wide_snaps"] / total).fillna(0)

        df["target_share"] = df["targets"] / df["routes"].replace(0, np.nan)
        df["target_share"] = df["target_share"].fillna(0)
        df["contested_rate"] = df["contested_targets"] / df["targets"].replace(0, np.nan)
        df["contested_rate"] = df["contested_rate"].fillna(0)

        df["career_mean_grade"] = df["grades_offense"].expanding().mean()

        # XGB lagged row
        row_last = df.iloc[-1]
        row_prev = df.iloc[-2] if len(df) > 1 else row_last

        xgb_input = {
            "lag_grades_offense":         float(row_last["grades_offense"]),
            "lag_grades_pass_route":      float(row_last["grades_pass_route"]),
            "lag_yprr":                   float(row_last["yprr"]),
            "lag_avg_depth_of_target":    float(row_last["avg_depth_of_target"]),
            "lag_caught_percent":         float(row_last["caught_percent"]),
            "lag_drop_rate":              float(row_last["drop_rate"]),
            "lag_yards":                  float(row_last["yards"]),
            "lag_targets":                float(row_last["targets"]),
            "lag_touchdowns":             float(row_last["touchdowns"]),
            "lag_yards_after_catch":      float(row_last["yards_after_catch"]),
            "lag_targeted_qb_rating":     float(row_last["targeted_qb_rating"]),
            "lag_slot_share":             float(row_last["slot_share"]),
            "lag_wide_share":             float(row_last["wide_share"]),
            "adjusted_value":             float(row_last["adjusted_value"]),
            "age":                        float(row_last["age"]) + 1,
            "years_in_league":            int(row_last["years_in_league"]) + 1,
            "delta_grade_lag":            float(row_last["grades_offense"]) - float(row_prev["grades_offense"]),
            "team_performance_proxy_lag": float(row_last["team_performance_proxy"]),
        }

        return df, pd.DataFrame([xgb_input])

    def predict(self, player_history, mode="ensemble", apply_calibration=True):
        if player_history.empty:
            return "No Data", {"error": "History is empty"}

        df_history, df_xgb = self._prepare_features(player_history)

        # Transformer
        p_history_tail = df_history.tail(self.max_seq_len)
        history_vals = self.scaler.transform(p_history_tail[self.transformer_features])

        actual_len = len(history_vals)
        pad = np.zeros((self.max_seq_len - actual_len, len(self.transformer_features)))
        padded_x = np.vstack([pad, history_vals])
        mask = [True] * (self.max_seq_len - actual_len) + [False] * actual_len

        with torch.no_grad():
            x_tensor = torch.tensor(padded_x, dtype=torch.float32).unsqueeze(0)
            m_tensor = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
            transformer_grade = self.model(x_tensor, mask=m_tensor).item()

        # XGBoost
        xgb_grade = 0.0
        if self.xgb_model:
            xgb_grade = self.xgb_model.predict(df_xgb[self.xgb_features])[0]

        # Ensemble (55% XGB, 45% Transformer)
        xgb_weight   = 0.55
        trans_weight = 0.45

        if mode == "transformer":
            final_grade = transformer_grade
        elif mode == "xgb":
            final_grade = xgb_grade
        else:
            final_grade = (transformer_grade * trans_weight) + (xgb_grade * xgb_weight)

        # Age decay (WRs peak ~25-29)
        age_adjustment = 0.0
        if "age" in df_history.columns:
            current_age = float(df_history.iloc[-1]["age"])
            age_adjustment = self.get_age_decay_factor(current_age)
            final_grade -= age_adjustment

        tier = self.get_tier(final_grade)
        vol_score = self.get_volatility_score(df_history)
        conf_interval = self.get_confidence_interval(final_grade, vol_score)

        return tier, {
            "predicted_grade":    round(final_grade, 2),
            "transformer_grade":  round(transformer_grade, 2),
            "xgb_grade":          round(xgb_grade, 2) if self.xgb_model else None,
            "age_adjustment":     round(age_adjustment, 2),
            "volatility_index":   round(vol_score, 3),
            "confidence_interval": conf_interval,
        }

    def get_prediction(self, player_history, mode="ensemble", apply_calibration=True):
        return self.predict(player_history, mode=mode, apply_calibration=apply_calibration)

    def get_volatility_score(self, df_history):
        if len(df_history) < 2:
            return 0.5
        grade_std = df_history["grades_offense"].std()
        return min(1.0, grade_std / 15.0)

    def get_confidence_interval(self, grade, vol_score):
        base_mae = 7.5  # Typical WR MAE
        bound = base_mae * (1.0 + vol_score)
        return (round(grade - bound, 2), round(grade + bound, 2))

    def get_age_decay_factor(self, age):
        if age <= 29:
            return 0.0
        elif age <= 33:
            return (age - 29) * 1.0
        else:
            base = (33 - 29) * 1.0
            additional = (age - 33) * 2.0
            return min(base + additional, 12.0)

    def get_tier(self, grade):
        if grade >= 78.0:
            return "Elite"
        elif grade >= 65.0:
            return "Starter"
        elif grade >= 55.0:
            return "Rotation"
        else:
            return "Reserve/Poor"
