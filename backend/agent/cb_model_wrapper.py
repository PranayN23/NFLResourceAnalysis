import os
import torch
import pandas as pd
import numpy as np
import joblib
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.ML.CB_Transformers.Player_Model_CB import CBTransformerRegressor

class CBModelInference:
    def __init__(self, transformer_path, scaler_path=None):
        self.device = torch.device('cpu')

        self.transformer_features = [
            'grades_defense', 'grades_coverage_defense', 'grades_tackle',
            'qb_rating_against', 'pass_break_ups', 'interceptions',
            'targets', 'snap_counts_corner', 'snap_counts_coverage',
            'snap_counts_slot', 'adjusted_value', 'Cap_Space',
            'snap_counts_defense', 'years_in_league', 'delta_grade'
        ]

        self.max_seq_len = 5
        self.model = CBTransformerRegressor(input_dim=len(self.transformer_features), seq_len=self.max_seq_len).to(self.device)
        self.model.load_state_dict(torch.load(transformer_path, map_location=self.device, weights_only=True))
        self.model.eval()

        self.scaler = None
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

    def _prepare_features(self, player_history):
        df = player_history.copy()
        df['adjusted_value'] = pd.to_numeric(df['adjusted_value'], errors='coerce').fillna(0)
        numeric_cols = [
            'grades_defense', 'grades_coverage_defense', 'grades_tackle',
            'qb_rating_against', 'pass_break_ups', 'interceptions',
            'targets', 'snap_counts_corner', 'snap_counts_coverage',
            'snap_counts_slot', 'snap_counts_defense', 'Cap_Space',
            'tackles', 'stops', 'missed_tackle_rate', 'age'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.sort_values('Year')
        df["years_in_league"] = range(len(df))
        df["delta_grade"] = df["grades_defense"].diff().fillna(0)
        return df

    def predict(self, player_history, apply_calibration=True):
        if player_history.empty:
            return "No Data", {"error": "History is empty"}

        df_history = self._prepare_features(player_history)

        p_history_tail = df_history.tail(self.max_seq_len)
        history_vals = self.scaler.transform(p_history_tail[self.transformer_features])

        actual_len = len(history_vals)
        pad = np.zeros((self.max_seq_len - actual_len, len(self.transformer_features)))
        padded_x = np.vstack([pad, history_vals])
        mask = [True] * (self.max_seq_len - actual_len) + [False] * actual_len

        # Last known grade for delta reconstruction
        last_grade = df_history.iloc[-1]['grades_defense']

        with torch.no_grad():
            x_tensor = torch.tensor(padded_x, dtype=torch.float32).unsqueeze(0)
            m_tensor = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
            predicted_delta = self.model(x_tensor, mask=m_tensor).item()

        # Reconstruct absolute grade: last_grade + predicted_delta
        final_grade = last_grade + predicted_delta

        # Age-aware decay — CBs peak early (mid-20s) and decline by ~30
        age_adjustment = 0.0
        if 'age' in df_history.columns:
            current_age = df_history.iloc[-1]['age']
            if pd.notna(current_age):
                age_adjustment = self.get_age_decay_factor(current_age)
                final_grade -= age_adjustment

        tier = self.get_tier(final_grade)
        vol_score = self.get_volatility_score(df_history)
        conf_interval = self.get_confidence_interval(final_grade, vol_score)

        return tier, {
            "predicted_grade": round(final_grade, 2),
            "transformer_grade": round(last_grade + predicted_delta, 2),
            "age_adjustment": round(age_adjustment, 2),
            "volatility_index": round(vol_score, 3),
            "confidence_interval": conf_interval
        }

    def get_prediction(self, player_history, apply_calibration=True):
        return self.predict(player_history, apply_calibration=apply_calibration)

    def get_volatility_score(self, df_history):
        if len(df_history) < 2: return 0.5
        std = df_history['grades_defense'].std()
        return min(1.0, std / 15.0)

    def get_confidence_interval(self, grade, vol_score):
        base_mae = 8.0
        bound = base_mae * (1.0 + vol_score)
        return (round(grade - bound, 2), round(grade + bound, 2))

    def get_age_decay_factor(self, age):
        """
        CBs peak in their mid-20s and decline earlier than most positions.
        - Peak years (<=29): No adjustment
        - Decline years (30-34): Gradual decay (~1.0 pts/year)
        - Late career (35+): Steeper decay (capped at 10 pts total)
        """
        if age <= 29:
            return 0.0
        elif age <= 34:
            return (age - 29) * 1.0
        else:
            base_penalty = (34 - 29) * 1.0
            additional = (age - 34) * 0.5
            return min(base_penalty + additional, 10.0)

    def get_tier(self, grade):
        if grade >= 80.0: return "Elite"
        elif grade >= 60.0: return "Starter"
        else: return "Reserve/Poor"
