import os
import torch
import pandas as pd
import numpy as np
import joblib
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.ML.QBtransformers.Player_Model_QB import PlayerTransformerRegressor

class PlayerModelInference:
    def __init__(self, transformer_path, scaler_path=None, xgb_path=None):
        self.device = torch.device('cpu')
        
        # Original features used in Player_Model_QB.py
        self.transformer_features = [
            'grades_pass', 'grades_offense', 'qb_rating', 'adjusted_value',
            'Cap_Space', 'ypa', 'twp_rate', 'btt_rate', 'completion_percent',
            'years_in_league', 'delta_grade', 'delta_epa', 'delta_btt',
            'team_performance_proxy', 'dropbacks'
        ]
        
        # XGBoost features (Strictly Lagged)
        self.xgb_features = [
            'lag_grades_offense', 'lag_Net_EPA', 'lag_btt_rate', 'lag_twp_rate',
            'lag_qb_rating', 'lag_ypa', 'adjusted_value', 'years_in_league',
            'delta_grade_lag', 'team_performance_proxy_lag', 'lag_dropbacks'
        ]
        
        self.max_seq_len = 5
        self.model = PlayerTransformerRegressor(input_dim=len(self.transformer_features), seq_len=self.max_seq_len).to(self.device)
        self.model.load_state_dict(torch.load(transformer_path, map_location=self.device))
        self.model.eval()

        self.scaler = None
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            
        self.xgb_model = None
        if xgb_path and os.path.exists(xgb_path):
            self.xgb_model = joblib.load(xgb_path)

    def _prepare_features(self, player_history):
        """Prepare both original and lagged features for a player's history."""
        df = player_history.copy()
        df['adjusted_value'] = pd.to_numeric(df['adjusted_value'], errors='coerce').fillna(0)
        df = df.sort_values('Year')
        
        # Engineering (Original Logic)
        df["years_in_league"] = range(len(df))
        df["delta_grade"] = df["grades_offense"].diff().fillna(0)
        df["delta_epa"]   = df["Net EPA"].diff().fillna(0)
        df["delta_btt"]   = df["btt_rate"].diff().fillna(0)
        df['team_performance_proxy'] = df.groupby(['Team', 'Year'])['Net EPA'].transform('mean')
        
        # Lagged Engineering for XGBoost (Predicting T using T-1)
        # Assuming the LAST row in player_history is Year T-1
        row_last = df.iloc[-1]
        row_prev = df.iloc[-2] if len(df) > 1 else row_last
        
        xgb_input = {
            'lag_grades_offense': row_last['grades_offense'],
            'lag_Net_EPA': row_last['Net EPA'],
            'lag_btt_rate': row_last['btt_rate'],
            'lag_twp_rate': row_last['twp_rate'],
            'lag_qb_rating': row_last['qb_rating'],
            'lag_ypa': row_last['ypa'],
            'adjusted_value': row_last['adjusted_value'],
            'years_in_league': row_last['years_in_league'] + 1,
            'delta_grade_lag': row_last['grades_offense'] - row_prev['grades_offense'],
            'team_performance_proxy_lag': row_last['team_performance_proxy'],
            'lag_dropbacks': row_last['dropbacks']
        }
        
        return df, pd.DataFrame([xgb_input])

    def predict(self, player_history, mode="ensemble", apply_calibration=True):
        """
        Predict performance for the NEXT year based on player_history.
        - mode: 'transformer', 'xgb', or 'ensemble'
        - apply_calibration: Whether to apply the volatility-aware bias reduction
        """
        if player_history.empty:
            return "No Data", {"error": "History is empty"}

        df_history, df_xgb = self._prepare_features(player_history)
        
        # 1. Transformer Prediction (Uses Sequence)
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

        # 2. XGBoost Prediction
        xgb_grade = 0.0
        if self.xgb_model:
            xgb_grade = self.xgb_model.predict(df_xgb[self.xgb_features])[0]

        # 3. BASE ENSEMBLE CALCULATION (Optimized Weights via Grid Search)
        # Grid search on 2014-2024 data found optimal: 87% XGB, 13% Transformer
        # Using 65/35 for balanced approach
        xgb_weight = 0.65
        trans_weight = 0.35
        
        if mode == "transformer":
            final_grade = transformer_grade
        elif mode == "xgb":
            final_grade = xgb_grade
        else:
            final_grade = (transformer_grade * trans_weight) + (xgb_grade * xgb_weight)

        # 4. AGE-AWARE DECAY (Post-Processing)
        age_adjustment = 0.0
        if 'age' in df_history.columns:
            current_age = df_history.iloc[-1]['age']
            age_adjustment = self.get_age_decay_factor(current_age)
            final_grade -= age_adjustment

        tier = self.get_tier(final_grade)
        
        # 5. RISK METADATA
        vol_score = self.get_volatility_score(df_history)
        conf_interval = self.get_confidence_interval(final_grade, vol_score)

        return tier, {
            "predicted_grade": round(final_grade, 2),
            "transformer_grade": round(transformer_grade, 2),
            "xgb_grade": round(xgb_grade, 2) if self.xgb_model else None,
            "age_adjustment": round(age_adjustment, 2),
            "volatility_index": round(vol_score, 3),
            "confidence_interval": conf_interval
        }

    def get_prediction(self, player_history, mode="ensemble", apply_calibration=True):
        """Alias for predict() to maintain backward compatibility."""
        return self.predict(player_history, mode=mode, apply_calibration=apply_calibration)

    def get_volatility_score(self, df_history):
        """Formal 0-1 scale of player performance unpredictability."""
        if len(df_history) < 2: return 0.5
        std = df_history['grades_offense'].std()
        # Scale 0-15 std range to 0-1 index
        return min(1.0, std / 15.0)

    def get_confidence_interval(self, grade, vol_score):
        """Calculate +/- bounds based on historical cohort MAE and player volatility."""
        base_mae = 6.6 # From 2024 validation
        # Inflate interval for volatile players
        bound = base_mae * (1.0 + vol_score)
        return (round(grade - bound, 2), round(grade + bound, 2))

    def get_age_decay_factor(self, age):
        """
        Empirical age-based performance adjustment to address survivorship bias.
        - Peak years (25-32): No adjustment
        - Decline years (33-40): Gradual decay (~0.8 pts/year)
        - Veteran years (41+): Gentle additional decay (capped at 10 pts total)
        """
        if age <= 32:
            return 0.0  # Prime years
        elif age <= 40:
            return (age - 32) * 0.8  # Gradual decline
        else:
            # Age 41+: slower decay with cap
            base_penalty = (40 - 32) * 0.8  # 6.4 pts at age 40
            additional = (age - 40) * 0.5   # +0.5 pts per year after 40
            return min(base_penalty + additional, 10.0)  # Cap at 10 pts max

    def get_tier(self, grade):
        if grade >= 80.0: return "Elite"
        elif grade >= 60.0: return "Starter"
        else: return "Reserve/Poor"
