import os
import torch
import pandas as pd
import numpy as np
import joblib
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.ML.RB_Pranay_Transformers.Player_Model_RB import PlayerTransformerRegressor

class RBModelInference:
    def __init__(self, transformer_path, scaler_path=None, xgb_path=None):
        self.device = torch.device('cpu')
        
        # RB Transformer features
        self.transformer_features = [
            'grades_offense', 'grades_run', 'grades_pass_route', 'elusive_rating',
            'yards', 'yards_after_contact', 'yco_attempt', 'breakaway_percent',
            'explosive', 'first_downs', 'receptions', 'targets', 'total_touches',
            'touchdowns', 'adjusted_value', 'Cap_Space', 'age', 'years_in_league',
            'delta_grade', 'delta_yards', 'delta_touches', 'team_performance_proxy'
        ]
        
        # XGBoost features
        self.xgb_base_features = [
            'lag_grades_offense', 'lag_yards', 'lag_yco_attempt', 'lag_elusive_rating',
            'lag_breakaway_percent', 'lag_explosive', 'lag_total_touches', 'lag_touchdowns',
            'adjusted_value', 'age', 'years_in_league', 'delta_grade_lag',
            'team_performance_proxy_lag', 'lag_receptions'
        ]
        self.t2v_signal_feature = 't2v_transformer_signal'
        self.xgb_features = self.xgb_base_features + [self.t2v_signal_feature]
        
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

    def _build_transformer_tensors(self, df_history):
        p_history_tail = df_history.tail(self.max_seq_len).copy()

        for col in self.transformer_features:
            if col not in p_history_tail.columns:
                p_history_tail[col] = 0.0

        p_history_tail[self.transformer_features] = (
            p_history_tail[self.transformer_features]
            .apply(pd.to_numeric, errors='coerce')
            .fillna(0.0)
        )

        feature_frame = p_history_tail[self.transformer_features]
        feature_vals = feature_frame.values
        if self.scaler is not None:
            feature_vals = self.scaler.transform(feature_frame)

        actual_len = len(feature_vals)
        pad = np.zeros((self.max_seq_len - actual_len, len(self.transformer_features)))
        padded_x = np.vstack([pad, feature_vals])
        mask = [True] * (self.max_seq_len - actual_len) + [False] * actual_len

        x_tensor = torch.tensor(padded_x, dtype=torch.float32).unsqueeze(0)
        m_tensor = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
        return x_tensor, m_tensor

    def _compute_transformer_signal(self, df_history):
        with torch.no_grad():
            x_tensor, m_tensor = self._build_transformer_tensors(df_history)
            return self.model(x_tensor, mask=m_tensor).item()

    def _prepare_features(self, player_history):
        """Prepare both original and lagged features for a player's history."""
        df = player_history.copy()
        df['adjusted_value'] = pd.to_numeric(df['adjusted_value'], errors='coerce').fillna(0)
        
        # Convert age and other numeric columns that might be strings
        numeric_cols = ['age', 'yards', 'yco_attempt', 'elusive_rating', 'breakaway_percent', 
                        'explosive', 'total_touches', 'touchdowns', 'receptions', 
                        'yards_after_contact', 'first_downs', 'targets', 'grades_offense',
                        'grades_run', 'grades_pass_route', 'Cap_Space', 'Net EPA']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.sort_values('Year')
        
        # Engineering
        df["years_in_league"] = range(len(df))
        df["delta_grade"] = df["grades_offense"].diff().fillna(0)
        df["delta_yards"] = df["yards"].diff().fillna(0)
        df["delta_touches"] = df["total_touches"].diff().fillna(0)
        df['team_performance_proxy'] = df.groupby(['Team', 'Year'])['Net EPA'].transform('mean')
        
        # Lagged Engineering for XGBoost
        row_last = df.iloc[-1]
        row_prev = df.iloc[-2] if len(df) > 1 else row_last
        transformer_signal = self._compute_transformer_signal(df)
        
        xgb_input = {
            'lag_grades_offense': float(row_last['grades_offense']),
            'lag_yards': float(row_last['yards']),
            'lag_yco_attempt': float(row_last['yco_attempt']),
            'lag_elusive_rating': float(row_last['elusive_rating']),
            'lag_breakaway_percent': float(row_last['breakaway_percent']),
            'lag_explosive': float(row_last['explosive']),
            'lag_total_touches': float(row_last['total_touches']),
            'lag_touchdowns': float(row_last['touchdowns']),
            'adjusted_value': float(row_last['adjusted_value']),
            'age': float(row_last['age']) + 1,
            'years_in_league': int(row_last['years_in_league']) + 1,
            'delta_grade_lag': float(row_last['grades_offense']) - float(row_prev['grades_offense']),
            'team_performance_proxy_lag': float(row_last['team_performance_proxy']),
            'lag_receptions': float(row_last['receptions']),
            self.t2v_signal_feature: float(transformer_signal)
        }
        
        return df, pd.DataFrame([xgb_input])

    def predict(self, player_history, mode="ensemble", apply_calibration=True):
        if player_history.empty:
            return "No Data", {"error": "History is empty"}

        df_history, df_xgb = self._prepare_features(player_history)
        
        # Transformer (includes Time2Vec representation internally)
        transformer_grade = self._compute_transformer_signal(df_history)

        # XGBoost
        xgb_grade = 0.0
        if self.xgb_model:
            xgb_columns = self.xgb_features
            if hasattr(self.xgb_model, "feature_names_in_"):
                xgb_columns = list(self.xgb_model.feature_names_in_)

            for col in xgb_columns:
                if col not in df_xgb.columns:
                    df_xgb[col] = 0.0

            xgb_grade = self.xgb_model.predict(df_xgb[xgb_columns])[0]

        # Ensemble (50/50)
        xgb_weight = 0.50
        trans_weight = 0.50
        
        if mode == "transformer":
            final_grade = transformer_grade
        elif mode == "xgb":
            final_grade = xgb_grade
        else:
            final_grade = (transformer_grade * trans_weight) + (xgb_grade * xgb_weight)

        # Age decay
        age_adjustment = 0.0
        if 'age' in df_history.columns:
            current_age = float(df_history.iloc[-1]['age'])
            age_adjustment = self.get_age_decay_factor(current_age)
            final_grade -= age_adjustment

        tier = self.get_tier(final_grade)
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
        return self.predict(player_history, mode=mode, apply_calibration=apply_calibration)

    def get_volatility_score(self, df_history):
        if len(df_history) < 2: 
            return 0.6
        
        grade_std = df_history['grades_offense'].std()
        grade_volatility = min(1.0, grade_std / 18.0)
        
        if 'total_touches' in df_history.columns:
            touches_std = df_history['total_touches'].std()
            workload_volatility = min(1.0, touches_std / 150.0)
            return min(1.0, (grade_volatility * 0.7 + workload_volatility * 0.3))
        
        return grade_volatility

    def get_confidence_interval(self, grade, vol_score):
        base_mae = 8.5
        bound = base_mae * (1.0 + vol_score * 1.2)
        return (round(grade - bound, 2), round(grade + bound, 2))

    def get_age_decay_factor(self, age):
        if age <= 26:
            return 0.0
        elif age <= 28:
            return (age - 26) * 1.5
        elif age <= 31:
            base_penalty = 2 * 1.5
            additional = (age - 28) * 2.5
            return base_penalty + additional
        else:
            base_penalty = 2 * 1.5 + 3 * 2.5
            additional = (age - 31) * 3.5
            return min(base_penalty + additional, 20.0)

    def get_tier(self, grade):
        if grade >= 75.0:
            return "Elite"
        elif grade >= 65.0:
            return "Starter"
        elif grade >= 55.0:
            return "Rotation"
        else:
            return "Reserve/Poor"