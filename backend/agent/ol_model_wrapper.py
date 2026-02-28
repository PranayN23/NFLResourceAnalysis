import os
import torch
import pandas as pd
import numpy as np
import joblib
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.ML.OL_Pranay_Transformers.Player_Model_OL import PlayerTransformerRegressor

class OLModelInference:
    def __init__(self, transformer_path, scaler_path=None, xgb_path=None):
        self.device = torch.device('cpu')

        # OL Transformer features
        self.transformer_features = [
            'adjusted_value', 'Cap_Space', 'age', 'years_in_league',
            'delta_grade', 'delta_run_block', 'delta_pass_block', 'team_performance_proxy',
            'sacks_allowed_rate', 'hits_allowed_rate', 'hurries_allowed_rate',
            'pressures_allowed_rate', 'penalties_rate', 'pass_block_efficiency',
            'snap_counts_block_share', 'snap_counts_run_block_share', 'snap_counts_pass_block_share',
            'pos_T', 'pos_G', 'pos_C'
        ]

        # XGBoost features (lagged)
        self.xgb_base_features = [
            'lag_grades_offense', 'lag_grades_run_block', 'lag_grades_pass_block',
            'adjusted_value', 'age', 'years_in_league',
            'delta_grade_lag', 'team_performance_proxy_lag',
            'sacks_allowed_rate', 'hits_allowed_rate', 'hurries_allowed_rate'
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

    def get_prediction(self, player_history, apply_calibration=True, mode="ensemble"):
        return self.predict(player_history, mode=mode, apply_calibration=apply_calibration)


    def _prepare_features(self, player_history):
        df = player_history.copy()

        # Convert numeric columns
        numeric_cols = self.transformer_features + ['grades_offense', 'grades_run_block', 'grades_pass_block', 'Net EPA']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        df = df.sort_values('Year')

        # Engineering
        df["years_in_league"] = range(len(df))
        df["delta_grade"] = df["grades_offense"].diff().fillna(0)
        df["delta_run_block"] = df["grades_run_block"].diff().fillna(0)
        df["delta_pass_block"] = df["grades_pass_block"].diff().fillna(0)
        df['team_performance_proxy'] = df.groupby(['Team', 'Year'])['Net EPA'].transform('mean')

        # Lagged for XGBoost
        row_last = df.iloc[-1]
        row_prev = df.iloc[-2] if len(df) > 1 else row_last
        transformer_signal = self._compute_transformer_signal(df)

        xgb_input = {
            'lag_grades_offense': float(row_prev['grades_offense']),          # last year's grade
            'lag_grades_run_block': float(row_prev['grades_run_block']),      # last year's grade
            'lag_grades_pass_block': float(row_prev['grades_pass_block']),    # last year's grade
            'adjusted_value': float(row_last['adjusted_value']),
            'age': float(row_last['age']) + 1,
            'years_in_league': int(row_last['years_in_league']) + 1,
            'delta_grade_lag': float(row_last['grades_offense'] - row_prev['grades_offense']),
            'team_performance_proxy_lag': float(row_last['team_performance_proxy']),
            'sacks_allowed_rate': float(row_last.get('sacks_allowed_rate', 0)),
            'hits_allowed_rate': float(row_last.get('hits_allowed_rate', 0)),
            'hurries_allowed_rate': float(row_last.get('hurries_allowed_rate', 0)),
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

        # Ensemble
        trans_weight, xgb_weight = 0.0, 1.0  # Optimized: pure XGBoost performs best (0.4536 vs 0.4481)
        if mode == "transformer":
            final_grade = transformer_grade
        elif mode == "xgb":
            final_grade = xgb_grade
        else:
            final_grade = transformer_grade * trans_weight + xgb_grade * xgb_weight

        # Age adjustment
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

    def get_volatility_score(self, df_history):
        if len(df_history) < 2: 
            return 0.6
        
        grade_std = df_history['grades_offense'].std()
        grade_volatility = min(1.0, grade_std / 18.0)
        
        if 'snap_counts_offense' in df_history.columns:
            snaps_std = df_history['snap_counts_offense'].std()
            workload_volatility = min(1.0, snaps_std / 150.0)
            return min(1.0, grade_volatility * 0.7 + workload_volatility * 0.3)
        
        return grade_volatility

    def get_confidence_interval(self, grade, vol_score):
        base_mae = 8.5
        bound = base_mae * (1.0 + vol_score * 1.2)
        return (round(grade - bound, 2), round(grade + bound, 2))

    def get_age_decay_factor(self, age):
        """
        OLs maintain performance longer than RBs.
        Minimal decay until ~29, slight decline 30-33, steeper after 34.
        """
        if age <= 28:
            return 0.0
        elif age <= 33:
            # gradual decline
            return (age - 28) * 0.5
        else:
            # faster decline after 33
            base_penalty = 0.5 * 5  # ages 29-33
            additional = (age - 33) * 1.0
            return min(base_penalty + additional, 10.0)  # cap max penalty


    def get_tier(self, grade):
        # Optimized thresholds from 2023 validation tuning: Elite≥83, Starter≥61, Rotation≥58 (+2.07% accuracy)
        if grade >= 83.0:
            return "Elite"
        elif grade >= 61.0:
            return "Starter"
        elif grade >= 58.0:
            return "Rotation"
        else:
            return "Reserve/Poor"
