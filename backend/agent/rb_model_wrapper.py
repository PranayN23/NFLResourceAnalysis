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
    # Base 22 features (no OL, no rolling stats) - matches model trained with USE_OL_FEATURES=False, USE_ROLLING_STATS=False
    TRANSFORMER_FEATURES_BASE = [
        'grades_offense', 'grades_run', 'grades_pass_route', 'elusive_rating',
        'yards', 'yards_after_contact', 'yco_attempt', 'breakaway_percent',
        'explosive', 'first_downs', 'receptions', 'targets', 'total_touches',
        'touchdowns', 'adjusted_value', 'Cap_Space', 'age', 'years_in_league',
        'delta_grade', 'delta_yards', 'delta_touches', 'team_performance_proxy',
    ]
    # Extended 31 features (with OL and rolling stats)
    TRANSFORMER_FEATURES_EXTENDED = TRANSFORMER_FEATURES_BASE + [
        'grades_run_block_ol_prev', 'grades_pass_block_ol_prev',
        'penalties_ol_prev', 'pressures_allowed_ol_prev',
        'rb_yards_rolling_std', 'rb_touches_rolling_std', 'rb_grades_rolling_std',
        'rb_yards_rolling_mean', 'rb_grades_rolling_mean'
    ]
    XGB_FEATURES_BASE = [
        'lag_grades_offense', 'lag_yards', 'lag_yco_attempt', 'lag_elusive_rating',
        'lag_breakaway_percent', 'lag_explosive', 'lag_total_touches', 'lag_touchdowns',
        'adjusted_value', 'age', 'years_in_league', 'delta_grade_lag',
        'team_performance_proxy_lag', 'lag_receptions',
    ]
    XGB_FEATURES_EXTENDED = XGB_FEATURES_BASE + [
        'lag_grades_run_block_ol', 'lag_grades_pass_block_ol',
        'lag_penalties_ol', 'lag_pressures_allowed_ol',
        'lag_rb_yards_rolling_std', 'lag_rb_touches_rolling_std', 'lag_rb_grades_rolling_std'
    ]

    def __init__(self, transformer_path, scaler_path=None, xgb_path=None, stack_model_path=None):
        self.device = torch.device('cpu')
        self.max_seq_len = 5

        # Load checkpoint and infer input_dim from saved weights (supports both 22 and 31 feature models)
        state_dict = torch.load(transformer_path, map_location=self.device)
        # time2vec.w0 shape is (input_dim, 1)
        input_dim = state_dict['time2vec.w0'].shape[0]
        if input_dim == 22:
            self.transformer_features = list(self.TRANSFORMER_FEATURES_BASE)
            self.xgb_features = list(self.XGB_FEATURES_BASE)
        elif input_dim == 31:
            self.transformer_features = list(self.TRANSFORMER_FEATURES_EXTENDED)
            self.xgb_features = list(self.XGB_FEATURES_EXTENDED)
        else:
            raise RuntimeError(
                f"Transformer checkpoint has input_dim={input_dim}. "
                "Expected 22 (base features) or 31 (with OL + rolling stats). "
                "Retrain with Player_Model_RB.py to match desired feature set."
            )

        self.model = PlayerTransformerRegressor(input_dim=input_dim, seq_len=self.max_seq_len).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.scaler = None
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            
        self.xgb_model = None
        if xgb_path and os.path.exists(xgb_path):
            self.xgb_model = joblib.load(xgb_path)
            # XGB may have been trained with base (14) or extended (21) features; match to saved model
            n_xgb = getattr(self.xgb_model, 'n_features_in_', None)
            if n_xgb is not None:
                if n_xgb == 14:
                    self.xgb_features = list(self.XGB_FEATURES_BASE)
                elif n_xgb == 21:
                    self.xgb_features = list(self.XGB_FEATURES_EXTENDED)
        
        # Optional stacking model (learned ensemble weights)
        self.stack_model = None
        self.use_stacking = False
        if stack_model_path and os.path.exists(stack_model_path):
            self.stack_model = joblib.load(stack_model_path)
            self.use_stacking = True
            print("Stacking ensemble enabled - using learned weights")

    def _add_ol_features(self, df, ol_df_paths=None):
        """
        Add previous-year OL features to player history using snap-weighted averages.
        Combines G (Guard), T (Tackle), and C (Center) data.
        Uses snap_counts_offense for weighted averaging.
        """
        try:
            # Default paths for G, T, C files
            if ol_df_paths is None:
                base_paths = [
                    ('backend/ML/G.csv', 'backend/ML/T.csv', 'backend/ML/C.csv'),
                    (os.path.join(os.path.dirname(__file__), '../ML/G.csv'),
                     os.path.join(os.path.dirname(__file__), '../ML/T.csv'),
                     os.path.join(os.path.dirname(__file__), '../ML/C.csv')),
                    (os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backend/ML/G.csv')),
                     os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backend/ML/T.csv')),
                     os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backend/ML/C.csv')))
                ]
                
                ol_g_path = None
                ol_t_path = None
                ol_c_path = None
                
                for g_path, t_path, c_path in base_paths:
                    if os.path.exists(g_path) and os.path.exists(t_path) and os.path.exists(c_path):
                        ol_g_path = g_path
                        ol_t_path = t_path
                        ol_c_path = c_path
                        break
                
                if ol_g_path is None:
                    raise FileNotFoundError("Could not find G.csv, T.csv, or C.csv in any expected location")
            else:
                ol_g_path, ol_t_path, ol_c_path = ol_df_paths
            
            # Load all three OL position files
            ol_g = pd.read_csv(ol_g_path)
            ol_t = pd.read_csv(ol_t_path)
            ol_c = pd.read_csv(ol_c_path)
            
            # Combine all OL positions
            ol_df = pd.concat([ol_g, ol_t, ol_c], ignore_index=True)
            
            ol_features = ['grades_run_block', 'grades_pass_block', 'penalties', 'pressures_allowed']
            weight_col = 'snap_counts_offense'
            
            for feat in ol_features:
                if feat in ol_df.columns:
                    ol_df[feat] = pd.to_numeric(ol_df[feat], errors='coerce')
            
            if weight_col in ol_df.columns:
                ol_df[weight_col] = pd.to_numeric(ol_df[weight_col], errors='coerce')
                ol_df[weight_col] = ol_df[weight_col].fillna(0)
            else:
                weight_col = None
            
            # Shift OL year forward by 1
            ol_df_shifted = ol_df.copy()
            ol_df_shifted['Year'] = ol_df_shifted['Year'] + 1
            
            # Calculate snap-weighted averages by Team and Year
            ol_agg_list = []
            
            for team_year, group in ol_df_shifted.groupby(['Team', 'Year']):
                team, year = team_year
                agg_dict = {'Team': team, 'Year': year}
                
                if weight_col and (group[weight_col] > 0).any():
                    # Snap-weighted average
                    total_snaps = group[weight_col].sum()
                    if total_snaps > 0:
                        for feat in ol_features:
                            if feat in group.columns:
                                weighted_sum = (group[feat] * group[weight_col]).sum()
                                agg_dict[f'{feat}_ol_prev'] = weighted_sum / total_snaps
                            else:
                                agg_dict[f'{feat}_ol_prev'] = 0.0
                    else:
                        # Fallback to simple mean if no snaps
                        for feat in ol_features:
                            if feat in group.columns:
                                agg_dict[f'{feat}_ol_prev'] = group[feat].mean()
                            else:
                                agg_dict[f'{feat}_ol_prev'] = 0.0
                else:
                    # Simple mean if no snap data
                    for feat in ol_features:
                        if feat in group.columns:
                            agg_dict[f'{feat}_ol_prev'] = group[feat].mean()
                        else:
                            agg_dict[f'{feat}_ol_prev'] = 0.0
                
                ol_agg_list.append(agg_dict)
            
            ol_agg = pd.DataFrame(ol_agg_list)
            
            df = df.merge(ol_agg, on=['Team', 'Year'], how='left')
            
            # Fill missing with league average
            for feat in ol_features:
                col_name = f'{feat}_ol_prev'
                if col_name in df.columns:
                    league_avg = df[col_name].mean()
                    df[col_name] = df[col_name].fillna(league_avg)
            
            return df
        except Exception as e:
            print(f"Warning: Could not load OL features: {e}")
            # Add dummy columns if OL data unavailable
            for feat in ['grades_run_block', 'grades_pass_block', 'penalties', 'pressures_allowed']:
                df[f'{feat}_ol_prev'] = 0.0
            return df
    
    def _add_rolling_stats(self, df):
        """Add rolling statistics features."""
        if len(df) == 0:
            return df
        
        # For single player history, compute rolling stats directly
        df['rb_yards_rolling_std'] = df['yards'].rolling(window=3, min_periods=1).std().fillna(0)
        df['rb_touches_rolling_std'] = df['total_touches'].rolling(window=3, min_periods=1).std().fillna(0)
        df['rb_grades_rolling_std'] = df['grades_offense'].rolling(window=3, min_periods=1).std().fillna(0)
        df['rb_yards_rolling_mean'] = df['yards'].rolling(window=3, min_periods=1).mean().fillna(0)
        df['rb_grades_rolling_mean'] = df['grades_offense'].rolling(window=3, min_periods=1).mean().fillna(0)
        
        # Ensure all columns exist
        rolling_cols = ['rb_yards_rolling_std', 'rb_touches_rolling_std', 'rb_grades_rolling_std',
                        'rb_yards_rolling_mean', 'rb_grades_rolling_mean']
        for col in rolling_cols:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(0)
        
        return df

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
        
        # Add OL and rolling only when model uses extended features (31)
        if len(self.transformer_features) > 22:
            df = self._add_ol_features(df)
            df = self._add_rolling_stats(df)
        
        # Ensure all transformer features exist
        for feat in self.transformer_features:
            if feat not in df.columns:
                df[feat] = 0.0
        
        # Lagged Engineering for XGBoost
        row_last = df.iloc[-1]
        row_prev = df.iloc[-2] if len(df) > 1 else row_last
        
        # Get OL features from last year (already previous year, but need to lag for XGB)
        ol_run_block = float(row_last.get('grades_run_block_ol_prev', 0))
        ol_pass_block = float(row_last.get('grades_pass_block_ol_prev', 0))
        ol_penalties = float(row_last.get('penalties_ol_prev', 0))
        ol_pressures = float(row_last.get('pressures_allowed_ol_prev', 0))
        
        # Get rolling stats from last year
        rolling_yards_std = float(row_last.get('rb_yards_rolling_std', 0))
        rolling_touches_std = float(row_last.get('rb_touches_rolling_std', 0))
        rolling_grades_std = float(row_last.get('rb_grades_rolling_std', 0))
        
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
            # OL features (lagged)
            'lag_grades_run_block_ol': ol_run_block,
            'lag_grades_pass_block_ol': ol_pass_block,
            'lag_penalties_ol': ol_penalties,
            'lag_pressures_allowed_ol': ol_pressures,
            # Rolling stats (lagged)
            'lag_rb_yards_rolling_std': rolling_yards_std,
            'lag_rb_touches_rolling_std': rolling_touches_std,
            'lag_rb_grades_rolling_std': rolling_grades_std
        }
        
        # Ensure all XGB features exist
        for feat in self.xgb_features:
            if feat not in xgb_input:
                xgb_input[feat] = 0.0
        
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

        # Ensemble weighting
        if mode == "transformer":
            final_grade = transformer_grade
        elif mode == "xgb":
            final_grade = xgb_grade
        else:
            # Use stacking if available, otherwise fixed weights
            if self.use_stacking and self.stack_model is not None:
                # Stacking: learn optimal weights from transformer and xgb predictions
                stack_input = np.array([[transformer_grade, xgb_grade]])
                final_grade = self.stack_model.predict(stack_input)[0]
            else:
                # Adjusted weights - XGBoost often performs better, give it more weight
                # Try 50/50 or favor XGB slightly
                xgb_weight = 0.50
                trans_weight = 0.50
                final_grade = (transformer_grade * trans_weight) + (xgb_grade * xgb_weight)

        # Age decay (less aggressive for elite players)
        age_adjustment = 0.0
        if 'age' in df_history.columns:
            current_age = float(df_history.iloc[-1]['age'])
            base_decay = self.get_age_decay_factor(current_age)
            
            # Elite players (predicted grade > 75) age slower - reduce penalty by 40%
            if final_grade > 75:
                age_adjustment = base_decay * 0.6  # Only 60% of normal penalty
            else:
                age_adjustment = base_decay
            
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
    
    def predict_batch(self, players_histories, mode="ensemble", apply_calibration=True):
        """
        Batch prediction for multiple players.
        
        Args:
            players_histories: dict of {player_name: player_history_df} or list of DataFrames
            mode: 'transformer', 'xgb', or 'ensemble'
            apply_calibration: Whether to apply calibration
        
        Returns:
            pd.DataFrame with predictions for all players
        """
        results = []
        
        # Handle both dict and list inputs
        if isinstance(players_histories, dict):
            items = players_histories.items()
        elif isinstance(players_histories, list):
            items = [(f"player_{i}", hist) for i, hist in enumerate(players_histories)]
        else:
            raise ValueError("players_histories must be dict or list")
        
        for player_name, history in items:
            if history.empty:
                continue
            
            tier, details = self.predict(history, mode=mode, apply_calibration=apply_calibration)
            
            result = {
                "player": player_name,
                "tier": tier,
                **details
            }
            
            # Add metadata if available
            if not history.empty:
                result["last_year"] = int(history.iloc[-1]['Year']) if 'Year' in history.columns else None
                result["last_team"] = history.iloc[-1]['Team'] if 'Team' in history.columns else None
                result["last_age"] = float(history.iloc[-1]['age']) if 'age' in history.columns else None
            
            results.append(result)
        
        return pd.DataFrame(results)

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
        # FIXED (MORE REALISTIC):
        if age <= 27:
            return 0.0  # Peak extended to 27
        elif age <= 29:
            return (age - 27) * 0.8  # 0.8 pts/year (was 1.5)
        elif age <= 31:
            base_penalty = 2 * 0.8  # 1.6 total
            additional = (age - 29) * 1.5  # 1.5 pts/year (was 2.5)
            return base_penalty + additional
        else:
            base_penalty = 2 * 0.8 + 2 * 1.5  # 4.6 total
            additional = (age - 31) * 2.0  # 2.0 pts/year (was 3.5)
            return min(base_penalty + additional, 15.0)  # Cap at 15 (was 20)

    def get_tier(self, grade):
        if grade >= 75.0:
            return "Elite"
        elif grade >= 65.0:
            return "Starter"
        elif grade >= 55.0:
            return "Rotation"
        else:
            return "Reserve/Poor"