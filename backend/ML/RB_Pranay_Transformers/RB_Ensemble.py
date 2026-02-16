import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Import classes and wrapper (assuming similar structure for RB)
from backend.ML.RB_Pranay_Transformers.Player_Model_RB import PlayerTransformerRegressor, Time2Vec
from backend.agent.rb_model_wrapper import RBModelInference

# ==========================================
# 1. SETUP & CONFIG
# ==========================================
DATA_FILE = os.path.join(os.path.dirname(__file__), "../HB.csv")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "rb_best_classifier.pth")
SCALER_OUT = os.path.join(os.path.dirname(__file__), "rb_player_scaler.joblib")
XGB_MODEL_OUT = os.path.join(os.path.dirname(__file__), "rb_best_xgb.joblib")

MODE = "VALIDATION"  # Predict 2024 (Train < 2024)
#MODE = "DREAM"       # Predict 2025 (Train <= 2024)

print(f"==== STARTING RB ENSEMBLE MODELING (Mode: {MODE}) ====")

TRANSFORMER_FEATURES = [
    'grades_offense', 'grades_run', 'grades_pass_route', 'elusive_rating',
    'yards', 'yards_after_contact', 'yco_attempt', 'breakaway_percent',
    'explosive', 'first_downs', 'receptions', 'targets', 'total_touches',
    'touchdowns', 'adjusted_value', 'Cap_Space', 'age', 'years_in_league',
    'delta_grade', 'delta_yards', 'delta_touches', 'team_performance_proxy',
    # OL features (previous year)
    'grades_run_block_ol_prev', 'grades_pass_block_ol_prev', 
    'penalties_ol_prev', 'pressures_allowed_ol_prev',
    # Rolling stats
    'rb_yards_rolling_std', 'rb_touches_rolling_std', 'rb_grades_rolling_std',
    'rb_yards_rolling_mean', 'rb_grades_rolling_mean'
]

XGB_FEATURES = [
    'lag_grades_offense', 'lag_yards', 'lag_yco_attempt', 'lag_elusive_rating',
    'lag_breakaway_percent', 'lag_explosive', 'lag_total_touches', 'lag_touchdowns',
    'adjusted_value', 'age', 'years_in_league', 'delta_grade_lag',
    'team_performance_proxy_lag', 'lag_receptions',
    # OL features (previous year - already lagged)
    'lag_grades_run_block_ol', 'lag_grades_pass_block_ol',
    'lag_penalties_ol', 'lag_pressures_allowed_ol',
    # Rolling stats (from last year)
    'lag_rb_yards_rolling_std', 'lag_rb_touches_rolling_std', 'lag_rb_grades_rolling_std'
]

# ==========================================
# 2. DATA PREPARATION (WITH TEMPORAL AWARENESS)
# ==========================================
def add_ol_features(rb_df, ol_df_paths=None):
    """
    Add previous-year OL features to RB dataframe using snap-weighted averages.
    Combines G (Guard), T (Tackle), and C (Center) data.
    Uses snap_counts_offense for weighted averaging.
    """
    try:
        # Default paths for G, T, C files
        if ol_df_paths is None:
            base_paths = [
                ('backend/ML/G.csv', 'backend/ML/T.csv', 'backend/ML/C.csv'),
                (os.path.join(os.path.dirname(__file__), '../../ML/G.csv'),
                 os.path.join(os.path.dirname(__file__), '../../ML/T.csv'),
                 os.path.join(os.path.dirname(__file__), '../../ML/C.csv')),
                (os.path.abspath(os.path.join(os.path.dirname(__file__), '../../ML/G.csv')),
                 os.path.abspath(os.path.join(os.path.dirname(__file__), '../../ML/T.csv')),
                 os.path.abspath(os.path.join(os.path.dirname(__file__), '../../ML/C.csv')))
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
        
        rb_df = rb_df.merge(ol_agg, on=['Team', 'Year'], how='left')
        
        # Fill missing with league average
        for feat in ol_features:
            col_name = f'{feat}_ol_prev'
            if col_name in rb_df.columns:
                league_avg = rb_df[col_name].mean()
                rb_df[col_name] = rb_df[col_name].fillna(league_avg)
        
        return rb_df
    except Exception as e:
        print(f"Warning: Could not load OL features: {e}")
        return rb_df

def add_rolling_stats(df):
    """Add rolling statistics features."""
    groups = df.groupby('player')
    
    # These columns should exist, so groups is fine here
    df['rb_yards_rolling_std'] = groups['yards'].rolling(window=3, min_periods=1).std().reset_index(0, drop=True)
    df['rb_touches_rolling_std'] = groups['total_touches'].rolling(window=3, min_periods=1).std().reset_index(0, drop=True)
    df['rb_grades_rolling_std'] = groups['grades_offense'].rolling(window=3, min_periods=1).std().reset_index(0, drop=True)
    df['rb_yards_rolling_mean'] = groups['yards'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    df['rb_grades_rolling_mean'] = groups['grades_offense'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    
    rolling_cols = ['rb_yards_rolling_std', 'rb_touches_rolling_std', 'rb_grades_rolling_std',
                    'rb_yards_rolling_mean', 'rb_grades_rolling_mean']
    for col in rolling_cols:
        df[col] = df[col].fillna(0)
    
    return df

def prepare_data():
    df = pd.read_csv(DATA_FILE)
    df['adjusted_value'] = pd.to_numeric(df['adjusted_value'], errors='coerce').fillna(0)
    df = df[df['total_touches'] >= 50].copy()  # RB minimum threshold
    df = df.sort_values(['player', 'Year'])
    groups = df.groupby('player')
    
    # Feature engineering - all using past data only (shift creates lag)
    df["years_in_league"] = groups.cumcount()
    df["delta_grade"] = groups["grades_offense"].diff().fillna(0)
    df["delta_yards"] = groups["yards"].diff().fillna(0)
    df["delta_touches"] = groups["total_touches"].diff().fillna(0)
    df['team_performance_proxy'] = df.groupby(['Team', 'Year'])['Net EPA'].transform('mean')
    
    # Add OL features (previous year)
    df = add_ol_features(df)
    
    # Add rolling statistics
    df = add_rolling_stats(df)
    
    # Lagged features for XGBoost (using previous year to predict current)
    df['lag_grades_offense'] = groups['grades_offense'].shift(1)
    df['lag_yards'] = groups['yards'].shift(1)
    df['lag_yco_attempt'] = groups['yco_attempt'].shift(1)
    df['lag_elusive_rating'] = groups['elusive_rating'].shift(1)
    df['lag_breakaway_percent'] = groups['breakaway_percent'].shift(1)
    df['lag_explosive'] = groups['explosive'].shift(1)
    df['lag_total_touches'] = groups['total_touches'].shift(1).fillna(50)  # Baseline workload
    df['lag_touchdowns'] = groups['touchdowns'].shift(1)
    df['lag_receptions'] = groups['receptions'].shift(1)
    # Recreate groups to include newly created lag columns, or use df.groupby directly
    df['delta_grade_lag'] = df.groupby('player')['lag_grades_offense'].diff().fillna(0)
    df['team_performance_proxy_lag'] = groups['team_performance_proxy'].shift(1)
    
    # Lag OL features (already previous year, but need to shift for XGB)
    # Use df.groupby directly since groups was created before OL features existed
    if 'grades_run_block_ol_prev' in df.columns:
        df['lag_grades_run_block_ol'] = df.groupby('player')['grades_run_block_ol_prev'].shift(1)
        df['lag_grades_pass_block_ol'] = df.groupby('player')['grades_pass_block_ol_prev'].shift(1)
        df['lag_penalties_ol'] = df.groupby('player')['penalties_ol_prev'].shift(1)
        df['lag_pressures_allowed_ol'] = df.groupby('player')['pressures_allowed_ol_prev'].shift(1)
    
    # Lag rolling stats
    if 'rb_yards_rolling_std' in df.columns:
        df['lag_rb_yards_rolling_std'] = df.groupby('player')['rb_yards_rolling_std'].shift(1)
        df['lag_rb_touches_rolling_std'] = df.groupby('player')['rb_touches_rolling_std'].shift(1)
        df['lag_rb_grades_rolling_std'] = df.groupby('player')['rb_grades_rolling_std'].shift(1)
    
    # Fill NaN for lagged features
    if 'lag_grades_run_block_ol' in df.columns:
        df['lag_grades_run_block_ol'] = df['lag_grades_run_block_ol'].fillna(df['grades_run_block_ol_prev'].mean() if 'grades_run_block_ol_prev' in df.columns else 0)
        df['lag_grades_pass_block_ol'] = df['lag_grades_pass_block_ol'].fillna(df['grades_pass_block_ol_prev'].mean() if 'grades_pass_block_ol_prev' in df.columns else 0)
        df['lag_penalties_ol'] = df['lag_penalties_ol'].fillna(df['penalties_ol_prev'].mean() if 'penalties_ol_prev' in df.columns else 0)
        df['lag_pressures_allowed_ol'] = df['lag_pressures_allowed_ol'].fillna(df['pressures_allowed_ol_prev'].mean() if 'pressures_allowed_ol_prev' in df.columns else 0)
    
    if 'lag_rb_yards_rolling_std' in df.columns:
        df['lag_rb_yards_rolling_std'] = df['lag_rb_yards_rolling_std'].fillna(0)
        df['lag_rb_touches_rolling_std'] = df['lag_rb_touches_rolling_std'].fillna(0)
        df['lag_rb_grades_rolling_std'] = df['lag_rb_grades_rolling_std'].fillna(0)
    
    target_col = 'grades_offense'
    df_clean = df.dropna(subset=XGB_FEATURES + [target_col]).copy()
    
    # CRITICAL: Convert all XGB features to numeric (fix object dtype error)
    for col in XGB_FEATURES:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Drop any rows with NaN after conversion
    df_clean = df_clean.dropna(subset=XGB_FEATURES + [target_col]).copy()
    
    return df_clean, df, target_col

df_clean, df_all, target_col = prepare_data()

# ==========================================
# 3. TEMPORAL SPLIT & MODEL TRAINING
# ==========================================

if MODE == "VALIDATION":
    print("\n=== VALIDATION MODE: Training on <2024, Testing on 2024 ===")
    
    # CRITICAL: Train XGBoost ONLY on data before 2024
    train_data = df_clean[df_clean["Year"] < 2024].copy()
    test_data = df_clean[df_clean["Year"] == 2024].copy()
    
    print(f"\n[1/3] Training XGBoost on {len(train_data)} samples (<2024)...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500, 
        learning_rate=0.05, 
        max_depth=5, 
        random_state=42
    )
    xgb_model.fit(train_data[XGB_FEATURES], train_data[target_col])
    joblib.dump(xgb_model, XGB_MODEL_OUT)
    
    # Evaluate XGBoost performance
    xgb_train_preds = xgb_model.predict(train_data[XGB_FEATURES])
    xgb_test_preds = xgb_model.predict(test_data[XGB_FEATURES])
    
    print(f"XGBoost Train MAE: {mean_absolute_error(train_data[target_col], xgb_train_preds):.4f}")
    print(f"XGBoost Test MAE: {mean_absolute_error(test_data[target_col], xgb_test_preds):.4f}")
    
    # Initialize production wrapper
    engine = RBModelInference(MODEL_OUT, scaler_path=SCALER_OUT, xgb_path=XGB_MODEL_OUT)
    
    # Generate predictions
    print("\n[2/3] Generating ensemble predictions for 2024...")
    results = []
    for _, row in test_data.iterrows():
        name = row['player']
        # CRITICAL: Only use history BEFORE the target year
        history = df_all[(df_all['player'] == name) & (df_all['Year'] < 2024)].copy()
        
        if len(history) == 0:
            print(f"Warning: No history for {name} before 2024, skipping...")
            continue
            
        tier, details = engine.get_prediction(history, apply_calibration=True)
        
        results.append({
            "player": name,
            "Team": row["Team"],
            "Year": 2024,
            "Actual_Grade": row["grades_offense"],
            "Pred_XGB": details["xgb_grade"],
            "Pred_Transformer": details["transformer_grade"],
            "Ensemble_Pred": details["predicted_grade"],
            "Error": row["grades_offense"] - details["predicted_grade"],
            "Abs_Error": abs(row["grades_offense"] - details["predicted_grade"]),
            "Conf_Lower": details["confidence_interval"][0],
            "Conf_Upper": details["confidence_interval"][1]
        })
    
    final_df = pd.DataFrame(results)
    
    # Calculate metrics
    mae = final_df["Abs_Error"].mean()
    rmse = np.sqrt((final_df["Error"]**2).mean())
    
    print("\n=== VALIDATION RESULTS (2024) ===")
    print(f"Ensemble MAE: {mae:.4f}")
    print(f"Ensemble RMSE: {rmse:.4f}")
    print(f"Mean Error (bias): {final_df['Error'].mean():.4f}")
    
    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "RB_2024_Validation_Results.csv")
    final_df = final_df.sort_values(by="Ensemble_Pred", ascending=False)
    final_df.to_csv(out_path, index=False)
    print(f"\n[3/3] Saved Validation Report to {out_path}")
    
    # Show top errors
    print("\n=== LARGEST PREDICTION ERRORS ===")
    display_cols = ["player", "Actual_Grade", "Ensemble_Pred", "Error", "Abs_Error"]
    print(final_df.nlargest(5, "Abs_Error")[display_cols].to_string(index=False))

elif MODE == "DREAM":
    print("\n=== DREAM MODE: Training on ≤2024, Predicting 2025 ===")
    
    # CRITICAL: Train XGBoost on ALL available data (≤2024) for 2025 predictions
    print(f"\n[1/3] Training XGBoost on {len(df_clean)} samples (≤2024)...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500, 
        learning_rate=0.05, 
        max_depth=5, 
        random_state=42
    )
    xgb_model.fit(df_clean[XGB_FEATURES], df_clean[target_col])
    joblib.dump(xgb_model, XGB_MODEL_OUT)
    
    # Evaluate on full dataset (for reference only)
    xgb_all_preds = xgb_model.predict(df_clean[XGB_FEATURES])
    print(f"XGBoost Full Dataset MAE: {mean_absolute_error(df_clean[target_col], xgb_all_preds):.4f}")
    
    # Initialize production wrapper
    engine = RBModelInference(MODEL_OUT, scaler_path=SCALER_OUT, xgb_path=XGB_MODEL_OUT)
    
    # Get active players from 2024
    active_2024 = df_all[df_all["Year"] == 2024].copy()
    
    print("\n[2/3] Generating 2025 'Dream' Projections with Calibration...")
    rows_2025 = []
    
    retired_players = []  # Example - update as needed
    
    for _, row in active_2024.iterrows():
        player = row['player']
        
        if player in retired_players:
            print(f"Skipping {player} (retired)")
            continue
        
        # Use full history up to and including 2024
        history = df_all[df_all["player"] == player].sort_values("Year").tail(5)
        
        if len(history) == 0:
            print(f"Warning: No history for {player}, skipping...")
            continue
        
        # The wrapper handles the 2025 projection logic
        tier, details = engine.get_prediction(history, apply_calibration=True)
        
        rows_2025.append({
            "player": player,
            "Team": row["Team"],
            "Tier": tier,
            "Pred_XGB": details["xgb_grade"],
            "Pred_Transformer": details["transformer_grade"],
            "Ensemble_Pred": details["predicted_grade"],
            "Conf_Lower": details["confidence_interval"][0],
            "Conf_Upper": details["confidence_interval"][1],
            "Vol_Index": details["volatility_index"],
            "Age_Adjustment": details["age_adjustment"],
            "Last_Grade_2024": history.iloc[-1]["grades_offense"],
            "YoY_Change_Pred": details["predicted_grade"] - history.iloc[-1]["grades_offense"],
            "Touches_2024": row["total_touches"],
            "Age_2024": row["age"]
        })
    
    final_2025 = pd.DataFrame(rows_2025).sort_values("Ensemble_Pred", ascending=False)
    final_2025['Age_2024'] = pd.to_numeric(final_2025['Age_2024'], errors='coerce')
    # Save full results
    final_out = os.path.join(os.path.dirname(__file__), "RB_2025_Final_Rankings.csv")
    final_2025.to_csv(final_out, index=False)
    
    # Display top rankings
    print("\n" + "="*80)
    print("🏆 TOP 15 RB RANKINGS FOR 2025 (ENSEMBLE + CALIBRATION)")
    print("="*80)
    display_cols = [
        "player", "Team", "Tier", "Ensemble_Pred", 
        "Conf_Lower", "Conf_Upper", "Vol_Index", "YoY_Change_Pred", "Age_2024"
    ]
    print(final_2025[display_cols].head(15).to_string(index=False))
    
    # Show risers and fallers
    print("\n" + "="*80)
    print("📈 BIGGEST PROJECTED RISERS (2024 → 2025)")
    print("="*80)
    riser_cols = ["player", "Last_Grade_2024", "Ensemble_Pred", "YoY_Change_Pred", "Age_2024"]
    print(final_2025.nlargest(5, "YoY_Change_Pred")[riser_cols].to_string(index=False))
    
    print("\n" + "="*80)
    print("📉 BIGGEST PROJECTED FALLERS (2024 → 2025)")
    print("="*80)
    print(final_2025.nsmallest(5, "YoY_Change_Pred")[riser_cols].to_string(index=False))
    
    # Show age-related insights
    print("\n" + "="*80)
    print("👴 AGING CONCERNS (Age 28+ with projected decline)")
    print("="*80)
    aging_concerns = final_2025[(final_2025['Age_2024'] >= 28) & 
                                (final_2025['YoY_Change_Pred'] < -3)]
    if len(aging_concerns) > 0:
        print(aging_concerns[["player", "Age_2024", "Last_Grade_2024", 
                              "Ensemble_Pred", "YoY_Change_Pred"]].to_string(index=False))
    else:
        print("No significant aging concerns identified")
    
    print(f"\n[3/3] Full rankings saved to {final_out}")

# ==========================================
# 4. SUMMARY & RECOMMENDATIONS
# ==========================================
print("\n" + "="*80)
print("MODEL TRAINING SUMMARY")
print("="*80)

if MODE == "VALIDATION":
    print(f"✓ XGBoost trained on: Years < 2024")
    print(f"✓ Transformer trained on: Years ≤ 2022 (from separate training)")
    print(f"✓ Tested on: Year 2024")
    print(f"✓ No data leakage: Future data excluded from training")
    
elif MODE == "DREAM":
    print(f"✓ XGBoost trained on: All available data (≤ 2024)")
    print(f"✓ Transformer trained on: Years ≤ 2022 (from separate training)")
    print(f"✓ Predicting: Year 2025")
    print(f"✓ Using: Most recent available data for production predictions")
    print(f"✓ Age considerations: Built into predictions for RB position")

print("="*80)

