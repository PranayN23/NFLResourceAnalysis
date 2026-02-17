import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from backend.agent.model_wrapper import PlayerModelInference

# Paths
DATA_FILE = os.path.join(os.path.dirname(__file__), "../QB.csv")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "best_classifier.pth")
SCALER_OUT = os.path.join(os.path.dirname(__file__), "player_scaler.joblib")
XGB_MODEL_OUT = os.path.join(os.path.dirname(__file__), "best_xgb.joblib")

# Load data
df = pd.read_csv(DATA_FILE)
df['adjusted_value'] = pd.to_numeric(df['adjusted_value'], errors='coerce').fillna(0)
df = df[df['dropbacks'] >= 100].copy()
df = df.sort_values(['player', 'Year'])

# Initialize inference engine
engine = PlayerModelInference(MODEL_OUT, scaler_path=SCALER_OUT, xgb_path=XGB_MODEL_OUT)

# Collect predictions for 2014-2024
all_results = []

print("=" * 80)
print("GENERATING PREDICTIONS FOR 2014-2024...")
print("=" * 80)

for year in range(2014, 2025):
    test_df = df[df['Year'] == year].copy()
    
    for _, row in test_df.iterrows():
        player = row['player']
        history = df[(df['player'] == player) & (df['Year'] < year)].copy()
        
        if history.empty:
            continue  # Skip rookies
        
        tier, details = engine.get_prediction(history, apply_calibration=False)
        
        all_results.append({
            'Year': year,
            'player': player,
            'y_true': row['grades_offense'],
            'y_xgb': details['xgb_grade'],
            'y_transformer': details['transformer_grade']
        })

df_results = pd.DataFrame(all_results)

# Remove any NaN predictions
df_clean = df_results.dropna(subset=['y_xgb', 'y_transformer', 'y_true'])

print(f"\nTotal predictions: {len(df_clean)} player-seasons")

y_true = df_clean['y_true'].values
y_xgb = df_clean['y_xgb'].values
y_transformer = df_clean['y_transformer'].values

# Step 1: Individual model metrics
rmse_xgb = np.sqrt(mean_squared_error(y_true, y_xgb))
rmse_transformer = np.sqrt(mean_squared_error(y_true, y_transformer))
mae_xgb = mean_absolute_error(y_true, y_xgb)
mae_transformer = mean_absolute_error(y_true, y_transformer)

# Residual correlation
residuals_xgb = y_true - y_xgb
residuals_transformer = y_true - y_transformer
corr, p_value = pearsonr(residuals_xgb, residuals_transformer)

print("\n" + "=" * 80)
print("STEP 1: INDIVIDUAL MODEL PERFORMANCE (2014-2024)")
print("=" * 80)
print(f"XGBoost RMSE: {rmse_xgb:.4f}")
print(f"XGBoost MAE: {mae_xgb:.4f}")
print(f"Transformer RMSE: {rmse_transformer:.4f}")
print(f"Transformer MAE: {mae_transformer:.4f}")
print(f"\nResidual Correlation: {corr:.4f} (p={p_value:.4f})")

# Step 2: Grid search
print("\n" + "=" * 80)
print("STEP 2: GRID SEARCH FOR OPTIMAL ENSEMBLE WEIGHT")
print("=" * 80)

best_rmse = float('inf')
best_weight = 0.0
best_mae = 0.0

results = []

for w_xgb in np.arange(0, 1.01, 0.01):
    w_transformer = 1.0 - w_xgb
    y_ensemble = w_xgb * y_xgb + w_transformer * y_transformer
    
    rmse = np.sqrt(mean_squared_error(y_true, y_ensemble))
    mae = mean_absolute_error(y_true, y_ensemble)
    
    results.append({
        'XGB_Weight': round(w_xgb, 2),
        'Transformer_Weight': round(w_transformer, 2),
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4)
    })
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_weight = w_xgb
        best_mae = mae

print(f"\n🏆 OPTIMAL ENSEMBLE WEIGHT:")
print(f"   XGBoost: {best_weight:.2f}")
print(f"   Transformer: {1.0 - best_weight:.2f}")
print(f"   RMSE: {best_rmse:.4f}")
print(f"   MAE: {best_mae:.4f}")

# Current 50/50 performance
current_rmse = np.sqrt(mean_squared_error(y_true, 0.5 * y_xgb + 0.5 * y_transformer))
current_mae = mean_absolute_error(y_true, 0.5 * y_xgb + 0.5 * y_transformer)

print(f"\n📊 CURRENT (50/50) PERFORMANCE:")
print(f"   RMSE: {current_rmse:.4f}")
print(f"   MAE: {current_mae:.4f}")

print(f"\n📈 IMPROVEMENT:")
print(f"   RMSE Reduction: {current_rmse - best_rmse:.4f} ({((current_rmse - best_rmse) / current_rmse * 100):.2f}%)")
print(f"   MAE Reduction: {current_mae - best_mae:.4f} ({((current_mae - best_mae) / current_mae * 100):.2f}%)")

# Show top 5 weights
df_results_grid = pd.DataFrame(results)
print("\n📋 TOP 5 ENSEMBLE WEIGHTS BY RMSE:")
print(df_results_grid.nsmallest(5, 'RMSE').to_string(index=False))

print("\n" + "=" * 80)
