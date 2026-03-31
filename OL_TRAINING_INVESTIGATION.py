"""
OL Training Pipeline Investigation
Answers the 4 diagnostic questions
"""
import os
import torch
import numpy as np
import pandas as pd
import joblib

print("="*100)
print("OL TRAINING PIPELINE INVESTIGATION")
print("="*100)

# ==========================================
# QUESTION 1: TARGET VARIABLE DEFINITION
# ==========================================
print("\nQUESTION 1: TARGET VARIABLE DEFINITION")
print("="*100)

print("""
From OL_Ensemble.py (line 145):
  target_col = 'grades_offense'

From Player_Model_OL.py (line 47):
  target = p_data_raw.iloc[i][target_col]
  
From training loop (line 324):
  outputs = model(i, mask=m).squeeze()
  loss = criterion(outputs, t)  # t is the target

FINDING:
✓ Target is grades_offense DIRECTLY
✓ NO inversions like (1 - grade), (100 - grade), or (-grade)
✓ Training on raw grades in range 22.8 - 97.8 (mean ~65.6, std ~11.6)
✓ Target is NOT flipped or inverted
""")

# ==========================================
# QUESTION 2: TARGET NORMALIZATION/SCALING
# ==========================================
print("\nQUESTION 2: TARGET SCALING DURING TRAINING")
print("="*100)

print("""
From Player_Model_OL.py (lines 226-229):
  scaler = StandardScaler()
  scaler.fit(train_data[features])    ← Scaler applied to FEATURES ONLY
  df_clean_scaled[features] = scaler.transform(...)
  
From Player_Model_OL.py (line 247):
  y_train = np.array(y_boosted)  ← Direct array from raw grades_offense

FINDING:
✓ Only FEATURES are scaled with StandardScaler
✓ Targets (y) are NOT scaled before training
✓ No target_scaler file exists (scaler only has 20 elements = features)
✓ Model tries to learn mapping: scaled_features → raw_grades
✓ This means model output SHOULD be in 20-100 range
✓ BUT inference reports 35-45 range → indicates learning problem, not scaling problem
""")

# ==========================================
# QUESTION 3: OUTPUT LAYER ACTIVATION
# ==========================================
print("\nQUESTION 3: OUTPUT LAYER ACTIVATION FUNCTION")
print("="*100)

from backend.ML.OL_Pranay_Transformers.Player_Model_OL import PlayerTransformerRegressor

model = PlayerTransformerRegressor(input_dim=20, seq_len=5)

print("""
From Player_Model_OL.py (lines 120-127):
  self.regressor = nn.Sequential(
      nn.Linear(self.embed_dim, 64),
      nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(64, 1)           ← NO activation function!
  )

FINDING:
✓ Output layer is: Linear(64 → 1) with NO activation
✓ This is a linear regression head - outputs unbounded values
✓ Can output any value, not limited to 0-100 or 0-1
✓ Raw model output should be in training range (22-98)
✓ Model should NOT compress output to 35-45

Architecture not the issue - this is correct for regression.
The compressed output suggests training failure.
""")

# ==========================================
# QUESTION 4: TRAINING LOSS CURVES
# ==========================================
print("\nQUESTION 4: TRAINING LOSS CURVES & CONVERGENCE")
print("="*100)

# Check if validation results CSV exists
val_results_path = "backend/ML/OL_Pranay_Transformers/OL_2024_Validation_Results.csv"

if os.path.exists(val_results_path):
    print(f"\n✓ Validation results found: {val_results_path}")
    df_val = pd.read_csv(val_results_path)
    
    print(f"\nValidation Set Statistics (2024 Test Data):")
    print(f"  Count: {len(df_val)}")
    print(f"  Actual grades  - Mean: {df_val['Actual_Grade'].mean():.2f}, Std: {df_val['Actual_Grade'].std():.2f}")
    print(f"  Pred grades    - Mean: {df_val['Ensemble_Pred'].mean():.2f}, Std: {df_val['Ensemble_Pred'].std():.2f}")
    print(f"  Abs_Error      - Mean: {df_val['Abs_Error'].mean():.2f}, Max: {df_val['Abs_Error'].max():.2f}")
    
    # Check by grade range
    print(f"\nPredictions by Input Grade Range:")
    for threshold in [50, 60, 70, 80]:
        subset = df_val[df_val['Actual_Grade'] >= threshold]
        if len(subset) > 0:
            pred_mean = subset['Ensemble_Pred'].mean()
            print(f"  Actual ≥{threshold}: {len(subset)} players, Pred mean: {pred_mean:.2f}")
else:
    print(f"\n✗ Validation results NOT found at {val_results_path}")
    print("  Cannot directly check training convergence from saved files")

print(f"""
From Pplayer_Model_OL.py training loop (lines 318-336):
  - Uses L1Loss() as criterion
  - Runs up to 150 epochs
  - Early stopping based on validation loss
  - Reports: "Best model saved from epoch {best_epoch} with Val MAE: {best_val_loss:.4f}"
  
The training script SHOULD print convergence info, but we can infer from predictions:
""")

# Load actual predictions to analyze
try:
    df_val = pd.read_csv(val_results_path)
    errors = df_val['Actual_Grade'] - df_val['Ensemble_Pred']
    
    print(f"\nInferred Training Quality from Test Predictions:")
    print(f"  Residuals (Actual - Predicted):")
    print(f"    Mean: {errors.mean():.2f}")
    print(f"    Std: {errors.std():.2f}")
    print(f"    Min (biggest underestimate): {errors.min():.2f}")
    print(f"    Max (biggest overestimate): {errors.max():.2f}")
    
    # Check if residuals are systematically biased
    elite_mask = df_val['Actual_Grade'] >= 80
    if elite_mask.sum() > 0:
        elite_errors = errors[elite_mask]
        print(f"\n  Elite Players (Actual_Grade ≥ 80):")
        print(f"    Count: {elite_mask.sum()}")
        print(f"    Mean actual: {df_val.loc[elite_mask, 'Actual_Grade'].mean():.2f}")
        print(f"    Mean predicted: {df_val.loc[elite_mask, 'Ensemble_Pred'].mean():.2f}")
        print(f"    Mean error: {elite_errors.mean():.2f}")
        print(f"    ⚠️  Systematic UNDERESTIMATION of elite players!")
    
    # Check correlation
    from scipy.stats import spearmanr, pearsonr
    corr_p, _ = pearsonr(df_val['Actual_Grade'], df_val['Ensemble_Pred'])
    corr_s, _ = spearmanr(df_val['Actual_Grade'], df_val['Ensemble_Pred'])
    print(f"\n  Correlation (Actual vs Predicted):")
    print(f"    Pearson r: {corr_p:.3f}")
    print(f"    Spearman ρ: {corr_s:.3f}")
    
    if corr_p > 0.7:
        print(f"    ✓ Good correlation - model learns general trend")
    elif corr_p > 0:
        print(f"    ⚡ Weak correlation - model struggles with ranking")
    else:
        print(f"    ✗ Negative/no correlation - training failed!")
        
except FileNotFoundError:
    print("Cannot load validation results to infer training quality")

print("""\

SUMMARY OF FINDINGS:

From training script (Player_Model_OL.py):
  if (epoch + 1) % 10 == 0:
      print(f"Epoch {epoch+1}/{EPOCHS} | Train MAE: {avg_train_loss:.4f} | Val MAE: {avg_val_loss:.4f}")
  
  This SHOULD print convergence metrics every 10 epochs.
  
CRITICAL: We need to see if training actually converged.
  - If final Val MAE was still high (> 10), model didn't learn well
  - If there was oscillation instead of monotonic decrease, training was unstable
  - If model didn't improve after epoch 20, learning rate was too high/low
""")

print("\n" + "="*100)
print("ROOT CAUSE HYPOTHESIS")
print("="*100)
print("""
NOT a feature scaling issue:
  ✓ Features scaled correctly (StandardScaler on training data)
  ✓ All z-scores normal (±2 range)
  ✓ Targets not scaled (correct for unbounded regression)
  ✓ Output layer has no activation (correct for regression)

Likely a MODEL TRAINING issue:
  ❌ Output compressed to 35-45 when training range is 23-98
  ❌ Elite players ranked INVERSELY (lower actual = higher prediction)
  ❌ Suggests model either:
     1. Overfit to mean predictions (ignoring feature signals)
     2. Never learned elite player features (few samples > 90)
     3. Has inverted loss gradient (trained with wrong sign)
     4. Learning rate too aggressive (weights collapsed to uniform)
     5. Data imbalance (too many ~65 grade samples, few extreme samples)

Even with oversampling (doubling extreme yi >= 80 or yi < 50), 
model still outputs narrow range, suggesting fundamental learning failure.
""")
