# RB Model Improvements Summary

## Changes Implemented

### 1. **Previous-Year OL Features (Snap-Weighted)**
- **Files**: G.csv, T.csv, C.csv (all OL positions)
- **Features Added**:
  - `grades_run_block_ol_prev` - Previous year run blocking grade
  - `grades_pass_block_ol_prev` - Previous year pass blocking grade  
  - `penalties_ol_prev` - Previous year penalties
  - `pressures_allowed_ol_prev` - Previous year pressures allowed
- **Method**: Snap-weighted averages using `snap_counts_offense`
- **Temporal Integrity**: OL year Y-1 → RB year Y (no data leakage)

### 2. **Rolling Statistics Features**
- **Features Added**:
  - `rb_yards_rolling_std` - 3-year rolling std dev of yards
  - `rb_touches_rolling_std` - 3-year rolling std dev of touches
  - `rb_grades_rolling_std` - 3-year rolling std dev of grades
  - `rb_yards_rolling_mean` - 3-year rolling mean of yards
  - `rb_grades_rolling_mean` - 3-year rolling mean of grades
- **Purpose**: Capture volatility and consistency trends

### 3. **Feature Counts**
- **Transformer Features**: 22 → 31 features (+9)
- **XGBoost Features**: 14 → 21 features (+7)

## Files Modified

1. **Player_Model_RB.py** - Training script
   - Added `add_ol_features()` with G+T+C support and snap-weighting
   - Added `add_rolling_stats()` function
   - Updated feature list

2. **RB_Ensemble.py** - Ensemble evaluation script
   - Updated feature lists
   - Added OL and rolling stats to data preparation
   - Handles lagged features for XGBoost

3. **rb_model_wrapper.py** - Inference wrapper
   - Updated feature lists
   - Added batch prediction support
   - Added optional stacking ensemble support
   - Robust path resolution for OL data

## Running the Pipeline

### Option 1: Use the provided script
```bash
cd /Users/pranaynandkeolyar/Documents/NFLSalaryCap
source .venv/bin/activate
./backend/ML/RB_Pranay_Transformers/run_rb_training.sh
```

### Option 2: Run manually

**Step 1: Train Transformer**
```bash
cd /Users/pranaynandkeolyar/Documents/NFLSalaryCap
source .venv/bin/activate
python backend/ML/RB_Pranay_Transformers/Player_Model_RB.py
```

**Step 2: Run Validation Mode**
```bash
# Edit RB_Ensemble.py to set MODE = "VALIDATION"
python backend/ML/RB_Pranay_Transformers/RB_Ensemble.py
```

**Step 3: Run Dream Mode**
```bash
# Edit RB_Ensemble.py to set MODE = "DREAM"
python backend/ML/RB_Pranay_Transformers/RB_Ensemble.py
```

## Expected Outputs

1. **Training Output**:
   - `rb_best_classifier.pth` - Trained Transformer model
   - `rb_player_scaler.joblib` - Feature scaler
   - Training/validation/test MAE metrics

2. **Validation Mode Output**:
   - `rb_best_xgb.joblib` - Trained XGBoost model
   - `RB_2024_Validation_Results.csv` - 2024 predictions vs actuals

3. **Dream Mode Output**:
   - `RB_2025_Final_Rankings.csv` - 2025 projections

## Expected Improvements

With OL features and rolling stats, the model should:
- Better account for blocking context (OL quality affects RB production)
- Capture player consistency/volatility trends
- Reduce prediction errors, especially for:
  - RBs behind elite OL (shouldn't over-penalize)
  - RBs behind weak OL (should account for limitations)
  - Inconsistent performers (volatility features)

## Troubleshooting

If you encounter segmentation faults (exit code 139):
1. Ensure virtual environment is activated
2. Check PyTorch installation: `python -c "import torch; print(torch.__version__)"`
3. Try forcing CPU mode (already set in Player_Model_RB.py)
4. Check if CUDA libraries are causing issues

## Next Steps

After training completes:
1. Compare validation MAE with previous model
2. Analyze feature importance in XGBoost
3. Review prediction errors in validation results
4. Consider training stacking ensemble for optimal weights
