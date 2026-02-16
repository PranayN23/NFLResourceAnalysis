# Quick Fix: Test Without New Features

## Problem
Model accuracy dropped from **0.63 → 0.47** after adding OL features and rolling stats.

## Quick Test: Disable New Features

To test if the new features are causing the problem, edit `Player_Model_RB.py`:

**Line ~285-286:**
```python
USE_OL_FEATURES = False   # Set to False to disable OL features
USE_ROLLING_STATS = False  # Set to False to disable rolling stats
```

Then retrain:
```bash
/opt/anaconda3/bin/python backend/ML/RB_Pranay_Transformers/Player_Model_RB.py
```

## If Performance Improves

This confirms the new features are hurting. Possible fixes:

1. **Feature Quality**: Check for missing data, outliers, incorrect calculations
2. **Feature Selection**: Keep only predictive features (test individually)
3. **Feature Engineering**: Improve how features are calculated
4. **Normalization**: Ensure new features are properly scaled

## If Performance Stays Bad

The issue might be:
1. Model needs retuning with new feature count
2. Training data quality issues
3. Model architecture needs adjustment

## Next Steps After Testing

1. **Gradual Addition**: Add features one at a time to find the culprit
2. **Feature Importance**: Use XGBoost feature importance to identify helpful features
3. **Correlation Analysis**: Check if new features correlate with target
4. **Data Quality**: Verify OL data completeness and accuracy
