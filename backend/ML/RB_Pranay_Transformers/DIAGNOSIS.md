# RB Model Performance Diagnosis

## Problem
After adding OL features and rolling stats, model accuracy dropped from **0.63 → 0.47**.

## Potential Issues

### 1. **Feature Quality Issues**
- OL features might have high missing data rates
- Rolling stats might be too noisy for short careers
- Features might not be properly normalized

### 2. **Feature Redundancy**
- OL features might be correlated with existing features (team_performance_proxy)
- Rolling stats might duplicate information already in delta features

### 3. **Model Capacity**
- Adding 9 features might require more model capacity
- Current architecture might not handle the additional complexity

### 4. **Data Leakage Concerns**
- Need to verify OL features are truly from previous year
- Rolling stats need to be calculated correctly for temporal splits

## Recommended Solutions

### Option 1: Remove Problematic Features (Quick Fix)
Temporarily disable OL/rolling features to verify they're the cause:

```python
# In Player_Model_RB.py, comment out:
# df, ol_feature_cols = add_ol_features(df)
# df, rolling_cols = add_rolling_stats(df)
# features.extend(ol_feature_cols)
# features.extend(rolling_cols)

# Set to empty lists:
ol_feature_cols = []
rolling_cols = []
```

### Option 2: Feature Selection
Keep only the most predictive features:
- Test each feature individually
- Remove features with low correlation to target
- Keep only features that improve validation MAE

### Option 3: Improve Feature Engineering
- Better handling of missing OL data (team-level vs league average)
- More robust rolling stats (handle edge cases)
- Feature interactions (OL × RB workload)

### Option 4: Model Tuning
- Increase model capacity (more layers/heads)
- Adjust learning rate for new feature space
- Longer training with new features

## Next Steps

1. **Verify feature quality** - Run debug_features.py to check data
2. **Test without new features** - Confirm they're the problem
3. **Feature importance analysis** - See which features help/hurt
4. **Gradual addition** - Add features one at a time to isolate issues
