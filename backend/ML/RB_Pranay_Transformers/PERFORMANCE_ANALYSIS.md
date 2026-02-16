# RB Model Performance Analysis

## Current Results

- **With new features (31)**: Accuracy 0.4455
- **Without new features (22)**: Accuracy 0.4125  
- **Original baseline**: Accuracy 0.63

## Key Issues Identified

### 1. **Model Too Conservative**
- Too many "Starter" predictions (69% recall)
- Too few "Elite" predictions (15-19% recall)
- Elite players severely underpredicted:
  - Derrick Henry: 71.9 predicted vs 94.2 actual (-22.3)
  - Josh Jacobs: 67.3 predicted vs 92.3 actual (-25.0)
  - Saquon Barkley: 66.0 predicted vs 87.6 actual (-21.6)

### 2. **Age Decay Too Aggressive**
Current age decay:
- Age 28-29: 0.8 pts/year
- Age 30-31: 1.5 pts/year  
- Age 32+: 2.0 pts/year

For Derrick Henry (age 30): Gets ~4.6 point penalty
For Josh Jacobs (age 26): Gets 0 penalty (should be fine)

### 3. **Ensemble Weights**
Current: 60% Transformer, 40% XGBoost
- May need adjustment based on individual model performance
- Transformer might be pulling predictions down

## Recommended Fixes

### Fix 1: Reduce Age Decay for Elite Players
Elite RBs (grade > 80) often maintain performance longer:

```python
def get_age_decay_factor(self, age, current_grade=None):
    base_decay = ...  # current logic
    
    # Elite players age slower
    if current_grade and current_grade > 80:
        return base_decay * 0.5  # Half the penalty
    
    return base_decay
```

### Fix 2: Adjust Ensemble Weights
If Transformer is underperforming, reduce its weight:

```python
# Current: 60% Transformer, 40% XGB
# Try: 50% Transformer, 50% XGB
# Or: 40% Transformer, 60% XGB
```

### Fix 3: Calibration/Post-Processing
Add calibration to shift predictions upward:

```python
# After ensemble, apply calibration
if final_grade < 70 and transformer_grade > 75:
    # Transformer thinks elite, but ensemble pulled down
    final_grade = final_grade * 1.1  # Boost slightly
```

### Fix 4: Check Model Training
- Verify Transformer validation MAE
- Check if model is underfitting (high train MAE)
- Consider more training epochs or different hyperparameters

### Fix 5: Feature Quality
- Run `analyze_feature_impact.py` to check correlations
- Remove features with negative correlation
- Verify OL features are actually helping

## Next Steps

1. **Check individual model performance**:
   ```python
   # Test Transformer alone
   tier, details = engine.get_prediction(history, mode="transformer")
   
   # Test XGBoost alone  
   tier, details = engine.get_prediction(history, mode="xgb")
   ```

2. **Analyze prediction errors**:
   - Are errors systematic (always low)?
   - Do they correlate with age?
   - Do they correlate with actual grade?

3. **Adjust age decay**:
   - Reduce penalties for high-performing players
   - Make decay more gradual

4. **Tune ensemble weights**:
   - Test different weight combinations
   - Use validation set to find optimal weights
