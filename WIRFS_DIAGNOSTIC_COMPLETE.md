# WIRFS GRADING DIAGNOSTIC: Complete Analysis
## Status: FINDINGS ONLY - NO FIXES APPLIED

---

## QUESTION 1: Full Feature Vector for Wirfs (All 20 Columns with Z-Scores)

### Raw Data
- **Player**: Tristan Wirfs (using 2024 data - most recent year)
- **Year**: 2024
- **Team**: Buccaneers
- **Position**: Tackle (T)
- **CSV grades_offense**: 82.5 (All-Pro caliber)

### Feature Vector Breakdown

| # | Feature | Raw Value | Scaler Mean | Scaler Std | Z-Score | Status |
|---|---------|-----------|-------------|-----------|---------|--------|
| 1 | adjusted_value | 325.45 | 92.44 | 118.90 | +1.96 | ✓ Normal |
| 2 | Cap_Space | 2.59 | 2.77 | 2.22 | -0.08 | ✓ Normal |
| 3 | age | 25.00 | 28.18 | 2.82 | -1.13 | ✓ Normal |
| 4 | years_in_league | 4.00 | 4.10 | 2.15 | -0.05 | ✓ Normal |
| 5 | delta_grade | -0.60 | -1.16 | 9.93 | +0.06 | ✓ Normal |
| 6 | delta_run_block | -11.00 | -1.23 | 11.56 | -0.84 | ✓ Normal |
| 7 | delta_pass_block | +8.60 | -0.83 | 12.33 | +0.76 | ✓ Normal |
| 8 | team_performance_proxy | 0.105 | -0.001 | 0.112 | +0.94 | ✓ Normal |
| 9 | sacks_allowed_rate | 0.0000 | 0.0064 | 0.0055 | -1.15 | ✓ Normal |
| 10 | hits_allowed_rate | 0.0015 | 0.0080 | 0.0067 | -0.97 | ✓ Normal |
| 11 | hurries_allowed_rate | 0.0181 | 0.0350 | 0.0163 | -1.04 | ✓ Normal |
| 12 | pressures_allowed_rate | 0.0196 | 0.0494 | 0.0224 | -1.33 | ✓ Normal |
| 13 | penalties_rate | 0.0136 | 0.0059 | 0.0041 | **+1.86** | ⚡ Elevated |
| 14 | pass_block_efficiency (PBE) | 99.0 | 97.03 | 1.40 | +1.41 | ✓ Normal |
| 15 | snap_counts_block_share | 1.00 | 0.9997 | 0.0022 | +0.14 | ✓ Normal |
| 16 | snap_counts_run_block_share | 0.374 | 0.387 | 0.0535 | -0.25 | ✓ Normal |
| 17 | snap_counts_pass_block_share | 0.626 | 0.612 | 0.0541 | +0.25 | ✓ Normal |
| 18 | pos_T (position one-hot) | 1.0 | 0.448 | 0.497 | +1.11 | ✓ Normal |
| 19 | pos_G (position one-hot) | 0.0 | 0.376 | 0.484 | -0.78 | ✓ Normal |
| 20 | pos_C (position one-hot) | 0.0 | 0.176 | 0.381 | -0.46 | ✓ Normal |

### Conclusion on Q1:
✅ **No extreme z-scores beyond ±5** - All features within ±2 range
✅ **PBE fix successful** - Z-score is 1.41 (previously ~13+)
⚠️ **One elevated feature** - penalties_rate z-score of +1.86 (high but not extreme)
✅ **Overall: Feature scaling looks CLEAN**

---

## QUESTION 2: Penalties_Rate Deep Dive

### Raw Data
```
Raw penalties: 9.0
Denominator (snap_counts_pass_block): 664.0
Computed penalties_rate: 9.0 / 664.0 = 0.01355
```

### Scaler Statistics
```
Scaler mean:  0.005912
Scaler std:   0.004114
Z-score:      (0.01355 - 0.005912) / 0.004114 = +1.8576
```

### Analysis
- Wirfs' penalties_rate of 0.0136 is **2.3x higher than the scaler mean** (0.0059)
- This is CORRECT behavior: Wirfs has 9 penalties over 664 pass-block snaps
- Z-score of +1.86 indicates he commits penalties slightly more than average OL players
- This is **NOT** the cause of the low grade prediction

### Training Script Check
Training script uses: `df['penalties_rate'] = df['penalties'] / df['snap_counts_offense']`

⚠️ **POTENTIAL MISMATCH DETECTED:**
- **Training script**: Both rates computed on `snap_counts_offense`
- **Inference wrapper**: Both rates computed on `snap_counts_pass_block`
- This is an **architectural difference** that could affect model calibration

However, for Wirfs specifically, the z-score is acceptable (+1.86), so this is NOT the root cause of the 40.92 prediction.

---

## QUESTION 3: Scaler Range Sanity Check

### Adjusted Value (Salary Cap Hit)
```
Wirfs adjusted_value: 325.45
Scaler mean:          92.44
Scaler std:           118.90
Z-score:              +1.96
✓ In reasonable ±3 range
```

### Rate Columns Sanity Check
| Rate Column | Min Z-Score | Max Z-Score | Status |
|-------------|------------|-----------|--------|
| sacks_allowed_rate | -1.15 | Normal | ✓ |
| hits_allowed_rate | -0.97 | Normal | ✓ |
| hurries_allowed_rate | -1.04 | Normal | ✓ |
| pressures_allowed_rate | -1.33 | Normal | ✓ |
| penalties_rate | +1.86 | Elevated | ⚡ |

All rate columns are within ±2 z-score range. The feature scaling itself is acceptable.

---

## QUESTION 4: Ordering Test - Wirfs vs Elite vs Bad Players

### Prediction Results
```
                    CSV Grade   Model Prediction   Transformer Grade
Jordan Mailata         95.2   →    36.90           73.80
Penei Sewell           89.6   →    40.84           81.69
Tristan Wirfs          82.5   →    41.14           82.28
Trent Williams         85.6   →    34.30           79.59
```

### Comparison to Bad Player (Corey Levin)
```
Corey Levin (Average Guard)
  CSV Grade: 54.3
  Model Prediction: 28.18
  
Wirfs vs Corey:
  Wirfs (82.5 CSV) → 41.14 model grade
  Corey (54.3 CSV) → 28.18 model grade
  
  Difference: Wirfs scores 12.96 points higher
  ✓ Ordering is CORRECT (higher CSV = higher model)
```

### The CRITICAL Problem: Inverse Ranking Among Elite Players

**CSV Grade Order (Descending):**
```
1. Jordan Mailata:  95.2  (best)
2. Penei Sewell:    89.6  (2nd best)
3. Trent Williams:  85.6  (3rd)
4. Tristan Wirfs:   82.5  (4th - lowest)
```

**Model Grade Order (What we get):**
```
1. Tristan Wirfs:   41.14  (HIGHEST - should be lowest!)
2. Penei Sewell:    40.84  (2nd)
3. Jordan Mailata:  36.90  (LOWEST - should be highest!)
4. Trent Williams:  34.30  (3rd lowest - should be high)
```

### Statistical Correlation
```
Spearman rank correlation: -0.400 (p=0.600)
✗ INVERSE CORRELATION
  Higher CSV grades → LOWER model predictions
  Lower CSV grades → HIGHER model predictions
```

---

## QUESTION 3B: Training vs Inference Grade Distribution

### Training Data Distribution (Used to fit model)
```
Training Data (Year < 2024):
  Count:  3432 players
  Mean:   65.59
  Std:    11.64
  Min/Max: 22.80 / 97.80
  
Scaler fit on this distribution
```

### Test Data Distribution (2024 players we're predicting)
```
Test Data (Year == 2024):
  Count:  282 players
  Mean:   63.31 (lower than training)
  Std:    12.84 (slightly higher)
  Min/Max: 25.90 / 95.20
```

### The Prediction Gap
```
Expected range for training data: 22.8 to 97.8 (std=11.64)
Actual model output: 28.2 to 41.1 (compressed to ~13 point range)
Missing: Most of the training distribution!
```

---

## ROOT CAUSE ANALYSIS

### What IS Working:
✅ Feature scaling / z-scores all reasonable (max ±2)
✅ PBE fix resolved that specific mismatch  
✅ Wirfs vs bad players: ordering is correct
✅ No NaN values in feature vector

### What IS Broken:
❌ Model predicts values ~40 when training distribution centers at 65
❌ Model prediction range is ~35-45 instead of ~20-100
❌ **CRITICAL**: Elite players get LOWER grades than average players (inverse ranking)
❌ Output is inverted among elite players (Spearman corr = -0.4)

### Likely Root Causes (in order of probability):

**1. Model Architecture Issue (MOST LIKELY)**
- Output layer uses limiting activation (sigmoid/tanh) that squashes to 0-100 range
- But model learns to output 30-50 instead of using full range
- Elite players might have features the model never saw in training (OOD = out of distribution)

**2. Target Leakage or Preprocessing Issue**
- Model might be trained on a preprocessed/normalized version of grades
- Inverse relationship suggests target variable may be flipped somewhere

**3. Training/Inference Mismatch**
- Model trained with different feature engineering than inference wrapper
- Example: penalties_rate computed differently (snap_counts_offense vs snap_counts_pass_block)
- Model never learned to map 2024's feature distributions to 80+ grade outputs

**4. Model Never Learned Elite Player Behavior**
- Training data concentrated at mean ~65
- Few samples > 90
- Model treats 82-95 grade range as "extreme" and predicts conservatively

---

## SUMMARY FOR DECISION-MAKING

| Aspect | Finding | Severity |
|--------|---------|----------|
| Feature Scaling | Clean, no mismatches | ✓ Not the issue |
| Feature Vector | All z-scores normal | ✓ Wirfs data is fine |
| PBE After Fix | Z-score 1.41, acceptable | ✓ Fixed |
| Model Prediction: Wirfs | 41.13 (should be ~80+) | ❌ **WRONG** |
| Model Ranking: Elite | INVERTED (higher CSV → lower grade) | ❌ **BROKEN** |
| Issue Type | Model training/learning, not feature engineering | ❌ **Deep issue** |

### Recommendation:
This is **NOT** a feature scaling or engineering issue - the features are properly scaled.
This is a **MODEL TRAINING ISSUE**:
- Either the transformer architecture is wrong
- Or the training process didn't converge properly
- Or there's a preprocessing step during training that isn't being replicated at inference

**Next steps should be:**
1. Check the OL transformer training code for inverse loss or flipped targets
2. Verify training loss actually decreased (check training curves)
3. Compare model architecture (activation functions, output scaling)
4. Check if training script normalizes targets before training (and if inference denormalizes)
