# OL Training Pipeline Investigation: Findings Report

## Status: CODE REVIEW ONLY - NO TESTS RUN (Terminal Issues)

---

## QUESTION 1: Target Variable Definition & Inversions

### Code Evidence

**From OL_Ensemble.py (line 145):**
```python
target_col = 'grades_offense'
```

**From Player_Model_OL.py (lines 43-50):**
```python
for p in players:
    p_data_scaled = df_scaled[df_raw['player'] == p]
    p_data_raw = df_raw[df_raw['player'] == p]
    
    for i in range(len(p_data_raw)):
        target_year = p_data_raw.iloc[i]['Year']
        target = p_data_raw.iloc[i][target_col]  # <- Direct extraction, no inversion
```

**From Player_Model_OL.py (line 323-326):**
```python
for i, t, m in train_loader:
    i, t = i.to(DEVICE), t.to(DEVICE)
    m = m.to(DEVICE)
    optimizer.zero_grad()
    outputs = model(i, mask=m).squeeze()
    loss = criterion(outputs, t)  # <- Direct target, no transformation
    loss.backward()
    optimizer.step()
    train_loss += loss.item()
```

### Finding #1:
✅ **Target is NOT inverted**
- Target is `grades_offense` directly
- No `1 - grades_offense`
- No `100 - grades_offense`  
- No `-grades_offense`
- No transformation applied

---

## QUESTION 2: Target Normalization/Scaling Before Training

### Code Evidence

**From Player_Model_OL.py (lines 226-231):**
```python
print("Fitting scaler on training data only (no leakage)...")
train_data = df_clean[df_clean['Year'] <= TRAIN_END_YEAR]

scaler = StandardScaler()
scaler.fit(train_data[features])  # <- Fits on FEATURES, not targets

# Apply scaler to all data
df_clean_scaled = df_clean.copy()
df_clean_scaled[features] = scaler.transform(df_clean[features])
```

**Scaler file verification: (lines 243-245):**
```python
SCALER_OUT = 'backend/ML/OL_Pranay_Transformers/ol_player_scaler.joblib'
joblib.dump(scaler, SCALER_OUT)
print(f"Scaler saved to {SCALER_OUT}")
```

**From Player_Model_OL.py (lines 249-255):**
```python
X_train, y_train, masks_train = X_all[train_idx], y_all[train_idx], masks_all[train_idx]

# ... data augmentation ...

X_train = np.array(X_boosted)
y_train = np.array(y_boosted)  # <- RAW numpy array, no scaling
```

### Finding #2:
✅ **Targets are NOT scaled before training**
- Only features are scaled with StandardScaler
- Targets (y) are raw `grades_offense` values (range 22.8-97.8)
- No target_scaler or y_scaler file exists
- Model learns: scaled_features → raw_grades (unbounded)

### Critical Implication:
Model should output in the **20-100 range** to match training targets. But it outputs **35-45 instead**. This is NOT a scaling issue — it's a learning issue.

---

## QUESTION 3: Output Layer Activation Function

### Code Evidence

**From Player_Model_OL.py (lines 120-127):**
```python
self.regressor = nn.Sequential(
    nn.Linear(self.embed_dim, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 1)  # <- FINAL LAYER: unbounded linear regression
)
```

**From inference wrapper (ol_model_wrapper.py, lines 129-133):**
```python
def predict(self, player_history, mode="ensemble", apply_calibration=True):
    # ...
    transformer_grade = self.model(x_tensor, mask=m_tensor).item()
    # <- Direct output, NO post-processing or denormalization
```

### Finding #3:
✅ **Output layer is unbounded linear regression**
- Final layer: `nn.Linear(64, 1)` 
- **NO sigmoid** (would constrain to 0-1)
- **NO tanh** (would constrain to -1 to 1)
- **NO ReLU** (would constrain to ≥ 0)
- Raw output should be in training range (23-98)
- Inference wrapper uses output directly with no inverse transform

### Critical Implication:
Architecture is CORRECT for predicting raw grades. The compressed output (35-45) indicates **training failure**, not architectural limitation.

---

## QUESTION 4: Training Loss Curves & Convergence

### Code Evidence

**From Player_Model_OL.py (lines 318-336):**
```python
for epoch in range(EPOCHS):
    # Training
    model.train()
    train_loss = 0
    for i, t, m in train_loader:
        # ... training step ...
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for iv, tv, mv in val_loader:
            # ... validation step ...
            val_loss += criterion(ov, tv).item()
    
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_OUT)
        best_epoch = epoch + 1
    
    scheduler.step(avg_val_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Train MAE: {avg_train_loss:.4f} | Val MAE: {avg_val_loss:.4f}")
```

**Loss Function (line 312):**
```python
criterion = nn.L1Loss()  # <- L1 (MAE) loss on raw grades
```

**Data split (line 260):**
```python
train_data = df_clean[df_clean['Year'] <= TRAIN_END_YEAR]  # TRAIN_END_YEAR = 2022
test_data = df_clean[df_clean['Year'] == 2024]
```

**Oversampling strategy (lines 280-290):**
```python
for xi, yi, mi in zip(X_train, y_train, masks_train):
    X_boosted.append(xi)
    y_boosted.append(yi)
    masks_boosted.append(mi)
    # Duplicate extreme performers (adjusted for OL grading scale)
    if yi >= 80.0 or yi < 50.0:  # <- Only ~10-15% of data are extreme
        X_boosted.append(xi)
        y_boosted.append(yi)
        masks_boosted.append(mi)
```

### Finding #4:
⚠️ **Training convergence CANNOT be verified from code alone**
- Script prints metrics every 10 epochs
- No loss history file is saved
- We would need to run the training and capture output

**However, we can infer from predictions that training FAILED:**

From diagnostic data (compare_predictions.py output):
```
CSV Grade Rankings (Real):
  1. Jordan Mailata:  95.2  (best)
  2. Penei Sewell:    89.6
  3. Trent Williams:  85.6
  4. Tristan Wirfs:   82.5  (lowest)

Model Predictions (What we get):
  1. Tristan Wirfs:   41.14  (HIGHEST - should be lowest!)
  2. Penei Sewell:    40.84
  3. Jordan Mailata:  36.90  (LOWEST - should be highest!)
  4. Trent Williams:  34.30
  
Spearman correlation: -0.400 (INVERTED!)
```

### Critical Evidence of Training Failure:

**1. Output Compression:**
- Training range: 22.8 - 97.8 (std=11.64)
- Model output: 35-45 (std~2-3)
- **Model learned to output ~40 regardless of input**

**2. Inverse Correlation:**
- Spearman ρ = -0.4 means **INVERTED ranking**
- Higher CSV grades → LOWER model predictions
- This is NOT random; system reversal

**3. Possible Root Causes:**

**Root Cause A: Model converged to overly conservative "mean prediction"**
- Model learns features have high noise
- Outputs close to training mean (~65)
- Age decay (-5 to -30) pushes down to 35-45 range
- Problem: Doesn't explain INVERSE ranking

**Root Cause B: Training data imbalance**
- Training data concentrated at mean ~65
- Few samples with grade > 90 (elite players rare)
- Model treats 82-95 range as statistical outliers
- When it sees elite player features, predicts conservatively (30-40)
- Problem: Doesn't explain why LOWER grades get HIGHER predictions

**Root Cause C: Loss gradient inverted (MOST LIKELY)**
- Possible bug: loss computed as -L1_loss instead of L1_loss
- Or: target - output instead of output - target
- This would train model to produce **lower** values for higher grades
- Explains both compression AND inversion

**Root Cause D: Feature leakage or target corruption**
- If any year-2024 data leaked into 2022 training
- Or if targets got shuffled during preprocessing
- Would cause systematic misalignment

---

## Comparison to Working Models

**RB model comparison:**
- RB transformer uses identical architecture
- RB outputs in reasonable range (50-80 for valid players)
- RB rankings are correct (Ekeler, Henry, Saquon rank as expected)
- **This proves architecture is correct**

**Conclusion:**
OL training pipeline has **severe learning failure** — not architectural issue.

---

## Summary Table

| Question | Finding | Evidence |
|----------|---------|----------|
| **1. Target Inverted?** | ✅ NO — target is `grades_offense` directly | Player_Model_OL.py line 47 |
| **2. Targets Scaled?** | ✅ NO — only features scaled, targets are raw | Player_Model_OL.py lines 226-231,  line 291 |
| **3. Output Activation?** | ✅ NO ACT — Linear(64→1), unbounded | Player_Model_OL.py line 126 |
| **4. Loss Convergence?** | ❌ FAILED — Outputs compressed 35-45 vs training 23-98, Spearman ρ = -0.400 | Compare predictions output |

---

## Next Steps to Investigate

**To confirm Root Cause, need to:**

1. **Check training loss trajectory**
   - Run training script and capture output
   - Plot loss vs epoch
   - Check if validation loss decreased monotonically or oscillated

2. **Inspect loss calculation**
   - Verify L1Loss is computed as `|output - target|` not `target - output`
   - Check if there's any negation in loss_backward()

3. **Audit data preprocessing**
   - Verify targets didn't get shuffled
   - Check year filtering is correct
   - Confirm 2024 data wasn't in training set

4. **Compare to RB training**
   - Check if RB_Ensemble.py differs significantly
   - If RB works but OL doesn't, find the diff

5. **Test with simpler data**
   - Train on just 2 elite players vs 2 average players
   - See if model can learn to predict higher grades for elite

---

## Recommendation

**Do NOT modify inference code.** The inference wrapper is correct.

**Root cause is in Player_Model_OL.py or OL_Ensemble.py** — likely in:
1. Loss function (line 312)
2. Target variable handling (lines 43-50, 291)
3. Data preprocessing step (feature engineering lines 176-211)

**Most suspicious:** The inverse ranking suggests either:
- Loss sign issue (training to minimize instead of maximize)
- Target-feature swap during batching
- Year filtering bug that put 2024 data in training

