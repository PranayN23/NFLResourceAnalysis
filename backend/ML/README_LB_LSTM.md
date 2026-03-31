# LB Position LSTM Prediction Model

This guide explains how to train an LSTM model to predict linebacker (LB) performance using weighted team-position averages.

---

## Overview

The model predicts future LB performance (e.g., `grades_defense`) using:
- **Previous year's weighted averages** for all stats (lagged features)
- **Current year context** (Win %, EPA, Cap Space, etc.)

---

## Step-by-Step Instructions

### Step 1: Generate the Data

First, run the data extraction script to pull LB data from MongoDB and calculate weighted averages with lagged features:

```bash
cd /Users/abhitandon/Downloads/Code/Projects/ML@P/NFLResourceAnalysis/backend/ML
python pull_lb_data.py
```

**What this does:**
- Pulls all LB player data from MongoDB (2010-2024)
- Calculates weighted averages per team-position-year
- Adds lagged features (previous year's data)
- Saves to: `lb_weighted_averages_by_team_pos_year.csv`
- Pushes back to MongoDB collection: `LB_Weighted_Averages`

**Expected output:**
- Console shows processing for each team and year
- Creates `lb_weighted_averages_by_team_pos_year.csv` (~450-500 rows)

---

### Step 2: Train the LSTM Model

Run the LSTM training script:

```bash
python lb_weighted_lstm.py
```

**What this does:**
1. Loads the weighted averages CSV file
2. Prepares data for LSTM (filters LB position, extracts features)
3. Splits data:
   - **Training set:** ~64% of data
   - **Validation set:** ~16% of data
   - **Test set:** ~20% of data
4. Normalizes features using StandardScaler (fit on training data only)
5. Builds and trains LSTM model with:
   - 2 LSTM layers (128 → 64 units)
   - 3 Dense layers (64 → 32 → 16 units)
   - Dropout and BatchNormalization for regularization
6. Evaluates on all three sets
7. Saves model and visualizations

**Expected output:**
- Training progress with loss/MAE metrics
- Final R², RMSE, MAE for train/val/test sets
- Saved files:
  - `lb_lstm_model_grades_defense.h5` (trained model)
  - `lb_lstm_training_history.png` (loss curves)
  - `lb_lstm_predictions.png` (predictions vs actual)
  - `lb_lstm_residuals.png` (residual plots)

---

## Model Configuration

### Default Target
- **Target:** `weighted_avg_grades_defense` (overall defensive performance grade)

To predict a different stat, modify line 288 in `lb_weighted_lstm.py`:
```python
target_stat = 'grades_defense'  # Change to: tackles, sacks, interceptions, etc.
```

### Input Features (35 total)

**Previous year weighted averages (31):**
- `prev_weighted_avg_grades_defense`
- `prev_weighted_avg_tackles`
- `prev_weighted_avg_sacks`
- `prev_weighted_avg_interceptions`
- ... (all performance stats)

**Previous year summary (4):**
- `prev_total_snap_counts_defense`
- `prev_total_players`
- `prev_sum_Cap_Space`
- `prev_sum_adjusted_value`

**Current year context (4):**
- `Win_Percent`
- `Net_EPA`
- `sum_Cap_Space`
- `sum_adjusted_value`

### Model Architecture

```
Input: (batch_size, 1 timestep, 35 features)
  ↓
LSTM(128 units) + Dropout(0.3) + BatchNorm
  ↓
LSTM(64 units) + Dropout(0.3) + BatchNorm
  ↓
Dense(64) + Dropout(0.3)
  ↓
Dense(32) + Dropout(0.2)
  ↓
Dense(16)
  ↓
Output: 1 value (predicted grade)
```

### Training Configuration

- **Optimizer:** Adam (learning_rate=0.001)
- **Loss:** Mean Squared Error (MSE)
- **Metrics:** Mean Absolute Error (MAE)
- **Epochs:** 200 (with early stopping)
- **Batch size:** 32
- **Early stopping:** Patience = 25 epochs on validation loss
- **Learning rate scheduler:** Reduces LR by 0.5 when val_loss plateaus (patience=10)

---

## Data Split Strategy

**Why this split?**
- **Training (64%):** Used to learn patterns
- **Validation (16%):** Monitors overfitting during training, tunes hyperparameters
- **Test (20%):** Final evaluation on completely unseen data

**Random state:** 42 (ensures reproducibility)

---

## Expected Performance

Performance will vary based on data quality, but typical results:

- **R² Score:** 0.4 - 0.7 (moderate to good predictive power)
- **RMSE:** 5-10 points on PFF grade scale
- **MAE:** 3-7 points on PFF grade scale

---

## Customization Options

### 1. Change Target Stat
Edit line 288 in `lb_weighted_lstm.py`:
```python
target_stat = 'tackles'  # or 'sacks', 'interceptions', etc.
```

### 2. Adjust Train/Val/Test Split
Edit lines 294-295 in `lb_weighted_lstm.py`:
```python
X_train, X_val, X_test, y_train, y_val, y_test = split_data(
    X, y, test_size=0.15, val_size=0.25  # Custom split
)
```

### 3. Modify Model Architecture
Edit the `build_lstm_model()` function (lines 162-198):
```python
# Example: Add more layers or change units
tf.keras.layers.LSTM(256, return_sequences=True, input_shape=input_shape),
```

### 4. Adjust Training Parameters
Edit lines 311-313 in `lb_weighted_lstm.py`:
```python
history = train_model(
    model, X_train_scaled, y_train, X_val_scaled, y_val,
    epochs=300, batch_size=16  # Custom parameters
)
```

---

## Troubleshooting

### Error: "File not found"
**Solution:** Run `python pull_lb_data.py` first to generate the data.

### Error: "No valid records found"
**Cause:** Not enough data with complete lagged features.
**Solution:** Check that you have data for multiple consecutive years.

### Warning: "Low R² score"
**Possible reasons:**
- Limited data (need more years/teams)
- High variance in LB performance
- Need more features or different model architecture

### Model overfitting (Train R² >> Test R²)
**Solutions:**
- Increase dropout rates
- Reduce model complexity
- Add more training data
- Use L2 regularization

---

## Output Files

After running, you'll have:

1. **lb_weighted_averages_by_team_pos_year.csv**
   - Processed data with weighted averages and lagged features

2. **lb_lstm_model_grades_defense.h5**
   - Trained model (can be loaded for predictions)

3. **lb_lstm_training_history.png**
   - Loss and MAE curves over training epochs

4. **lb_lstm_predictions.png**
   - Scatter plots of predicted vs actual values for all sets

5. **lb_lstm_residuals.png**
   - Residual plots to check for prediction bias

---

## Next Steps

### Make Predictions on New Data

```python
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load model
model = tf.keras.models.load_model('lb_lstm_model_grades_defense.h5')

# Load new data (must have same features)
new_data = pd.read_csv('new_lb_data.csv')

# Prepare features (same as training)
# ... (extract features, normalize, reshape)

# Predict
predictions = model.predict(new_data_prepared)
```

### Compare Multiple Models

Train models with different architectures or targets, then compare performance.

### Feature Importance Analysis

Use techniques like SHAP or permutation importance to understand which features matter most.

---

## Questions?

Check the code comments in `lb_weighted_lstm.py` for detailed explanations of each function.
