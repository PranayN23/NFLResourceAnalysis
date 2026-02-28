# RB Pranay Transformers: Model Performance + File Guide

## Which model performs best?

Using the same evaluation window and settings:
- Years: **2014–2024**
- Top RBs per year: **32** by touches
- Total evaluated rows: **303**

### Aggregate Tier Accuracy (higher is better)

| Mode | Accuracy |
|---|---:|
| **XGBoost (`xgb`)** | **0.7690** |
| Ensemble (`ensemble`) | 0.5545 |
| Transformer (`transformer`) | 0.3432 |

### Conclusion
**`xgb` is currently the best-performing model** for tier classification accuracy in this pipeline.

---

## What each file in this folder does

### Core scripts

- `Player_Model_RB.py`
  - Trains the RB **Transformer** model (Time2Vec + Transformer encoder + regressor).
  - Builds temporal sequences from player history.
  - Applies temporal split (`train <= 2022`, `val = 2023`, `test = 2024`) to avoid leakage.
  - Saves:
    - `rb_player_scaler.joblib`
    - `rb_best_classifier.pth`
  - Prints test metrics including MAE and tier accuracy.

- `RB_Ensemble.py`
  - Trains/refreshes **XGBoost** (`rb_best_xgb.joblib`) and combines predictions with Transformer through `RBModelInference`.
  - Supports two modes:
    - `VALIDATION`: evaluates on 2024 and writes validation report.
    - `DREAM`: projects 2025 rankings.
  - Writes report CSVs used for ranking/analysis.

- `test_model.py`
  - Main evaluation harness for comparing `transformer`, `xgb`, and `ensemble`.
  - Runs per-year predictions and computes aggregate classification metrics.
  - Optional `--export` writes detailed prediction rows to CSV for analysis.

### Model/scaler artifacts

- `rb_best_classifier.pth`
  - Best saved Transformer checkpoint weights.

- `rb_player_scaler.joblib`
  - Scaler fit on training-era features; required for Transformer inference consistency.

- `rb_best_xgb.joblib`
  - Trained XGBoost model used in `xgb` and `ensemble` modes.

### Output reports

- `RB_2024_Validation_Results.csv`
  - Validation-mode ensemble predictions vs actuals for 2024.

- `RB_2025_Final_Rankings.csv`
  - Dream-mode forward projections/rankings for 2025.

- `RB_Test_Results_transformer_2014_2024.csv`
- `RB_Test_Results_xgb_2014_2024.csv`
- `RB_Test_Results_ensemble_2014_2024.csv`
  - Comparable benchmark exports from `test_model.py --export`.

### Runtime cache

- `__pycache__/`
  - Python bytecode cache; not source logic.

---

## Relevance of this folder in the full RB pipeline

1. **Train base Transformer** in `Player_Model_RB.py`.
2. **Train/refresh XGB + run blend** in `RB_Ensemble.py`.
3. **Benchmark all modes** in `test_model.py`.
4. Use exported CSVs to compare errors and decide deployment mode.

Given current results, `xgb` is strongest for pure tier accuracy, while `ensemble` remains useful if you want a blend with transformer signal and confidence outputs.
