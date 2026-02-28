# OL Pranay Transformers: Model Performance + File Guide

## What was repeated from RB to OL

The same hybrid approach was implemented for OL:
- Keep Transformer (with Time2Vec) for temporal sequence modeling.
- Keep XGBoost for strong tabular prediction.
- Add a **stacking feature**: `t2v_transformer_signal` so XGBoost directly learns from Transformer signal.

This was added end-to-end in:
- `backend/ML/OL_Pranay_Transformers/OL_Ensemble.py` (training-time feature generation)
- `backend/agent/ol_model_wrapper.py` (inference-time feature generation + backward compatibility)

---

## Latest OL benchmark metrics

Benchmark command pattern:
- `python backend/ML/OL_Pranay_Transformers/test_model.py --mode <mode> --year_start 2014 --year_end 2024 --export`

Total evaluations: **1890**

| Mode | Accuracy |
|---|---:|
| **XGBoost (`xgb`)** | **0.5138** |
| Ensemble (`ensemble`) | 0.4984 |
| Transformer (`transformer`) | 0.4513 |

### Conclusion
For OL tier classification, **XGBoost is currently best**, but the ensemble remains close and now includes explicit Time2Vec transformer signal.

---

## What each file in this folder does

### Core scripts

- `Player_Model_OL.py`
  - Trains OL Transformer model and saves:
    - `ol_best_classifier.pth`
    - `ol_player_scaler.joblib`
  - Uses engineered OL temporal features and positional one-hot columns.

- `OL_Ensemble.py`
  - Loads/combines OL datasets (`G.csv`, `C.csv`, `T.csv`), engineers lag/rate/share features.
  - Adds transformer-derived stacking feature: `t2v_transformer_signal`.
  - Trains `ol_best_xgb.joblib`.
  - Produces validation or 2025 projection outputs depending on `MODE`.

- `test_model.py`
  - Evaluates `transformer`, `xgb`, and `ensemble` modes over year ranges.
  - Computes aggregate tier accuracy/classification report.
  - Exports detailed rows with `--export`.

### Model/scaler artifacts

- `ol_best_classifier.pth` — Trained OL Transformer checkpoint.
- `ol_player_scaler.joblib` — Scaler for OL transformer features.
- `ol_best_xgb.joblib` — Trained OL XGBoost model (includes `t2v_transformer_signal` in new runs).

### Output reports

- `OL_2024_Validation_Results.csv` — Validation-mode output report.
- `OL_2025_Final_Rankings.csv` — Dream-mode 2025 ranking/projection report.
- `OL_Test_Results_transformer_2014_2024.csv`
- `OL_Test_Results_xgb_2014_2024.csv`
- `OL_Test_Results_ensemble_2014_2024.csv`
  - Benchmark exports from `test_model.py --export`.

### Misc

- `__pycache__/` — Python bytecode cache.
- `rb_*` files in this folder are legacy/cross-position artifacts and are not OL source logic.

---

## Practical guidance

- If your goal is maximum current tier accuracy: use `xgb` mode.
- If your goal is to preserve transformer temporal signal while staying competitive: use `ensemble` and tune blend weights.
- Since OL ensemble now includes transformer signal inside XGBoost features, the next step is **weight tuning + threshold calibration** on a held-out validation year.
