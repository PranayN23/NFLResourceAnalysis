# RB Pranay Transformers: Full Model Guide

**Owner:** Siddarth Nachannagari — Running Back (RB) position

---

## Model Performance

Evaluated on top-32 RBs by touches per year, 2014–2024 (303 total player-seasons):

| Mode | Tier Accuracy |
|---|---:|
| **XGBoost (`xgb`)** | **0.7690** |
| Ensemble (old 50/50) | 0.5545 |
| Transformer alone | 0.3432 |

**Current ensemble weight: 100% XGB / 0% Transformer.**
The transformer signal is not discarded — it is stacked as the `t2v_transformer_signal` feature inside XGBoost, so the temporal information is fully preserved while avoiding the accuracy drag from a 50/50 blend.

---

## Model Architecture

### Pipeline

```
Player History (last 5 seasons)
        ↓
Feature Engineering
  - Delta stats (grade, yards, touches YoY changes)
  - Team performance proxy (mean Net EPA per team-year)
  - Years in league (career stage)
        ↓
Time2Vec Transformer (22 features, seq_len=5)
  - CLS token → grade regression head
  - Output: t2v_transformer_signal
        ↓
XGBoost (15 features, including t2v_transformer_signal)
  - All features are lagged (prev year → predict current year)
  - No data leakage
        ↓
Predicted Grade (0–100)
        ↓
Age Decay Adjustment (kicks in at age 27, max −20 pts)
        ↓
Tier Classification
  Elite        ≥ 75   (Franchise RB — top 10 in league)
  Starter    65–74    (Reliable starter, above average)
  Rotation   55–64    (Role player / committee back)
  Reserve/Poor < 55   (Depth only)
```

### Key Design Choices

| Component | Choice | Reason |
|-----------|--------|--------|
| Sequence length | 5 years | Full career arc without over-weighting old seasons |
| Time2Vec | Linear + sinusoidal encoding | Captures trend (career arc) and cyclical patterns (age) |
| CLS token | BERT-style sequence summary | Better than mean-pooling for variable-length histories |
| Ensemble weight | 100% XGB | Transformer = 34% accuracy vs XGB = 77%; transformer signal baked into XGB |
| Age decay | Peaks at 26, max −20 pts | RBs peak earlier and decline faster than any other skill position |
| Sample weighting | Inverse-frequency + boundary boost | Prevents model from ignoring rare tiers and players near tier boundaries (55/65/75) |

---

## Training Configuration

| Setting | Value |
|---------|-------|
| Train years | 2010–2022 |
| Validation year | 2023 |
| Test year | 2024 |
| Minimum touches filter | ≥ 50 per season |
| Epochs | 180 (early stop: 20 no-improve epochs) |
| Optimizer | Adam (lr=0.001, weight_decay=1e-5) |
| Loss | SmoothL1 (Huber), per-sample weighted |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=6) |
| Scaler | StandardScaler — fit on training data only |

---

## Age Decay Curve

RBs have the sharpest aging curve in professional football:

| Age Range | Penalty per Year |
|-----------|:---:|
| ≤ 26 | 0.0 pts (peak years) |
| 27–28 | −1.5 pts/yr |
| 29–31 | −2.5 pts/yr |
| 32+ | −3.5 pts/yr (capped at −20 total) |

---

## Features

### Transformer Input (22 features — raw historical stats)

`grades_offense`, `grades_run`, `grades_pass_route`, `elusive_rating`,
`yards`, `yards_after_contact`, `yco_attempt`, `breakaway_percent`,
`explosive`, `first_downs`, `receptions`, `targets`, `total_touches`,
`touchdowns`, `adjusted_value`, `Cap_Space`, `age`, `years_in_league`,
`delta_grade`, `delta_yards`, `delta_touches`, `team_performance_proxy`

### XGBoost Input (15 features — all lagged)

`lag_grades_offense`, `lag_yards`, `lag_yco_attempt`, `lag_elusive_rating`,
`lag_breakaway_percent`, `lag_explosive`, `lag_total_touches`, `lag_touchdowns`,
`adjusted_value`, `age`, `years_in_league`, `delta_grade_lag`,
`team_performance_proxy_lag`, `lag_receptions`, `t2v_transformer_signal`

---

## RB Grade → Team Success Correlation Analysis

Run `RB_correlation_analysis.py` to regenerate. Results across 480 team-seasons (2010–2024).

### Team-Level Correlations

| RB Metric | vs. Win % | vs. Net EPA |
|-----------|:---------:|:-----------:|
| Best RB grade (team starter) | r = +0.264 | r = +0.308 |
| Touches-weighted avg grade | r = +0.271 | r = +0.299 |
| Top-2 avg grade (depth) | r = +0.257 | r = +0.304 |

All correlations statistically significant (p < 0.0001).

### Win Rates by Tier

| Team's Best RB Tier | Avg Win % | Avg Net EPA | Playoff Rate |
|---------------------|:---------:|:-----------:|:------------:|
| Elite (≥75) | 53.3% | +0.030 | 45.2% |
| Starter (65–74) | 46.3% | −0.022 | 33.8% |
| Rotation (55–64) | 42.5% | −0.049 | 20.9% |

### Strongest Individual RB Correlates of Team Winning

| Stat | vs. Win % | vs. Net EPA |
|------|:---------:|:-----------:|
| Touchdowns | +0.211 | +0.208 |
| Offensive grade | +0.196 | +0.213 |
| Run grade | +0.150 | +0.173 |
| Yards | +0.136 | +0.119 |

### Key Takeaways for GM Agent Reasoning

- Teams with an **Elite RB** win 53% of games and make the playoffs 45% of the time
- Teams with only a **Rotation-level RB** win just 42.5% and make the playoffs 21% of the time
- The **~11 point Win % gap** between Elite and Rotation RBs equals roughly 1.8 wins per 16-game season
- **Touchdowns** are the strongest individual RB correlate of team success

---

## GM Agent Integration

The RB predictor feeds into the GM LLM agent via `backend/agent/rb_agent_graph.py`. The agent receives a player name + salary ask, runs the grade predictor, and returns a **SIGN / PASS** recommendation.

**RB Salary Valuation Anchors** (current NFL market):

| Tier | Estimated Fair Value |
|------|:-------------------:|
| Elite (≥75) | ~$18M/year |
| Starter (65–74) | ~$10M/year |
| Rotation (55–64) | ~$4M/year |
| Reserve/Poor (<55) | ~$1M/year |

> RBs are systematically undervalued in the current NFL market. These anchors reflect realistic AAV for free-agent contracts.

---

## Files

### Core Scripts

| File | Description |
|------|-------------|
| `Player_Model_RB.py` | Trains the Transformer model; saves `.pth` and `.joblib` scaler |
| `RB_Ensemble.py` | Trains XGBoost, stacks transformer signal, runs VALIDATION or DREAM mode |
| `test_model.py` | CLI harness: `--mode [transformer\|xgb\|ensemble]`, `--export` flag |
| `RB_correlation_analysis.py` | Computes RB grade → team success correlations; outputs two CSVs |

### Model Artifacts

| File | Description |
|------|-------------|
| `rb_best_classifier.pth` | Best saved Transformer checkpoint |
| `rb_player_scaler.joblib` | StandardScaler (fit on 2010–2022 only) |
| `rb_best_xgb.joblib` | Trained XGBoost model |

### Output Reports

| File | Description |
|------|-------------|
| `RB_2024_Validation_Results.csv` | 2024 player predictions vs. actuals |
| `RB_2025_Final_Rankings.csv` | 2025 dream-mode projections |
| `RB_Test_Results_xgb_2014_2024.csv` | Year-by-year XGBoost benchmark |
| `RB_Test_Results_ensemble_2014_2024.csv` | Year-by-year ensemble benchmark |
| `RB_team_correlation.csv` | Pearson correlations: RB grades vs. Win % / Net EPA |
| `RB_tier_win_rates.csv` | Win rates and playoff rates broken down by RB tier |

### Related Files

| File | Description |
|------|-------------|
| `backend/agent/rb_model_wrapper.py` | Inference wrapper used by the GM agent |
| `backend/agent/rb_agent_graph.py` | LangGraph RB GM agent (SIGN/PASS decisions) |
| `backend/ML/HB.csv` | Source data (all qualifying RB seasons, 2010–2024) |

---

## Full Pipeline Order

1. **Train Transformer** — run `Player_Model_RB.py` (saves `.pth` + scaler)
2. **Train XGBoost + validate** — run `RB_Ensemble.py` (MODE = "VALIDATION")
3. **Project 2025** — run `RB_Ensemble.py` (MODE = "DREAM")
4. **Benchmark all modes** — run `test_model.py --export`
5. **Correlation analysis** — run `RB_correlation_analysis.py`
6. **GM Agent** — `rb_agent_graph.py` uses `rb_model_wrapper.py` for live inference
