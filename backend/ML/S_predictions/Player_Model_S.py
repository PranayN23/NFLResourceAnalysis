"""
Player_Model_S.py: Safety (S) player grade prediction.

WHAT THE CODE DOES
------------------
1. DATA: Loads S.csv (Safety position), keeps rows with required stats. Builds features
   (PFF grades, rates, snap shares, deltas, career mean grade, etc.) and target = grades_defense.

2. TEMPORAL SPLIT: Train on years <= 2022, validate on 2023, test on 2024 (no future leakage).

3. TRANSFORMER MODEL: For each player-year we form a sequence of the last RECENT_YEARS
   (default 2) years of (scaled) features. A Time2Vec + Transformer encoder uses that
   sequence to predict next-year grade (recent performance only). Training uses a weighted sampler to balance grade buckets and
   early stopping on validation Huber loss.

4. XGBOOST: Uses prior-year features only (one row = previous year's stats) to predict
   current-year grade (true forecasting, like the Transformer). Then blended with the
   Transformer (weight chosen on validation MAE).

5. OUTPUT: Best Transformer and XGBoost models saved; per-player predictions for the test
   year written to S_predictions_by_player.csv (player, Year, actual grade, prior_year_actual,
   pred_*, jump_* = pred minus prior-year actual for each model).
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import os
import joblib

# Optional: XGBoost for ensemble (install: pip install xgboost; on macOS segfault try: pip install xgboost-cpu)
try:
    import xgboost as xgb
    from optuna import create_study
    from optuna.samplers import TPESampler
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ==========================
# DEVICE
# ==========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==========================
# TEMPORAL SPLIT CONFIG
# ==========================
TRAIN_END_YEAR = 2022
VAL_YEAR       = 2023
TEST_YEAR      = 2024

# ==========================
# TRAINING HYPERPARAMETERS
# ==========================
# Use only last N years for predictions (recent performance)
RECENT_YEARS  = 3
SEQ_LEN       = RECENT_YEARS  # Transformer: sequence of last RECENT_YEARS years
EPOCHS        = 200
LR            = 0.0002
BATCH_SIZE    = 16
EARLY_STOP    = 40
GRAD_CLIP     = 0.5

# XGBoost Optuna tuning
XGB_N_TRIALS  = 25
XGB_TIMEOUT   = 400  # seconds

# ==========================
# I/O PATHS (script-relative so script runs from any cwd)
# ==========================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_CANDIDATES = [
    os.path.join(_SCRIPT_DIR, "S.csv"),
    os.path.join(_SCRIPT_DIR, "..", "S.csv"),
    os.path.join(os.getcwd(), "backend", "ML", "S.csv"),
    os.path.join(os.getcwd(), "backend", "ML", "S_predictions", "S.csv"),
]
DATA_PATH = None
for _p in _DATA_CANDIDATES:
    if os.path.isfile(_p):
        DATA_PATH = _p
        break
if DATA_PATH is None:
    raise FileNotFoundError("S.csv not found. Tried: " + str(_DATA_CANDIDATES))

OUT_DIR = _SCRIPT_DIR
SCALER_OUT = os.path.join(OUT_DIR, "s_player_scaler.joblib")
MODEL_OUT  = os.path.join(OUT_DIR, "s_best_transformer.pth")
XGB_OUT    = os.path.join(OUT_DIR, "s_xgb_model.json")
PREDICTIONS_CSV = os.path.join(OUT_DIR, "S_predictions_by_player.csv")

# Set to True to run Optuna tuning (can cause segfault on some macOS/envs; use False for stability)
USE_OPTUNA = False

# ==========================
# 1. Dataset Class
# ==========================
class PlayerDataset(Dataset):
    def __init__(self, sequences, targets, masks):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets   = torch.tensor(targets,   dtype=torch.float32)
        self.masks     = torch.tensor(masks,     dtype=torch.bool)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx], self.masks[idx]

# ==========================
# 2. Sequence Utility
# ==========================
def create_sequences_temporal(df_scaled, df_raw, seq_len, features, target_col):
    sequences, targets, masks, years, row_indices = [], [], [], [], []

    for player in df_raw["player"].unique():
        p_scaled = df_scaled[df_raw["player"] == player]
        p_raw    = df_raw[df_raw["player"]    == player]

        for i in range(len(p_raw)):
            history_scaled = p_scaled.iloc[:i]
            if len(history_scaled) == 0:
                continue

            target_val  = p_raw.iloc[i][target_col]
            target_year = p_raw.iloc[i]["Year"]
            row_idx     = p_raw.index[i]

            history_scaled = history_scaled.tail(seq_len)
            actual_len     = len(history_scaled)
            pad_len        = seq_len - actual_len

            seq  = np.vstack([
                np.zeros((pad_len, len(features))),
                history_scaled[features].values
            ])
            mask = [True] * pad_len + [False] * actual_len

            sequences.append(seq)
            targets.append(target_val)
            masks.append(mask)
            years.append(target_year)
            row_indices.append(row_idx)

    return np.array(sequences), np.array(targets), np.array(masks), np.array(years), np.array(row_indices)

# ==========================
# 3. Weighted Sampler
# ==========================
def make_weighted_sampler(y_train):
    thresholds = [62, 70, 78, 84]
    buckets = np.digitize(y_train, thresholds)
    bucket_counts = np.bincount(buckets, minlength=5).astype(float)
    bucket_counts = np.where(bucket_counts == 0, 1, bucket_counts)
    bucket_weights = 1.0 / bucket_counts
    bucket_weights[3] *= 1.5
    bucket_weights[4] *= 2.0
    sample_weights = bucket_weights[buckets]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(sample_weights),
        replacement=True,
    )

# ==========================
# 4. Time2Vec Layer
# ==========================
class Time2Vec(nn.Module):
    def __init__(self, input_dim, kernel_size=1):
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(input_dim, 1))
        self.b0 = nn.Parameter(torch.randn(input_dim, 1))
        self.wk = nn.Parameter(torch.randn(input_dim, kernel_size))
        self.bk = nn.Parameter(torch.randn(input_dim, kernel_size))

    def forward(self, x):
        x_uns    = x.unsqueeze(-1)
        linear   = x_uns * self.w0  + self.b0
        periodic = torch.sin(x_uns * self.wk + self.bk)
        out      = torch.cat([linear, periodic], dim=-1)
        return out.reshape(x.size(0), x.size(1), -1)

# ==========================
# 5. Transformer Regressor
# ==========================
class PlayerTransformerRegressor(nn.Module):
    def __init__(self, input_dim, seq_len, kernel_size=1, num_heads=4, ff_dim=64, num_layers=2, dropout=0.15):
        super().__init__()
        self.time2vec  = Time2Vec(input_dim, kernel_size)
        self.embed_dim = input_dim * (kernel_size + 1)

        if self.embed_dim % num_heads != 0:
            new_dim        = (self.embed_dim // num_heads + 1) * num_heads
            self.pad_proj  = nn.Linear(self.embed_dim, new_dim)
            self.embed_dim = new_dim
        else:
            self.pad_proj = nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.pos_embedding       = nn.Parameter(torch.randn(1, seq_len + 1, self.embed_dim) * 0.02)
        self.cls_token           = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.regressor = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )

    def forward(self, x, mask=None):
        x          = self.time2vec(x)
        x          = self.pad_proj(x)
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x          = torch.cat((cls_tokens, x), dim=1)

        if mask is not None:
            cls_mask  = torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device)
            full_mask = torch.cat([cls_mask, mask], dim=1)
        else:
            full_mask = None

        x = x + self.pos_embedding
        x = self.transformer_encoder(x, src_key_padding_mask=full_mask)
        x = x[:, 0, :]
        return self.regressor(x)

# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    # Deterministic seeds (transformer, XGBoost uses random_state=42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("\nStarting training with career-mean insights...")

    # --------------------------------------------------
    # Data Loading
    # --------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    df = df[df["position"] == "S"].copy()
    df.sort_values(by=["player", "Year"], inplace=True)

    numeric_cols = [
        "grades_defense","grades_coverage_defense","grades_tackle","grades_defense_penalty",
        "interceptions","pass_break_ups","receptions","targets","yards","tackles","assists",
        "missed_tackles","missed_tackle_rate","stops","qb_rating_against",
        "snap_counts_defense","snap_counts_fs","snap_counts_box", "snap_counts_coverage","snap_counts_slot",
        "tackles_for_loss","penalties","forced_fumbles","fumble_recoveries","age",
        "Net EPA","adjusted_value","Cap_Space"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------
    df["years_in_league"] = df.groupby("player").cumcount()
    df["delta_grade"]     = df.groupby("player")["grades_defense"].diff().fillna(0)
    df["delta_coverage"]  = df.groupby("player")["grades_coverage_defense"].diff().fillna(0)
    df["delta_epa"]       = df.groupby("player")["Net EPA"].diff().fillna(0)
    df["team_performance_proxy"] = df.groupby(["Team","Year"])["Net EPA"].transform("mean")

    def safe_div(num, den):
        den_safe = den.copy().astype(float)
        den_safe[den_safe==0] = np.nan
        return (num / den_safe).fillna(0)

    snap = df["snap_counts_defense"]
    df["targets_per_snap"]  = safe_div(df["targets"], snap)
    df["stop_rate"]         = safe_div(df["stops"], snap)
    df["yards_per_target"]  = safe_div(df["yards"], df["targets"])
    df["reception_rate"]    = safe_div(df["receptions"], df["targets"])
    df["int_rate"]          = safe_div(df["interceptions"], snap)
    df["pbu_rate"]          = safe_div(df["pass_break_ups"], snap)
    df["penalty_rate"]      = safe_div(df["penalties"], snap)
    df["tfl_rate"]          = safe_div(df["tackles_for_loss"], snap)
    df["slot_share"]        = safe_div(df["snap_counts_slot"], snap)
    df["fs_share"]      = safe_div(df["snap_counts_fs"], snap)
    df["box_share"]      = safe_div(df["snap_counts_box"], snap)
    df["coverage_success_rate"] = safe_div(df["interceptions"] + df["pass_break_ups"], df["targets"])

    # -------------------------
    # Career mean as feature
    # -------------------------
    df["career_mean_grade"] = df.groupby("player")["grades_defense"] \
                            .transform(lambda x: x.shift(1).expanding().mean())


    features = [
        "grades_defense","grades_coverage_defense","grades_tackle","grades_defense_penalty",
        "qb_rating_against","targets_per_snap","yards_per_target","reception_rate","coverage_success_rate",
        "int_rate","pbu_rate","stop_rate","missed_tackle_rate","tfl_rate","penalty_rate",
        "slot_share","fs_share","box_share","age","years_in_league","adjusted_value","Cap_Space","team_performance_proxy",
        "delta_grade","delta_coverage","delta_epa",
        "career_mean_grade"
    ]
    target_col = "grades_defense"

    df_clean = df.dropna(subset=features + [target_col]).copy()
    print(f"\nClean rows: {len(df_clean)}  | Grade mean: {df_clean[target_col].mean():.1f}")

    # -------------------------
    # Scaling
    # -------------------------
    train_rows = df_clean[df_clean["Year"] <= TRAIN_END_YEAR]
    scaler     = StandardScaler()
    scaler.fit(train_rows[features])
    df_clean_scaled           = df_clean.copy()
    df_clean_scaled[features] = scaler.transform(df_clean[features])
    os.makedirs(os.path.dirname(SCALER_OUT), exist_ok=True)
    joblib.dump(scaler, SCALER_OUT)
    print(f"Scaler saved to {SCALER_OUT}")

    # -------------------------
    # Sequence creation (returns row indices so tabular splits match sequence splits)
    # -------------------------
    X_all, y_all, masks_all, years_all, row_indices_all = create_sequences_temporal(
        df_clean_scaled, df_clean, SEQ_LEN, features, target_col
    )

    train_idx = years_all <= TRAIN_END_YEAR
    val_idx   = years_all == VAL_YEAR
    test_idx  = years_all == TEST_YEAR

    X_train, y_train, m_train = X_all[train_idx], y_all[train_idx], masks_all[train_idx]
    X_val,   y_val,   m_val   = X_all[val_idx],   y_all[val_idx],   masks_all[val_idx]
    X_test,  y_test,  m_test  = X_all[test_idx],  y_all[test_idx],  masks_all[test_idx]

    # Tabular splits for XGBoost: use last RECENT_YEARS prior years (averaged) for true forecasting
    features_xgb = [f for f in features if f != target_col]
    train_ix = row_indices_all[train_idx]
    val_ix   = row_indices_all[val_idx]
    test_ix  = row_indices_all[test_idx]

    def get_prior_years_avg(indices):
        """For each (player, year), average features over the last RECENT_YEARS prior years."""
        X_list = []
        prior_ix_list = []  # store prior-year index(es) for first available (for jump: get prior actual)
        for ix in indices:
            p, y = df_clean.loc[ix, "player"], int(df_clean.loc[ix, "Year"])
            sub = df_clean[(df_clean["player"] == p) & (df_clean["Year"] < y)].sort_values("Year", ascending=False).head(RECENT_YEARS)
            if sub.empty:
                raise ValueError(f"No prior year in data for {p} year {y}")
            feats = df_clean_scaled.loc[sub.index, features_xgb].mean(axis=0).values
            X_list.append(feats)
            prior_ix_list.append(sub.index[0])   # most recent prior year (for prior actual grade)
        return np.array(X_list), np.array(prior_ix_list)

    X_train_t, prior_train_ix = get_prior_years_avg(train_ix)
    X_val_t,   prior_val_ix   = get_prior_years_avg(val_ix)
    X_test_t,  prior_test_ix  = get_prior_years_avg(test_ix)
    y_train_t = df_clean.loc[train_ix, target_col].values
    y_val_t   = df_clean.loc[val_ix, target_col].values
    y_test_t  = df_clean.loc[test_ix, target_col].values
    print(f"XGBoost: using last {RECENT_YEARS} prior years (avg) for true forecast. Train {len(y_train_t)}, Val {len(y_val_t)}, Test {len(y_test_t)}.")

    # -------------------------
    # Weighted sampler
    # -------------------------
    train_dataset = PlayerDataset(X_train, y_train, m_train)
    sampler       = make_weighted_sampler(y_train)
    g = torch.Generator()
    g.manual_seed(42)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, generator=g)
    val_loader    = DataLoader(PlayerDataset(X_val, y_val, m_val), batch_size=BATCH_SIZE, shuffle=False)

    # -------------------------
    # Model
    # -------------------------
    model = PlayerTransformerRegressor(input_dim=len(features), seq_len=SEQ_LEN).to(DEVICE)
    criterion = nn.HuberLoss(delta=5.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR*0.05)

    best_val_loss = float("inf")
    best_epoch    = 0
    epochs_no_improve = 0

    print("="*65)
    print("Starting temporal training (career-mean insights)...")
    print("="*65)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for seqs, tgts, msk in train_loader:
            seqs, tgts, msk = seqs.to(DEVICE), tgts.to(DEVICE), msk.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(seqs, mask=msk).squeeze(-1)
            loss = criterion(outputs, tgts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_mae  = 0.0
        with torch.no_grad():
            for seqs, tgts, msk in val_loader:
                seqs, tgts, msk = seqs.to(DEVICE), tgts.to(DEVICE), msk.to(DEVICE)
                outputs = model(seqs, mask=msk).squeeze(-1)
                val_loss += criterion(outputs, tgts).item()
                val_mae  += torch.abs(outputs - tgts).mean().item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss / len(val_loader)
        avg_mae   = val_mae / len(val_loader)

        if avg_val < best_val_loss:
            best_val_loss     = avg_val
            best_epoch        = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_OUT)
        else:
            epochs_no_improve += 1

        scheduler.step()

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:3}/{EPOCHS} | Train Huber: {avg_train:.4f} | Val Huber: {avg_val:.4f} | Val MAE: {avg_mae:.4f} | No-improve: {epochs_no_improve}/{EARLY_STOP}")

        if epochs_no_improve >= EARLY_STOP:
            print(f"\nEarly stopping at epoch {epoch+1}.")
            break

    print(f"\nBest model from epoch {best_epoch}  Val Huber: {best_val_loss:.4f}")

    # -------------------------
    # Final Evaluation
    # -------------------------
    model.load_state_dict(torch.load(MODEL_OUT))
    model.eval()

    with torch.no_grad():
        tx = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        tm = torch.tensor(m_test, dtype=torch.bool).to(DEVICE)
        y_pred = model(tx, mask=tm).squeeze(-1).cpu().numpy()

    test_mae  = mean_absolute_error(y_test, y_pred)
    residuals = y_test - y_pred

    print(f"\nTest Set MAE ({TEST_YEAR}):    {test_mae:.4f}")
    print(f"Mean Residual:              {residuals.mean():.4f}")
    print(f"Std Residual:               {residuals.std():.4f}")
    print(f"Max Overestimate:           {residuals.min():.4f}")
    print(f"Max Underestimate:          {residuals.max():.4f}")

    # Performance by grade threshold
    for threshold in [60,70,78,83,87]:
        mask_sub = y_test >= threshold
        if mask_sub.sum() > 0:
            mae_sub = mean_absolute_error(y_test[mask_sub], y_pred[mask_sub])
            print(f"  Grade ≥{threshold}: MAE = {mae_sub:.4f}  (n={mask_sub.sum()})")

    # Naive career-mean baseline
    career_means = df_clean[df_clean["Year"] <= TRAIN_END_YEAR].groupby("player")[target_col].mean()
    global_mean  = df_clean[target_col].mean()
    test_rows_baseline = df_clean.loc[test_ix]
    baseline_preds = [career_means.get(r["player"], global_mean) for _, r in test_rows_baseline.iterrows()]
    baseline_mae   = mean_absolute_error(test_rows_baseline[target_col].values, baseline_preds)

    improvement = baseline_mae - test_mae
    print(f"\nNaive career-mean baseline MAE: {baseline_mae:.4f}")
    print(f"Transformer MAE:                 {test_mae:.4f}")
    print(f"Improvement over baseline:      {improvement:+.4f}  "
          f"({'✓ better' if improvement>0 else '✗ worse'})")

    # Transformer val predictions for ensemble
    model.load_state_dict(torch.load(MODEL_OUT))
    model.eval()
    y_pred_val_trans = []
    with torch.no_grad():
        for seqs, tgts, msk in val_loader:
            seqs, msk = seqs.to(DEVICE), msk.to(DEVICE)
            out = model(seqs, mask=msk).squeeze(-1).cpu().numpy()
            y_pred_val_trans.append(out)
    y_pred_val_trans = np.concatenate(y_pred_val_trans, axis=0)

    # -------------------------
    # XGBoost + optional Optuna + Ensemble
    # Segfault fix: OMP_NUM_THREADS=1, C-contiguous arrays, tree_method='hist', device='cpu', n_jobs=1.
    # If still crashing on macOS, try: pip uninstall xgboost && pip install xgboost-cpu
    # -------------------------
    y_pred_test_xgb = None
    pred_test_ens = None
    if HAS_XGB:
        _omp = os.environ.get("OMP_NUM_THREADS")
        os.environ["OMP_NUM_THREADS"] = "1"
        X_train_t = np.ascontiguousarray(X_train_t, dtype=np.float64)
        y_train_t = np.ascontiguousarray(y_train_t, dtype=np.float64)
        X_val_t   = np.ascontiguousarray(X_val_t, dtype=np.float64)
        y_val_t   = np.ascontiguousarray(y_val_t, dtype=np.float64)
        X_test_t  = np.ascontiguousarray(X_test_t, dtype=np.float64)

        xgb_params = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_weight": 3,
            "random_state": 42,
            "n_jobs": 1,
            "tree_method": "hist",
        }
        if getattr(xgb, "__version__", "0").split(".")[0] >= "2":
            xgb_params["device"] = "cpu"
        if USE_OPTUNA:
            try:
                def xgb_objective(trial):
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                        "max_depth": trial.suggest_int("max_depth", 3, 10),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                        "random_state": 42,
                        "n_jobs": 1,
                        "tree_method": "hist",
                    }
                    m = xgb.XGBRegressor(**params)
                    m.fit(X_train_t, y_train_t, eval_set=[(X_val_t, y_val_t)], verbose=False)
                    pred = m.predict(X_val_t)
                    return mean_absolute_error(y_val_t, pred)

                print("\n" + "="*65)
                print("XGBoost hyperparameter tuning (Optuna)...")
                print("="*65)
                study = create_study(direction="minimize", sampler=TPESampler(seed=42, n_startup_trials=10))
                study.optimize(xgb_objective, n_trials=XGB_N_TRIALS, timeout=XGB_TIMEOUT, show_progress_bar=False)
                xgb_params = {**study.best_params, "random_state": 42, "n_jobs": 1, "tree_method": "hist"}
                if getattr(xgb, "__version__", "0").split(".")[0] >= "2":
                    xgb_params["device"] = "cpu"
                print(f"Best XGB val MAE: {study.best_value:.4f}")
            except Exception as e:
                print(f"Optuna failed ({e}), using fixed XGBoost params.")

        print("\nTraining XGBoost (fixed params)...")
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_train_t, y_train_t, eval_set=[(X_val_t, y_val_t)], verbose=False)
        xgb_model.save_model(XGB_OUT)

        y_pred_val_xgb = xgb_model.predict(X_val_t)
        y_pred_test_xgb = xgb_model.predict(X_test_t)
        xgb_val_mae = mean_absolute_error(y_val_t, y_pred_val_xgb)
        xgb_test_mae = mean_absolute_error(y_test_t, y_pred_test_xgb)
        print(f"XGBoost Val MAE:  {xgb_val_mae:.4f}")
        print(f"XGBoost Test MAE: {xgb_test_mae:.4f}")

        best_w = 0.5
        best_val_mae = float("inf")
        for w in np.linspace(0, 1, 21):
            pred_val_ens = w * y_pred_val_trans + (1 - w) * y_pred_val_xgb
            mae = mean_absolute_error(y_val_t, pred_val_ens)
            if mae < best_val_mae:
                best_val_mae = mae
                best_w = w
        pred_test_ens = best_w * y_pred + (1 - best_w) * y_pred_test_xgb
        ensemble_test_mae = mean_absolute_error(y_test, pred_test_ens)
        print(f"\nEnsemble (weight Transformer={best_w:.2f}, XGB={1-best_w:.2f})  Test MAE: {ensemble_test_mae:.4f}")
        print(f"Models saved to {MODEL_OUT}  and  {XGB_OUT}")
        if _omp is not None:
            os.environ["OMP_NUM_THREADS"] = _omp
        else:
            os.environ.pop("OMP_NUM_THREADS", None)
    else:
        print(f"\nModel saved to {MODEL_OUT}")
        print("Install xgboost for XGBoost ensemble: pip install xgboost")

    # -------------------------
    # Per-player predictions CSV (test year = TEST_YEAR; aligned with sequence test set)
    # -------------------------
    test_rows = df_clean.loc[test_ix].copy()
    test_rows = test_rows.reset_index(drop=True)
    prior_year_actual = df_clean.loc[prior_test_ix, target_col].values  # prior-year grade (for jumps)

    out_df = pd.DataFrame({
        "player": test_rows["player"].values,
        "Year": test_rows["Year"].values,
        "actual_" + target_col: test_rows[target_col].values,
        "prior_year_actual": prior_year_actual,
        "pred_transformer": y_pred,
    })
    if "Team" in test_rows.columns:
        out_df.insert(2, "Team", test_rows["Team"].values)
    if y_pred_test_xgb is not None:
        out_df["pred_xgb"] = y_pred_test_xgb
    if pred_test_ens is not None:
        out_df["pred_ensemble"] = pred_test_ens
        out_df["jump_ensemble"] = out_df["pred_ensemble"] - out_df["prior_year_actual"]
        out_df["ensemble_minus_actual"] = out_df["pred_ensemble"] - out_df["actual_" + target_col]
    # Sort by predicted grade descending (best predicted at top)
    sort_col = "pred_ensemble" if "pred_ensemble" in out_df.columns else ("pred_xgb" if "pred_xgb" in out_df.columns else "pred_transformer")
    out_df = out_df.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
    out_df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"\nPer-player predictions saved to {PREDICTIONS_CSV}  (n={len(out_df)}, sorted by {sort_col} descending)")
