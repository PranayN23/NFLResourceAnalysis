import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from collections import defaultdict
import os
import joblib

# ==========================
# DEVICE
# ==========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==========================
# TEMPORAL SPLIT CONFIG
# ==========================
TRAIN_END_YEAR = 2022   # Train on data through 2022
VAL_YEAR       = 2023   # Validate on 2023
TEST_YEAR      = 2024   # Hold-out test on 2024

# ==========================
# TRAINING HYPERPARAMETERS
# ==========================
SEQ_LEN       = 6       # Look-back window
EPOCHS        = 150
LR            = 0.0001
BATCH_SIZE    = 32
EARLY_STOP    = 30
GRAD_CLIP     = 1.0

# ==========================
# I/O PATHS
# ==========================
DATA_PATH  = "backend/ML/ED.csv"
SCALER_OUT = "backend/ML/ED_Transformers/ed_player_scaler.joblib"
MODEL_OUT  = "backend/ML/ED_Transformers/ed_best_classifier.pth"


# ==========================
# 1. Dataset Class
# ==========================
class PlayerDataset(Dataset):
    """Wraps pre-built sequence arrays into a PyTorch Dataset."""

    def __init__(self, sequences, targets, masks):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets   = torch.tensor(targets,   dtype=torch.float32)
        self.masks     = torch.tensor(masks,     dtype=torch.bool)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx], self.masks[idx]


# ==========================
# 2. Sequence Utility with Year Tracking
# ==========================
def create_sequences_temporal(df_scaled, df_raw, seq_len, features, target_col):
    """
    Build fixed-length look-back sequences for each player-year observation.

    For each row i of a player's career, the *input* sequence is the
    player's scaled stat history from years 0..i-1 (no target leakage),
    and the *label* is the raw target value in year i.

    Parameters
    ----------
    df_scaled  : DataFrame with feature columns already scaled.
    df_raw     : Original unscaled DataFrame (same row order as df_scaled).
    seq_len    : Maximum look-back window length.
    features   : List of input feature column names.
    target_col : Name of the regression target column.

    Returns
    -------
    sequences  : ndarray, shape (N, seq_len, n_features)
    targets    : ndarray, shape (N,)
    masks      : ndarray, shape (N, seq_len)  — True where padding was applied
    years      : ndarray, shape (N,)          — calendar year of each target
    """
    sequences, targets, masks, years = [], [], [], []

    for player in df_raw["player"].unique():
        p_scaled = df_scaled[df_raw["player"] == player]
        p_raw    = df_raw[df_raw["player"]    == player]

        for i in range(len(p_raw)):
            history_scaled = p_scaled.iloc[:i]
            if len(history_scaled) == 0:
                continue

            target_val  = p_raw.iloc[i][target_col]
            target_year = p_raw.iloc[i]["Year"]

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

    return (
        np.array(sequences),
        np.array(targets),
        np.array(masks),
        np.array(years),
    )


# ==========================
# 3. Time2Vec Layer
# ==========================
class Time2Vec(nn.Module):
    """
    Learnable time-aware feature encoding.

    Produces one linear projection plus `kernel_size` sinusoidal
    projections per input dimension, concatenated along the last axis.
    Output shape: (batch, seq_len, input_dim * (kernel_size + 1)).
    """

    def __init__(self, input_dim, kernel_size=1):
        super(Time2Vec, self).__init__()
        self.w0 = nn.Parameter(torch.randn(input_dim, 1))
        self.b0 = nn.Parameter(torch.randn(input_dim, 1))
        self.wk = nn.Parameter(torch.randn(input_dim, kernel_size))
        self.bk = nn.Parameter(torch.randn(input_dim, kernel_size))

    def forward(self, x):
        x_uns    = x.unsqueeze(-1)
        linear   = x_uns * self.w0  + self.b0
        periodic = torch.sin(x_uns * self.wk + self.bk)
        out      = torch.cat([linear, periodic], dim=-1)
        out      = out.reshape(x.size(0), x.size(1), -1)
        return out


# ==========================
# 4. Transformer Regressor
# ==========================
class PlayerTransformerRegressor(nn.Module):
    """
    CLS-token Transformer for per-player seasonal grade regression.

    Architecture
    ------------
    Input  -> Time2Vec -> optional projection -> positional embedding
           -> [CLS] prepend -> TransformerEncoder -> CLS output -> MLP head
    """

    def __init__(
        self,
        input_dim,
        seq_len,
        kernel_size=1,
        num_heads=4,
        ff_dim=64,
        num_layers=1,
        dropout=0.3,
    ):
        super(PlayerTransformerRegressor, self).__init__()

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
        )

        self.pos_embedding       = nn.Parameter(torch.randn(1, seq_len + 1, self.embed_dim))
        self.cls_token           = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.regressor = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 32),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
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
        x = self.regressor(x)
        return x


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":

    # --------------------------------------------------
    # 5. Data Loading & Filtering
    # --------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    df = df[df["position"] == "ED"].copy()
    df.sort_values(by=["player", "Year"], inplace=True)

    numeric_cols = [
        "grades_pass_rush_defense", "grades_run_defense", "grades_defense",
        "pressures", "sacks", "tackles", "assists", "missed_tackles",
        "age", "snap_counts_defense", "hits", "hurries", "stops",
        "tackles_for_loss", "total_pressures", "penalties",
        "snap_counts_pass_rush", "snap_counts_run_defense",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --------------------------------------------------
    # 6. Feature Engineering
    # --------------------------------------------------
    df["years_in_league"] = df.groupby("player").cumcount()
    df["delta_grade"]     = df.groupby("player")["grades_defense"].diff().fillna(0)
    df["delta_pass_rush"] = df.groupby("player")["grades_pass_rush_defense"].diff().fillna(0)
    df["delta_run_def"]   = df.groupby("player")["grades_run_defense"].diff().fillna(0)
    df["team_performance_proxy"] = (
        df.groupby(["Team", "Year"])["Net EPA"].transform("mean")
    )

    def safe_div(a, b):
        return np.where(b == 0, 0, a / b)

    snap    = df["snap_counts_defense"]
    snap_dl = df["snap_counts_dl"]

    # Edge rushers: pressure-focused rates (same columns as DI but ED-specific context)
    df["pressure_rate"]   = safe_div(df["total_pressures"],    snap)
    df["sack_rate"]       = safe_div(df["sacks"],              snap)
    df["hit_rate"]        = safe_div(df["hits"],               snap)
    df["hurry_rate"]      = safe_div(df["hurries"],            snap)
    df["stop_rate"]       = safe_div(df["stops"],              snap)
    df["tfl_rate"]        = safe_div(df["tackles_for_loss"],   snap)
    df["penalty_rate"]    = safe_div(df["penalties"],          snap)

    # ED alignment shares — outside-T and over-T are most relevant for edge
    df["outside_t_share"] = safe_div(df["snap_counts_dl_outside_t"], snap_dl)
    df["over_t_share"]    = safe_div(df["snap_counts_dl_over_t"],    snap_dl)

    # NOTE: grades_defense as a feature is VALID — create_sequences_temporal
    # uses iloc[:i], so only PAST seasons' grades predict the CURRENT season.
    features = [
        "grades_defense",
        "grades_run_defense",
        "grades_pass_rush_defense",
        "pressure_rate",
        "sack_rate",
        "hit_rate",
        "hurry_rate",
        "stop_rate",
        "tfl_rate",
        "penalty_rate",
        "outside_t_share",
        "over_t_share",
        "age",
        "years_in_league",
        "adjusted_value",
        "Cap_Space",
        "team_performance_proxy",
        "delta_grade",
        "delta_pass_rush",
        "delta_run_def",
    ]
    target_col = "grades_defense"

    # Coerce all model columns; sentinels like 'MISSING' become NaN
    all_model_cols = features + [target_col]
    for col in all_model_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    bad_counts = {c: df[c].isna().sum() for c in all_model_cols if df[c].isna().any()}
    if bad_counts:
        print("\nColumns with NaN / non-numeric values (will be dropped via dropna):")
        for col, n in bad_counts.items():
            print(f"  {col}: {n} rows")
    else:
        print("All feature columns are clean numeric values.")

    df_clean = df.dropna(subset=all_model_cols).copy()

    # --------------------------------------------------
    # 7. Scaling — fit ONLY on training data (no leakage)
    # --------------------------------------------------
    print("\nFitting scaler on training data only (no leakage)...")
    train_rows = df_clean[df_clean["Year"] <= TRAIN_END_YEAR]
    scaler     = StandardScaler()
    scaler.fit(train_rows[features])

    df_clean_scaled           = df_clean.copy()
    df_clean_scaled[features] = scaler.transform(df_clean[features])

    os.makedirs(os.path.dirname(SCALER_OUT), exist_ok=True)
    joblib.dump(scaler, SCALER_OUT)
    print(f"Scaler saved to {SCALER_OUT}")

    # --------------------------------------------------
    # 8. Sequence Creation
    # --------------------------------------------------
    X_all, y_all, masks_all, years_all = create_sequences_temporal(
        df_clean_scaled, df_clean, SEQ_LEN, features, target_col
    )

    print(f"\nTotal sequences created: {len(X_all)}")
    print(f"Year range: {years_all.min()} - {years_all.max()}")

    # --------------------------------------------------
    # 9. Temporal Split
    # --------------------------------------------------
    train_idx = years_all <= TRAIN_END_YEAR
    val_idx   = years_all == VAL_YEAR
    test_idx  = years_all == TEST_YEAR

    print(f"\nTemporal Split:")
    print(f"  Train (≤{TRAIN_END_YEAR}): {train_idx.sum()} samples")
    print(f"  Val   ({VAL_YEAR}):        {val_idx.sum()} samples")
    print(f"  Test  ({TEST_YEAR}):       {test_idx.sum()} samples")

    X_train, y_train, m_train = X_all[train_idx], y_all[train_idx], masks_all[train_idx]
    X_val,   y_val,   m_val   = X_all[val_idx],   y_all[val_idx],   masks_all[val_idx]
    X_test,  y_test,  m_test  = X_all[test_idx],  y_all[test_idx],  masks_all[test_idx]

    # --------------------------------------------------
    # 10. Calibrated Oversampling — training data only
    # --------------------------------------------------
    print("\nApplying oversampling to training data...")
    X_boosted, y_boosted, m_boosted = [], [], []

    for xi, yi, mi in zip(X_train, y_train, m_train):
        X_boosted.append(xi); y_boosted.append(yi); m_boosted.append(mi)
        if yi >= 80.0 or yi <= 62.0:
            X_boosted.append(xi); y_boosted.append(yi); m_boosted.append(mi)
        if yi >= 85.0:
            X_boosted.append(xi); y_boosted.append(yi); m_boosted.append(mi)

    X_train = np.array(X_boosted)
    y_train = np.array(y_boosted)
    m_train = np.array(m_boosted)
    print(f"  Training samples after oversampling: {len(X_train)}")

    # --------------------------------------------------
    # 11. DataLoaders
    # --------------------------------------------------
    train_loader = DataLoader(
        PlayerDataset(X_train, y_train, m_train), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        PlayerDataset(X_val, y_val, m_val), batch_size=BATCH_SIZE, shuffle=False
    )

    # --------------------------------------------------
    # 12. Model, Loss, Optimiser, Scheduler
    # --------------------------------------------------
    model = PlayerTransformerRegressor(
        input_dim=len(features),
        seq_len=SEQ_LEN,
        num_layers=1,
        dropout=0.3,
    ).to(DEVICE).float()

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    best_val_loss     = float("inf")
    best_epoch        = 0
    epochs_no_improve = 0

    # --------------------------------------------------
    # 13. Training Loop
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("Starting temporal training (no leakage)...")
    print("=" * 60)

    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0.0
        for seqs, tgts, msk in train_loader:
            seqs, tgts, msk = seqs.to(DEVICE), tgts.to(DEVICE), msk.to(DEVICE)
            optimizer.zero_grad()
            outputs    = model(seqs, mask=msk).squeeze(-1)
            loss       = criterion(outputs, tgts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seqs, tgts, msk in val_loader:
                seqs, tgts, msk = seqs.to(DEVICE), tgts.to(DEVICE), msk.to(DEVICE)
                outputs   = model(seqs, mask=msk).squeeze(-1)
                val_loss += criterion(outputs, tgts).item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)

        if avg_val < best_val_loss:
            best_val_loss     = avg_val
            best_epoch        = epoch + 1
            epochs_no_improve = 0
            os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
            torch.save(model.state_dict(), MODEL_OUT)
        else:
            epochs_no_improve += 1

        scheduler.step(avg_val)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:>3}/{EPOCHS} | "
                f"Train MAE: {avg_train:.4f} | "
                f"Val MAE: {avg_val:.4f} | "
                f"No-improve: {epochs_no_improve}/{EARLY_STOP}"
            )

        if epochs_no_improve >= EARLY_STOP:
            print(f"\nEarly stopping at epoch {epoch+1} — "
                  f"no val improvement for {EARLY_STOP} epochs.")
            break

    print(f"\nBest model saved from epoch {best_epoch} with Val MAE: {best_val_loss:.4f}")

    # --------------------------------------------------
    # 14. Final Evaluation
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Final Evaluation on Test Set ({TEST_YEAR})")
    print("=" * 60)

    model.load_state_dict(torch.load(MODEL_OUT, weights_only=True))
    model.eval()

    with torch.no_grad():
        tx     = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        tm     = torch.tensor(m_test, dtype=torch.bool).to(DEVICE)
        y_pred = model(tx, mask=tm).squeeze(-1).cpu().numpy()

    test_mae  = mean_absolute_error(y_test, y_pred)
    residuals = y_test - y_pred

    print(f"\nTest Set MAE ({TEST_YEAR}):    {test_mae:.4f}")
    print(f"Mean Residual:              {residuals.mean():.4f}")
    print(f"Std Residual:               {residuals.std():.4f}")
    print(f"Max Overestimate:           {residuals.min():.4f}")
    print(f"Max Underestimate:          {residuals.max():.4f}")

    print("\nPerformance by Grade Range:")
    for threshold in [60, 70, 80, 85]:
        mask_sub = y_test >= threshold
        if mask_sub.sum() > 0:
            mae_sub = mean_absolute_error(y_test[mask_sub], y_pred[mask_sub])
            print(f"  Grade ≥{threshold}: MAE = {mae_sub:.4f}  (n={mask_sub.sum()})")

    # Naive career-mean baseline
    career_means = {
        p: df_clean[df_clean["player"] == p][target_col].mean()
        for p in df_clean[df_clean["Year"] <= TRAIN_END_YEAR]["player"].unique()
    }
    global_mean    = df_clean[target_col].mean()
    test_rows      = df_clean[df_clean["Year"] == TEST_YEAR]
    baseline_preds = [career_means.get(r["player"], global_mean) for _, r in test_rows.iterrows()]
    baseline_mae   = mean_absolute_error(test_rows[target_col].values, baseline_preds)

    print(f"\nNaive career-mean baseline MAE: {baseline_mae:.4f}")
    print(f"Model improvement over baseline: {baseline_mae - test_mae:+.4f}")
    print(f"\nModel saved to {MODEL_OUT}")