import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, f1_score
import os
import joblib

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

TRAIN_END_YEAR = 2022
VAL_YEAR = 2023
TEST_YEAR = 2024

class PlayerDataset(Dataset):
    def __init__(self, sequences, targets, masks, weights):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets   = torch.tensor(targets,   dtype=torch.float32)
        self.masks     = torch.tensor(masks,     dtype=torch.bool)
        self.weights   = torch.tensor(weights,   dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx], self.masks[idx], self.weights[idx]


def create_sequences_temporal(df_scaled, df_raw, seq_len, features, target_col):
    sequences, targets, masks, years, last_grades = [], [], [], [], []
    players = df_raw['player'].unique()

    for p in players:
        p_data_scaled = df_scaled[df_raw['player'] == p]
        p_data_raw = df_raw[df_raw['player'] == p]

        for i in range(len(p_data_raw)):
            target_year = p_data_raw.iloc[i]['Year']
            target = p_data_raw.iloc[i][target_col]
            history_raw = p_data_raw.iloc[:i]
            history_scaled = p_data_scaled.iloc[:i]

            if len(history_scaled) == 0:
                continue

            last_grade = history_raw.iloc[-1][target_col]  # last known grade before target year

            history_scaled = history_scaled.tail(seq_len)
            actual_len = len(history_scaled)
            pad_len = seq_len - actual_len
            seq = np.vstack([np.zeros((pad_len, len(features))), history_scaled[features].values])
            mask = [True] * pad_len + [False] * actual_len

            sequences.append(seq)
            targets.append(target)
            masks.append(mask)
            years.append(target_year)
            last_grades.append(last_grade)

    return np.array(sequences), np.array(targets), np.array(masks), np.array(years), np.array(last_grades)


class Time2Vec(nn.Module):
    def __init__(self, input_dim, kernel_size=1):
        super(Time2Vec, self).__init__()
        self.k = kernel_size
        self.input_dim = input_dim
        self.w0 = nn.Parameter(torch.randn(input_dim, 1))
        self.b0 = nn.Parameter(torch.randn(input_dim, 1))
        self.wk = nn.Parameter(torch.randn(input_dim, kernel_size))
        self.bk = nn.Parameter(torch.randn(input_dim, kernel_size))

    def forward(self, x):
        x_uns = x.unsqueeze(-1)
        linear = x_uns * self.w0 + self.b0
        periodic = torch.sin(x_uns * self.wk + self.bk)
        out = torch.cat([linear, periodic], dim=-1)
        out = out.reshape(x.size(0), x.size(1), -1)
        return out


class CBTransformerRegressor(nn.Module):
    def __init__(self, input_dim, seq_len, kernel_size=1, num_heads=4, ff_dim=64, num_layers=2, dropout=0.1):
        super(CBTransformerRegressor, self).__init__()
        self.time2vec = Time2Vec(input_dim, kernel_size)
        self.embed_dim = input_dim * (kernel_size + 1)

        if self.embed_dim % num_heads != 0:
            new_dim = (self.embed_dim // num_heads + 1) * num_heads
            self.pad_proj = nn.Linear(self.embed_dim, new_dim)
            self.embed_dim = new_dim
        else:
            self.pad_proj = nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, self.embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.regressor = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x, mask=None):
        x = self.time2vec(x)
        x = self.pad_proj(x)
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if mask is not None:
            cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device)
            full_mask = torch.cat([cls_mask, mask], dim=1)
        else:
            full_mask = None

        x = x + self.pos_embedding
        x = self.transformer_encoder(x, src_key_padding_mask=full_mask)
        x = x[:, 0, :]
        x = self.regressor(x)
        return x


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), '../CB.csv')
    df = pd.read_csv(data_path)
    df['adjusted_value'] = pd.to_numeric(df['adjusted_value'], errors='coerce')

    numeric_cols = [
        'grades_defense', 'grades_coverage_defense', 'grades_tackle',
        'qb_rating_against', 'pass_break_ups', 'interceptions',
        'targets', 'snap_counts_corner', 'snap_counts_coverage',
        'snap_counts_slot', 'snap_counts_defense', 'Cap_Space',
        'tackles', 'stops', 'missed_tackle_rate'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df[df['snap_counts_defense'] >= 200].copy()
    df.sort_values(by=['player', 'Year'], inplace=True)

    df["years_in_league"] = df.groupby("player").cumcount()
    df["delta_grade"] = df.groupby("player")["grades_defense"].diff().fillna(0)

    features = [
        'grades_defense', 'grades_coverage_defense', 'grades_tackle',
        'qb_rating_against', 'pass_break_ups', 'interceptions',
        'targets', 'snap_counts_corner', 'snap_counts_coverage',
        'snap_counts_slot', 'adjusted_value', 'Cap_Space',
        'snap_counts_defense', 'years_in_league', 'delta_grade'
    ]
    target_col = 'grades_defense'

    df_clean = df.dropna(subset=features + [target_col]).copy()

    print("Fitting scaler on training data only (no leakage)...")
    train_data = df_clean[df_clean['Year'] <= TRAIN_END_YEAR]
    scaler = StandardScaler()
    scaler.fit(train_data[features])

    df_clean_scaled = df_clean.copy()
    df_clean_scaled[features] = scaler.transform(df_clean[features])

    SCALER_OUT = os.path.join(os.path.dirname(__file__), 'cb_scaler.joblib')
    MODEL_OUT  = os.path.join(os.path.dirname(__file__), 'best_cb_classifier.pth')
    joblib.dump(scaler, SCALER_OUT)
    print(f"Scaler saved to {SCALER_OUT}")

    X_all, y_all, masks_all, years_all, last_grades_all = create_sequences_temporal(
        df_clean_scaled, df_clean, 5, features, target_col
    )

    # Train on delta (change from last known grade) — breaks mean collapse
    y_delta_all = y_all - last_grades_all

    print(f"\nTotal sequences created: {len(X_all)}")
    print(f"Year range: {years_all.min()} - {years_all.max()}")
    print(f"Delta target mean: {y_delta_all.mean():.3f}, std: {y_delta_all.std():.3f}")

    train_idx = years_all <= TRAIN_END_YEAR
    val_idx   = years_all == VAL_YEAR
    test_idx  = years_all == TEST_YEAR

    print(f"\nTemporal Split:")
    print(f"  Train (≤{TRAIN_END_YEAR}): {train_idx.sum()} samples")
    print(f"  Val ({VAL_YEAR}):  {val_idx.sum()} samples")
    print(f"  Test ({TEST_YEAR}): {test_idx.sum()} samples")

    X_train, y_train, masks_train = X_all[train_idx], y_delta_all[train_idx], masks_all[train_idx]
    X_val,   y_val,   masks_val   = X_all[val_idx],   y_delta_all[val_idx],   masks_all[val_idx]
    X_test,  y_test,  masks_test  = X_all[test_idx],  y_delta_all[test_idx],  masks_all[test_idx]

    # Keep raw grades for evaluation reconstruction
    last_grades_val  = last_grades_all[val_idx]
    last_grades_test = last_grades_all[test_idx]
    y_raw_val  = y_all[val_idx]
    y_raw_test = y_all[test_idx]

    # Moderate oversampling + loss weights to balance class imbalance without overfitting
    # Elite (8% of data) → 3x oversample, 2.5x loss weight
    # Reserve (<50)      → 2x oversample, 1.8x loss weight
    # Borderline (70-79 or 50-59) → 1.5x, 1.3x
    # Starter core      → 1x, 1.0x
    y_abs_train = y_all[train_idx]
    X_boosted, y_boosted, masks_boosted, w_boosted = [], [], [], []
    for xi, yi, mi, yi_abs in zip(X_train, y_train, masks_train, y_abs_train):
        if yi_abs >= 80.0:
            repeats, w = 3, 2.5
        elif yi_abs < 50.0:
            repeats, w = 2, 1.8
        elif yi_abs >= 70.0 or yi_abs < 60.0:
            repeats, w = 2, 1.3
        else:
            repeats, w = 1, 1.0
        for _ in range(repeats):
            X_boosted.append(xi); y_boosted.append(yi)
            masks_boosted.append(mi); w_boosted.append(w)

    X_train     = np.array(X_boosted)
    y_train     = np.array(y_boosted)
    masks_train = np.array(masks_boosted)
    w_train     = np.array(w_boosted)

    # Val / test use uniform weight=1
    w_val  = np.ones(len(y_val))
    w_test = np.ones(len(y_test))

    print(f"  Training samples after oversampling: {len(X_train)}")

    train_loader = DataLoader(PlayerDataset(X_train, y_train, masks_train, w_train), batch_size=32, shuffle=True)
    val_loader   = DataLoader(PlayerDataset(X_val,   y_val,   masks_val,   w_val),   batch_size=32, shuffle=False)
    test_loader  = DataLoader(PlayerDataset(X_test,  y_test,  masks_test,  w_test),  batch_size=32, shuffle=False)

    def weighted_mse(pred, target, weight):
        return ((pred - target) ** 2 * weight).mean()

    model = CBTransformerRegressor(input_dim=len(features), seq_len=5).to(DEVICE).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=12)

    def grade_to_tier(grades_arr):
        tiers = []
        for g in grades_arr:
            if g >= 80.0:   tiers.append(2)   # Elite
            elif g >= 60.0: tiers.append(1)   # Starter
            else:           tiers.append(0)   # Reserve
        return tiers

    best_val_f1 = -1.0
    EPOCHS = 250

    print("\n" + "="*60)
    print("Starting temporal training (no leakage)...")
    print("="*60)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for i, t, m, w in train_loader:
            i, t, m, w = i.to(DEVICE), t.to(DEVICE), m.to(DEVICE), w.to(DEVICE)
            optimizer.zero_grad()
            loss = weighted_mse(model(i, mask=m).squeeze(), t, w)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_delta_preds = []
        with torch.no_grad():
            for iv, tv, mv, wv in val_loader:
                iv, mv = iv.to(DEVICE), mv.to(DEVICE)
                out = model(iv, mask=mv).squeeze()
                val_delta_preds.append(out.cpu().numpy())

        val_delta_preds = np.concatenate(val_delta_preds)
        # Reconstruct absolute grades for tier computation
        val_abs_preds = last_grades_val[:len(val_delta_preds)] + val_delta_preds
        val_abs_true  = y_raw_val[:len(val_delta_preds)]

        val_tier_pred = grade_to_tier(val_abs_preds)
        val_tier_true = grade_to_tier(val_abs_true)
        val_macro_f1  = f1_score(val_tier_true, val_tier_pred, average='macro', zero_division=0)
        val_mae       = mean_absolute_error(val_abs_true, val_abs_preds)

        avg_train = train_loss / len(train_loader)

        # Checkpoint on macro-F1 — directly optimizes tier classification
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            torch.save(model.state_dict(), MODEL_OUT)
            best_epoch = epoch + 1

        scheduler.step(val_mae)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train wMSE: {avg_train:.4f} | Val MAE: {val_mae:.4f} | Val F1: {val_macro_f1:.4f}")

    print(f"\nBest model saved from epoch {best_epoch} with Val macro-F1: {best_val_f1:.4f}")

    model.load_state_dict(torch.load(MODEL_OUT))
    model.eval()
    with torch.no_grad():
        test_x = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        test_m = torch.tensor(masks_test, dtype=torch.bool).to(DEVICE)
        y_pred_delta = model(test_x, mask=test_m).squeeze().cpu().numpy()

    # Reconstruct absolute grades: last_known + predicted_delta
    y_pred = last_grades_test + y_pred_delta

    test_mae = mean_absolute_error(y_raw_test, y_pred)
    print(f"\nTest Set MAE (2024): {test_mae:.4f}")
    residuals = y_raw_test - y_pred
    print(f"Mean Residual: {residuals.mean():.4f}")
    print(f"Std Residual:  {residuals.std():.4f}")
    print(f"Pred range: {y_pred.min():.1f} – {y_pred.max():.1f}")

    print("\nPerformance by Grade Range:")
    for threshold in [60, 70, 80]:
        mask_t = y_raw_test >= threshold
        if mask_t.sum() > 0:
            print(f"  Grade ≥{threshold}: MAE = {mean_absolute_error(y_raw_test[mask_t], y_pred[mask_t]):.4f} (n={mask_t.sum()})")

    print(f"\nModel training complete. Saved to {MODEL_OUT}")
