import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import os
import joblib

# Check device stability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ==========================
# TEMPORAL SPLIT CONFIG
# ==========================
TRAIN_END_YEAR = 2022  # Train on 2010-2022
VAL_YEAR = 2023        # Validate on 2023
TEST_YEAR = 2024       # Test on 2024

# 1. Dataset Class
class PlayerDataset(Dataset):
    def __init__(self, sequences, targets, masks):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.bool)
        
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx], self.masks[idx]

# 2. Sequence Utility with Year Tracking
def create_sequences_temporal(df_scaled, df_raw, seq_len, features, target_col):
    """Create sequences with year labels for temporal splitting"""
    sequences, targets, masks, years = [], [], [], []
    players = df_raw['player'].unique()
    
    for p in players:
        p_data_scaled = df_scaled[df_raw['player'] == p]
        p_data_raw = df_raw[df_raw['player'] == p]
        
        for i in range(len(p_data_raw)):
            # Target is current year (i)
            target_year = p_data_raw.iloc[i]['Year']
            target = p_data_raw.iloc[i][target_col]
            
            # Historical sequence up to previous years (before i)
            history_scaled = p_data_scaled.iloc[:i]
            
            if len(history_scaled) == 0:
                continue # No history to predict from

            # Take last seq_len years
            history_scaled = history_scaled.tail(seq_len)
            actual_len = len(history_scaled)
            
            # Padding
            pad_len = seq_len - actual_len
            seq = np.vstack([np.zeros((pad_len, len(features))), history_scaled[features].values])
            mask = [True] * pad_len + [False] * actual_len
            
            sequences.append(seq)
            targets.append(target)
            masks.append(mask)
            years.append(target_year)
            
    return np.array(sequences), np.array(targets), np.array(masks), np.array(years)

# 3. Time2Vec Layer (Structural)
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

# 4. Transformer Regressor
class PlayerTransformerRegressor(nn.Module):
    def __init__(self, input_dim, seq_len, kernel_size=1, num_heads=4, ff_dim=64, num_layers=2, dropout=0.1):
        super(PlayerTransformerRegressor, self).__init__()
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
    # 5. Data Processing

    # Load CSVs for OL positions
    df_guard = pd.read_csv("backend/ML/G.csv")
    df_center = pd.read_csv("backend/ML/C.csv")
    df_tackle = pd.read_csv("backend/ML/T.csv")  

    # Combine datasets
    df = pd.concat([df_guard, df_center, df_tackle], axis=0, ignore_index=True)

    # One-hot encode positions
    df = pd.get_dummies(df, columns=['position'], prefix='pos')
    grade_cols = ['grades_offense', 'grades_run_block', 'grades_pass_block']

    for col in grade_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')


    # Ensure numeric columns
    df['adjusted_value'] = pd.to_numeric(df['adjusted_value'], errors='coerce')

    # Filter players with minimal workload (example: total snaps >= 100)
    df = df[df['snap_counts_offense'] >= 100].copy()

    # Sort by player and year
    df.sort_values(by=['player', 'Year'], inplace=True)

    # =========================
    # Feature Engineering
    # =========================
    df["years_in_league"] = df.groupby("player").cumcount()
    df["delta_grade"] = df.groupby("player")["grades_offense"].diff().fillna(0)
    df["delta_run_block"] = df.groupby("player")["grades_run_block"].diff().fillna(0)
    df["delta_pass_block"] = df.groupby("player")["grades_pass_block"].diff().fillna(0)
    df['team_performance_proxy'] = df.groupby(['Team', 'Year'])['Net EPA'].transform('mean')

    # =========================
    # Efficiency / Rate Metrics
    # =========================
    df['sacks_allowed_rate'] = df['sacks_allowed'] / df['snap_counts_pass_block']
    df['hits_allowed_rate'] = df['hits_allowed'] / df['snap_counts_pass_block']
    df['hurries_allowed_rate'] = df['hurries_allowed'] / df['snap_counts_pass_block']
    df['pressures_allowed_rate'] = df['pressures_allowed'] / df['snap_counts_pass_block']
    df['penalties_rate'] = df['penalties'] / df['snap_counts_pass_block']  # efficiency metric
    df['pass_block_efficiency'] = df['pbe']  # already a rate metric

    # Optional: Snap share per OL position (normalized by total OL snaps)
    df['snap_counts_block_share'] = df['snap_counts_block'] / df['snap_counts_offense']
    df['snap_counts_run_block_share'] = df['snap_counts_run_block'] / df['snap_counts_offense']
    df['snap_counts_pass_block_share'] = df['snap_counts_pass_block'] / df['snap_counts_offense']

    # =========================
    # Final Features for Model
    # =========================
    features = [
        'adjusted_value',
        'Cap_Space',
        'age',
        'years_in_league',
        'delta_grade',
        'delta_run_block',
        'delta_pass_block',
        'team_performance_proxy',
        # Efficiency metrics
        'sacks_allowed_rate',
        'hits_allowed_rate',
        'hurries_allowed_rate',
        'pressures_allowed_rate',
        'penalties_rate',
        'pass_block_efficiency',
        # Snap shares
        'snap_counts_block_share',
        'snap_counts_run_block_share',
        'snap_counts_pass_block_share',
        # Position (one-hot encoded)
        'pos_T',
        'pos_G',
        'pos_C'
    ]
    target_col = 'grades_offense'

    # Drop rows with missing values in features or target
    df_clean = df.dropna(subset=features + [target_col]).copy()

    # =========================
    # Normalize Features
    # =========================
    print("Fitting scaler on training data only (no leakage)...")
    train_data = df_clean[df_clean['Year'] <= TRAIN_END_YEAR]

    scaler = StandardScaler()
    scaler.fit(train_data[features])

    # Apply scaler to all data
    df_clean_scaled = df_clean.copy()
    df_clean_scaled[features] = scaler.transform(df_clean[features])


    SCALER_OUT = 'backend/ML/OL_Pranay_Transformers/ol_player_scaler.joblib'
    MODEL_OUT  = 'backend/ML/OL_Pranay_Transformers/ol_best_classifier.pth'
    joblib.dump(scaler, SCALER_OUT)
    print(f"Scaler saved to {SCALER_OUT}")

    # Create sequences with year tracking
    X_all, y_all, masks_all, years_all = create_sequences_temporal(
        df_clean_scaled, df_clean, 5, features, target_col
    )

    print(f"\nTotal sequences created: {len(X_all)}")
    print(f"Year range: {years_all.min()} - {years_all.max()}")

    # ==========================
    # TEMPORAL SPLIT (NO LEAKAGE)
    # ==========================
    train_idx = years_all <= TRAIN_END_YEAR
    val_idx = years_all == VAL_YEAR
    test_idx = years_all == TEST_YEAR

    print(f"\nTemporal Split:")
    print(f"  Train (≤{TRAIN_END_YEAR}): {train_idx.sum()} samples")
    print(f"  Val ({VAL_YEAR}): {val_idx.sum()} samples")
    print(f"  Test ({TEST_YEAR}): {test_idx.sum()} samples")

    X_train, y_train, masks_train = X_all[train_idx], y_all[train_idx], masks_all[train_idx]
    X_val, y_val, masks_val = X_all[val_idx], y_all[val_idx], masks_all[val_idx]
    X_test, y_test, masks_test = X_all[test_idx], y_all[test_idx], masks_all[test_idx]

    # Calibrated Oversampling (ONLY on training data)
    print("\nApplying oversampling to training data...")
    X_boosted, y_boosted, masks_boosted = [], [], []
    for xi, yi, mi in zip(X_train, y_train, masks_train):
        X_boosted.append(xi)
        y_boosted.append(yi)
        masks_boosted.append(mi)
        # Duplicate extreme performers (adjusted for OL grading scale)
        if yi >= 80.0 or yi < 50.0:
            X_boosted.append(xi)
            y_boosted.append(yi)
            masks_boosted.append(mi)

    X_train = np.array(X_boosted)
    y_train = np.array(y_boosted)
    masks_train = np.array(masks_boosted)
    
    print(f"  Training samples after oversampling: {len(X_train)}")

    # Create DataLoaders
    train_loader = DataLoader(
        PlayerDataset(X_train, y_train, masks_train), 
        batch_size=32, 
        shuffle=True
    )
    val_loader = DataLoader(
        PlayerDataset(X_val, y_val, masks_val), 
        batch_size=32, 
        shuffle=False
    )
    test_loader = DataLoader(
        PlayerDataset(X_test, y_test, masks_test), 
        batch_size=32, 
        shuffle=False
    )

    # Model initialization
    model = PlayerTransformerRegressor(
        input_dim=len(features), 
        seq_len=5
    ).to(DEVICE).float()
    
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    best_val_loss = float('inf')
    EPOCHS = 150
    
    print("\n" + "="*60)
    print("Starting temporal training (no leakage)...")
    print("="*60)

    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for i, t, m in train_loader:
            i, t = i.to(DEVICE), t.to(DEVICE)
            m = m.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(i, mask=m).squeeze()
            loss = criterion(outputs, t)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for iv, tv, mv in val_loader:
                iv, tv, mv = iv.to(DEVICE), tv.to(DEVICE), mv.to(DEVICE)
                ov = model(iv, mask=mv).squeeze()
                val_loss += criterion(ov, tv).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_OUT)
            best_epoch = epoch + 1
        
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train MAE: {avg_train_loss:.4f} | Val MAE: {avg_val_loss:.4f}")

    print(f"\nBest model saved from epoch {best_epoch} with Val MAE: {best_val_loss:.4f}")

    # ==========================
    # FINAL EVALUATION ON TEST SET
    # ==========================
    print("\n" + "="*60)
    print("Final Evaluation on Test Set (2024)")
    print("="*60)
    
    model.load_state_dict(torch.load(MODEL_OUT))
    model.eval()
    
    with torch.no_grad():
        test_x = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        test_m = torch.tensor(masks_test, dtype=torch.bool).to(DEVICE)
        y_pred = model(test_x, mask=test_m).squeeze().cpu().numpy()
    
    test_mae = mean_absolute_error(y_test, y_pred)
    print(f"\nTest Set MAE (2024): {test_mae:.4f}")
    
    # Additional metrics
    residuals = y_test - y_pred
    print(f"Mean Residual: {residuals.mean():.4f}")
    print(f"Std Residual: {residuals.std():.4f}")
    print(f"Max Overestimate: {residuals.min():.4f}")
    print(f"Max Underestimate: {residuals.max():.4f}")
    
    # Performance by grade range (adjusted for OL grading)
    print("\nPerformance by Grade Range:")
    for threshold in [50, 60, 70, 80]:
        mask = y_test >= threshold
        if mask.sum() > 0:
            mae_subset = mean_absolute_error(y_test[mask], y_pred[mask])
            print(f"  Grade ≥{threshold}: MAE = {mae_subset:.4f} (n={mask.sum()})")
    
    print(f"\nModel training complete. Best model saved to {MODEL_OUT}")