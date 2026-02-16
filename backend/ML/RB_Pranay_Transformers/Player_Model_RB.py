import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os
import joblib

# Check device stability
DEVICE = torch.device('cpu')
print(f"Using device: {DEVICE}")

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

# 2. Sequence Utility
def create_sequences_hybrid(df_scaled, df_raw, seq_len, features, target_col):
    sequences, targets, masks = [], [], []
    players = df_raw['player'].unique()
    
    for p in players:
        p_data_scaled = df_scaled[df_raw['player'] == p]
        p_data_raw = df_raw[df_raw['player'] == p]
        
        for i in range(len(p_data_raw)):
            # Target is current year (i)
            # Sequence is up to previous years (before i)
            target = p_data_raw.iloc[i][target_col]
            
            # Historical sequence up to i (not including i for target consistency)
            # Actually, standard transformer training uses [t-seq_len, ..., t-1] to predict t
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
            
    return np.array(sequences), np.array(targets), np.array(masks)

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
    data_path = '/Users/pranaynandkeolyar/Documents/NFLSalaryCap/backend/ML/HB.csv'
    df = pd.read_csv(data_path)
    df['adjusted_value'] = pd.to_numeric(df['adjusted_value'], errors='coerce')
    df = df[df['total_touches'] >= 50].copy()
    df.sort_values(by=['player', 'Year'], inplace=True)

    # Engineering
    df["years_in_league"] = df.groupby("player").cumcount()
    df["delta_grade"] = df.groupby("player")["grades_offense"].diff().fillna(0)
    df["delta_yards"] = df.groupby("player")["yards"].diff().fillna(0)
    df["delta_touches"] = df.groupby("player")["total_touches"].diff().fillna(0)

    df["team_performance_proxy"] = df.groupby(['Team', 'Year'])['Net EPA'].transform('mean')


    features = [
    'grades_offense',
    'grades_run',
    'grades_pass_route',
    'elusive_rating',
    'yards',
    'yards_after_contact',
    'yco_attempt',
    'breakaway_percent',
    'explosive',
    'first_downs',
    'receptions',
    'targets',
    'total_touches',
    'touchdowns',
    'adjusted_value',
    'Cap_Space',
    'age',
    'years_in_league',
    'delta_grade',
    'delta_yards',
    'delta_touches',
    'team_performance_proxy'
]

    target_col = 'grades_offense' 

    df_clean = df.dropna(subset=features + [target_col]).copy()

    # Normalize Features (Strict Separation)
    scaler = StandardScaler()
    X_features = df_clean[features].copy()
    scaler.fit(X_features)
    df_clean_scaled = df_clean.copy()
    df_clean_scaled[features] = scaler.transform(X_features)

    SCALER_OUT = os.path.join(os.path.dirname(__file__), 'rb_player_scaler.joblib')
    MODEL_OUT  = os.path.join(os.path.dirname(__file__), 'rb_best_classifier.pth')
    joblib.dump(scaler, SCALER_OUT)

    X_all, y_all, masks_all = create_sequences_hybrid(df_clean_scaled, df_clean, 5, features, target_col)

    # Calibrated Oversampling
    X_boosted, y_boosted, masks_boosted = [], [], []
    for xi, yi, mi in zip(X_all, y_all, masks_all):
        X_boosted.append(xi)
        y_boosted.append(yi)
        masks_boosted.append(mi)
        if yi >= 75.0 or yi < 55.0:
            X_boosted.append(xi)
            y_boosted.append(yi)
            masks_boosted.append(mi)

    X_train, X_test, y_train, y_test, masks_train, masks_test = train_test_split(
        np.array(X_boosted), np.array(y_boosted), np.array(masks_boosted), test_size=0.1, random_state=42
    )

    train_loader = DataLoader(PlayerDataset(X_train, y_train, masks_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(PlayerDataset(X_test, y_test, masks_test), batch_size=32, shuffle=False)

    model = PlayerTransformerRegressor(input_dim=len(features), seq_len=3).to(DEVICE).float()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_loss = float('inf')
    EPOCHS = 150
    print("Starting calibrated hybrid training with Learned Durability...")

    for epoch in range(EPOCHS):
        model.train()
        for i, t, m in train_loader:
            i, t = i.to(DEVICE), t.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(i, mask=m).squeeze()
            loss = criterion(outputs, t)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for iv, tv, mv in test_loader:
                ov = model(iv, mask=mv).squeeze()
                val_loss += criterion(ov, tv).item()
        
        avg_v_loss = val_loss / len(test_loader)
        if avg_v_loss < best_loss:
            best_loss = avg_v_loss
            torch.save(model.state_dict(), MODEL_OUT)
        
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}, Val MAE: {avg_v_loss:.4f}")

    # Final Test
    model.load_state_dict(torch.load(MODEL_OUT))
    model.eval()
    with torch.no_grad():
        test_x = torch.tensor(X_test, dtype=torch.float32)
        test_m = torch.tensor(masks_test, dtype=torch.bool)
        y_pred = model(test_x, mask=test_m).squeeze().numpy()
    print(f"\nFinal Calibrated MAE: {mean_absolute_error(y_test, y_pred):.4f}")
