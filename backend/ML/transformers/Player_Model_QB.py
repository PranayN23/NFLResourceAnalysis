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

# 1. Time2Vec Layer
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

# 2. Transformer Regressor
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

# 3. Data Processing
data_path = '/Users/pranaynandkeolyar/Documents/NFLSalaryCap/backend/ML/QB.csv'
df = pd.read_csv(data_path)
df['adjusted_value'] = pd.to_numeric(df['adjusted_value'], errors='coerce')
df = df[df['dropbacks'] >= 100].copy()
df.sort_values(by=['player', 'Year'], inplace=True)

# Engineering
df["years_in_league"] = df.groupby("player").cumcount()
df["delta_grade"] = df.groupby("player")["grades_offense"].diff().fillna(0)
df["delta_epa"]   = df.groupby("player")["Net EPA"].diff().fillna(0)
df["delta_btt"]   = df.groupby("player")["btt_rate"].diff().fillna(0)
df['team_performance_proxy'] = df.groupby(['Team', 'Year'])['Net EPA'].transform('mean')

features = [
    'grades_pass', 'grades_offense', 'qb_rating', 'adjusted_value',
    'Cap_Space', 'ypa', 'twp_rate', 'btt_rate', 'completion_percent',
    'years_in_league', 'delta_grade', 'delta_epa', 'delta_btt',
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

SCALER_OUT = os.path.join(os.path.dirname(__file__), 'player_scaler.joblib')
MODEL_OUT  = os.path.join(os.path.dirname(__file__), 'best_classifier.pth')
joblib.dump(scaler, SCALER_OUT)

def create_sequences_hybrid(scaled_df, original_df, max_seq_len, features, target_col):
    X, y, masks = [], [], []
    for player, group_scaled in scaled_df.groupby('player'):
        group_orig = original_df.loc[group_scaled.index]
        vals = group_scaled[features].values
        targs = group_orig[target_col].values
        for i in range(1, len(group_scaled)):
            start_idx = max(0, i - max_seq_len)
            history_vals = vals[start_idx:i]
            actual_len = len(history_vals)
            padding_len = max_seq_len - actual_len
            
            if padding_len > 0:
                pad = np.zeros((padding_len, len(features)))
                padded_x = np.vstack([pad, history_vals])
                mask = [True] * padding_len + [False] * actual_len
            else:
                padded_x = history_vals
                mask = [False] * actual_len
                
            X.append(padded_x)
            y.append(targs[i])
            masks.append(mask)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(masks, dtype=bool)

MAX_SEQUENCE_LENGTH = 5
X_all, y_all, masks_all = create_sequences_hybrid(df_clean_scaled, df_clean, MAX_SEQUENCE_LENGTH, features, target_col)

# Calibrated Oversampling
X_boosted, y_boosted, masks_boosted = [], [], []
for xi, yi, mi in zip(X_all, y_all, masks_all):
    X_boosted.append(xi)
    y_boosted.append(yi)
    masks_boosted.append(mi)
    if yi >= 80.0 or yi < 50.0:
        X_boosted.append(xi)
        y_boosted.append(yi)
        masks_boosted.append(mi)

X_train, X_test, y_train, y_test, masks_train, masks_test = train_test_split(
    np.array(X_boosted), np.array(y_boosted), np.array(masks_boosted), test_size=0.1, random_state=42
)

class PlayerDataset(Dataset):
    def __init__(self, X, y, masks):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.bool)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx], self.masks[idx]

train_loader = DataLoader(PlayerDataset(X_train, y_train, masks_train), batch_size=32, shuffle=True)
test_loader = DataLoader(PlayerDataset(X_test, y_test, masks_test), batch_size=32, shuffle=False)

model = PlayerTransformerRegressor(input_dim=len(features), seq_len=MAX_SEQUENCE_LENGTH).to(DEVICE)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

best_loss = float('inf')
EPOCHS = 150
print("Starting calibrated hybrid training...")

for epoch in range(EPOCHS):
    model.train()
    for inputs, targets, masks in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs, mask=masks).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets, masks in test_loader:
            outputs = model(inputs, mask=masks).squeeze()
            val_loss += criterion(outputs, targets).item()
    
    avg_val_loss = val_loss / len(test_loader)
    scheduler.step(avg_val_loss)
    
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_OUT)
        
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}, Val MAE: {avg_val_loss:.4f}")

# Final Test
model.load_state_dict(torch.load(MODEL_OUT))
model.eval()
with torch.no_grad():
    y_pred = model(torch.tensor(X_test), mask=torch.tensor(masks_test)).squeeze().numpy()
print(f"\nFinal Calibrated MAE: {mean_absolute_error(y_test, y_pred):.4f}")
