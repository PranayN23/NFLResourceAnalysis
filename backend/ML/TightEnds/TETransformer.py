
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ==========================================
# 1. TIME2VEC LAYER (Learnable)
# ==========================================
class Time2Vec(nn.Module):
    def __init__(self, output_dim):
        super(Time2Vec, self).__init__()
        self.output_dim = output_dim
        
        # Learnable parameters
        # Linear component: w0 * t + b0
        self.w0 = nn.Parameter(torch.randn(1, 1))
        self.b0 = nn.Parameter(torch.randn(1, 1))
        
        # Periodic component: sin(w * t + b)
        # We need output_dim - 1 periodic features
        self.w = nn.Parameter(torch.randn(1, output_dim - 1))
        self.b = nn.Parameter(torch.randn(1, output_dim - 1))
        
        self.f = torch.sin 

    def forward(self, x):
        # x shape: [batch, seq_len, 1]
        
        # Linear: Broadcast w0, b0
        # [batch, seq_len, 1]
        v0 = x * self.w0 + self.b0
        
        # Periodic
        # x: [batch, seq_len, 1]
        # w: [1, output_dim-1]
        # x @ w -> [batch, seq_len, output_dim-1]
        v1 = self.f(torch.matmul(x, self.w) + self.b)
        
        # Concatenate -> [batch, seq_len, output_dim]
        return torch.cat([v0, v1], dim=-1)

# ==========================================
# 2. TRANSFORMER ARCHITECTURE
# ==========================================
class TETransformer(nn.Module):
    def __init__(self, num_features, time_emb_dim=16, d_model=64, nhead=4, num_layers=2):
        super(TETransformer, self).__init__()
        
        self.t2v = Time2Vec(time_emb_dim)
        
        # Input Embedding: Project features to d_model space
        # Total dims = num_features + time_emb_dim
        self.input_proj = nn.Linear(num_features + time_emb_dim, d_model)
        
        # Positional Encoding is handled by Time2Vec conceptually, 
        # but Transformers usually need explicit sequence position too.
        # We will let Time2Vec handle the "Calendar Time" (Year), 
        # and standard PositionalEncoding handle "Sequence Index" if needed.
        # For short sequences (3-5 years), Time2Vec on "Year" is usually sufficient and superior.
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.05, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x_features, x_time):
        # x_features: [batch, seq_len, F]
        # x_time: [batch, seq_len, 1]
        
        # 1. Get Time Embeddings
        t_emb = self.t2v(x_time) # [batch, seq, T_dim]
        
        # 2. Concatenate & Project
        combined = torch.cat([x_features, t_emb], dim=-1) # [batch, seq, F+T_dim]
        x = self.input_proj(combined) # [batch, seq, d_model]
        
        # 3. Transformer
        out = self.transformer_encoder(x) # [batch, seq, d_model]
        
        # 4. Global Average Pooling + Last Token (to give more weight to history & current state)
        avg_pool = torch.mean(out, dim=1)
        last_token = out[:, -1, :]
        combined_rep = (avg_pool + last_token) / 2
        
        # 5. Regression Head
        pred = self.head(combined_rep)
        return pred

# ==========================================
# 3. DATA LOADER
# ==========================================
def load_sequences(filepath, seq_len=4):
    df = pd.read_csv(filepath)
    df = df.replace("MISSING", np.nan)
    
    metadata_cols = ["player_id", "player", "Team", "position_x", "position", "franchise_id", "Year"]
    for col in df.columns:
        if col not in metadata_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    
    df = df.sort_values(["player_id", "Year"])
    
    # Target & Breakout indicators
    df["weighted_grade"] = df["grades_offense"].fillna(0) * df["total_snaps"].fillna(0) / 1000.0
    df["efficiency"] = df["grades_offense"] / 100.0
    df["volume"] = df["total_snaps"] / 1000.0
    df["yprr_trend"] = df.groupby("player_id")["yprr"].diff().fillna(0)
    df["efficiency_per_snap"] = df["weighted_grade"] / df["total_snaps"].replace(0, np.nan)
    df["efficiency_per_snap"] = df["efficiency_per_snap"].fillna(0)
    
    # Prime Years (23-26) vs Decline (30+)
    df["is_prime"] = ((df["age"] >= 23) & (df["age"] <= 26)).astype(float)
    df["is_decline"] = (df["age"] >= 30).astype(float)
    
    target_col = "weighted_grade"
    
    # Input Features (Enhanced for Breakouts)
    features = [
        "weighted_grade", "grades_offense", "yards", "touchdowns", 
        "first_downs", "receptions", "targets", "age",
        "efficiency", "volume", "yprr", "yprr_trend", "efficiency_per_snap",
        "is_prime", "is_decline"
    ]
    df[features] = df[features].fillna(0)
    
    X_seqs = []
    T_seqs = [] # Absolute Year (normalized later) or relative? Let's use Normalized Year.
    y_vals = []
    meta = []
    
    # Normalize Year globally for Time2Vec consistency
    min_year = df["Year"].min()
    df["Year_Norm"] = df["Year"] - min_year
    
    groups = df.groupby("player_id")
    
    for pid, group in groups:
        group = group.sort_values("Year")
        if len(group) < 2: continue
        
        data = group[features].values
        times = group["Year_Norm"].values
        years = group["Year"].values
        
        for i in range(1, len(group)):
            # Target is step i
            target_val = group.iloc[i][target_col]
            target_year = years[i]
            
            # Input is history up to i
            start_idx = max(0, i - seq_len)
            seq_data = data[start_idx:i]
            seq_time = times[start_idx:i]
            
            # Pad
            curr_len = seq_data.shape[0]
            if curr_len < seq_len:
                pad_len = seq_len - curr_len
                # Pad with zeros
                seq_data = np.vstack([np.zeros((pad_len, len(features))), seq_data])
                # Pad time? Use previous times? Or just zeros?
                # Using 0 for time might confuse it with year 0. 
                # Better to use relative time padding or just replicates.
                # Let's simple pad with -1 to indicate 'no time'.
                seq_time = np.concatenate([np.full(pad_len, -1), seq_time])
            
            seq_time = seq_time.reshape(-1, 1)
            
            X_seqs.append(seq_data)
            T_seqs.append(seq_time)
            y_vals.append(target_val)
            meta.append({"player": group.iloc[i]["player"], "Team": group.iloc[i]["Team"], "Year": target_year})
            
    return np.array(X_seqs), np.array(T_seqs), np.array(y_vals), pd.DataFrame(meta), features

# ==========================================
# 4. TRAINING
# ==========================================
print("Loading Sequence Data...")
SEQ_LEN = 4 # Look at last 4 years
X, T, y, meta_df, feats = load_sequences("backend/ML/TightEnds/TE.csv", seq_len=SEQ_LEN)

# Normalize Features
N, S, F = X.shape
scaler = StandardScaler()
X_flat = X.reshape(-1, F)
X_scaled = scaler.fit_transform(X_flat).reshape(N, S, F)

# Train/Test Split (2024 is Test)
train_mask = meta_df["Year"] < 2024
test_mask = meta_df["Year"] == 2024

X_train = torch.FloatTensor(X_scaled[train_mask])
T_train = torch.FloatTensor(T[train_mask])
y_train = torch.FloatTensor(y[train_mask]).unsqueeze(1)

X_test = torch.FloatTensor(X_scaled[test_mask])
T_test = torch.FloatTensor(T[test_mask])
y_test = torch.tensor(y[test_mask], dtype=torch.float32).unsqueeze(1) # Fix for MPS potentially
test_meta = meta_df[test_mask]

print(f"Train Samples: {len(X_train)} | Test Samples: {len(X_test)}")

# Model Setup
device = torch.device("cpu")
model = TETransformer(num_features=F, time_emb_dim=16, d_model=64, nhead=4, num_layers=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) # Weight decay for regularization
criterion = nn.MSELoss()

dataset = torch.utils.data.TensorDataset(X_train, T_train, y_train)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

print("Training Transformer...")
EPOCHS = 80
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for bx, bt, by in loader:
        optimizer.zero_grad()
        out = model(bx, bt)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss {epoch_loss/len(loader):.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    preds = model(X_test, T_test).flatten().numpy()
    y_true = y_test.flatten().numpy()

r2 = r2_score(y_true, preds)
mae = mean_absolute_error(y_true, preds)
rmse = np.sqrt(mean_squared_error(y_true, preds))

print("\n==== 2024 TRANSFORMER + TIME2VEC RESULTS ====")
print(f"R-Squared (R²): {r2:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

val_df = test_meta.copy()
val_df["Actual"] = y_true
val_df["Predicted"] = preds
val_df["Error"] = val_df["Actual"] - val_df["Predicted"]
val_df["Abs_Error"] = val_df["Error"].abs()

print("\n==== TOP 2024 PREDICTIONS ====")
print(val_df[["player", "Team", "Actual", "Predicted", "Error"]].sort_values("Predicted", ascending=False).head(15).to_string(index=False))

print("\n==== BIGGEST MISSES (Potential Breakouts/Busts) ====")
# Sort first, then select columns to display
print(val_df.sort_values("Abs_Error", ascending=False)[["player", "Team", "Actual", "Predicted", "Error"]].head(10).to_string(index=False))

# 2025 Prediction logic would go here similar to previous scripts
