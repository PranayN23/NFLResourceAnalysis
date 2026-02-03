
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ====================================================
# 1. DATA LOADING & PREPROCESSING
# ====================================================
def load_and_process_data(filepath, features, target, seq_length=2):
    df = pd.read_csv(filepath)
    df = df.replace("MISSING", np.nan)
    
    # Convert to numeric
    # Convert to numeric (exclude metadata)
    metadata_cols = ["player_id", "player", "Team", "position_x", "position", "franchise_id"]
    for col in df.columns:
        if col not in metadata_cols and col not in ["Year"]: # Keep Year as integer if capable, but to_numeric is fine
             df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Ensure Year is numeric
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    
    # Sort by player and year
    df = df.sort_values(["player_id", "Year"]).reset_index(drop=True)
    
    # Fill NaNs - forward fill per player, then fill 0
    df[features] = df.groupby("player_id")[features].ffill().fillna(0)
    df = df.dropna(subset=[target]) # Target must exist
    
    # We need to create sequences
    # Sequence: [Year T, Year T+1] -> Target: Year T+2 (Net EPA)
    # Actually, standard logic: Input [Year T, Year T+1] -> Predict Year T+2's Net EPA
    # But for training, we need target at T+2.
    
    sequences = []
    targets = []
    meta = [] # Store player_id, year for tracking
    
    # Debug info
    print(f"Initial rows: {len(df)}")
    print(f"Target '{target}' non-null count: {df[target].count()}")
    
    player_groups = df.groupby("player_id")
    print(f"Total players: {len(player_groups)}")
    
    dropped_short = 0
    
    for pid, group in player_groups:
        if len(group) < seq_length + 1:
            dropped_short += 1
            continue
            
        group_vals = group[features].values
        target_vals = group[target].values
        years = group["Year"].values
        
        # Create sliding windows
        for i in range(len(group) - seq_length):
            # Check if years are consecutive (optional but good for LSTM)
            # If we just want any previous year predicting next available year, we skip this check
            # But strict LSTM usually implies fixed time steps.
            # Let's check year difference.
            if years[i+seq_length] - years[i+seq_length-1] != 1:
                 # Gap in data
                 continue

            # Input: T to T+seq_length-1
            seq = group_vals[i : i+seq_length]
            # Target: T+seq_length
            tgt = target_vals[i+seq_length]
            
    
            sequences.append(seq)
            targets.append(tgt)
            meta.append((pid, years[i+seq_length])) # Year of the target
            
    return np.array(sequences), np.array(targets), meta

# Feature Selection (based on correlation analysis)
FEATURES = [
    "Net EPA", "grades_offense", "yards", "touchdowns", 
    "wide_snaps", "targeted_qb_rating", "avoided_tackles", 
    "weighted_grade", "grades_pass_route", "first_downs"
]
TARGET = "Net EPA"
SEQ_LENGTH = 1

print("Loading and training data...")
X, y, metadata = load_and_process_data("backend/ML/TightEnds/TE.csv", FEATURES, TARGET, SEQ_LENGTH)

if len(X) == 0:
    print("Not enough data for sequence length of", SEQ_LENGTH)
    exit()

# Normalize Features
N, L, F = X.shape
scaler = StandardScaler()
X_reshaped = X.reshape(-1, F)
X_scaled = scaler.fit_transform(X_reshaped).reshape(N, L, F)

# Train/Test Split (Time-based split)
metadata_years = np.array([m[1] for m in metadata])
test_mask = metadata_years == 2024
train_mask = ~test_mask

X_train, y_train = X_scaled[train_mask], y[train_mask]
X_test, y_test = X_scaled[test_mask], y[test_mask]

# Convert to PyTorch Tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

# ====================================================
# 2. LSTM MODEL DEFINITION
# ====================================================
class EPA_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(EPA_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# ====================================================
# 3. TRAINING
# ====================================================
INPUT_DIM = len(FEATURES)
HIDDEN_DIM = 64
NUM_LAYERS = 2
OUTPUT_DIM = 1
LEARNING_RATE = 0.001
EPOCHS = 150 # Increased epochs
BATCH_SIZE = 32

model = EPA_LSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Training model...")
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

# ====================================================
# 4. EVALUATION (2024)
# ====================================================
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).numpy()
    actuals = y_test_tensor.numpy()

rmse = np.sqrt(mean_squared_error(actuals, predictions))
r2 = r2_score(actuals, predictions)

print("\n==== 2024 VALIDATION RESULTS ====")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# ====================================================
# 5. PREDICT 2025
# ====================================================
# To predict 2025, we need the sequences ending in 2024.
# Input: [2024] -> Output: 2025 (Predicted)

df = pd.read_csv("backend/ML/TightEnds/TE.csv")
df = df.replace("MISSING", np.nan)
metadata_cols = ["player_id", "player", "Team", "position_x", "position", "franchise_id"]
for col in df.columns:
    if col not in metadata_cols and col not in ["Year"]:
         df[col] = pd.to_numeric(df[col], errors="coerce")
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df = df.sort_values(["player_id", "Year"])
df[FEATURES] = df.groupby("player_id")[FEATURES].ffill().fillna(0) # Same preprocessing

# Get latest year for each player
latest_df = df.sort_values("Year").groupby("player_id").tail(SEQ_LENGTH)
# Filter for players where the latest year is 2024
latest_df = latest_df[latest_df["Year"] == 2024]

# Create sequences
pred_2025_input = []
pred_meta = []

for pid, group in latest_df.groupby("player_id"):
    if len(group) < SEQ_LENGTH:
        continue
    seq = group[FEATURES].values
    pred_2025_input.append(seq)
    # Store player name/id
    player_name = group["player"].iloc[-1]
    pred_meta.append({"player_id": pid, "player": player_name, "Team": group["Team"].iloc[-1]})

if len(pred_2025_input) > 0:
    X_pred = np.array(pred_2025_input)
    # Scale
    N_p, L_p, F_p = X_pred.shape
    X_pred_reshaped = X_pred.reshape(-1, F_p)
    X_pred_scaled = scaler.transform(X_pred_reshaped).reshape(N_p, L_p, F_p)
    
    X_pred_tensor = torch.FloatTensor(X_pred_scaled)
    
    with torch.no_grad():
        preds_2025 = model(X_pred_tensor).numpy().flatten()
    
    results_2025 = pd.DataFrame(pred_meta)
    results_2025["Predicted_Net_EPA_2025"] = preds_2025
    results_2025 = results_2025.sort_values("Predicted_Net_EPA_2025", ascending=False)
    
    print("\n==== TOP 20 PREDICTED TEs FOR 2025 ====")
    print(results_2025.head(20).to_string(index=False))
    
    # Save results
    results_2025.to_csv("backend/ML/TightEnds/TELSTMPredictions.csv", index=False)
    print("\nFull 2025 predictions saved to backend/ML/TightEnds/TELSTMPredictions.csv")
else:
    print("No 2024 data found to predict 2025.")
