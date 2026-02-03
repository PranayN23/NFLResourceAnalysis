
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. LOAD & PROCESS DATA
# ==========================================
def load_and_process(filepath):
    df = pd.read_csv(filepath)
    df = df.replace("MISSING", np.nan)
    
    # Metadata columns to exclude from numeric conversion
    metadata_cols = ["player_id", "player", "Team", "position_x", "position", "franchise_id"]
    for col in df.columns:
        if col not in metadata_cols and col != "Year":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.sort_values(["player_id", "Year"])
    
    # --- Feature Engineering (Same as XGBoost) ---
    df["age_sq"] = df["age"] ** 2
    df["targets_per_snap"] = df["targets"] / df["total_snaps"].replace(0, np.nan)
    df["epa_per_target"] = df["Net EPA"] / df["targets"].replace(0, np.nan)
    
    lag_features = [
        "TE_Value_Score", "Net EPA", "grades_offense", "yards", "touchdowns", 
        "first_downs", "yprr", "receptions", "targets",
        "grades_pass_route", "wide_snaps", "slot_snaps",
        "epa_per_target", "targets_per_snap"
    ]
    
    for col in lag_features:
        if col in df.columns:
            df[f"{col}_prev"] = df.groupby("player_id")[col].shift(1)
            df[f"{col}_prev2"] = df.groupby("player_id")[col].shift(2)
            df[f"{col}_trend"] = df[f"{col}_prev"] - df[f"{col}_prev2"]
            df[f"{col}_rolling2"] = (df[f"{col}_prev"] + df[f"{col}_prev2"]) / 2
            
    # Features & Target
    # Note: We explicitly include 'Team' now for embeddings
    target_col = "Net EPA"
    
    # Filter valid rows
    model_df = df.dropna(subset=[f"{target_col}_prev", target_col])
    
    # Select Predictors
    num_predictors = [c for c in model_df.columns if "_prev" in c or "_trend" in c or "_rolling" in c]
    num_predictors += ["age", "age_sq", "Cap_Space"]
    num_predictors = [c for c in num_predictors if c in model_df.columns]
    
    # --- PREPARE TENSORS ---
    # 1. Team Encoding
    # We need a consistent encoder for Team.
    team_encoder = LabelEncoder()
    # Fit on full dataset to ensure all teams are known
    all_teams = df["Team"].fillna("Unknown").unique()
    team_encoder.fit(all_teams)
    
    model_df["Team_Encoded"] = team_encoder.transform(model_df["Team"].fillna("Unknown"))
    
    # 2. Train/Test Split
    train = model_df[model_df["Year"] < 2024]
    test = model_df[model_df["Year"] == 2024]
    
    # 3. Scaling Numerical Data
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(train[num_predictors].fillna(0)) # Fill NaNs with 0 for NN
    X_test_num = scaler.transform(test[num_predictors].fillna(0))
    
    X_train_cat = train["Team_Encoded"].values
    X_test_cat = test["Team_Encoded"].values
    
    y_train = train[target_col].values
    y_test = test[target_col].values
    
    return (X_train_num, X_train_cat, y_train), (X_test_num, X_test_cat, y_test), (scaler, team_encoder, num_predictors, test)

# ==========================================
# 2. DEFINE NEURAL NET with EMBEDDINGS
# ==========================================
class NFL_Predictor(nn.Module):
    def __init__(self, num_features, num_teams, embedding_dim=10):
        super(NFL_Predictor, self).__init__()
        
        # Embedding Layer for Team
        # This allows the model to learn a vector representation of "Ravens", "Chiefs", etc.
        self.team_embedding = nn.Embedding(num_teams, embedding_dim)
        
        # Main Network
        input_dim = num_features + embedding_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128), # Batch Norm helps convergence
            nn.ReLU(),
            nn.Dropout(0.3), # Dropout prevents overfitting on small data
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 1) # Output: Net EPA
        )
        
    def forward(self, x_num, x_cat):
        # x_num: [batch, num_features]
        # x_cat: [batch] (Team IDs)
        
        embedded_team = self.team_embedding(x_cat) # [batch, emb_dim]
        
        # Concatenate numerical features with team embedding
        combined = torch.cat([x_num, embedded_team], dim=1)
        
        return self.net(combined)

# ==========================================
# 3. TRAINING LOOP
# ==========================================
print("Loading Data...")
(train_data, test_data, tools) = load_and_process("backend/ML/TightEnds/TE.csv")
X_train_num, X_train_cat, y_train = train_data
X_test_num, X_test_cat, y_test = test_data
scaler, team_encoder, feature_names, test_df_raw = tools

# Convert to Tensors
Xt_num = torch.FloatTensor(X_train_num)
Xt_cat = torch.LongTensor(X_train_cat)
yt = torch.FloatTensor(y_train).unsqueeze(1)

Xv_num = torch.FloatTensor(X_test_num)
Xv_cat = torch.LongTensor(X_test_cat)
yv_test_numpy = y_test # Keep for sklearn metrics

# Model Init
num_teams = len(team_encoder.classes_)
model = NFL_Predictor(num_features=X_train_num.shape[1], num_teams=num_teams, embedding_dim=8)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) # Weight decay = L2 regularization
criterion = nn.MSELoss()

print(f"Training on {len(Xt_num)} samples, Validating on {len(Xv_num)} samples (2024)...")

EPOCHS = 100
BATCH_SIZE = 32

dataset = torch.utils.data.TensorDataset(Xt_num, Xt_cat, yt)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for bn, bc, by in loader:
        optimizer.zero_grad()
        output = model(bn, bc)
        loss = criterion(output, by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}: Loss {total_loss/len(loader):.5f}")

# ==========================================
# 4. EVALUATION
# ==========================================
model.eval()
with torch.no_grad():
    preds = model(Xv_num, Xv_cat).numpy().flatten()

rmse = np.sqrt(mean_squared_error(yv_test_numpy, preds))
r2 = r2_score(yv_test_numpy, preds)

print("\n==== 2024 NEURAL NET RESULTS ====")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")

# Save Validation Results
val_res = pd.DataFrame({
    "player": test_df_raw["player"],
    "Team": test_df_raw["Team"],
    "Net EPA": yv_test_numpy,
    "Predicted": preds,
    "Error": yv_test_numpy - preds
})
print("\nTop 10 Predictions (validation):")
print(val_res.sort_values("Predicted", ascending=False).head(10).to_string(index=False))

# ==========================================
# 5. PREDICT 2025
# ==========================================
# (Requires reconstructing the 2025 input vector like in XGBoost script)
# For brevity, we'll demonstrate that this architecture works first.
