
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Check device stability
# For Mac M1/M2, 'mps' is available but sometimes float64 isn't supported. 
# We'll stick to CPU for maximum stability given the previous crashes, 
# unless we explicitly want to try 'mps'. Let's stay safe with CPU for now.
DEVICE = torch.device('cpu')
print(f"Using device: {DEVICE}")

# --- PFF CLASS DEFINITIONS (3 TIERS) ---
# TEAM REVIEW NOTE:
# We consolidated 6 PFF tiers into 3 for better model stability.
# Tier 2 (Elite/High Quality): 80.0+ (Includes both Elite & High Quality)
# Tier 1 (Starter/Average): 60.0 - 79.9 (Includes Above Avg, Avg, Below Avg)
# Tier 0 (Reserve/Poor): < 60.0 (Includes Poor)

def get_tier(grade):
    if grade >= 80.0: return 2
    elif grade >= 60.0: return 1
    else: return 0

TIER_NAMES = ["Reserve/Poor", "Starter/Average", "Elite/High Quality"]

# 1. Time2Vec Layer (PyTorch Implementation)
class Time2Vec(nn.Module):
    def __init__(self, input_dim, kernel_size=1):
        super(Time2Vec, self).__init__()
        self.k = kernel_size
        self.input_dim = input_dim
        
        # Initializing weights (matches the logic of Time2Vec paper/implementations)
        # Linear term weights: w0, b0
        self.w0 = nn.Parameter(torch.randn(input_dim, 1)) 
        self.b0 = nn.Parameter(torch.randn(input_dim, 1))
        
        # Periodic term weights: w1..k, b1..k (captured in one tensor)
        self.wk = nn.Parameter(torch.randn(input_dim, kernel_size))
        self.bk = nn.Parameter(torch.randn(input_dim, kernel_size))
        
    def forward(self, x):
        # x shape: [batch, seq_len, features]
        
        # We process each feature independently for time2vec embeddings
        # Simplest approach: Apply Time2Vec to each element
        # But usually Time2Vec is for the *Time* feature. 
        # Here we are applying it as a feature embedding layer for ALL features as per the previous keras code.
        
        # Linear term: x * w0 + b0 (elementwise scaling and shift)
        # We need to broadcast properly. 
        # w0 shape: [features, 1] - we want to multiply each feature channel
        
        # Reshape x for easy multiplication: [batch, seq_len, features, 1]
        x_uns = x.unsqueeze(-1)
        
        # Linear: [batch, seq_len, features, 1]
        linear = x_uns * self.w0 + self.b0
        
        # Periodic: [batch, seq_len, features, k]
        # x_uns * wk + bk
        periodic = torch.sin(x_uns * self.wk + self.bk)
        
        # Concatenate: [batch, seq_len, features, k+1]
        out = torch.cat([linear, periodic], dim=-1)
        
        # Flatten the last two dims: [batch, seq_len, features * (k+1)]
        out = out.reshape(x.size(0), x.size(1), -1)
        return out

# 2. Transformer Classification Model
class PlayerTransformerClassifier(nn.Module):
    def __init__(self, input_dim, seq_len, num_classes=3, kernel_size=1, num_heads=4, ff_dim=64, num_layers=2, dropout=0.1):
        super(PlayerTransformerClassifier, self).__init__()
        
        # Time2Vec Embedding
        self.time2vec = Time2Vec(input_dim, kernel_size)
        
        # Helper to compute expanded dimension
        self.embed_dim = input_dim * (kernel_size + 1)
        
        # Ensure embed_dim is divisible by num_heads for PyTorch Transformer
        if self.embed_dim % num_heads != 0:
            # Simple padding layer to make it compatible
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
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes) # Logic Output (logits)
        )
        
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        
        # Embedding
        x = self.time2vec(x) # [batch, seq_len, input_dim * (k+1)]
        x = self.pad_proj(x)
        
        # Transformer
        x = self.transformer_encoder(x) # [batch, seq_len, embed_dim]
        
        # Global Average Pooling
        # PyTorch pooling expects [batch, channels, length]
        x = x.permute(0, 2, 1) # [batch, embed_dim, seq_len]
        x = self.global_avg_pool(x).squeeze(-1) # [batch, embed_dim]
        
        # Classifier
        x = self.classifier(x)
        return x

# 3. Data Processing
data_path = '../../ML/QB.csv' 
if not os.path.exists(data_path):
    data_path = '/Users/pranaynandkeolyar/Documents/NFLSalaryCap/backend/ML/QB.csv'

print(f"Reading data from: {data_path}")
df = pd.read_csv(data_path)

# TEAM REVIEW NOTE (STAGE 1 - DATA):
# We filter out players with < 100 dropbacks to avoid noise from backups.
# REVIEW: Is 100 the right cutoff? Should it be higher (e.g., 200) for "Starter" analysis?
MIN_DROPBACKS = 100
print(f"Filtering for Minimum Dropbacks: {MIN_DROPBACKS}")
if 'dropbacks' in df.columns:
    df = df[df['dropbacks'] >= MIN_DROPBACKS].copy()

if 'Year' not in df.columns and 'year' in df.columns: 
    df.rename(columns={'year': 'Year'}, inplace=True)

if 'player' not in df.columns:
     raise KeyError(f"Could not find 'player' column.")

print(f"Unique Players: {df['player'].nunique()}")
df.sort_values(by=['player', 'Year'], inplace=True)

SEQUENCE_LENGTH = 3 

features = [
    'grades_pass', 
    'grades_offense',
    'qb_rating',
    'adjusted_value',
    'Cap_Space',
    'ypa',
    'twp_rate',
    'btt_rate',
    'completion_percent'
]
target_col = 'grades_offense' 

available_features = [f for f in features if f in df.columns]
if len(available_features) < len(features):
    features = available_features

df_clean = df.dropna(subset=features + [target_col])

# Create Classes BEFORE normalization
print("Raw Grades Distribution:")
print(df_clean[target_col].describe())
df_clean['tier'] = df_clean[target_col].apply(get_tier)
print("Target Class Distribution:")
print(df_clean['tier'].value_counts().sort_index())

# Normalize Features
scaler = StandardScaler()
df_clean[features] = scaler.fit_transform(df_clean[features])

def create_sequences(dataset, seq_len, features, target_class_col):
    X = []
    y = []
    for player, group in dataset.groupby('player'):
        group = group.sort_values('Year')
        if len(group) <= seq_len:
            continue
        vals = group[features].values
        targs = group[target_class_col].values
        for i in range(len(group) - seq_len):
            X.append(vals[i:i+seq_len])
            y.append(targs[i+seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

X, y = create_sequences(df_clean, SEQUENCE_LENGTH, features, 'tier')
print(f"Generated Sequences Shape: X={X.shape}, y={y.shape}")

if len(X) == 0:
    print("Error: Not enough history.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verify Class Balance in Split
unique, counts = np.unique(y_train, return_counts=True)
class_counts = dict(zip(unique, counts))
print(f"Train Class Dist: {class_counts}")

# TEAM REVIEW NOTE (STAGE 2 - MODEL OPTIMIZATION):
# We use "Inverse Frequency" weighting to force the model to care about rare "Elite" seasons.
# This increases Recall (finding elites) but decreases Precision (more false positives).
# REVIEW: Adjust this strategy if the GM Agent makes too many "High Risk" signings.
total_samples = len(y_train)
class_weights = []
for i in range(3):
    count = class_counts.get(i, 0)
    if count > 0:
        weight = total_samples / (3 * count) # Sklearn style: n_samples / (n_classes * n_samples_j)
    else:
        weight = 1.0
    class_weights.append(weight)

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
print(f"Computed Class Weights: {class_weights}")

# Convert to PyTorch tensors
train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 4. Train
num_classes = 3
model = PlayerTransformerClassifier(input_dim=len(features), seq_len=SEQUENCE_LENGTH, num_classes=num_classes).to(DEVICE)
print(model)

# Weighted Loss to handle imbalance
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

EPOCHS = 150
best_acc = 0.0

print("Starting training...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct / total
    
    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            val_total += targets.size(0)
            val_correct += (predicted == targets).sum().item()
    
    val_acc = val_correct / val_total
    
    # Step Scheduler based on Val Accuracy
    scheduler.step(val_acc)
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Acc: {val_acc:.4f}")
        
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_classifier.pth')

# 5. Eval
model.load_state_dict(torch.load('best_classifier.pth'))
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.numpy())

print(f"\nFinal Test Accuracy: {accuracy_score(all_targets, all_preds):.4f}")
print("\nClassification Report (Weighted):")
print(classification_report(all_targets, all_preds, target_names=[TIER_NAMES[i] for i in sorted(list(set(all_targets)))]))

# Confusion Matrix
cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=TIER_NAMES, yticklabels=TIER_NAMES)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - PFF Tier Prediction (Optimized)')
plt.savefig('player_prediction_classifier_cm_optimized.png')
print("Confusion Matrix saved to player_prediction_classifier_cm_optimized.png")
