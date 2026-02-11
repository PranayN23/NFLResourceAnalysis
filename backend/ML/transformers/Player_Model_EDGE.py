import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns

# For stability, keep everything on CPU for now.
DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")


# --- PFF TIERING (3 TIERS, DEFENSE) ---
def get_tier(grade: float) -> int:
    if grade >= 80.0:
        return 2
    elif grade >= 60.0:
        return 1
    else:
        return 0


TIER_NAMES = ["Reserve/Poor", "Starter/Average", "Elite/High Quality"]


# 1. Time2Vec Layer (same architecture as QB model)
class Time2Vec(nn.Module):
    def __init__(self, input_dim: int, kernel_size: int = 1):
        super().__init__()
        self.k = kernel_size
        self.input_dim = input_dim

        self.w0 = nn.Parameter(torch.randn(input_dim, 1))
        self.b0 = nn.Parameter(torch.randn(input_dim, 1))

        self.wk = nn.Parameter(torch.randn(input_dim, kernel_size))
        self.bk = nn.Parameter(torch.randn(input_dim, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, features]
        x_uns = x.unsqueeze(-1)  # [batch, seq_len, features, 1]
        linear = x_uns * self.w0 + self.b0  # [batch, seq_len, features, 1]
        periodic = torch.sin(x_uns * self.wk + self.bk)  # [batch, seq_len, features, k]
        out = torch.cat([linear, periodic], dim=-1)  # [batch, seq_len, features, k+1]
        out = out.reshape(x.size(0), x.size(1), -1)  # [batch, seq_len, features * (k+1)]
        return out


# 2. Transformer Classification Model (same hyperparameters as QB)
class EdgeTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        num_classes: int = 3,
        kernel_size: int = 1,
        num_heads: int = 4,
        ff_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

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
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.time2vec(x)
        x = self.pad_proj(x)
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1)  # [batch, embed_dim, seq_len]
        x = self.global_avg_pool(x).squeeze(-1)  # [batch, embed_dim]
        x = self.classifier(x)
        return x


# 3. Data Processing for EDGE players
data_path = "../../ML/ED.csv"
if not os.path.exists(data_path):
    data_path = "/Users/pranaynandkeolyar/Documents/NFLSalaryCap/backend/ML/ED.csv"

print(f"Reading EDGE data from: {data_path}")
df = pd.read_csv(data_path)

# Filter to EDGE position only, just in case the CSV has multiple positions.
if "position" in df.columns:
    df = df[df["position"] == "ED"].copy()

# Minimum defensive snaps to focus on real contributors (analogous to QB dropbacks filter).
MIN_SNAPS = 300
if "snap_counts_defense" in df.columns:
    print(f"Filtering for Minimum Defensive Snaps: {MIN_SNAPS}")
    df = df[df["snap_counts_defense"] >= MIN_SNAPS].copy()

if "Year" not in df.columns and "year" in df.columns:
    df.rename(columns={"year": "Year"}, inplace=True)

if "player" not in df.columns:
    raise KeyError("Could not find 'player' column in EDGE data.")

print(f"Unique EDGE Players: {df['player'].nunique()}")
df.sort_values(by=["player", "Year"], inplace=True)

SEQUENCE_LENGTH = 3

# Mirror QB-style feature selection but with defensive metrics.
features = [
    "grades_defense",
    "grades_pass_rush_defense",
    "grades_run_defense",
    "grades_tackle",
    "Cap_Space",
    "Net EPA",
    "snap_counts_pass_rush",
    "snap_counts_run_defense",
    "total_pressures",
]
target_col = "grades_defense"

available_features = [f for f in features if f in df.columns]
if len(available_features) < len(features):
    print(
        "Warning: Some configured EDGE features are missing. "
        f"Using available subset: {available_features}"
    )
    features = available_features

df_clean = df.dropna(subset=features + [target_col])

print("Raw Defensive Grades Distribution:")
print(df_clean[target_col].describe())
df_clean["tier"] = df_clean[target_col].apply(get_tier)
print("Target Class Distribution (EDGE):")
print(df_clean["tier"].value_counts().sort_index())

scaler = StandardScaler()
df_clean[features] = scaler.fit_transform(df_clean[features])


def create_sequences(dataset: pd.DataFrame, seq_len: int, features, target_class_col):
    X = []
    y = []
    for player, group in dataset.groupby("player"):
        group = group.sort_values("Year")
        if len(group) <= seq_len:
            continue
        vals = group[features].values
        targs = group[target_class_col].values
        for i in range(len(group) - seq_len):
            X.append(vals[i : i + seq_len])
            y.append(targs[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


X, y = create_sequences(df_clean, SEQUENCE_LENGTH, features, "tier")
print(f"Generated EDGE Sequences Shape: X={X.shape}, y={y.shape}")

if len(X) == 0:
    print("Error: Not enough EDGE history for sequence modeling.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

unique, counts = np.unique(y_train, return_counts=True)
class_counts = dict(zip(unique, counts))
print(f"Train Class Dist (EDGE): {class_counts}")

total_samples = len(y_train)
class_weights = []
for i in range(3):
    count = class_counts.get(i, 0)
    if count > 0:
        weight = total_samples / (3 * count)
    else:
        weight = 1.0
    class_weights.append(weight)

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
print(f"Computed Class Weights (EDGE): {class_weights}")

train_dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(X_train), torch.from_numpy(y_train)
)
test_dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(X_test), torch.from_numpy(y_test)
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

num_classes = 3
model = EdgeTransformerClassifier(
    input_dim=len(features), seq_len=SEQUENCE_LENGTH, num_classes=num_classes
).to(DEVICE)
print(model)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=10
)

EPOCHS = 150
best_acc = 0.0

print("Starting EDGE training...")

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
    scheduler.step(val_acc)

    if (epoch + 1) % 10 == 0:
        print(
            f"[EDGE] Epoch {epoch+1}/{EPOCHS}, "
            f"Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Acc: {val_acc:.4f}"
        )

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_classifier_edge.pth")

model.load_state_dict(torch.load("best_classifier_edge.pth"))
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

print(f"\nFinal EDGE Test Accuracy: {accuracy_score(all_targets, all_preds):.4f}")
print("\nEDGE Classification Report (Weighted):")
print(
    classification_report(
        all_targets,
        all_preds,
        target_names=[TIER_NAMES[i] for i in sorted(list(set(all_targets)))],
    )
)

cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=TIER_NAMES, yticklabels=TIER_NAMES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - EDGE Tier Prediction (Optimized)")
plt.savefig("player_prediction_classifier_cm_optimized_edge.png")
print(
    "Confusion Matrix saved to player_prediction_classifier_cm_optimized_edge.png and model to best_classifier_edge.pth"
)

