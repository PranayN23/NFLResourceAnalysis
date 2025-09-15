import numpy as np
import pandas as pd
from pymongo import MongoClient
import tensorflow as tf
from tensorflow.keras import layers, models

# --- Connect to MongoDB ---
client = MongoClient("mongodb+srv://<db_username>:<db_password>@cluster0.4nbxj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["football_stats"]  # DB name for football stats

# --- Load all collections (teams) ---
all_data = []
for team in db.list_collection_names():
    data = list(db[team].find())
    for d in data:
        d["team"] = team
    all_data.extend(data)

df = pd.DataFrame(all_data)

year_col = "year" if "year" in df.columns else "season"
target = "sacks"  # example IDL-relevant stat; could also be tackles or pressures


ignore_cols = ["_id", "team", year_col, target]
features = [c for c in df.columns if c not in ignore_cols and np.issubdtype(df[c].dtype, np.number)]

df = df.sort_values([year_col, "team"]).reset_index(drop=True)

X = df[features].fillna(0).to_numpy(dtype=np.float32)
y = df[target].fillna(0).to_numpy(dtype=np.float32)
teams = df["team"].values
years = df[year_col].values

SEQ_LEN = 10
Xs, ys, seq_years = [], [], []

for t in df["team"].unique():
    idx = np.where(teams == t)[0]
    for i in range(len(idx) - SEQ_LEN):
        Xs.append(X[idx[i:i+SEQ_LEN]])
        ys.append(y[idx[i+SEQ_LEN]])
        seq_years.append(years[idx[i+SEQ_LEN]])

X_seq, y_seq, seq_years = np.array(Xs), np.array(ys), np.array(seq_years)

# --- 15-year split: 11 train, 2 val, 2 test ---
unique_years = np.sort(df[year_col].unique())
train_years = unique_years[:11]
val_years = unique_years[11:13]
test_years = unique_years[13:15]

train_idx = np.isin(seq_years, train_years)
val_idx = np.isin(seq_years, val_years)
test_idx = np.isin(seq_years, test_years)

X_train, y_train = X_seq[train_idx], y_seq[train_idx]
X_val, y_val = X_seq[val_idx], y_seq[val_idx]
X_test, y_test = X_seq[test_idx], y_seq[test_idx]

#model
model = models.Sequential([
    layers.Input(shape=(SEQ_LEN, X_train.shape[2])),
    layers.LSTM(64),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)
])

model.compile(optimizer="adam",
              loss="mse",
              metrics=["mae", tf.keras.metrics.RootMeanSquaredError()])

#train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32
)

#test
test_loss, test_mae, test_rmse = model.evaluate(X_test, y_test)
print(f"Test MSE: {test_loss:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
