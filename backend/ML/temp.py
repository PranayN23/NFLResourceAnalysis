# ==============================
# NFL QB Performance Prediction (Top Features)
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Masking, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pymongo import MongoClient

# ------------------------------
# 1. Load Data from MongoDB
# ------------------------------
mongo_uri = "mongodb+srv://pranaynandkeolyar:nfl@cluster0.4nbxj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(mongo_uri)
pos_db = client['QB']
all_teams = pos_db.list_collection_names()

all_data = []
for team in all_teams:
    collection = pos_db[team]
    cursor = collection.find({'Year': {'$exists': True}})
    for doc in cursor:
        player_data = {k: v for k, v in doc.items() if k != '_id'}
        all_data.append(player_data)

df = pd.DataFrame(all_data)
print(f"Loaded {len(df)} rows from MongoDB")
print(f"Columns: {df.columns.tolist()}")

# ------------------------------
# 2. Clean & Process Columns
# ------------------------------
percent_cols = ['accuracy_percent', 'Win %', 'pressure_to_sack_rate']
for col in percent_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('%','',regex=False), errors='coerce').fillna(0)/100

df['adjusted_value'] = pd.to_numeric(df['adjusted_value'].replace("MISSING", 0), errors='coerce').fillna(0)
df['dropbacks'] = pd.to_numeric(df['dropbacks'], errors='coerce').fillna(0)
df['grades_offense'] = pd.to_numeric(df['grades_offense'], errors='coerce').fillna(df['grades_offense'].mean())

df = df[df['passing_snaps'] >= 100].copy()
df = df.sort_values(['player', 'Year'])

# ------------------------------
# 3. Lagged Features
# ------------------------------
exclude_cols = [
    'Net EPA', 'Win %', 'weighted_grade', 'weighted_average_grade', 
    'player', 'position', 'position_x'
]

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
lag_cols = [c for c in numeric_cols if c not in exclude_cols and c not in ['Cap_Space', 'adjusted_value']]

lag_feature_cols = [f'Previous_{c}' for c in lag_cols]
df[lag_feature_cols] = df[lag_cols].shift(1)
df[lag_feature_cols] = df[lag_feature_cols].fillna(0)

# ------------------------------
# 4. Correlation Matrix (Optional Visualization)
# ------------------------------
corr_cols = ['Cap_Space', 'adjusted_value'] + lag_feature_cols + ['grades_offense']
corr_matrix = df[corr_cols].corr()
sorted_corr = corr_matrix['grades_offense'].sort_values(ascending=False)
print("Sorted correlation with grades_offense:")
print(sorted_corr)

plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix (Lagged Features)")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
print("Saved correlation matrix to correlation_matrix.png")

# ------------------------------
# 5. Prepare Sequences for LSTM (Top Features Only)
# ------------------------------
top_features = [
    'Cap_Space',
    'Previous_grades_offense',
    'Previous_touchdowns',
    'Previous_grades_pass',
    'Previous_first_downs',
    'Previous_yards',
    'Previous_completions',
    'Previous_qb_rating',
    'Previous_passing_snaps',
    'Previous_attempts',
    'Previous_aimed_passes',
    'Previous_dropbacks',
    'Previous_big_time_throws',
    'Previous_def_gen_pressures',
    'Previous_sacks',
    'Previous_penalties'
]

top_features = [f for f in top_features if f in df.columns]

sequences = []
targets = []
years_seq = []
players_seq = []

for player, group in df.groupby('player'):
    if len(group) >= 4:
        group = group.sort_values('Year')
        for i in range(len(group)-3):
            seq = group.iloc[i:i+3][top_features].values
            tgt = group.iloc[i+3]['grades_offense']
            sequences.append(seq)
            targets.append(tgt)
            years_seq.append(group.iloc[i+3]['Year'])
            players_seq.append(player)

X = np.array(sequences)
y = np.array(targets)
years_seq = np.array(years_seq)
players_seq = np.array(players_seq)
print(f"Total sequences: {len(X)}, X shape: {X.shape}, y shape: {y.shape}")

# ------------------------------
# 6. Train / Validation / Test Split
# ------------------------------
train_mask = (years_seq >= 2015) & (years_seq <= 2020)
val_mask = (years_seq >= 2021) & (years_seq <= 2022)
test_mask = (years_seq >= 2023) & (years_seq <= 2024)

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

n_samples, n_timesteps, n_features = X_train.shape
scaler = StandardScaler()
X_train_flat = X_train.reshape(n_samples, n_timesteps*n_features)
X_val_flat = X_val.reshape(X_val.shape[0], n_timesteps*n_features)
X_test_flat = X_test.reshape(X_test.shape[0], n_timesteps*n_features)

scaler.fit(X_train_flat)
X_train_scaled = scaler.transform(X_train_flat).reshape(n_samples, n_timesteps, n_features)
X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape[0], n_timesteps, n_features)
X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape[0], n_timesteps, n_features)

# ------------------------------
# 7. Epoch Sweep Training
# ------------------------------
epoch_list = [50, 100, 150, 200, 300]
results = []

for epochs in epoch_list:
    print(f"\nTraining with {epochs} epochs...\n")
    
    inputs = Input(shape=(n_timesteps, n_features))
    masked = Masking(mask_value=0.0)(inputs)
    lstm_out = LSTM(32, return_sequences=True)(masked)
    gap = GlobalAveragePooling1D()(lstm_out)
    dense1 = Dense(32, activation='relu')(gap)
    drop1 = Dropout(0.2)(dense1)
    output = Dense(1, activation='linear')(drop1)
    model = Model(inputs=inputs, outputs=output)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='huber')
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=epochs,
        batch_size=8,
        verbose=1,
        callbacks=[early_stopping]
    )

    y_val_pred = model.predict(X_val_scaled).flatten()
    r2 = r2_score(y_val, y_val_pred)
    mae = mean_absolute_error(y_val, y_val_pred)
    rmse = mean_squared_error(y_val, y_val_pred, squared=False)
    print(f"Epochs={epochs}: R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

    results.append({'epochs': epochs, 'r2': r2, 'mae': mae, 'rmse': rmse})

# Save epoch sweep results
results_df = pd.DataFrame(results)
results_df.to_csv("epoch_sweep_results.csv", index=False)
print("Saved epoch sweep results to epoch_sweep_results.csv")

# ------------------------------
# 8. Plot metrics vs epochs
# ------------------------------
plt.figure(figsize=(10,5))
plt.plot(results_df['epochs'], results_df['r2'], marker='o', label='R²')
plt.plot(results_df['epochs'], results_df['mae'], marker='x', label='MAE')
plt.plot(results_df['epochs'], results_df['rmse'], marker='s', label='RMSE')
plt.xlabel('Epochs')
plt.ylabel('Metric Value')
plt.title('Validation Metrics vs Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("metrics_vs_epochs.png")
print("Saved metrics plot to metrics_vs_epochs.png")
