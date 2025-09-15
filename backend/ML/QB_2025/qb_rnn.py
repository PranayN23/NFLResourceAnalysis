import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient
from sklearn.metrics import r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, SimpleRNN, Dropout, MultiHeadAttention, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# -----------------------------
# MongoDB Setup
# -----------------------------
mongo_uri = "mongodb+srv://pranaynandkeolyar:nfl@cluster0.4nbxj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(mongo_uri)
pos_db = client['QB']
all_teams = pos_db.list_collection_names()

# -----------------------------
# Fetch data
# -----------------------------
all_data = []
for team in all_teams:
    collection = pos_db[team]
    cursor = collection.find({'Year': {'$exists': True}})
    for doc in cursor:
        player_data = {k: v for k, v in doc.items() if k != '_id'}
        all_data.append(player_data)

df = pd.DataFrame(all_data)
print(f"Columns in dataset: {df.columns.tolist()}")
print(df['adjusted_value'].dtype)
print(df['adjusted_value'].head(10))

# -----------------------------
# Feature and target setup
# -----------------------------

feature_columns = [
    'twp_rate', 'ypa', 'qb_rating', 'accuracy_percent',
    'btt_rate', 'Cap_Space', 'age', 'dropbacks', 'adjusted_value'
]
target_column = 'grades_offense'

# List of columns that might contain percentages
percent_columns = ['accuracy_percent', 'Win %', 'pressure_to_sack_rate']

for col in percent_columns:
    df[col] = df[col].astype(str).str.replace('%', '', regex=False)  # remove %
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) / 100  # convert to float


df['adjusted_value'] = df['adjusted_value'].replace("MISSING", 0)
df['adjusted_value'] = pd.to_numeric(df['adjusted_value'], errors='coerce').fillna(0)



# -----------------------------
# Fill NaNs with sensible defaults
# -----------------------------
fill_values = {
    'twp_rate': df['twp_rate'].mean(),
    'ypa': df['ypa'].mean(),
    'qb_rating': df['qb_rating'].mean(),
    'accuracy_percent': df['accuracy_percent'].mean(),
    'btt_rate': df['btt_rate'].mean(),
    'completion_percent': df['completion_percent'].mean(),
    'yards': 0,
    'touchdowns': 0,
    'interceptions': 0,
    'sack_percent': df['sack_percent'].mean(),
    'Cap_Space': df['Cap_Space'].mean(),
    'age': df['age'].mean(),
    'dropbacks': df['dropbacks'].mean(),
    'passing_snaps': df['passing_snaps'].mean(),
    'scrambles': df['scrambles'].mean(),
    'pressure_to_sack_rate': df['pressure_to_sack_rate'].mean(),
    'adjusted_value': df['adjusted_value'].mean()
}



for col, val in fill_values.items():
    
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(val)

df[target_column] = pd.to_numeric(df[target_column], errors='coerce').fillna(df[target_column].mean())

# -----------------------------
# Create sequences per player
# -----------------------------
sequences = []
targets = []
player_info = []

player_groups = df.groupby('player')

for player, group in player_groups:
    group = group.sort_values('Year')
    if len(group) >= 4:
        for i in range(len(group) - 3):
            seq = group.iloc[i:i+3][feature_columns].values.astype(float)
            target = group.iloc[i+3][target_column]
            sequences.append(seq)
            targets.append(float(target))
            player_info.append({
                'player': player,
                'target_year': group.iloc[i+3]['Year'],
                'team': group.iloc[i+3].get('Team', 'Unknown')
            })

X = np.array(sequences, dtype=np.float32)
y = np.array(targets, dtype=np.float32)
info_df = pd.DataFrame(player_info)

# -----------------------------
# Chronological train/val/test split
# -----------------------------
sort_indices = np.argsort(info_df['target_year'].values)
X = X[sort_indices]
y = y[sort_indices]
info_df = info_df.iloc[sort_indices].reset_index(drop=True)

n_samples = len(X)
train_end = int(n_samples * 0.6)
val_end = int(n_samples * 0.8)

X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# -----------------------------
# Build faster LSTM
# -----------------------------
inputs = Input(shape=(3, len(feature_columns)), dtype=tf.float32)
x = SimpleRNN(32, activation='relu')(inputs)   # Faster on short sequences
x = Dropout(0.1)(x)
output = Dense(1, dtype=tf.float32)(x)

model = Model(inputs, output)
model.compile(optimizer='adam', loss='mean_squared_error')

# -----------------------------
# Train model (Mac optimized)
# -----------------------------
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,      # optimized for Mac CPU/GPU
    callbacks=[early_stopping],
    verbose=2
)

# -----------------------------
# Evaluate model
# -----------------------------
y_pred = model.predict(X_test, batch_size=32).flatten()
r2 = r2_score(y_test, y_pred)
print(f"Test R² Score: {r2:.4f}")

# -----------------------------
# Plot Actual vs Predicted
# -----------------------------
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual grades_offense')
plt.ylabel('Predicted grades_offense')
plt.title(f'QB grades_offense Prediction - Test Set R²: {r2:.4f}')
plt.show()


# Prediction Error Distribution
error = np.abs(y_test - y_pred)
plt.figure(figsize=(10,6))
plt.hist(error, bins=20, alpha=0.7, color='orange')
plt.xlabel('Absolute Error')
plt.ylabel('Frequency')
plt.title('Prediction Error Distribution')
plt.show()

# Optional: Sample predictions
test_info = info_df.iloc[-len(y_test):].copy()
test_info['Actual'] = y_test
test_info['Predicted'] = y_pred
print("\nSample Predictions:")
print(test_info[['player','team','target_year','Actual','Predicted']].sample(10))
