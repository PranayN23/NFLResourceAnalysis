"""
Transformer model using INDIVIDUAL player data (not aggregated)
Key idea: Treat each team-year's LB roster as a SET of players
The transformer learns which players/stats matter most via attention
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_individual_player_data():
    """Load raw individual player data from LB.csv"""
    print("="*80)
    print("TRANSFORMER ON INDIVIDUAL PLAYER DATA")
    print("="*80)

    df = pd.read_csv('LB.csv')
    print(f"\nLoaded {len(df)} individual player records")

    # Replace 'MISSING' and 'NaN' strings with actual NaN
    df = df.replace(['MISSING', 'NaN', 'nan', ''], np.nan)

    # Convert numeric columns to numeric types
    numeric_cols = [
        'grades_defense', 'tackles', 'assists', 'sacks', 'stops',
        'grades_coverage_defense', 'grades_run_defense', 'grades_tackle',
        'snap_counts_defense', 'player_game_count', 'age',
        'interceptions', 'pass_break_ups', 'tackles_for_loss', 'Year'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"Years: {sorted(df['Year'].dropna().unique())}")
    print(f"Unique players: {df['player_id'].nunique()}")
    print(f"Teams: {df['Team'].nunique()}")

    return df

def select_features(df):
    """Select key features for each player"""

    # Key stats from previous analysis
    player_features = [
        # Performance metrics
        'grades_defense', 'tackles', 'assists', 'sacks', 'stops',
        'grades_coverage_defense', 'grades_run_defense', 'grades_tackle',
        'snap_counts_defense', 'player_game_count', 'age',
        'interceptions', 'pass_break_ups', 'tackles_for_loss'
    ]

    # Filter to available features
    available_features = [f for f in player_features if f in df.columns]
    print(f"\nUsing {len(available_features)} player features:")
    for f in available_features:
        print(f"  - {f}")

    return available_features

def create_team_year_sequences(df, features, max_players_per_team=10):
    """
    Create sequences where each sample is a team-year's roster of LBs

    Returns:
        X: (n_samples, max_players, n_features) - padded sequences
        y: (n_samples,) - next year's team average grades_defense
        mask: (n_samples, max_players) - which positions are real players
        metadata: team, year info
    """
    print(f"\nCreating team-year sequences (max {max_players_per_team} players per team)...")

    sequences = []
    targets = []
    masks = []
    metadata = []

    # Group by team and year
    team_years = df.groupby(['Team', 'Year'])

    for (team, year), group in team_years:
        # Get this year's players
        players = group.copy()

        # Remove rows with missing features
        players_clean = players[features].dropna()

        if len(players_clean) == 0:
            continue

        # Get next year's data for this team
        next_year_data = df[(df['Team'] == team) & (df['Year'] == year + 1)]

        if len(next_year_data) == 0:
            continue  # No next year data available

        # Target: next year's average grades_defense for this team
        next_year_grades = next_year_data['grades_defense'].dropna()
        if len(next_year_grades) == 0:
            continue

        target = next_year_grades.mean()  # Average of all LBs next year

        # Get feature matrix for this team-year
        X_team = players_clean[features].values
        n_players = len(X_team)

        # Pad or truncate to max_players
        if n_players > max_players_per_team:
            # Keep top players by snap count
            snap_counts = players_clean['snap_counts_defense'].values
            top_indices = np.argsort(snap_counts)[-max_players_per_team:]
            X_team = X_team[top_indices]
            n_players = max_players_per_team
            player_mask = np.ones(max_players_per_team, dtype=np.float32)
        else:
            # Pad with zeros
            padding = np.zeros((max_players_per_team - n_players, len(features)))
            X_team = np.vstack([X_team, padding])
            player_mask = np.concatenate([
                np.ones(n_players, dtype=np.float32),
                np.zeros(max_players_per_team - n_players, dtype=np.float32)
            ])

        sequences.append(X_team)
        targets.append(target)
        masks.append(player_mask)
        metadata.append({'team': team, 'year': year, 'n_players': n_players})

    X = np.array(sequences)  # (n_samples, max_players, n_features)
    y = np.array(targets)     # (n_samples,)
    mask = np.array(masks)    # (n_samples, max_players)

    print(f"\nCreated {len(X)} team-year sequences")
    print(f"Shape: {X.shape} (samples, players, features)")
    print(f"Target range: [{y.min():.1f}, {y.max():.1f}]")

    return X, y, mask, metadata

def time_based_split(X, y, mask, metadata, test_size=0.2, val_size=0.2):
    """Split by year (most recent for test)"""

    years = np.array([m['year'] for m in metadata])
    unique_years = sorted(np.unique(years))

    n_years = len(unique_years)
    n_test_years = max(1, int(n_years * test_size))
    n_val_years = max(1, int(n_years * val_size))

    test_years = unique_years[-n_test_years:]
    val_years = unique_years[-(n_test_years + n_val_years):-n_test_years]
    train_years = unique_years[:-(n_test_years + n_val_years)]

    print(f"\nTime-based split:")
    print(f"  Train: {train_years} ({len(train_years)} years)")
    print(f"  Val: {val_years} ({len(val_years)} years)")
    print(f"  Test: {test_years} ({len(test_years)} years)")

    train_mask = np.isin(years, train_years)
    val_mask = np.isin(years, val_years)
    test_mask = np.isin(years, test_years)

    return (X[train_mask], X[val_mask], X[test_mask],
            y[train_mask], y[val_mask], y[test_mask],
            mask[train_mask], mask[val_mask], mask[test_mask])

class TransformerEncoder(layers.Layer):
    """Transformer encoder layer"""

    def __init__(self, d_model, num_heads, ff_dim, dropout=0.4):
        super().__init__()
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout
        )
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, mask=None, training=False):
        # Multi-head attention
        attn_output = self.attention(inputs, inputs, attention_mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-forward network
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def build_player_transformer(max_players, n_features, d_model=32, num_heads=4, ff_dim=64, dropout=0.4):
    """
    Build transformer that processes a SET of players

    Architecture:
      Input: (batch, max_players, n_features)
      → Linear projection to d_model
      → Transformer encoder (learns player interactions via attention)
      → Global average pooling (aggregate all players)
      → Dense layers
      → Output: predicted team performance
    """
    print("\n" + "="*80)
    print("BUILDING PLAYER-LEVEL TRANSFORMER")
    print("="*80)
    print(f"  Max players: {max_players}")
    print(f"  Features per player: {n_features}")
    print(f"  Embedding dim: {d_model}, Heads: {num_heads}, FF dim: {ff_dim}")

    # Inputs
    player_input = layers.Input(shape=(max_players, n_features), name='player_features')
    mask_input = layers.Input(shape=(max_players,), name='player_mask')

    # Project to embedding space
    x = layers.Dense(d_model, activation='relu')(player_input)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Transformer encoder
    x = TransformerEncoder(d_model, num_heads, ff_dim, dropout)(x, mask=mask_input[:, tf.newaxis, :])

    # Apply mask before pooling (zero out padded players)
    mask_expanded = mask_input[:, :, tf.newaxis]  # (batch, max_players, 1)
    x_masked = x * mask_expanded

    # Global average pooling (aggregate all players)
    # Sum and divide by number of real players
    sum_players = tf.reduce_sum(x_masked, axis=1)  # (batch, d_model)
    count_players = tf.reduce_sum(mask_input, axis=1, keepdims=True)  # (batch, 1)
    x_pooled = sum_players / (count_players + 1e-8)  # Avoid division by zero

    # Dense layers for final prediction
    x = layers.Dense(64, activation='relu')(x_pooled)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dropout(dropout/2)(x)

    # Output
    output = layers.Dense(1, name='grade_prediction')(x)

    model = keras.Model(inputs=[player_input, mask_input], outputs=output)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    print("\nModel summary:")
    model.summary()

    return model

def build_baseline_model(max_players, n_features):
    """Simple baseline: flatten all players and use dense network"""
    print("\n" + "="*80)
    print("BUILDING BASELINE (Flattened Dense)")
    print("="*80)

    player_input = layers.Input(shape=(max_players, n_features))
    mask_input = layers.Input(shape=(max_players,))

    # Apply mask before flattening (zero out padded players)
    mask_expanded = layers.Reshape((max_players, 1))(mask_input)  # Use Keras layer
    x_masked = layers.Multiply()([player_input, mask_expanded])

    # Flatten
    x = layers.Flatten()(x_masked)

    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    output = layers.Dense(1)(x)

    model = keras.Model(inputs=[player_input, mask_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model

def train_model(model, X_train, y_train, mask_train, X_val, y_val, mask_val,
                model_name, epochs=200):
    """Train model"""
    print(f"\nTraining {model_name}...")

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=30,
                                      restore_best_weights=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                          patience=15, min_lr=1e-7, verbose=0)
    ]

    history = model.fit(
        [X_train, mask_train], y_train,
        validation_data=([X_val, mask_val], y_val),
        epochs=epochs,
        batch_size=16,
        callbacks=callbacks,
        verbose=0
    )

    print(f"  Stopped at epoch {len(history.history['loss'])}")
    print(f"  Best val loss: {min(history.history['val_loss']):.4f}")

    return history

def evaluate_model(model, X, y, mask, split_name):
    """Evaluate model"""
    pred = model.predict([X, mask], verbose=0).flatten()
    r2 = r2_score(y, pred)
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))

    print(f"  {split_name} - R²: {r2:.4f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    return {'r2': r2, 'mae': mae, 'rmse': rmse, 'pred': pred}

def plot_results(baseline_results, transformer_results, y_test, baseline_hist, trans_hist):
    """Plot comparison"""

    fig = plt.figure(figsize=(16, 5))

    # 1. R² comparison
    ax1 = plt.subplot(1, 4, 1)
    models = ['Baseline', 'Transformer']
    r2_scores = [baseline_results['test']['r2'], transformer_results['test']['r2']]
    colors = ['steelblue', 'orange']
    bars = ax1.bar(models, r2_scores, color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_ylabel('Test R²')
    ax1.set_title('Model Comparison')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, r2 in zip(bars, r2_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{r2:.3f}', ha='center', va='bottom' if height > 0 else 'top')

    # 2. Training curves - Baseline
    ax2 = plt.subplot(1, 4, 2)
    ax2.plot(baseline_hist.history['loss'], label='Train', alpha=0.7)
    ax2.plot(baseline_hist.history['val_loss'], label='Val', alpha=0.7)
    ax2.set_title('Baseline Training')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Training curves - Transformer
    ax3 = plt.subplot(1, 4, 3)
    ax3.plot(trans_hist.history['loss'], label='Train', alpha=0.7)
    ax3.plot(trans_hist.history['val_loss'], label='Val', alpha=0.7)
    ax3.set_title('Transformer Training')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Best model predictions
    best = 'Transformer' if transformer_results['test']['r2'] > baseline_results['test']['r2'] else 'Baseline'
    results = transformer_results if best == 'Transformer' else baseline_results

    ax4 = plt.subplot(1, 4, 4)
    pred = results['test']['pred']
    ax4.scatter(y_test, pred, alpha=0.6)
    min_val = min(y_test.min(), pred.min())
    max_val = max(y_test.max(), pred.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax4.set_xlabel('Actual')
    ax4.set_ylabel('Predicted')
    ax4.set_title(f'{best} - Test\nR²={results["test"]["r2"]:.3f}')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('player_transformer_comparison.png', dpi=300, bbox_inches='tight')
    print("\n  Saved: player_transformer_comparison.png")
    plt.show()

def main():
    # 1. Load data
    df = load_individual_player_data()

    # 2. Select features
    features = select_features(df)

    # 3. Create sequences
    X, y, mask, metadata = create_team_year_sequences(df, features, max_players_per_team=10)

    # 4. Normalize features
    print("\nNormalizing features...")
    n_samples, max_players, n_features = X.shape
    X_reshaped = X.reshape(-1, n_features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped).reshape(n_samples, max_players, n_features)

    # 5. Split data
    X_train, X_val, X_test, y_train, y_val, y_test, mask_train, mask_val, mask_test = time_based_split(
        X_scaled, y, mask, metadata
    )

    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val: {len(X_val)}")
    print(f"  Test: {len(X_test)}")

    # 6. Build and train baseline
    baseline = build_baseline_model(max_players, n_features)
    baseline_hist = train_model(baseline, X_train, y_train, mask_train,
                                 X_val, y_val, mask_val, 'Baseline', epochs=200)

    # 7. Build and train transformer
    transformer = build_player_transformer(max_players, n_features)
    trans_hist = train_model(transformer, X_train, y_train, mask_train,
                             X_val, y_val, mask_val, 'Transformer', epochs=200)

    # 8. Evaluate both models
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)

    print("\nBaseline:")
    baseline_results = {
        'train': evaluate_model(baseline, X_train, y_train, mask_train, 'Train'),
        'val': evaluate_model(baseline, X_val, y_val, mask_val, 'Val'),
        'test': evaluate_model(baseline, X_test, y_test, mask_test, 'Test')
    }

    print("\nTransformer:")
    transformer_results = {
        'train': evaluate_model(transformer, X_train, y_train, mask_train, 'Train'),
        'val': evaluate_model(transformer, X_val, y_val, mask_val, 'Val'),
        'test': evaluate_model(transformer, X_test, y_test, mask_test, 'Test')
    }

    # 9. Plot results
    plot_results(baseline_results, transformer_results, y_test, baseline_hist, trans_hist)

    # 10. Final summary
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    print(f"\nBaseline (Flattened Dense):")
    print(f"  Test R²: {baseline_results['test']['r2']:.4f}")
    print(f"  Test MAE: {baseline_results['test']['mae']:.2f}")

    print(f"\nTransformer (Player-level Attention):")
    print(f"  Test R²: {transformer_results['test']['r2']:.4f}")
    print(f"  Test MAE: {transformer_results['test']['mae']:.2f}")

    improvement = transformer_results['test']['r2'] - baseline_results['test']['r2']

    if improvement > 0.05:
        print(f"\n✓ Transformer WINS by {improvement:.4f} R² points!")
        print("  → Attention mechanism captures player interactions")
    elif improvement > 0:
        print(f"\n≈ Transformer marginally better (+{improvement:.4f})")
    else:
        print(f"\n✗ Baseline wins (Transformer worse by {abs(improvement):.4f})")
        print("  → Player interactions don't matter much, or too little data")

    # Save best model
    if improvement > 0:
        transformer.save('player_transformer_model.h5')
        print("\n  Saved: player_transformer_model.h5")
    else:
        baseline.save('player_baseline_model.h5')
        print("\n  Saved: player_baseline_model.h5")

if __name__ == "__main__":
    main()
