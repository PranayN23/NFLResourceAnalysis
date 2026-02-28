import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_lb_weighted_data():
    """Load the weighted averages data with lagged features"""
    print("Loading LB weighted averages data...")

    try:
        df = pd.read_csv('lb_weighted_averages_by_team_pos_year.csv')
        print(f"Loaded data: {df.shape}")
        print(f"Years: {sorted(df['Year'].unique())}")
        print(f"Teams: {df['Team'].nunique()}")
        print(f"Positions: {df['position'].unique()}")
        return df
    except FileNotFoundError:
        print("lb_weighted_averages_by_team_pos_year.csv not found.")
        print("Please run pull_lb_data.py first to generate the data.")
        return None

def prepare_lstm_data_improved(df, target_stat='grades_defense'):
    """
    Improved data preparation with better feature selection
    """
    print(f"\nPreparing LSTM data to predict: weighted_avg_{target_stat}")

    # Filter only LB position
    df_lb = df[df['position'] == 'LB'].copy()
    print(f"LB records: {len(df_lb)}")

    # Select ONLY the most important previous year features (reduce dimensionality)
    important_prev_features = [
        'prev_weighted_avg_grades_defense',
        'prev_weighted_avg_tackles',
        'prev_weighted_avg_sacks',
        'prev_weighted_avg_stops',
        'prev_weighted_avg_grades_coverage_defense',
        'prev_weighted_avg_grades_run_defense',
        'prev_weighted_avg_missed_tackle_rate',
        'prev_total_snap_counts_defense',
        'prev_sum_Cap_Space',
        'prev_sum_adjusted_value'
    ]

    # Current year context
    context_features = ['Win_Percent', 'Net_EPA']

    # Combine features
    feature_columns = [col for col in important_prev_features if col in df_lb.columns]
    feature_columns.extend([col for col in context_features if col in df_lb.columns])

    print(f"Using {len(feature_columns)} carefully selected features (reduced from 35+)")

    # Target column
    target_column = f'weighted_avg_{target_stat}'

    # Remove rows with missing target or features
    required_columns = feature_columns + [target_column, 'Year', 'Team']
    df_clean = df_lb[required_columns].dropna()

    print(f"Records after removing missing values: {len(df_clean)}")

    if len(df_clean) == 0:
        print("ERROR: No valid records found!")
        return None, None, None, None

    # Prepare features and target
    X = df_clean[feature_columns].values
    y = df_clean[target_column].values
    years = df_clean['Year'].values
    teams = df_clean['Team'].values

    # Reshape for LSTM: (samples, timesteps, features)
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    print(f"\nFinal data shape:")
    print(f"  X (features): {X.shape}")
    print(f"  y (target): {y.shape}")

    return X, y, feature_columns, years

def split_data_by_year(X, y, years, test_year_cutoff=2022, val_year_cutoff=2020):
    """
    Time-based split to prevent data leakage
    Train: Years < val_year_cutoff
    Val: val_year_cutoff <= Years < test_year_cutoff
    Test: Years >= test_year_cutoff
    """
    print(f"\nTime-based data split:")
    print(f"  Train: Years < {val_year_cutoff}")
    print(f"  Validation: {val_year_cutoff} <= Years < {test_year_cutoff}")
    print(f"  Test: Years >= {test_year_cutoff}")

    # Create masks
    train_mask = years < val_year_cutoff
    val_mask = (years >= val_year_cutoff) & (years < test_year_cutoff)
    test_mask = years >= test_year_cutoff

    # Split data
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"\nSplit results:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Validation set: {X_val.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test

def normalize_data(X_train, X_val, X_test):
    """
    Normalize features using StandardScaler
    Fit on training data only to prevent data leakage
    """
    print("\nNormalizing features...")

    # Reshape for scaling
    n_samples_train, n_timesteps, n_features = X_train.shape
    n_samples_val = X_val.shape[0]
    n_samples_test = X_test.shape[0]

    X_train_reshaped = X_train.reshape(-1, n_features)
    X_val_reshaped = X_val.reshape(-1, n_features)
    X_test_reshaped = X_test.reshape(-1, n_features)

    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_val_scaled = scaler.transform(X_val_reshaped)
    X_test_scaled = scaler.transform(X_test_reshaped)

    # Reshape back to LSTM format
    X_train_scaled = X_train_scaled.reshape(n_samples_train, n_timesteps, n_features)
    X_val_scaled = X_val_scaled.reshape(n_samples_val, n_timesteps, n_features)
    X_test_scaled = X_test_scaled.reshape(n_samples_test, n_timesteps, n_features)

    print("  Normalization complete")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def build_simpler_lstm_model(input_shape, learning_rate=0.0005):
    """
    Simpler LSTM model with stronger regularization to prevent overfitting
    """
    print("\nBuilding SIMPLIFIED LSTM model with strong regularization...")

    # L2 regularization
    l2_reg = tf.keras.regularizers.l2(0.01)

    model = tf.keras.Sequential([
        # Single LSTM layer (reduced complexity)
        tf.keras.layers.LSTM(
            64,
            return_sequences=False,
            input_shape=input_shape,
            kernel_regularizer=l2_reg,
            recurrent_regularizer=l2_reg
        ),
        tf.keras.layers.Dropout(0.4),  # Increased dropout
        tf.keras.layers.BatchNormalization(),

        # Smaller dense layers
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2_reg),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=l2_reg),
        tf.keras.layers.Dropout(0.3),

        # Output layer
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )

    print("  Model architecture (SIMPLIFIED):")
    model.summary()

    return model

def build_simple_dense_baseline(input_shape, learning_rate=0.001):
    """
    Simple dense baseline model (no LSTM) for comparison
    """
    print("\nBuilding SIMPLE DENSE BASELINE model...")

    # Flatten input
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )

    print("  Baseline model architecture:")
    model.summary()

    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=150, batch_size=16):
    """Train the model with conservative settings"""
    print(f"\nTraining model for up to {epochs} epochs...")

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,  # Reduced patience
        restore_best_weights=True,
        verbose=1
    )

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        verbose=1
    )

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )

    return history

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Evaluate model performance on all sets"""
    print("\nEvaluating model...")

    # Predictions
    train_pred = model.predict(X_train, verbose=0).flatten()
    val_pred = model.predict(X_val, verbose=0).flatten()
    test_pred = model.predict(X_test, verbose=0).flatten()

    # Calculate metrics
    results = {}

    for name, y_true, y_pred in [
        ('Train', y_train, train_pred),
        ('Validation', y_val, val_pred),
        ('Test', y_test, test_pred)
    ]:
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        results[name] = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'predictions': y_pred,
            'actual': y_true
        }

        print(f"\n{name} Set Metrics:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")

    # Check for overfitting
    train_val_gap = results['Train']['r2'] - results['Validation']['r2']
    print(f"\n⚠️  Overfitting Check:")
    print(f"  Train-Val R² Gap: {train_val_gap:.4f}")
    if train_val_gap > 0.15:
        print(f"  ⚠️  WARNING: Significant overfitting detected!")
    else:
        print(f"  ✓ Overfitting is under control")

    return results

def plot_training_history(history, model_name):
    """Plot training history"""
    print("\nPlotting training history...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss plot
    ax1.plot(history.history['loss'], label='Training Loss', alpha=0.8)
    ax1.plot(history.history['val_loss'], label='Validation Loss', alpha=0.8)
    ax1.set_title(f'{model_name} - Loss Over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # MAE plot
    ax2.plot(history.history['mae'], label='Training MAE', alpha=0.8)
    ax2.plot(history.history['val_mae'], label='Validation MAE', alpha=0.8)
    ax2.set_title(f'{model_name} - MAE Over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'lb_lstm_{model_name.lower().replace(" ", "_")}_training_history.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.show()

def plot_predictions(results, target_stat='grades_defense', model_name='Improved'):
    """Plot predictions vs actual for all sets"""
    print("\nPlotting predictions vs actual...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (name, ax) in enumerate(zip(['Train', 'Validation', 'Test'], axes)):
        y_true = results[name]['actual']
        y_pred = results[name]['predictions']

        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, s=30)

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

        # Labels and title
        ax.set_xlabel(f'Actual {target_stat}')
        ax.set_ylabel(f'Predicted {target_stat}')
        ax.set_title(f'{name} Set - {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add metrics text
        r2 = results[name]['r2']
        rmse = results[name]['rmse']
        mae = results[name]['mae']
        textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    filename = f'lb_lstm_{model_name.lower().replace(" ", "_")}_predictions.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.show()

def main():
    """Main function to run improved LB LSTM training"""
    print("="*80)
    print("LB WEIGHTED AVERAGES - IMPROVED LSTM MODEL")
    print("="*80)

    # 1. Load data
    df = load_lb_weighted_data()
    if df is None:
        return

    # 2. Prepare data with reduced features
    target_stat = 'grades_defense'
    X, y, feature_names, years = prepare_lstm_data_improved(df, target_stat=target_stat)

    if X is None:
        return

    # 3. Time-based split (prevents data leakage)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_by_year(
        X, y, years, test_year_cutoff=2022, val_year_cutoff=2020
    )

    # 4. Normalize data
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = normalize_data(
        X_train, X_val, X_test
    )

    # 5. Train BOTH models for comparison
    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])

    print("\n" + "="*80)
    print("TRAINING MODEL 1: SIMPLE DENSE BASELINE")
    print("="*80)
    baseline_model = build_simple_dense_baseline(input_shape, learning_rate=0.001)
    baseline_history = train_model(
        baseline_model, X_train_scaled, y_train, X_val_scaled, y_val,
        epochs=150, batch_size=16
    )
    baseline_results = evaluate_model(
        baseline_model, X_train_scaled, y_train, X_val_scaled, y_val,
        X_test_scaled, y_test
    )

    print("\n" + "="*80)
    print("TRAINING MODEL 2: SIMPLIFIED LSTM")
    print("="*80)
    lstm_model = build_simpler_lstm_model(input_shape, learning_rate=0.0005)
    lstm_history = train_model(
        lstm_model, X_train_scaled, y_train, X_val_scaled, y_val,
        epochs=150, batch_size=16
    )
    lstm_results = evaluate_model(
        lstm_model, X_train_scaled, y_train, X_val_scaled, y_val,
        X_test_scaled, y_test
    )

    # 6. Plot results for both models
    plot_training_history(baseline_history, 'Baseline Dense')
    plot_predictions(baseline_results, target_stat=target_stat, model_name='Baseline')

    plot_training_history(lstm_history, 'Simplified LSTM')
    plot_predictions(lstm_results, target_stat=target_stat, model_name='LSTM')

    # 7. Compare models
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(f"\nBaseline Dense Model:")
    print(f"  Test R²: {baseline_results['Test']['r2']:.4f}")
    print(f"  Test RMSE: {baseline_results['Test']['rmse']:.4f}")
    print(f"  Test MAE: {baseline_results['Test']['mae']:.4f}")

    print(f"\nSimplified LSTM Model:")
    print(f"  Test R²: {lstm_results['Test']['r2']:.4f}")
    print(f"  Test RMSE: {lstm_results['Test']['rmse']:.4f}")
    print(f"  Test MAE: {lstm_results['Test']['mae']:.4f}")

    # Choose best model
    if baseline_results['Test']['r2'] > lstm_results['Test']['r2']:
        print(f"\n✓ Best Model: Baseline Dense (simpler is better!)")
        best_model = baseline_model
        best_name = 'baseline_dense'
    else:
        print(f"\n✓ Best Model: Simplified LSTM")
        best_model = lstm_model
        best_name = 'simplified_lstm'

    # 8. Save best model
    model_filename = f'lb_{best_name}_model_{target_stat}.h5'
    best_model.save(model_filename)
    print(f"\nBest model saved: {model_filename}")

    print("\n" + "="*80)
    print("KEY IMPROVEMENTS IN THIS VERSION:")
    print("="*80)
    print("✓ Reduced features from 35+ to 12 (less overfitting)")
    print("✓ Time-based split (no data leakage)")
    print("✓ Stronger L2 regularization")
    print("✓ Higher dropout rates (0.3-0.4)")
    print("✓ Simpler architecture (fewer layers)")
    print("✓ Baseline model for comparison")
    print("✓ Smaller batch size (16 vs 32)")

if __name__ == "__main__":
    main()
