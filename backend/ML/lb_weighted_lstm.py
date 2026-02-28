import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
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

def prepare_lstm_data(df, target_stat='grades_defense', sequence_length=1, top_n_features=15):
    """
    Prepare data for LSTM training with feature selection

    Args:
        df: DataFrame with weighted averages and lagged features
        target_stat: The stat to predict (default: grades_defense)
        sequence_length: Number of previous years to use (default: 1)
        top_n_features: Number of top features to keep (default: 15)

    Returns:
        X, y, feature_names, scaler, df_clean
    """
    print(f"\nPreparing LSTM data to predict: weighted_avg_{target_stat}")
    print(f"Using sequence length: {sequence_length} year(s)")

    # Filter only LB position
    df_lb = df[df['position'] == 'LB'].copy()
    print(f"LB records: {len(df_lb)}")

    # Define feature columns (ONLY previous year's weighted averages - NO current year context)
    prev_feature_columns = [col for col in df_lb.columns if col.startswith('prev_weighted_avg_')]

    # Add previous year summary stats
    prev_summary_features = [
        'prev_total_snap_counts_defense',
        'prev_total_players',
        'prev_sum_Cap_Space',
        'prev_sum_adjusted_value'
    ]
    prev_feature_columns.extend([col for col in prev_summary_features if col in df_lb.columns])

    # REMOVED: Current year context features to prevent data leakage
    print(f"Using {len(prev_feature_columns)} features (previous year only)")

    # Target column
    target_column = f'weighted_avg_{target_stat}'

    # Keep Year column for time-based splitting
    required_columns = prev_feature_columns + [target_column, 'Year']
    df_clean = df_lb[required_columns].dropna()

    print(f"Records after removing missing values: {len(df_clean)}")

    if len(df_clean) == 0:
        print("ERROR: No valid records found!")
        return None, None, None, None

    # Prepare features and target
    X = df_clean[prev_feature_columns].values
    y = df_clean[target_column].values

    # Feature Selection using Random Forest
    print(f"\nPerforming feature selection...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    feature_importance = pd.DataFrame({
        'feature': prev_feature_columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop {top_n_features} Features:")
    print(feature_importance.head(top_n_features))

    # Select top features
    top_features = feature_importance.head(top_n_features)['feature'].tolist()
    X_selected = df_clean[top_features].values

    print(f"\nReduced features from {len(prev_feature_columns)} to {len(top_features)}")

    # Reshape for LSTM: (samples, timesteps, features)
    X_selected = X_selected.reshape((X_selected.shape[0], 1, X_selected.shape[1]))

    print(f"\nFinal data shape:")
    print(f"  X (features): {X_selected.shape}")
    print(f"  y (target): {y.shape}")

    return X_selected, y, top_features, df_clean

def time_based_split(df_clean, X, y, test_size=0.2, val_size=0.2):
    """
    Split data by year (chronological) instead of random split
    Most recent year(s) for test, next for validation, rest for training
    """
    print(f"\nPerforming time-based split...")
    
    years = sorted(df_clean['Year'].unique())
    n_years = len(years)
    
    n_test_years = max(1, int(n_years * test_size))
    n_val_years = max(1, int(n_years * val_size))
    
    test_years = years[-n_test_years:]
    val_years = years[-(n_test_years + n_val_years):-n_test_years]
    train_years = years[:-(n_test_years + n_val_years)]
    
    print(f"  Training years: {train_years}")
    print(f"  Validation years: {val_years}")
    print(f"  Test years: {test_years}")
    
    # Create masks
    train_mask = df_clean['Year'].isin(train_years).values
    val_mask = df_clean['Year'].isin(val_years).values
    test_mask = df_clean['Year'].isin(test_years).values
    
    X_train = X[train_mask]
    X_val = X[val_mask]
    X_test = X[test_mask]
    
    y_train = y[train_mask]
    y_val = y[val_mask]
    y_test = y[test_mask]
    
    print(f"\nSplit results:")
    print(f"  Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"  Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"  Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def split_data_random(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split data randomly (original method kept as option)
    """
    print(f"\nPerforming random split:")
    print(f"  Test size: {test_size*100:.0f}%")
    print(f"  Validation size: {val_size*100:.0f}% of remaining")

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Second split: separate validation from training
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state
    )

    print(f"\nSplit results:")
    print(f"  Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"  Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"  Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

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

def baseline_comparison(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Run baseline Ridge regression to check if problem is solvable
    """
    print("\n" + "="*80)
    print("BASELINE COMPARISON - Ridge Regression")
    print("="*80)
    
    # Flatten for sklearn models
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Try different alpha values
    best_r2 = -np.inf
    best_alpha = None
    
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        ridge = Ridge(alpha=alpha, random_state=42)
        ridge.fit(X_train_flat, y_train)
        
        val_r2 = ridge.score(X_val_flat, y_val)
        
        if val_r2 > best_r2:
            best_r2 = val_r2
            best_alpha = alpha
    
    # Train final model with best alpha
    print(f"Best alpha: {best_alpha} (Validation R²: {best_r2:.4f})")
    
    ridge = Ridge(alpha=best_alpha, random_state=42)
    ridge.fit(X_train_flat, y_train)
    
    # Evaluate on all sets
    train_r2 = ridge.score(X_train_flat, y_train)
    val_r2 = ridge.score(X_val_flat, y_val)
    test_r2 = ridge.score(X_test_flat, y_test)
    
    train_pred = ridge.predict(X_train_flat)
    val_pred = ridge.predict(X_val_flat)
    test_pred = ridge.predict(X_test_flat)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    print(f"\nBaseline Ridge Regression Results:")
    print(f"  Train R²: {train_r2:.4f}, MAE: {train_mae:.4f}")
    print(f"  Val R²: {val_r2:.4f}, MAE: {val_mae:.4f}")
    print(f"  Test R²: {test_r2:.4f}, MAE: {test_mae:.4f}")
    
    if test_r2 < 0:
        print("\n⚠️  WARNING: Even baseline Ridge has negative test R²!")
        print("   This suggests the features don't predict the target well.")
        print("   Consider:")
        print("   - Different features")
        print("   - Different target variable")
        print("   - More data")
        print("   - This might just be a hard prediction problem")
    
    return ridge, test_r2

def build_lstm_model(input_shape, learning_rate=0.001, l2_reg=0.01):
    """
    Build simplified LSTM model with regularization
    Much simpler architecture to prevent overfitting
    """
    print("\nBuilding simplified LSTM model...")
    
    from tensorflow.keras import regularizers

    model = tf.keras.Sequential([
        # Single LSTM layer - much simpler
        tf.keras.layers.LSTM(
            32, 
            return_sequences=False, 
            input_shape=input_shape,
            kernel_regularizer=regularizers.l2(l2_reg),
            recurrent_regularizer=regularizers.l2(l2_reg)
        ),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),

        # Single dense layer
        tf.keras.layers.Dense(
            16, 
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg)
        ),
        tf.keras.layers.Dropout(0.2),

        # Output layer
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )

    print("  Simplified model architecture:")
    model.summary()

    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=200, batch_size=32):
    """Train the LSTM model"""
    print(f"\nTraining model for up to {epochs} epochs...")

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=30,  # Increased patience
        restore_best_weights=True,
        verbose=1
    )

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,  # Increased patience
        min_lr=1e-7,
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

    return results

def plot_training_history(history):
    """Plot training history"""
    print("\nPlotting training history...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss plot
    ax1.plot(history.history['loss'], label='Training Loss', alpha=0.8)
    ax1.plot(history.history['val_loss'], label='Validation Loss', alpha=0.8)
    ax1.set_title('Model Loss Over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # MAE plot
    ax2.plot(history.history['mae'], label='Training MAE', alpha=0.8)
    ax2.plot(history.history['val_mae'], label='Validation MAE', alpha=0.8)
    ax2.set_title('Model MAE Over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lb_lstm_training_history_improved.png', dpi=300, bbox_inches='tight')
    print("  Saved: lb_lstm_training_history_improved.png")
    plt.show()

def plot_predictions(results, target_stat='grades_defense'):
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
        ax.set_title(f'{name} Set')
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
    plt.savefig('lb_lstm_predictions_improved.png', dpi=300, bbox_inches='tight')
    print("  Saved: lb_lstm_predictions_improved.png")
    plt.show()

def plot_residuals(results, target_stat='grades_defense'):
    """Plot residuals for all sets"""
    print("\nPlotting residuals...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (name, ax) in enumerate(zip(['Train', 'Validation', 'Test'], axes)):
        y_true = results[name]['actual']
        y_pred = results[name]['predictions']
        residuals = y_true - y_pred

        # Residual plot
        ax.scatter(y_pred, residuals, alpha=0.6, s=30)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)

        # Labels and title
        ax.set_xlabel(f'Predicted {target_stat}')
        ax.set_ylabel('Residuals')
        ax.set_title(f'{name} Set - Residuals')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lb_lstm_residuals_improved.png', dpi=300, bbox_inches='tight')
    print("  Saved: lb_lstm_residuals_improved.png")
    plt.show()

def main():
    """Main function to run improved LB LSTM training"""
    print("="*80)
    print("IMPROVED LB WEIGHTED AVERAGES LSTM PREDICTION MODEL")
    print("="*80)
    print("\nImprovements:")
    print("  ✓ Removed current-year context features (data leakage fix)")
    print("  ✓ Feature selection with Random Forest")
    print("  ✓ Simplified LSTM architecture")
    print("  ✓ Added L2 regularization")
    print("  ✓ Time-based data splitting")
    print("  ✓ Baseline Ridge regression comparison")
    print("="*80)

    # 1. Load data
    df = load_lb_weighted_data()
    if df is None:
        return

    # 2. Prepare data for LSTM with feature selection
    target_stat = 'grades_defense'
    X, y, feature_names, df_clean = prepare_lstm_data(
        df, 
        target_stat=target_stat,
        top_n_features=15  # Use top 15 features
    )

    if X is None:
        return

    # 3. Split data (time-based by default, can switch to random if needed)
    use_time_split = True  # Set to False for random split
    
    if use_time_split:
        X_train, X_val, X_test, y_train, y_val, y_test = time_based_split(
            df_clean, X, y, test_size=0.2, val_size=0.2
        )
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = split_data_random(
            X, y, test_size=0.2, val_size=0.2
        )

    # 4. Normalize data
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = normalize_data(
        X_train, X_val, X_test
    )

    # 5. Run baseline comparison
    baseline_model, baseline_test_r2 = baseline_comparison(
        X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test
    )

    # 6. Build simplified LSTM model
    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
    model = build_lstm_model(input_shape, learning_rate=0.001, l2_reg=0.01)

    # 7. Train model
    history = train_model(
        model, X_train_scaled, y_train, X_val_scaled, y_val,
        epochs=200, batch_size=32
    )

    # 8. Evaluate model
    results = evaluate_model(
        model, X_train_scaled, y_train, X_val_scaled, y_val,
        X_test_scaled, y_test
    )

    # 9. Plot results
    plot_training_history(history)
    plot_predictions(results, target_stat=target_stat)
    plot_residuals(results, target_stat=target_stat)

    # 10. Save model
    model_filename = f'lb_lstm_model_{target_stat}_improved.h5'
    model.save(model_filename)
    print(f"\nModel saved: {model_filename}")

    # 11. Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Target: weighted_avg_{target_stat}")
    print(f"Features used: {len(feature_names)}")
    
    print(f"\nBaseline Ridge Regression Test R²: {baseline_test_r2:.4f}")
    print(f"\nLSTM Final Test Set Performance:")
    print(f"  R² Score: {results['Test']['r2']:.4f}")
    print(f"  RMSE: {results['Test']['rmse']:.4f}")
    print(f"  MAE: {results['Test']['mae']:.4f}")
    
    # Compare to baseline
    improvement = results['Test']['r2'] - baseline_test_r2
    if improvement > 0:
        print(f"\n✓ LSTM improved over baseline by {improvement:.4f} R² points")
    else:
        print(f"\n⚠️  LSTM did not beat baseline (worse by {abs(improvement):.4f} R² points)")
        print("   Consider using the simpler Ridge model instead")
    
    print("\nFiles saved:")
    print(f"  - {model_filename}")
    print(f"  - lb_lstm_training_history_improved.png")
    print(f"  - lb_lstm_predictions_improved.png")
    print(f"  - lb_lstm_residuals_improved.png")
    
    print("\nSelected Features:")
    for i, feat in enumerate(feature_names, 1):
        print(f"  {i}. {feat}")

if __name__ == "__main__":
    main()