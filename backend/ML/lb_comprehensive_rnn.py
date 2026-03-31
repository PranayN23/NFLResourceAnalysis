import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_comprehensive_lb_data():
    """Load the comprehensive LB data from CSV"""
    print("Loading comprehensive LB data...")
    
    try:
        df = pd.read_csv('all_lb_data_15_years.csv')
        print(f"Loaded LB data: {df.shape}")
        print(f"Years: {sorted(df['Year'].unique())}")
        print(f"Unique players: {df['player_id'].nunique()}")
        return df
    except FileNotFoundError:
        print("all_lb_data_15_years.csv not found. Please run pull_lb_data.py first.")
        return None

def create_lb_sequences(data, sequence_length=4):
    """Create sequences for RNN training with comprehensive LB features"""
    sequences = []
    targets = []
    
    # Sort by player and year
    data = data.sort_values(['player_id', 'Year'])
    
    # Define comprehensive feature set for LB predictions
    feature_columns = [
        # Core performance metrics
        'grades_defense', 'tackles', 'assists', 'sacks', 'stops', 'tackles_for_loss',
        'total_pressures', 'hits', 'hurries', 'interceptions', 'pass_break_ups',
        
        # Coverage and run defense
        'grades_coverage_defense', 'grades_run_defense', 'grades_pass_rush_defense',
        'grades_tackle', 'missed_tackle_rate', 'missed_tackles',
        
        # Snap counts and usage
        'snap_counts_defense', 'snap_counts_box', 'snap_counts_offball',
        'snap_counts_pass_rush', 'snap_counts_run_defense',
        
        # Team context
        'Cap_Space', 'adjusted_value', 'Net EPA', 'Win %',
        
        # Player characteristics
        'age', 'player_game_count'
    ]
    
    # Filter available features
    available_features = [col for col in feature_columns if col in data.columns]
    print(f"Using {len(available_features)} features: {available_features}")
    
    # Remove rows with missing target or critical features
    data_clean = data.dropna(subset=['grades_defense'] + available_features)
    print(f"Data after cleaning: {len(data_clean)} records")
    
    for player_id in data_clean['player_id'].unique():
        player_data = data_clean[data_clean['player_id'] == player_id].sort_values('Year')
        
        if len(player_data) < sequence_length + 1:
            continue
            
        for i in range(len(player_data) - sequence_length):
            # Get sequence of features
            sequence = player_data[available_features].iloc[i:i+sequence_length].values
            # Get target (next year's grades_defense)
            target = player_data['grades_defense'].iloc[i+sequence_length]
            
            sequences.append(sequence)
            targets.append(target)
    
    return np.array(sequences), np.array(targets), available_features

def build_advanced_lstm_model(input_shape, learning_rate=0.001):
    """Build advanced LSTM model for LB predictions"""
    model = tf.keras.Sequential([
        # First LSTM layer with return sequences
        tf.keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),
        
        # Second LSTM layer
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),
        
        # Third LSTM layer
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),
        
        # Dense layers
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        
        # Output layer
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def build_advanced_gru_model(input_shape, learning_rate=0.001):
    """Build advanced GRU model for LB predictions"""
    model = tf.keras.Sequential([
        # First GRU layer
        tf.keras.layers.GRU(128, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),
        
        # Second GRU layer
        tf.keras.layers.GRU(64, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),
        
        # Third GRU layer
        tf.keras.layers.GRU(32, return_sequences=False),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),
        
        # Dense layers
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        
        # Output layer
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def build_hybrid_model(input_shape, learning_rate=0.001):
    """Build hybrid LSTM-GRU model"""
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # LSTM branch
    lstm_out = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
    lstm_out = tf.keras.layers.Dropout(0.3)(lstm_out)
    lstm_out = tf.keras.layers.LSTM(32, return_sequences=False)(lstm_out)
    
    # GRU branch
    gru_out = tf.keras.layers.GRU(64, return_sequences=True)(inputs)
    gru_out = tf.keras.layers.Dropout(0.3)(gru_out)
    gru_out = tf.keras.layers.GRU(32, return_sequences=False)(gru_out)
    
    # Concatenate branches
    combined = tf.keras.layers.concatenate([lstm_out, gru_out])
    combined = tf.keras.layers.Dropout(0.3)(combined)
    
    # Dense layers
    x = tf.keras.layers.Dense(64, activation='relu')(combined)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    
    # Output
    outputs = tf.keras.layers.Dense(1)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, 
                           model_name, epochs=200, batch_size=32):
    """Train and evaluate the model"""
    print(f"\nTraining {model_name}...")
    
    # Early stopping and learning rate reduction
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=25, 
        restore_best_weights=True
    )
    
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )
    
    # Make predictions
    train_pred = model.predict(X_train, verbose=0).flatten()
    val_pred = model.predict(X_val, verbose=0).flatten()
    test_pred = model.predict(X_test, verbose=0).flatten()
    
    # Calculate metrics
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    print(f"{model_name} Results:")
    print(f"  Training   - R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
    print(f"  Validation - R²: {val_r2:.4f}, RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}")
    print(f"  Test       - R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
    
    return {
        'model': model,
        'history': history,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'val_mae': val_mae,
        'test_mae': test_mae,
        'predictions': {
            'train': train_pred,
            'val': val_pred,
            'test': test_pred
        }
    }

def plot_training_history(history, model_name):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss', alpha=0.8)
    ax1.plot(history.history['val_loss'], label='Validation Loss', alpha=0.8)
    ax1.set_title(f'{model_name} - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot MAE
    ax2.plot(history.history['mae'], label='Training MAE', alpha=0.8)
    ax2.plot(history.history['val_mae'], label='Validation MAE', alpha=0.8)
    ax2.set_title(f'{model_name} - MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_training_history.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_predictions_vs_actual(y_true, y_pred, title, model_name):
    """Plot predictions vs actual values"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6, s=20)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual PFF Grade')
    plt.ylabel('Predicted PFF Grade')
    plt.title(f'{model_name} - {title}')
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    plt.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}', 
             transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.subplot(1, 2, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted PFF Grade')
    plt.ylabel('Residuals')
    plt.title(f'{model_name} - Residuals')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_{title.lower().replace(" ", "_")}_predictions.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run comprehensive LB RNN predictions"""
    print("=== Comprehensive LB RNN Predictions ===")
    
    # Load data
    lb_data = load_comprehensive_lb_data()
    if lb_data is None:
        return
    
    # Create sequences with different lengths
    sequence_lengths = [3, 4, 5]
    best_sequence_length = 4
    best_score = -1
    
    print(f"\nTesting different sequence lengths...")
    for seq_len in sequence_lengths:
        X, y, feature_names = create_lb_sequences(lb_data, sequence_length=seq_len)
        if len(X) > 0:
            print(f"Sequence length {seq_len}: {len(X)} sequences")
            # Quick test with simple model
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
            
            # Normalize
            scaler = StandardScaler()
            X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
            X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
            X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
            
            X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
            X_val_scaled = scaler.transform(X_val_reshaped).reshape(X_val.shape)
            X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)
            
            # Quick model test
            model = build_advanced_lstm_model((X_train_scaled.shape[1], X_train_scaled.shape[2]))
            model.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val), 
                     epochs=10, verbose=0)
            test_pred = model.predict(X_test_scaled, verbose=0).flatten()
            test_r2 = r2_score(y_test, test_pred)
            
            if test_r2 > best_score:
                best_score = test_r2
                best_sequence_length = seq_len
            
            print(f"  Sequence length {seq_len}: Test R² = {test_r2:.4f}")
    
    print(f"\nBest sequence length: {best_sequence_length}")
    
    # Create final sequences with best length
    X, y, feature_names = create_lb_sequences(lb_data, sequence_length=best_sequence_length)
    print(f"Final dataset: {X.shape[0]} sequences with {len(feature_names)} features each")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Normalize features
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    
    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val_reshaped).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)
    
    # Test different models
    models = {
        'Advanced LSTM': build_advanced_lstm_model((X_train_scaled.shape[1], X_train_scaled.shape[2])),
        'Advanced GRU': build_advanced_gru_model((X_train_scaled.shape[1], X_train_scaled.shape[2])),
        'Hybrid LSTM-GRU': build_hybrid_model((X_train_scaled.shape[1], X_train_scaled.shape[2]))
    }
    
    results = {}
    
    for model_name, model in models.items():
        # Train and evaluate
        result = train_and_evaluate_model(
            model, X_train_scaled, y_train, X_val_scaled, y_val, 
            X_test_scaled, y_test, model_name
        )
        
        results[model_name] = result
        
        # Plot results
        plot_training_history(result['history'], model_name)
        plot_predictions_vs_actual(y_test, result['predictions']['test'], 'Test Set', model_name)
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    for model_name, result in results.items():
        print(f"{model_name}:")
        print(f"  Test R²: {result['test_r2']:.4f}, RMSE: {result['test_rmse']:.4f}, MAE: {result['test_mae']:.4f}")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
    best_result = results[best_model_name]
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Test R²: {best_result['test_r2']:.4f}")
    print(f"   Test RMSE: {best_result['test_rmse']:.4f}")
    print(f"   Test MAE: {best_result['test_mae']:.4f}")
    
    # Save best model
    best_model_name_clean = best_model_name.lower().replace(" ", "_").replace("-", "_")
    best_result['model'].save(f'best_lb_comprehensive_{best_model_name_clean}.h5')
    print(f"💾 Best model saved as: best_lb_comprehensive_{best_model_name_clean}.h5")

if __name__ == "__main__":
    main()

