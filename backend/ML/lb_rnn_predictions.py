import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_lb_data():
    """Load and prepare LB data from multiple sources"""
    print("Loading LB data...")
    
    # Load the main LB data
    lb_data = pd.read_csv('LB.csv')
    print(f"Loaded LB data: {lb_data.shape}")
    
    # Load PFF data if available
    try:
        lb_pff = pd.read_csv('LBPFF.csv')
        print(f"Loaded LB PFF data: {lb_pff.shape}")
    except FileNotFoundError:
        print("LBPFF.csv not found, using only LB.csv")
        lb_pff = None
    
    # Load secondary defense data if available
    try:
        lb_secondary = pd.read_csv('LB_Secondary_Defense.csv')
        print(f"Loaded LB Secondary Defense data: {lb_secondary.shape}")
    except FileNotFoundError:
        print("LB_Secondary_Defense.csv not found")
        lb_secondary = None
    
    # Merge datasets if available
    if lb_pff is not None:
        # Merge with PFF data
        lb_data = lb_data.merge(lb_pff, on=['player_id', 'Year', 'Team'], how='left', suffixes=('', '_pff'))
        print(f"After PFF merge: {lb_data.shape}")
    
    if lb_secondary is not None:
        # Merge with secondary defense data
        lb_data = lb_data.merge(lb_secondary, on=['Team', 'Year'], how='left', suffixes=('', '_sec'))
        print(f"After secondary defense merge: {lb_data.shape}")
    
    return lb_data

def create_sequences(data, feature_columns, target_column, sequence_length=3):
    """Create sequences for RNN training"""
    sequences = []
    targets = []
    
    # Sort by player and year
    data = data.sort_values(['player_id', 'Year'])
    
    for player_id in data['player_id'].unique():
        player_data = data[data['player_id'] == player_id].sort_values('Year')
        
        if len(player_data) < sequence_length + 1:
            continue
            
        for i in range(len(player_data) - sequence_length):
            # Get sequence of features
            sequence = player_data[feature_columns].iloc[i:i+sequence_length].values
            # Get target (next year's performance)
            target = player_data[target_column].iloc[i+sequence_length]
            
            sequences.append(sequence)
            targets.append(target)
    
    return np.array(sequences), np.array(targets)

def build_rnn_model(input_shape, learning_rate=0.001, rnn_units=50, dense_units=[50, 25]):
    """Build RNN model with LSTM layers"""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(rnn_units//2, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
    ])
    
    # Add dense layers
    for units in dense_units:
        model.add(tf.keras.layers.Dense(units, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
    
    # Output layer
    model.add(tf.keras.layers.Dense(1))
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def build_gru_model(input_shape, learning_rate=0.001, gru_units=50, dense_units=[50, 25]):
    """Build RNN model with GRU layers"""
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(gru_units, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GRU(gru_units//2, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
    ])
    
    # Add dense layers
    for units in dense_units:
        model.add(tf.keras.layers.Dense(units, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
    
    # Output layer
    model.add(tf.keras.layers.Dense(1))
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, 
                           model_name, epochs=100, batch_size=32):
    """Train and evaluate the model"""
    print(f"\nTraining {model_name}...")
    
    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=15, 
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0
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
    
    print(f"{model_name} Results:")
    print(f"  Training R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}")
    print(f"  Validation R²: {val_r2:.4f}, RMSE: {val_rmse:.4f}")
    print(f"  Test R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}")
    
    return {
        'model': model,
        'history': history,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'test_rmse': test_rmse,
        'predictions': {
            'train': train_pred,
            'val': val_pred,
            'test': test_pred
        }
    }

def plot_training_history(history, model_name):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title(f'{model_name} - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot MAE
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title(f'{model_name} - MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_predictions_vs_actual(y_true, y_pred, title, model_name):
    """Plot predictions vs actual values"""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{model_name} - {title}')
    
    # Add R² score to plot
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_{title.lower().replace(" ", "_")}_predictions.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run LB RNN predictions"""
    print("=== LB RNN Predictions ===")
    
    # Load and prepare data
    lb_data = load_and_prepare_lb_data()
    
    # Display basic info about the data
    print(f"\nData shape: {lb_data.shape}")
    print(f"Years available: {sorted(lb_data['Year'].unique())}")
    print(f"Number of unique players: {lb_data['player_id'].nunique()}")
    
    # Define feature columns (adjust based on your data)
    feature_columns = [
        'Value_cap_space', 'Value_draft_data', 'Previous_AV', 'Previous_PFF',
        'age', 'snap_counts_defense', 'tackles', 'assists', 'sacks', 
        'interceptions', 'pass_break_ups', 'tackles_for_loss', 'stops',
        'grades_defense', 'grades_tackle', 'grades_run_defense', 'grades_coverage_defense'
    ]
    
    # Filter out columns that don't exist
    available_features = [col for col in feature_columns if col in lb_data.columns]
    print(f"Using features: {available_features}")
    
    # Target column
    target_column = 'Current_PFF'  # or 'Current_AV' or other metric
    
    # Remove rows with missing target values
    lb_data = lb_data.dropna(subset=[target_column])
    print(f"Data shape after removing missing targets: {lb_data.shape}")
    
    # Create sequences for RNN
    sequence_length = 3
    X, y = create_sequences(lb_data, available_features, target_column, sequence_length)
    
    print(f"Created sequences: {X.shape}, targets: {y.shape}")
    
    if len(X) == 0:
        print("No sequences created. Check your data and feature columns.")
        return
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
    
    # Normalize features
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    
    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val_reshaped).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)
    
    # Model configurations to test
    model_configs = [
        {'name': 'LSTM Model', 'type': 'lstm', 'rnn_units': 50, 'dense_units': [50, 25]},
        {'name': 'GRU Model', 'type': 'gru', 'rnn_units': 50, 'dense_units': [50, 25]},
        {'name': 'Deep LSTM Model', 'type': 'lstm', 'rnn_units': 100, 'dense_units': [100, 50, 25]},
        {'name': 'Deep GRU Model', 'type': 'gru', 'rnn_units': 100, 'dense_units': [100, 50, 25]}
    ]
    
    learning_rates = [0.001, 0.01]
    results = {}
    
    # Train and evaluate models
    for config in model_configs:
        for lr in learning_rates:
            model_name = f"{config['name']} (LR={lr})"
            
            if config['type'] == 'lstm':
                model = build_rnn_model(
                    input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
                    learning_rate=lr,
                    rnn_units=config['rnn_units'],
                    dense_units=config['dense_units']
                )
            else:  # GRU
                model = build_gru_model(
                    input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
                    learning_rate=lr,
                    gru_units=config['rnn_units'],
                    dense_units=config['dense_units']
                )
            
            result = train_and_evaluate_model(
                model, X_train_scaled, y_train, X_val_scaled, y_val, 
                X_test_scaled, y_test, model_name
            )
            
            results[model_name] = result
            
            # Plot training history
            plot_training_history(result['history'], model_name)
            
            # Plot predictions vs actual
            plot_predictions_vs_actual(y_test, result['predictions']['test'], 'Test Set', model_name)
    
    # Print summary of all results
    print("\n" + "="*60)
    print("SUMMARY OF ALL MODELS")
    print("="*60)
    
    for model_name, result in results.items():
        print(f"{model_name}:")
        print(f"  Test R²: {result['test_r2']:.4f}, Test RMSE: {result['test_rmse']:.4f}")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
    best_result = results[best_model_name]
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Best Test R²: {best_result['test_r2']:.4f}")
    print(f"Best Test RMSE: {best_result['test_rmse']:.4f}")
    
    # Save the best model
    best_model_name_clean = best_model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    best_result['model'].save(f'best_lb_rnn_model_{best_model_name_clean}.h5')
    print(f"Best model saved as: best_lb_rnn_model_{best_model_name_clean}.h5")

if __name__ == "__main__":
    main()

