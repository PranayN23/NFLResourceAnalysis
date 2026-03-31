import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

def load_lb_data():
    """Load LB data and prepare for RNN"""
    print("Loading LB data...")
    
    # Load the main LB data
    lb_data = pd.read_csv('LB.csv')
    print(f"Loaded LB data: {lb_data.shape}")
    
    # Filter for recent years and remove missing values
    lb_data = lb_data[lb_data['Year'] >= 2015]  # Focus on recent data
    lb_data = lb_data.dropna(subset=['Current_PFF', 'Previous_PFF', 'Previous_AV'])
    
    print(f"Data after filtering: {lb_data.shape}")
    print(f"Years: {sorted(lb_data['Year'].unique())}")
    print(f"Unique players: {lb_data['player_id'].nunique()}")
    
    return lb_data

def create_sequences(data, sequence_length=3):
    """Create sequences for RNN training"""
    sequences = []
    targets = []
    
    # Sort by player and year
    data = data.sort_values(['player_id', 'Year'])
    
    # Define features to use
    feature_columns = [
        'Previous_PFF', 'Previous_AV', 'Value_cap_space', 'Value_draft_data',
        'age', 'snap_counts_defense', 'tackles', 'assists', 'sacks'
    ]
    
    # Filter available features
    available_features = [col for col in feature_columns if col in data.columns]
    print(f"Using features: {available_features}")
    
    for player_id in data['player_id'].unique():
        player_data = data[data['player_id'] == player_id].sort_values('Year')
        
        if len(player_data) < sequence_length + 1:
            continue
            
        for i in range(len(player_data) - sequence_length):
            # Get sequence of features
            sequence = player_data[available_features].iloc[i:i+sequence_length].values
            # Get target (next year's PFF grade)
            target = player_data['Current_PFF'].iloc[i+sequence_length]
            
            sequences.append(sequence)
            targets.append(target)
    
    return np.array(sequences), np.array(targets), available_features

def build_lstm_model(input_shape, learning_rate=0.001):
    """Build LSTM model"""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def build_gru_model(input_shape, learning_rate=0.001):
    """Build GRU model"""
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(64, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.GRU(32, return_sequences=False),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, model_name, epochs=100):
    """Train the model"""
    print(f"\nTraining {model_name}...")
    
    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=20, 
        restore_best_weights=True
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate the model"""
    predictions = model.predict(X_test, verbose=0).flatten()
    
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = np.mean(np.abs(y_test - predictions))
    
    print(f"\n{model_name} Results:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    return r2, rmse, mae, predictions

def plot_results(y_true, y_pred, model_name):
    """Plot predictions vs actual"""
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual PFF Grade')
    plt.ylabel('Predicted PFF Grade')
    plt.title(f'{model_name} - Predictions vs Actual')
    
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.subplot(1, 2, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted PFF Grade')
    plt.ylabel('Residuals')
    plt.title(f'{model_name} - Residuals')
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function"""
    print("=== LB RNN Predictions (Simple Version) ===")
    
    # Load data
    lb_data = load_lb_data()
    
    # Create sequences
    X, y, feature_names = create_sequences(lb_data, sequence_length=3)
    
    if len(X) == 0:
        print("No sequences created. Check your data.")
        return
    
    print(f"Created {len(X)} sequences with {len(feature_names)} features each")
    
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
        'LSTM': build_lstm_model((X_train_scaled.shape[1], X_train_scaled.shape[2])),
        'GRU': build_gru_model((X_train_scaled.shape[1], X_train_scaled.shape[2]))
    }
    
    results = {}
    
    for model_name, model in models.items():
        # Train
        history = train_model(model, X_train_scaled, y_train, X_val_scaled, y_val, model_name)
        
        # Evaluate
        r2, rmse, mae, predictions = evaluate_model(model, X_test_scaled, y_test, model_name)
        
        results[model_name] = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'predictions': predictions
        }
        
        # Plot results
        plot_results(y_test, predictions, model_name)
    
    # Print summary
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    
    for model_name, result in results.items():
        print(f"{model_name}: R² = {result['r2']:.4f}, RMSE = {result['rmse']:.4f}")
    
    # Find best model
    best_model = max(results.keys(), key=lambda x: results[x]['r2'])
    print(f"\nBest Model: {best_model} (R² = {results[best_model]['r2']:.4f})")
    
    # Save best model
    best_model_obj = models[best_model]
    best_model_obj.save(f'best_lb_{best_model.lower()}_model.h5')
    print(f"Best model saved as: best_lb_{best_model.lower()}_model.h5")

if __name__ == "__main__":
    main()

