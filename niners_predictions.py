import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import tensorflow as tf

# Filter data for the 49ers team
def filter_niners_data(data):
    return data[data['Team'] == '49ers']

# sklearn MLP Regressor
def sklearn_mlp(df):
    niners_data = filter_niners_data(df)
    
    # Features and target
    X = niners_data[['Value_cap_space', 'Value_draft_data', 'Previous_AV', 'Previous_PFF', 'win-loss-pct']]
    y = niners_data['Current_AV']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = mlp.predict(X_test_scaled)
    
    # Evaluate the model
    r2 = r2_score(y_test, y_pred)
    print(f"R2 Score for Sklearn MLP: {r2}")
    
    return mlp, r2

# TensorFlow MLP Regressor
def tensorflow_mlp(df):
    niners_data = filter_niners_data(df)
    
    # Features and target
    X = niners_data[['Value_cap_space', 'Value_draft_data', 'Previous_AV', 'Previous_PFF', 'win-loss-pct']]
    y = niners_data['Current_AV']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build the TensorFlow model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer for regression
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    
    # Train the model
    model.fit(X_train_scaled, y_train, epochs=100, batch_size=10, verbose=0)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate the model
    r2 = r2_score(y_test, y_pred)
    print(f"R2 Score for TensorFlow MLP: {r2}")
    
    return model, r2

# Main method to call both sklearn and tensorflow MLP models
def main():
    # Load the data
    df = pd.read_csv('data.csv')
    
    # Run sklearn MLP
    sklearn_model, sklearn_r2 = sklearn_mlp(df)
    
    # Run TensorFlow MLP
    tensorflow_model, tensorflow_r2 = tensorflow_mlp(df)

if __name__ == "__main__":
    main()
