import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

def main():
    df = pd.read_csv('data.csv')
    decision_tree(df)
    sklearn_mlp(df)
    tensorflow_mlp(df)

def decision_tree(df):
    from sklearn.tree import DecisionTreeRegressor
    
    features = df[['Year', 'Position', 'Value_cap_space', 'Value_draft_data', 'Previous_AV']]
    features = pd.get_dummies(features)
    labels = df['Current_AV']
    features_train, features_test = features[:864], features[864:]
    labels_train, labels_test = labels[:864], labels[864:]
    
    model = DecisionTreeRegressor()
    model.fit(features_train, labels_train)
    
    # Predict and calculate R² score
    train_predictions = model.predict(features_train)
    test_predictions = model.predict(features_test)
    
    train_r2 = r2_score(labels_train, train_predictions)
    test_r2 = r2_score(labels_test, test_predictions)
    
    print(f'Decision Tree - Train R²: {train_r2:.2f}')
    print(f'Decision Tree - Test R²: {test_r2:.2f}')

def sklearn_mlp(df):
    features = df[['Year', 'Position', 'Value_cap_space', 'Value_draft_data', 'Previous_AV']]
    features = pd.get_dummies(features)
    labels = df['Current_AV']
    features_train, features_test = features[:864], features[864:]
    labels_train, labels_test = labels[:864], labels[864:]
    
    learning_rates = [0.001, 0.01, 0.5]
    sizes = [(10,), (50,), (10, 10, 10, 10)]
    
    for learning_rate in learning_rates:
        for size in sizes:
            print(f'Scikit-learn MLP - Learning Rate {learning_rate}, Size {size}')
            mlp = MLPRegressor(hidden_layer_sizes=size, max_iter=200,  # Increase max_iter
                               random_state=1, learning_rate_init=learning_rate)
            mlp.fit(features_train, labels_train)
            
            # Predict and calculate R² score
            train_predictions = mlp.predict(features_train)
            test_predictions = mlp.predict(features_test)
            
            train_r2 = r2_score(labels_train, train_predictions)
            test_r2 = r2_score(labels_test, test_predictions)
            
            print(f'    Training set R²: {train_r2:.2f}')
            print(f'    Test set R²: {test_r2:.2f}')

def tensorflow_mlp(df):
    features = df[['Year', 'Position', 'Value_cap_space', 'Value_draft_data', 'Previous_AV']]
    labels = df['Current_AV'].values  # Ensure labels are numpy arrays
    
    # Convert categorical features to numeric
    features = pd.get_dummies(features)
    
    # Standardize the features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    features_train, features_test = features[:864], features[864:]
    labels_train, labels_test = labels[:864], labels[864:]

    learning_rates = [0.001, 0.01, 0.5]
    sizes = [(10,), (50,), (10, 10, 10, 10)]
    
    for learning_rate in learning_rates:
        for size in sizes:
            print(f'TensorFlow MLP - Learning Rate {learning_rate}, Size {size}')
            
            # Create the model
            model = tf.keras.Sequential()
            for neurons in size:
                model.add(tf.keras.layers.Dense(neurons, activation='relu'))
            model.add(tf.keras.layers.Dense(1))  # Output layer

            # Compile the model
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          loss='mean_squared_error')

            # Train the model
            history = model.fit(features_train, labels_train, epochs=100,  # Increase epochs
                                batch_size=32, validation_split=0.2, verbose=0)

            # Predict and calculate R² score
            train_predictions = model.predict(features_train).flatten()  # Flatten to match the expected shape
            test_predictions = model.predict(features_test).flatten()    # Flatten to match the expected shape
            
            train_r2 = r2_score(labels_train, train_predictions)
            test_r2 = r2_score(labels_test, test_predictions)
            
            print(f'    Training set R²: {train_r2:.2f}')
            print(f'    Test set R²: {test_r2:.2f}')

if __name__ == "__main__":
    main()
