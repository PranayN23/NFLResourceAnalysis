import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np

def attention_layer(inputs):
    """Simple attention mechanism."""
    # Using dot-product attention without softmax (not needed for regression)
    attention = tf.keras.layers.Attention()([inputs, inputs])
    return tf.reduce_sum(attention, axis=1)

def tensorflow_rnn(df, metric):
    features_train = df[df['Year'] <= 2021][['Previous_twp_rate', 'Previous_ypa', 'Previous_qb_rating', 'Previous_grades_pass', 'Value_cap_space', 'Previous_PFF', 'Previous_AV']]
    labels_train = df[df['Year'] <= 2021]['Current_' + metric]

    # For testing, use data from 2022
    features_test = df[df['Year'] == 2022][['Previous_twp_rate', 'Previous_ypa', 'Previous_qb_rating', 'Previous_grades_pass', 'Value_cap_space', 'Previous_PFF', 'Previous_AV']]
    labels_test = df[df['Year'] == 2022]['Current_' + metric]

    # Apply one-hot encoding to any categorical features if necessary (e.g., if you have 'Team')
    features_train = pd.get_dummies(features_train)
    features_test = pd.get_dummies(features_test)

    # Ensure that both training and test sets have the same number of features after encoding
    features_train, features_test = features_train.align(features_test, join='left', axis=1, fill_value=0)

    # Standardize the features
    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)
    features_test = scaler.transform(features_test)

    # Reshape the features for RNN input (samples, time steps, features)
    features_train = features_train.reshape((features_train.shape[0], 1, features_train.shape[1]))
    features_test = features_test.reshape((features_test.shape[0], 1, features_test.shape[1]))

    learning_rates = [0.001, 0.01, 0.5]
    sizes = [(10,), (50,), (10, 10, 10, 10)]

    for learning_rate in learning_rates:
        for size in sizes:
            print(f'TensorFlow RNN with Attention - Learning Rate {learning_rate}, Size {size} Metric {metric}')

            # Create the model
            model = tf.keras.Sequential()

            # Add RNN Layer (LSTM or GRU)
            model.add(tf.keras.layers.LSTM(50, activation='relu', input_shape=(features_train.shape[1], features_train.shape[2]), return_sequences=True))
            
            # Add Attention Layer
            model.add(tf.keras.layers.Lambda(attention_layer))
            
            # Add Dense Output Layer (No activation function for regression)
            model.add(tf.keras.layers.Dense(1))  # Output layer

            # Compile the model
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          loss='mean_squared_error')

            # Train the model
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            history = model.fit(features_train, labels_train, epochs=200, batch_size=32, 
                                validation_split=0.2, callbacks=[early_stopping], verbose=0)

            # Predict and calculate R² score
            train_predictions = model.predict(features_train).flatten()  # Flatten to match the expected shape
            test_predictions = model.predict(features_test).flatten()    # Flatten to match the expected shape

            train_r2 = r2_score(labels_train, train_predictions)
            test_r2 = r2_score(labels_test, test_predictions)

            print(f'    Training set R²: {train_r2:.2f}')
            print(f'    Test set R²: {test_r2:.2f}')

def main():
    # Example: load your data into `df`
    df = pd.read_csv('Combined_QB.csv')  # Replace with your actual data
    metric = 'PFF'  # Specify the metric you want to predict
    tensorflow_rnn(df, metric)

if __name__ == "__main__":
    main()
