import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


def main():
    df = pd.read_csv('RB19-22.csv')
    df = df[df['Position'] == "QB"]
    metrics = ['PFF', 'AV']
    for metric in metrics:
        # check_correlation(df, metric)
        sklearn_mlp(df, metric)
        tensorflow_mlp(df, metric)


def check_correlation(df, metric):
    features = ['Year', 'Value_cap_space', 'Value_draft_data', 'Previous_PFF', 'Previous_AV', 'Current_' + metric,
                'Total DVOA', 'win-loss-pct', 'Net EPA']
    df['Total DVOA'] = df['Total DVOA'].astype(str).str.rstrip('%').astype(float) / 100.0

    # Filter only the relevant columns
    corr_df = df[features]

    # Compute the correlation matrix
    corr_matrix = corr_df.corr()

    # Print the correlation matrix
    print(f'Correlation Matrix for {metric}:\n', corr_matrix, '\n')


def sklearn_mlp(df, metric):
    features_train = df[df['Year'] <= 2021][
        ['Value_cap_space', 'Previous_PFF', 'Previous_AV', 'Total DVOA', 'win-loss-pct', 'Net EPA']]
    features_train['Total DVOA'] = features_train['Total DVOA'].astype(str).str.rstrip('%').astype(float) / 100.0

    labels_train = df[df['Year'] <= 2021]['Current_' + metric]

    # For testing, use data from 2022
    features_test = df[df['Year'] == 2022][['Value_cap_space', 'Previous_PFF', 'Previous_AV']]
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

    learning_rates = [0.001, 0.01, 0.5]
    sizes = [(10,), (50,), (10, 10, 10, 10)]

    for learning_rate in learning_rates:
        for size in sizes:
            print(f'Scikit-learn MLP ' + metric + '- Learning Rate {learning_rate}, Size {size}')
            mlp = MLPRegressor(hidden_layer_sizes=size, max_iter=1000, alpha=0.01, learning_rate_init=learning_rate)
            mlp.fit(features_train, labels_train)

            # Predict and calculate R² score
            train_predictions = mlp.predict(features_train)
            test_predictions = mlp.predict(features_test)

            train_r2 = r2_score(labels_train, train_predictions)
            test_r2 = r2_score(labels_test, test_predictions)

            print(f'    Training set R²: {train_r2:.2f}')
            print(f'    Test set R²: {test_r2:.2f}')


def tensorflow_mlp(df, metric):
    features_train = df[df['Year'] <= 2021][
        ['Value_cap_space', 'Previous_PFF', 'Previous_AV', 'Total DVOA', 'win-loss-pct', 'Net EPA']]
    features_train['Total DVOA'] = features_train['Total DVOA'].astype(str).str.rstrip('%').astype(float) / 100.0

    labels_train = df[df['Year'] <= 2021]['Current_' + metric]

    # For testing, use data from 2022
    features_test = df[df['Year'] == 2022][
        ['Value_cap_space', 'Previous_PFF', 'Previous_AV', 'Total DVOA', 'win-loss-pct', 'Net EPA']]
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

    learning_rates = [0.001, 0.01, 0.5]
    sizes = [(10,), (50,), (10, 10, 10, 10)]

    for learning_rate in learning_rates:
        for size in sizes:
            print(f'TensorFlow MLP - Learning Rate {learning_rate}, Size {size} Metric {metric}')

            # Create the model
            model = tf.keras.Sequential()
            for neurons in size:
                model.add(tf.keras.layers.Dense(neurons, activation='relu'))
            model.add(tf.keras.layers.Dense(1))  # Output layer

            # Compile the model
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          loss='mean_squared_error')

            # Train the model
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                                              restore_best_weights=True)

            history = model.fit(features_train, labels_train, epochs=200, batch_size=32,
                                validation_split=0.2, callbacks=[early_stopping], verbose=0)

            # Predict and calculate R² score
            train_predictions = model.predict(features_train).flatten()  # Flatten to match the expected shape
            test_predictions = model.predict(features_test).flatten()  # Flatten to match the expected shape

            train_r2 = r2_score(labels_train, train_predictions)
            test_r2 = r2_score(labels_test, test_predictions)

            print(f'    Training set R²: {train_r2:.2f}')
            print(f'    Test set R²: {test_r2:.2f}')


if __name__ == "__main__":
    main()