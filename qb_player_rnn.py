import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt




# Load the dataset
file_path = 'Combined_QB.csv'
df = pd.read_csv(file_path)


# Define input features and target output
input_features = [col for col in df.columns if col.startswith('weighted_')]
output_column = 'Current_PFF'  # Modify this if predicting a different metric


# Filter columns
df = df[input_features + [output_column, 'Year']]


# Split data into training and testing sets
train_data = df[df['Year'].isin([2019, 2020, 2021])].drop(columns=['Year'])
test_data = df[df['Year'] == 2022].drop(columns=['Year'])


# Normalize the data
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()


X_train = scaler_x.fit_transform(train_data[input_features])
y_train = scaler_y.fit_transform(train_data[[output_column]])


X_test = scaler_x.transform(test_data[input_features])
y_test_actual = test_data[output_column].values  # for later comparison


# Reshape for RNN input (samples, time steps, features)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


# Build the RNN model using tf.keras
model = tf.keras.Sequential([
   tf.keras.layers.LSTM(100, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
   tf.keras.layers.Dropout(0.2),
   tf.keras.layers.LSTM(50, activation='relu'),
   tf.keras.layers.Dense(1)
])




# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')


# Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test_actual))


# Store loss values for plotting
train_loss = history.history['loss']
val_loss = history.history['val_loss']  # Optional: if you want to compare train vs validation loss




# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


# Inverse transform predictions to original scale
y_train_pred = scaler_y.inverse_transform(y_train_pred)
y_test_pred = scaler_y.inverse_transform(y_test_pred)


# Print R² scores for training and test data
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test_actual, y_test_pred)
mse = mean_squared_error(y_test_actual, y_test_pred)
mae = mean_absolute_error(y_test_actual, y_test_pred)




print(f"Training R²: {r2_train}")
print(f"Testing R²: {r2_test}")


print(f"MSE: {mse}")
print(f"MAE: {mae}")




# Save predictions with actual values in a new CSV file
test_data['Predicted_PFF'] = y_test_pred
test_data['Actual_PFF'] = y_test_actual
test_data.to_csv('predictions_with_actuals.csv', index=False)


print("Predictions saved in 'predictions_with_actuals.csv'")


# Plot the actual vs predicted values for the test set
plt.figure(figsize=(10, 6))
plt.scatter(y_test_actual, y_test_pred, color='blue', alpha=0.5)
plt.plot([min(y_test_actual), max(y_test_actual)], [min(y_test_actual), max(y_test_actual)], color='red', linewidth=2)
plt.xlabel('Actual PFF')
plt.ylabel('Predicted PFF')
plt.title('Actual vs Predicted PFF for Test Set')
plt.grid(True)
plt.show()
plt.plot(train_loss, label='Training Loss')
if 'val_loss' in history.history:
   plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss During Training')
plt.legend()
plt.show()

