# ------------------------------
# GRU-Based Model for Stock Prediction
# ------------------------------

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import matplotlib.pyplot as plt

# Assume 'stock_data' contains the processed DataFrame from your LSTM section
# with features ['Close', 'Lag_1', 'Open', 'High', 'Low', 'Volume', 'Sentiment']

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(stock_data[['Close', 'Lag_1', 'Open', 'High', 'Low', 'Volume', 'Sentiment']])

look_back = 60
X, y = [], []
for i in range(look_back, len(scaled_data)):
    X.append(scaled_data[i - look_back:i])
    y.append(scaled_data[i, 0])  # Predicting Close price

X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# GRU model
model_gru = Sequential()
model_gru.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model_gru.add(GRU(units=50, return_sequences=False))
model_gru.add(Dense(units=25))
model_gru.add(Dense(units=1))
model_gru.compile(optimizer='adam', loss='mean_squared_error')

model_gru.fit(X_train, y_train, batch_size=32, epochs=50, verbose=1)

# Predictions
predictions = model_gru.predict(X_test)
predictions = scaler.inverse_transform(np.hstack((predictions, np.zeros((predictions.shape[0], 6)))))[:, 0]
y_test_scaled = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 6)))))[:, 0]

# Evaluation
mse = mean_squared_error(y_test_scaled, predictions)
r2 = r2_score(y_test_scaled, predictions)
accuracy = np.mean(np.abs((predictions - y_test_scaled) / y_test_scaled) <= 0.05) * 100

print("GRU Model Metrics:")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2:.4f}")
print(f"Accuracy: {accuracy:.2f}%")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(y_test_scaled, label='Actual')
plt.plot(predictions, label='Predicted')
plt.title('GRU - Actual vs Predicted Stock Prices')
plt.xlabel('Test Data Points')
plt.ylabel('Stock Price (Close)')
plt.legend()
plt.show()

# ------------------------------
# ARIMA-Based Model (SARIMAX) for Stock Prediction
# ------------------------------

from statsmodels.tsa.statespace.sarimax import SARIMAX

# Prepare DataFrame for ARIMA
df_arima = stock_data[['Close', 'Sentiment']].copy().reset_index(drop=True)

# Train-test split using index instead of datetime
train_size = int(len(df_arima) * 0.8)
train = df_arima.iloc[:train_size]
test = df_arima.iloc[train_size:]

# Fit SARIMAX model
model_arima = SARIMAX(train['Close'], exog=train[['Sentiment']], order=(1, 1, 1))
model_fit = model_arima.fit(disp=False)

# Predict using integer-based start/end
preds_arima = model_fit.predict(start=len(train), end=len(df_arima)-1, exog=test[['Sentiment']])

# Evaluation
mse_arima = mean_squared_error(test['Close'], preds_arima)
r2_arima = r2_score(test['Close'], preds_arima)
accuracy_arima = np.mean(np.abs((preds_arima - test['Close']) / test['Close']) <= 0.05) * 100

print("\nARIMA Model Metrics:")
print(f"Mean Squared Error: {mse_arima}")
print(f"R-squared: {r2_arima:.4f}")
print(f"Accuracy: {accuracy_arima:.2f}%")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(test['Close'].values, label='Actual')
plt.plot(preds_arima.values, label='Predicted')
plt.title('ARIMA - Actual vs Predicted Stock Prices')
plt.xlabel('Test Data Points')
plt.ylabel('Stock Price (Close)')
plt.legend()
plt.show()

