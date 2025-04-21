import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from keras.models import Sequential
from keras.layers import LSTM, GRU, Dropout, Dense
from keras.callbacks import EarlyStopping


# 1. Download historical IBM data

def download_data(ticker='IBM', start_date='2000-01-01', end_date='2024-04-01'):
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    print("Data downloaded!")
    return data


# 2. Preprocessing & normalization
def preprocess_data(df, feature='Close'):
    """
    Accepts a DataFrame and the column name (feature) to use.
    Returns the scaled data, the scaler, and the original DataFrame (trimmed).
    """
    df = df[[feature]].copy()
    df.dropna(inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    return scaled_data, scaler, df


# 3. Sequence creation

def create_sequences(data, window_size=15):
    """
    Creates input/output pairs:
    - X of shape (samples, timesteps)
    - y of shape (samples,)
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# 4. Build & Train GRU model

def build_gru_model(input_shape):
    """
    Builds a GRU model with 4 GRU layers (50 units each) and dropout.
    """
    model = Sequential()
    model.add(GRU(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(GRU(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# 5. Build & Train LSTM model

def build_lstm_model(input_shape):
    """
    Builds an LSTM model with 4 LSTM layers (50 units each) and dropout.
    """
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 6. Plot utility
def plot_predictions(actual, predicted, title):
    plt.figure()
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.title(title)
    plt.xlabel('Time Steps (Test Samples)')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# 7. Main demonstration, with metrics
def main():
    # Step 1: Download IBM data
    df_raw = download_data('IBM', '2000-01-01', '2024-04-01')

    # Step 2: Preprocess (focus on 'Close' price)
    scaled_data, scaler, df = preprocess_data(df_raw, feature='Close')

    # Step 3: Create sequences
    window_size = 15
    X, y = create_sequences(scaled_data, window_size=window_size)

    # Reshape X to (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Train/test split (80% training, 20% test)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Step 4: GRU model
    gru_model = build_gru_model((X_train.shape[1], 1))
    print("\n--- GRU Model Summary ---")
    gru_model.summary()

    early_stop_gru = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    gru_model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=24,
        validation_data=(X_test, y_test),
        callbacks=[early_stop_gru],
        verbose=1
    )

    # GRU Predictions
    preds_gru_scaled = gru_model.predict(X_test)
    preds_gru = scaler.inverse_transform(preds_gru_scaled)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))  # so it's (samples, 1)

    plot_predictions(y_test_actual, preds_gru, "GRU: Actual vs Predicted")

    # Step 5: LSTM model
    lstm_model = build_lstm_model((X_train.shape[1], 1))
    print("\n--- LSTM Model Summary ---")
    lstm_model.summary()

    early_stop_lstm = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lstm_model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=24,
        validation_data=(X_test, y_test),
        callbacks=[early_stop_lstm],
        verbose=1
    )

    # LSTM Predictions
    preds_lstm_scaled = lstm_model.predict(X_test)
    preds_lstm = scaler.inverse_transform(preds_lstm_scaled)

    plot_predictions(y_test_actual, preds_lstm, "LSTM: Actual vs Predicted")

    # GRU metrics
    gru_mse = mean_squared_error(y_test_actual, preds_gru)
    gru_rmse = np.sqrt(gru_mse)
    gru_mae = mean_absolute_error(y_test_actual, preds_gru)
    gru_r2 = r2_score(y_test_actual, preds_gru)
    gru_mape = np.mean(np.abs((y_test_actual - preds_gru) / y_test_actual)) * 100

    # LSTM metrics
    lstm_mse = mean_squared_error(y_test_actual, preds_lstm)
    lstm_rmse = np.sqrt(lstm_mse)
    lstm_mae = mean_absolute_error(y_test_actual, preds_lstm)
    lstm_r2 = r2_score(y_test_actual, preds_lstm)
    lstm_mape = np.mean(np.abs((y_test_actual - preds_lstm) / y_test_actual)) * 100

    print("\n===== GRU Model Performance =====")
    print(f"MSE:  {gru_mse:.4f}")
    print(f"RMSE: {gru_rmse:.4f}")
    print(f"MAE:  {gru_mae:.4f}")
    print(f"R^2:  {gru_r2:.4f}")
    print(f"MAPE: {gru_mape:.2f}%")

    print("\n===== LSTM Model Performance =====")
    print(f"MSE:  {lstm_mse:.4f}")
    print(f"RMSE: {lstm_rmse:.4f}")
    print(f"MAE:  {lstm_mae:.4f}")
    print(f"R^2:  {lstm_r2:.4f}")
    print(f"MAPE: {lstm_mape:.2f}%")

if __name__ == "__main__":
    main()
