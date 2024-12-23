# 01. Load Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf

# 02: Download Stock Data (e.g., Apple)
stock_data = yf.download('AAPL', start='2010-01-01', end='2024-01-01')


# Step 2: Use only the 'Close' price for prediction
data = stock_data[['Close']]

# Step 3: Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Step 4: Prepare data for training the model
# Use previous 60 days to predict the 61st day's stock price
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])  # X = Previous 60 days' prices
        y.append(data[i + time_step, 0])  # y = The price after 60 days
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data)

# Reshape X to be compatible with LSTM input [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)


# Split data into training and test sets (80% training, 20% testing)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# Step 5: Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
model.add(Dense(units=1))  # Output layer (predicted price)


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 6: Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64)

# Step 7: Predict the stock prices on the test set
predictions = model.predict(X_test)

# Step 8: Inverse transform the predictions and actual prices back to the original scale
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform([y_test])


# Step 9: Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(data.index[-len(y_test[0]):], y_test[0], color='blue', label='Actual Prices')
plt.plot(data.index[-len(predictions):], predictions[:, 0], color='red', label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.show()