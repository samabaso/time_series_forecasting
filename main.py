import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# Step 1: Fetch stock data from Yahoo Finance
ticker = 'AAPL'  # Apple stock as an example
data = yf.download(ticker, start='2015-01-01', end='2023-01-01')

# Step 2: Visualize the stock price
plt.figure(figsize=(10, 6))
plt.plot(data['Close'], label='Close Price')
plt.title(f'{ticker} Stock Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Step 3: Decompose the time series into trend, seasonal, and residual components
result = seasonal_decompose(data['Close'], model='multiplicative', period=252)
result.plot()
plt.show()

# Step 4: Prepare the data for ARIMA model (use the 'Close' price for the model)
stock_prices = data['Close'].fillna(method='ffill')  # Fill missing values

# Train-test split
train_size = int(len(stock_prices) * 0.80)
train, test = stock_prices[:train_size], stock_prices[train_size:]

# Step 5: Build and train the ARIMA model (using p, d, q parameters)
# These are hyperparameters and can be tuned
model = ARIMA(train, order=(5, 1, 0))
arima_result = model.fit()

# Step 6: Make predictions using the ARIMA model
forecast = arima_result.forecast(steps=len(test))
test_dates = test.index

# Step 7: Plot predictions vs actual data
plt.figure(figsize=(10, 6))
plt.plot(test_dates, test, label='Actual Prices', color='blue')
plt.plot(test_dates, forecast, label='ARIMA Predicted Prices', color='red')
plt.title(f'{ticker} Stock Price Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Step 8: Calculate the error (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(test, forecast))
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
