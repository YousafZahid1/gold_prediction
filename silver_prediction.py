import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import train_test_split

def fetch_live_silver_data(ticker):
    try:
        data = yf.download(ticker, period='1y')
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        print(f"Failed to fetch live silver data: {e}")
        return None

def train_forecasting_model(data):
    try:
        data['Date'] = pd.to_datetime(data['Date'])
        data['Date'] = data['Date'].apply(lambda date: date.timestamp())
        X = data[['Date']]
        y = data['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"Failed to train forecasting model: {e}")
        return None

def plot_prices(data, model):
    try:
        data['Date'] = pd.to_datetime(data['Date'])
        data['Date'] = data['Date'].apply(lambda date: date.timestamp())
        future_dates = np.array([data['Date'].max() + i * 86400 for i in range(1, 365*4)])  # Predict for the next 4 years
        future_dates_df = pd.DataFrame(future_dates, columns=['Date'])
        predicted_prices = model.predict(future_dates_df)
        plt.figure(figsize=(10, 6))
        plt.plot(data['Date'], data['Close'], label='Historical Prices')
        plt.plot(future_dates, predicted_prices, label='Predicted Prices')
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"Failed to plot prices: {e}")

if __name__ == "__main__":
    live_data = fetch_live_silver_data('SI=F')
    model = train_forecasting_model(live_data)
    plot_prices(live_data, model)