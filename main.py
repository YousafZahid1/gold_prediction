import pandas as pd
import yfinance as yf
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

def load_historical_gold_data(file_path):
    try:
        data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'])
        return data
    except Exception as e:
        print(f"Failed to load historical gold data: {e}")
        return None

def fetch_live_gold_data(ticker):
    try:
        data = yf.download(ticker, period='1y')
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        print(f"Failed to fetch live gold data: {e}")
        return None

def merge_gold_data(historical_data, live_data):
    try:
        historical_data['Close'] = historical_data['Close'].astype(float)
        live_data['Close'] = live_data['Close'].astype(float)
        merged_data = pd.concat([historical_data[['Date', 'Close']], live_data[['Date', 'Close']]], ignore_index=True)
        merged_data.drop_duplicates(subset='Date', keep='last', inplace=True)
        merged_data.sort_values(by='Date', inplace=True)
        return merged_data
    except Exception as e:
        print(f"Failed to merge gold data: {e}")
        return None

def fetch_news_sentiment(ticker):
    try:
        news = yf.Ticker(ticker).news
        sia = SentimentIntensityAnalyzer()
        sentiments = []
        for article in news:
            text = article['title']
            sentiment = sia.polarity_scores(text)
            sentiments.append(sentiment['compound'])
        return np.mean(sentiments)
    except Exception as e:
        print(f"Failed to fetch news sentiment: {e}")
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
        y_pred = model.predict(X_test)
        print(f"Model MSE: {mean_squared_error(y_test, y_pred)}")
        return model
    except Exception as e:
        print(f"Failed to train forecasting model: {e}")
        return None

def generate_signal(model, data):
    try:
        latest_date = data['Date'].max()
        future_date = latest_date + 86400  # Predict for the next day
        future_data = pd.DataFrame([[future_date]], columns=['Date'])
        predicted_price = model.predict(future_data)[0]
        current_price = data[data['Date'] == latest_date]['Close'].values[0]
        if predicted_price > current_price:
            return "Buy"
        elif predicted_price == current_price:
            return "Hold"
        else:
            return "Avoid"
    except Exception as e:
        print(f"Failed to generate signal: {e}")
        return None

def plot_prices(data, model):
    try:
        data['Date'] = pd.to_datetime(data['Date'])
        data['Date'] = data['Date'].apply(lambda date: date.timestamp())
        future_dates = np.array([data['Date'].max() + i * 86400 for i in range(1, 31)])  # Predict for the next 30 days
        future_dates_df = pd.DataFrame(future_dates, columns=['Date'])
        predicted_prices = model.predict(future_dates_df)
        plt.figure(figsize=(10, 6))
        plt.plot(data['Date'], data['Close'], label='Historical Prices')
        plt.plot(future_dates, predicted_prices, label='Predicted Prices')
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"Failed to plot prices: {e}")

def load_historical_silver_data(ticker):
    try:
        data = yf.download(ticker, period='10y')
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date'])
        return data
    except Exception as e:
        print(f"Failed to load historical silver data: {e}")
        return None

def train_silver_forecasting_model(data):
    try:
        data['Date'] = pd.to_datetime(data['Date'])
        data['Date'] = data['Date'].apply(lambda date: date.timestamp())
        X = data[['Date']]
        y = data['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"Silver Model MSE: {mean_squared_error(y_test, y_pred)}")
        return model
    except Exception as e:
        print(f"Failed to train silver forecasting model: {e}")
        return None

def plot_silver_prices(data, model):
    try:
        data['Date'] = pd.to_datetime(data['Date'])
        data['Date'] = data['Date'].apply(lambda date: date.timestamp())
        future_dates = np.array([data['Date'].max() + i * 86400 for i in range(1, 365*4)])  # Predict for the next 4 years
        future_dates_df = pd.DataFrame(future_dates, columns=['Date'])
        predicted_prices = model.predict(future_dates_df)
        plt.figure(figsize=(10, 6))
        plt.plot(data['Date'], data['Close'], label='Historical Silver Prices')
        plt.plot(future_dates, predicted_prices, label='Predicted Silver Prices')
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"Failed to plot silver prices: {e}")

if __name__ == "__main__":
    historical_data = load_historical_gold_data('BSE-BOM590111.csv')
    live_data = fetch_live_gold_data('GC=F')
    merged_data = merge_gold_data(historical_data, live_data)
    news_sentiment = fetch_news_sentiment('GC=F')
    print(f"News Sentiment: {news_sentiment}")
    model = train_forecasting_model(merged_data)
    signal = generate_signal(model, merged_data)
    print(f"Signal: {signal}")
    plot_prices(merged_data, model)
    silver_data = load_historical_silver_data('SI=F')
    silver_model = train_silver_forecasting_model(silver_data)
    plot_silver_prices(silver_data, silver_model)