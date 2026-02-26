import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import yfinance as yf
from newsapi import NewsApiClient
import requests

def load_historical_gold_data(file_path):
    try:
        data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        return data
    except Exception as e:
        print(f"Failed to load historical gold data: {e}")
        return None

def fetch_live_gold_data():
    try:
        gold_data = yf.download('GC=F', period='1d')
        return gold_data
    except Exception as e:
        print(f"Failed to fetch live gold data: {e}")
        return None

def prepare_data(data):
    try:
        data['Close_Lag1'] = data['Close'].shift(1)
        data['Close_Lag2'] = data['Close'].shift(2)
        data.dropna(inplace=True)
        X = data[['Close_Lag1', 'Close_Lag2']]
        y = data['Close']
        return train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        print(f"Failed to prepare data: {e}")
        return None, None, None, None

def train_model(X_train, y_train):
    try:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"Failed to train model: {e}")
        return None

def predict_future_prices(model, data, days=365):
    try:
        future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=days)
        future_data = pd.DataFrame(index=future_dates, columns=['Close_Lag1', 'Close_Lag2'])
        future_data['Close_Lag1'] = data['Close'].iloc[-1]
        future_data['Close_Lag2'] = data['Close'].iloc[-2]
        predictions = []
        for i in range(days):
            pred = model.predict([[future_data['Close_Lag1'].iloc[i], future_data['Close_Lag2'].iloc[i]]])[0]
            predictions.append(pred)
            if i < days - 1:
                future_data['Close_Lag1'].iloc[i+1] = pred
                future_data['Close_Lag2'].iloc[i+1] = future_data['Close_Lag1'].iloc[i]
        return pd.Series(predictions, index=future_dates)
    except Exception as e:
        print(f"Failed to predict future prices: {e}")
        return None

def plot_predicted_prices(data, predictions):
    try:
        plt.figure(figsize=(10,6))
        plt.plot(data.index, data['Close'], label='Historical')
        plt.plot(predictions.index, predictions, label='Predicted')
        plt.legend()
        plt.title('Gold Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.show()
    except Exception as e:
        print(f"Failed to plot predicted prices: {e}")

def fetch_news_sentiment(api_key, query='gold'):
    try:
        newsapi = NewsApiClient(api_key=api_key)
        response = newsapi.get_everything(q=query)
        articles = response['articles']
        sentiments = [article['title'] for article in articles if 'title' in article]
        return sentiments
    except Exception as e:
        print(f"Failed to fetch news sentiment: {e}")
        return None

def main():
    file_path = 'BSE-BOM590111.csv'
    data = load_historical_gold_data(file_path)
    if data is not None:
        X_train, X_test, y_train, y_test = prepare_data(data)
        if X_train is not None:
            model = train_model(X_train, y_train)
            if model is not None:
                predictions = predict_future_prices(model, data)
                if predictions is not None:
                    plot_predicted_prices(data, predictions)
                
                live_data = fetch_live_gold_data()
                if live_data is not None:
                    print("Live Gold Data:")
                    print(live_data)
                
                news_api_key = 'YOUR_NEWS_API_KEY'
                sentiments = fetch_news_sentiment(news_api_key)
                if sentiments is not None:
                    print("News Sentiment:")
                    print(sentiments)

if __name__ == "__main__":
    main()