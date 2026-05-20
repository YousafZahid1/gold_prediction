import pandas as pd
import yfinance as yf
import nltk
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
# No change needed here, but ensure torch is installed
# You can add torch to your requirements.txt or install it via pip
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
            if 'title' in article:
                text = article['title']
                sentiment = sia.polarity_scores(text)
                sentiments.append(sentiment['compound'])
        if sentiments:
            return np.mean(sentiments)
        else:
            return 0  # or any other default value
    except Exception as e:
        print(f"Failed to fetch news sentiment: {e}")
        return None

def train_forecasting_model(data):
    try:
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data = data.dropna(subset=['Date'])
        if not data.empty:
            data['Date'] = data['Date'].apply(lambda date: date.timestamp())
            X = data[['Date']]
            y = data['Close']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print(f"Model MSE: {mean_squared_error(y_test, y_pred)}")
            return model
        else:
            print("No valid dates in data for training.")
            return None
    except Exception as e:
        print(f"Failed to train forecasting model: {e}")
        return None

def generate_signal(model, data):
    try:
        latest_date = pd.to_datetime(data['Date'].max(), errors='coerce')
        if pd.isnull(latest_date):
            print("Invalid latest date for generating signal.")
            return None
        future_date = latest_date + pd.Timedelta(days=1)
        future_date_timestamp = future_date.timestamp()
        future_data = pd.DataFrame([[future_date_timestamp]], columns=['Date'])
        predicted_price = model.predict(future_data)[0]
        current_price = data.iloc[-1]['Close']
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
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data = data.dropna(subset=['Date'])
        if not data.empty:
            data['Date'] = data['Date'].apply(lambda date: date.timestamp())
            future_dates = np.array([data['Date'].max() + i * 86400 for i in range(1, 31)])
            future_dates_df = pd.DataFrame(future_dates, columns=['Date'])
            predicted_prices = model.predict(future_dates_df)
            plt.figure(figsize=(10, 6))
            plt.plot(data['Date'], data['Close'], label='Historical Prices')
            plt.plot(future_dates, predicted_prices, label='Predicted Prices')
            plt.legend()
            plt.show()
        else:
            print("No valid data to plot.")
    except Exception as e:
        print(f"Failed to plot prices: {e}")

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
