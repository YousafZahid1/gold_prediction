import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np

nltk.download('vader_lexicon')

def load_historical_gold_data(file_path):
    try:
        gold_data = pd.read_csv(file_path)
        gold_data['Date'] = pd.to_datetime(gold_data['Date'])
        gold_data.set_index('Date', inplace=True)
        return gold_data
    except Exception as e:
        print(f"Failed to load historical gold data: {e}")
        return None

def fetch_live_gold_data(ticker, period):
    try:
        live_data = yf.download(ticker, period=period)
        return live_data
    except Exception as e:
        print(f"Failed to fetch live gold data: {e}")
        return None

def analyze_news_sentiment(news_data):
    try:
        sia = SentimentIntensityAnalyzer()
        sentiment_scores = [sia.polarity_scores(item)['compound'] for item in news_data]
        return np.mean(sentiment_scores) if sentiment_scores else None
    except Exception as e:
        print(f"Failed to fetch news sentiment: {e}")
        return None

def train_forecasting_model(data):
    try:
        data['Prediction'] = data['Close'].shift(-1)
        data.dropna(inplace=True)
        X = data[['Close']]
        y = data['Prediction']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"Model RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
        return model
    except Exception as e:
        print(f"Failed to train forecasting model: {e}")
        return None

def generate_signal(data, model):
    try:
        latest_close = data['Close'].iloc[-1]
        prediction = model.predict([[latest_close]])
        return prediction[0]
    except Exception as e:
        print(f"Failed to generate signal: {e}")
        return None

def plot_prices(data):
    try:
        plt.figure(figsize=(10,5))
        plt.plot(data.index, data['Close'], label='Close Price')
        plt.title('Gold Price Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"Failed to plot prices: {e}")

def main():
    file_path = 'BSE-BOM590111.csv'
    ticker = 'GC=F'  # Gold futures ticker
    period = '1y'

    gold_data = load_historical_gold_data(file_path)
    live_data = fetch_live_gold_data(ticker, period)

    if gold_data is not None and live_data is not None:
        news_data = ['Gold prices surged today', 'Market experts predict stable gold prices']  # Example news headlines
        news_sentiment = analyze_news_sentiment(news_data)
        print(f"News Sentiment: {news_sentiment}")

        combined_data = pd.concat([gold_data['Close'], live_data['Close']])
        model = train_forecasting_model(combined_data.to_frame())

        if model is not None:
            signal = generate_signal(combined_data.to_frame(), model)
            print(f"Signal: {signal}")

            plot_prices(combined_data.to_frame())

if __name__ == "__main__":
    main()