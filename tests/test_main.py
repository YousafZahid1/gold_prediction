import sys
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['yfinance'] = MagicMock()

from main import fetch_news_sentiment, train_forecasting_model, generate_signal, plot_prices

class TestGoldPricePrediction(unittest.TestCase):

    @patch('yfinance.Ticker')
    def test_fetch_news_sentiment(self, mock_ticker):
        mock_news = [{'title': 'Gold prices rise'}, {'title': 'Gold market analysis'}]
        mock_ticker.return_value.news = mock_news
        sentiment = fetch_news_sentiment('GC=F')
        self.assertIsNotNone(sentiment)

    def test_train_forecasting_model(self):
        data = pd.DataFrame({
            'Date': ['2023-01-01', '2023-01-02'],
            'Close': [100, 105]
        })
        model = train_forecasting_model(data)
        self.assertIsNotNone(model)

    @patch('main.RandomForestRegressor')
    def test_generate_signal(self, mock_model):
        mock_model.predict.return_value = [110]
        data = pd.DataFrame({
            'Date': [1643723400, 1643813400],
            'Close': [100, 105]
        })
        signal = generate_signal(mock_model, data)
        self.assertIn(signal, ['Buy', 'Hold', 'Avoid'])

    @patch('matplotlib.pyplot.show')
    def test_plot_prices(self, mock_show):
        data = pd.DataFrame({
            'Date': [1643723400, 1643813400],
            'Close': [100, 105]
        })
        model = MagicMock()
        model.predict.return_value = [110, 115]
        plot_prices(data, model)
        mock_show.assert_called_once()

if __name__ == '__main__':
    unittest.main()