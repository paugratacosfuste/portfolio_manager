import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from utils.data_fetcher import get_current_prices, get_historical_data, get_news

@patch('utils.data_fetcher.yf.Ticker')
def test_get_current_prices(mock_ticker):
    # Mocking single ticker
    mock_instance = MagicMock()
    mock_instance.fast_info = {'lastPrice': 150.0}
    mock_ticker.return_value = mock_instance
    
    prices = get_current_prices(['AAPL'])
    
    assert 'AAPL' in prices
    assert prices['AAPL'] == 150.0
    mock_ticker.assert_called_with('AAPL')

@patch('utils.data_fetcher.yf.download')
def test_get_historical_data(mock_download):
    # Mock download returning a typical dataframe
    mock_df = pd.DataFrame({
        'AAPL': [100.0, 101.0, 102.0],
        'MSFT': [200.0, 201.0, 202.0]
    })
    
    # We simulate a dataframe coming back with 'Adj Close'
    mock_download.return_value = mock_df
    
    # Note: data_fetcher might have logic dealing with 'Adj Close' specifically,
    # assuming we want to return whatever yf.download gives.
    data = get_historical_data(['AAPL', 'MSFT'], period='1mo')
    
    assert not data.empty
    assert 'AAPL' in data.columns
    assert len(data['AAPL']) == 3
    
@patch('utils.data_fetcher.yf.Ticker')
def test_get_news(mock_ticker):
    mock_instance = MagicMock()
    mock_instance.news = [
        {'title': 'Test Article 1', 'publisher': 'Test Pub', 'link': 'http://test.com/1'}
    ]
    mock_ticker.return_value = mock_instance
    
    news = get_news('AAPL')
    
    assert len(news) == 1
    assert news[0]['title'] == 'Test Article 1'
    assert news[0]['publisher'] == 'Test Pub'
