import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from utils.data_fetcher import fetch_current_prices, fetch_historical_data, fetch_recent_news

@patch('utils.data_fetcher.yf.download')
def test_get_current_prices(mock_download):
    # Mocking single ticker yf.download
    mock_df = pd.DataFrame({'Close': [149.0, 150.0]})
    mock_download.return_value = mock_df
    
    prices = fetch_current_prices(['AAPL'])
    
    assert 'AAPL' in prices
    assert prices['AAPL'] == 150.0
    mock_download.assert_called_with(['AAPL'], period="1d", progress=False)

@patch('utils.data_fetcher.yf.download')
def test_get_historical_data(mock_download):
    # Mock download returning a dataframe with 'Adj Close'
    mock_df = pd.DataFrame({
        'Adj Close': [100.0, 101.0, 102.0]
    })
    # yfinance multi-index for multiple tickers is a bit complex to mock linearly. 
    # For a single ticker it returns simply the columns.
    mock_download.return_value = mock_df
    
    data = fetch_historical_data(['AAPL'], period='1mo')
    
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
    
    news_dict = fetch_recent_news(['AAPL'])
    
    assert 'AAPL' in news_dict
    news = news_dict['AAPL']
    assert len(news) == 1
    assert news[0]['title'] == 'Test Article 1'
    assert news[0]['publisher'] == 'Test Pub'
