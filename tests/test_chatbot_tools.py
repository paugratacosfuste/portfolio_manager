import pytest
import json
import pandas as pd
from unittest.mock import patch, MagicMock
from utils.chatbot_tools import execute_tool, _execute_get_portfolio_summary, _execute_get_asset_price


@patch('utils.chatbot_tools.fetch_asset_metadata')
@patch('utils.chatbot_tools.fetch_current_prices')
def test_get_portfolio_summary(mock_prices, mock_meta):
    mock_prices.return_value = {'AAPL': 150.0, 'BTC-USD': 50000.0}
    mock_meta.return_value = {
        'AAPL': {'asset_class': 'Stocks', 'sector': 'Technology', 'region': 'US'},
        'BTC-USD': {'asset_class': 'Crypto', 'sector': 'Digital Assets', 'region': 'Global'},
    }

    holdings = {'AAPL': 10, 'BTC-USD': 0.5}
    result = json.loads(_execute_get_portfolio_summary(holdings))

    assert result['total_value'] == 26500.0
    assert len(result['holdings']) == 2
    assert result['holdings'][0]['ticker'] == 'AAPL'
    assert result['holdings'][0]['value'] == 1500.0


@patch('utils.chatbot_tools.fetch_current_prices')
def test_get_asset_price(mock_prices):
    mock_prices.return_value = {'TSLA': 250.0}
    result = json.loads(_execute_get_asset_price('TSLA'))

    assert result['ticker'] == 'TSLA'
    assert result['price'] == 250.0


def test_execute_tool_unknown():
    result = json.loads(execute_tool('unknown_tool', {}, {}))
    assert 'error' in result


@patch('utils.chatbot_tools.fetch_asset_metadata')
@patch('utils.chatbot_tools.fetch_current_prices')
def test_execute_tool_dispatcher(mock_prices, mock_meta):
    mock_prices.return_value = {'AAPL': 150.0}
    mock_meta.return_value = {'AAPL': {'asset_class': 'Stocks', 'sector': 'Tech', 'region': 'US'}}

    result = json.loads(execute_tool('get_portfolio_summary', {}, {'AAPL': 10}))
    assert 'total_value' in result


@patch('utils.chatbot_tools.fetch_current_prices')
def test_execute_tool_get_asset_price(mock_prices):
    mock_prices.return_value = {'GOOGL': 140.0}
    result = json.loads(execute_tool('get_asset_price', {'ticker': 'GOOGL'}, {}))
    assert result['price'] == 140.0


@patch('utils.chatbot_tools.fetch_historical_data')
@patch('utils.chatbot_tools.fetch_current_prices')
def test_what_if_add(mock_prices, mock_hist):
    mock_prices.return_value = {'AAPL': 150.0, 'TSLA': 250.0}
    mock_hist.return_value = pd.DataFrame()  # empty to trigger fallback

    result = json.loads(execute_tool(
        'what_if_analysis',
        {'action': 'add', 'ticker': 'TSLA', 'quantity': 5},
        {'AAPL': 10}
    ))
    assert 'new_total_value' in result
    assert 'TSLA' in result.get('new_weights', {})


def test_what_if_remove_nonexistent():
    result = json.loads(execute_tool(
        'what_if_analysis',
        {'action': 'remove', 'ticker': 'TSLA', 'quantity': 5},
        {'AAPL': 10}
    ))
    assert 'error' in result


@patch('utils.chatbot_tools.fetch_recent_news')
def test_get_market_news(mock_news):
    mock_news.return_value = {
        'AAPL': [{'title': 'Apple launches new product', 'publisher': 'Reuters', 'link': '#', 'timestamp': '2024-01-01'}]
    }
    result = json.loads(execute_tool('get_market_news', {'ticker': 'AAPL'}, {}))
    assert result['ticker'] == 'AAPL'
    assert len(result['news']) == 1
