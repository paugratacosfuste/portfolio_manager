import pytest
from unittest.mock import patch, MagicMock
from utils.ai_advisor import (
    generate_portfolio_advice,
    generate_news_summary,
    compare_portfolio_with_standard
)

@patch('utils.ai_advisor.client')
def test_generate_portfolio_advice(mock_client):
    # Mocking the client method chain
    mock_messages = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Mocked advice from Claude.")]
    mock_messages.create.return_value = mock_response
    mock_client.messages = mock_messages
    
    portfolio_data = {'holdings': {'AAPL': 1000}, 'weights': {'AAPL': 1.0}, 'total_value': 1000}
    risk_metrics = {'volatility': 0.15, 'beta': 1.0, 'hhi': 10000, 'risk_score': 50}
    macro_prediction = {'probability': 0.7, 'prediction': 1}
    user_profile = {'name': 'Test User', 'risk_tolerance': 'Moderate', 'horizon': '3-7 years'}
    
    result = generate_portfolio_advice(
        portfolio_data, risk_metrics, macro_prediction, user_profile, eli10_mode=False
    )
    
    assert "Mocked advice from Claude." in result
    mock_client.messages.create.assert_called_once()

@patch('utils.ai_advisor.client')
def test_generate_news_summary(mock_client):
    mock_messages = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Mocked news summary.")]
    mock_messages.create.return_value = mock_response
    mock_client.messages = mock_messages
    
    news_items = [{'title': 'Apple is doing great', 'publisher': 'News API'}]
    holdings = ['AAPL']
    
    result = generate_news_summary(news_items, holdings, eli10_mode=False)
    
    assert "Mocked news summary." in result
    mock_client.messages.create.assert_called_once()

@patch('utils.ai_advisor.client')
def test_compare_portfolio_with_standard(mock_client):
    mock_messages = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Mocked comparison.")]
    mock_messages.create.return_value = mock_response
    mock_client.messages = mock_messages
    
    portfolio_data = {'holdings': {'AAPL': 1000}, 'weights': {'AAPL': 1.0}}
    
    result = compare_portfolio_with_standard(
        portfolio_data, 50, "Classic 60/40", "Some desc", False
    )
    
    assert "Mocked comparison." in result
    mock_client.messages.create.assert_called_once()
