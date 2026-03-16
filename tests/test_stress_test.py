import pytest
import json
from unittest.mock import patch, MagicMock
from utils.stress_test import apply_stress_to_portfolio, generate_stress_scenario


def test_apply_stress_basic():
    """Test that stress is correctly applied to portfolio values."""
    holdings = {'AAPL': 10, 'BTC-USD': 1}
    prices = {'AAPL': 150.0, 'BTC-USD': 50000.0}

    scenario_data = {
        'scenario_name': 'Tech Crash',
        'scenario_summary': 'Technology stocks crash.',
        'overall_market_impact_pct': -20,
        'correlation_spike': 0.8,
        'impacts': [
            {'ticker': 'AAPL', 'sector': 'Technology', 'drawdown_pct': -40, 'rationale': 'Tech crash hits hard.'},
            {'ticker': 'BTC-USD', 'sector': 'Digital Assets', 'drawdown_pct': -60, 'rationale': 'Risk-off for crypto.'},
        ]
    }

    results = apply_stress_to_portfolio(holdings, prices, scenario_data)

    assert results['current_total'] == 51500.0
    # AAPL: 1500 * (1 - 0.40) = 900
    # BTC: 50000 * (1 - 0.60) = 20000
    assert results['stressed_total'] == 20900.0
    assert results['total_loss'] == 20900.0 - 51500.0
    assert len(results['holdings_impact']) == 2


def test_apply_stress_missing_ticker():
    """Test that missing ticker in scenario falls back to overall_market_impact."""
    holdings = {'AAPL': 10, 'MSFT': 5}
    prices = {'AAPL': 100.0, 'MSFT': 200.0}

    scenario_data = {
        'scenario_name': 'Mild Correction',
        'scenario_summary': 'A mild correction.',
        'overall_market_impact_pct': -10,
        'correlation_spike': 0.5,
        'impacts': [
            {'ticker': 'AAPL', 'drawdown_pct': -15, 'rationale': 'Test'}
        ]
    }

    results = apply_stress_to_portfolio(holdings, prices, scenario_data)

    # MSFT should use the fallback overall_market_impact_pct of -10
    msft_impact = next(h for h in results['holdings_impact'] if h['ticker'] == 'MSFT')
    assert msft_impact['drawdown_pct'] == -10


def test_apply_stress_empty_portfolio():
    """Test with empty portfolio."""
    results = apply_stress_to_portfolio({}, {}, {'impacts': [], 'overall_market_impact_pct': -20})

    assert results['current_total'] == 0.0
    assert results['stressed_total'] == 0.0


@patch('utils.stress_test.client')
def test_generate_stress_scenario_success(mock_client):
    """Test that generate_stress_scenario parses Claude's JSON response."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps({
        'scenario_name': 'Test Crash',
        'scenario_summary': 'A test crash.',
        'overall_market_impact_pct': -25,
        'correlation_spike': 0.9,
        'impacts': [
            {'ticker': 'AAPL', 'sector': 'Technology', 'drawdown_pct': -30, 'rationale': 'Tech hit.'}
        ]
    }))]
    mock_client.messages.create.return_value = mock_response

    result = generate_stress_scenario("A test crash", [{'ticker': 'AAPL', 'sector': 'Technology', 'asset_class': 'Stocks', 'weight_pct': 100}])

    assert result['scenario_name'] == 'Test Crash'
    assert len(result['impacts']) == 1


@patch('utils.stress_test.client', None)
def test_generate_stress_scenario_no_client():
    """Test that missing client returns error."""
    result = generate_stress_scenario("crash", [])
    assert 'error' in result
