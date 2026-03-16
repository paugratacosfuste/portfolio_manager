import pytest
import pandas as pd
import numpy as np
from utils.portfolio_metrics import (
    calculate_portfolio_volatility,
    calculate_hhi_index,
    calculate_beta,
    calculate_portfolio_beta,
    assess_risk_score,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
)

def test_calculate_hhi_index():
    # Perfectly diversified (2 assets, 50% each)
    weights = {'AAPL': 0.5, 'MSFT': 0.5}
    hhi = calculate_hhi_index(weights)
    assert hhi == 5000.0  # 50^2 + 50^2 = 2500 + 2500 = 5000
    
    # Highly concentrated (1 asset)
    weights = {'AAPL': 1.0}
    hhi = calculate_hhi_index(weights)
    assert hhi == 10000.0

def test_assess_risk_score():
    # Moderate portfolio
    score = assess_risk_score(volatility=0.15, hhi=5000, beta=1.0, total_value=10000)
    # Expected: 
    # vol_score: (0.15/0.15)*20 = 20
    # beta_score: (1.0-0.5)*20 = 10
    # hhi_score: (5000/10000)*30 = 15
    # total = 45
    assert score == 45
    
    # Very safe portfolio
    score_safe = assess_risk_score(volatility=0.01, hhi=100, beta=0.1, total_value=10000)
    assert score_safe < 45
    
    # Very risky portfolio
    score_risk = assess_risk_score(volatility=0.5, hhi=10000, beta=3.0, total_value=10000)
    assert score_risk > 45
    assert score_risk <= 100

def test_calculate_portfolio_volatility():
    # Dummy price dataframe
    dates = pd.date_range("2023-01-01", periods=5)
    df = pd.DataFrame({
        'AAPL': [100, 101, 102, 101, 100],
        'MSFT': [200, 202, 204, 202, 200]
    }, index=dates)
    
    weights = {'AAPL': 0.5, 'MSFT': 0.5}
    vol = calculate_portfolio_volatility(df, weights)
    
    # We just want to ensure it calculates without error and returns a float >= 0
    assert isinstance(vol, float)
    assert vol >= 0.0

def test_calculate_beta():
    # Perfectly correlated
    asset = pd.Series([0.01, 0.02, -0.01])
    market = pd.Series([0.01, 0.02, -0.01])

    beta = calculate_beta(asset, market)
    assert np.isclose(beta, 1.0)


def test_calculate_sharpe_ratio():
    # Create a simple upward-trending price series
    dates = pd.date_range("2023-01-01", periods=252)
    np.random.seed(42)
    prices_a = 100 * (1 + np.random.normal(0.0005, 0.01, 252)).cumprod()
    prices_b = 100 * (1 + np.random.normal(0.0003, 0.015, 252)).cumprod()
    df = pd.DataFrame({'AAPL': prices_a, 'MSFT': prices_b}, index=dates)

    weights = {'AAPL': 0.6, 'MSFT': 0.4}
    sharpe = calculate_sharpe_ratio(df, weights)

    assert isinstance(sharpe, float)
    # With random seed 42 and positive drift, Sharpe should be a real number
    assert not np.isnan(sharpe)


def test_calculate_sharpe_ratio_empty():
    df = pd.DataFrame()
    assert calculate_sharpe_ratio(df, {}) == 0.0


def test_calculate_max_drawdown():
    # Create a price series with a known drawdown
    dates = pd.date_range("2023-01-01", periods=10)
    # Goes up to 120, then drops to 90 (25% drawdown from peak), then recovers
    df = pd.DataFrame({
        'AAPL': [100, 110, 120, 110, 100, 90, 95, 100, 105, 110]
    }, index=dates)

    weights = {'AAPL': 1.0}
    max_dd = calculate_max_drawdown(df, weights)

    assert isinstance(max_dd, float)
    assert max_dd < 0  # Drawdown should be negative
    assert max_dd >= -1.0  # Can't lose more than 100%


def test_calculate_max_drawdown_empty():
    df = pd.DataFrame()
    assert calculate_max_drawdown(df, {}) == 0.0
