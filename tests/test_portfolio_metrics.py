import pytest
import pandas as pd
import numpy as np
from utils.portfolio_metrics import (
    calculate_portfolio_volatility,
    calculate_hhi_index,
    calculate_beta,
    calculate_portfolio_beta,
    assess_risk_score
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
