import pandas as pd
import numpy as np
from typing import Dict, Any

def calculate_portfolio_volatility(historical_prices: pd.DataFrame, weights: Dict[str, float]) -> float:
    """
    Calculates the annualized volatility of the portfolio given historical prices and asset weights.
    Returns a percentage (e.g. 0.15 for 15%).
    """
    if historical_prices.empty or not weights:
        return 0.0

    # Align columns with weights
    tickers = list(weights.keys())
    prices = historical_prices[tickers].dropna()
    
    if prices.empty:
        return 0.0
        
    # Calculate daily returns
    returns = prices.pct_change().dropna()
    
    # Calculate covariance matrix
    cov_matrix = returns.cov()
    
    # Create weights array in the same order as columns
    w_array = np.array([weights[t] for t in prices.columns])
    
    w_sum = np.sum(w_array)
    if w_sum == 0:
        return 0.0
    w_array = w_array / w_sum
    
    # Calculate portfolio variance (daily)
    port_variance = np.dot(w_array.T, np.dot(cov_matrix, w_array))
    
    # Annualize volatility (assuming 252 trading days)
    annual_volatility = np.sqrt(port_variance) * np.sqrt(252)
    
    return float(annual_volatility)

def calculate_hhi_index(weights: Dict[str, float]) -> float:
    """
    Calculates the Herfindahl-Hirschman Index (HHI) for the portfolio to measure concentration risk.
    HHI = sum of squared market shares (weights in percentage points).
    Max is 10000 (100% in one asset), Min approaches 0 for highly diversified.
    Returns a value between 0 and 10000.
    """
    if not weights:
        return 0.0
        
    w_array = np.array(list(weights.values()))
    # Ensure weights sum to 1
    w_sum = np.sum(w_array)
    if w_sum == 0:
        return 0.0
    w_array = w_array / w_sum
    
    # Convert to percentages (0-100) and square
    hhi = np.sum((w_array * 100) ** 2)
    return float(hhi)

def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Calculates the Beta of an asset or portfolio relative to a market benchmark.
    Beta > 1: more volatile than market. Beta < 1: less volatile.
    """
    # Align data
    aligned = pd.concat([asset_returns, market_returns], axis=1).dropna()
    if aligned.empty or len(aligned) < 2:
        return 1.0
        
    cov = aligned.cov().iloc[0, 1]
    var = aligned.iloc[:, 1].var()
    
    if var == 0:
        return 1.0
        
    return float(cov / var)

def calculate_portfolio_beta(historical_prices: pd.DataFrame, market_benchmark: pd.Series, weights: Dict[str, float]) -> float:
    """
    Calculates the weighted average beta of the overall portfolio.
    """
    if historical_prices.empty or market_benchmark.empty or not weights:
        return 1.0
        
    tickers = list(weights.keys())
    prices = historical_prices[tickers].dropna()
    
    if prices.empty:
        return 1.0
        
    returns = prices.pct_change().dropna()
    mkt_returns = market_benchmark.pct_change().dropna()
    
    # Align dates
    aligned_returns = pd.merge(returns, mkt_returns.rename('Market'), left_index=True, right_index=True).dropna()
    
    if aligned_returns.empty:
        return 1.0
        
    mkt = aligned_returns['Market']
    port_beta = 0.0
    
    # Normalize weights
    total_w = sum(weights.values())
    if total_w == 0:
        return 0.0
    norm_weights = {k: v/total_w for k, v in weights.items()}
    
    for ticker in tickers:
        if ticker in aligned_returns.columns:
            asset_beta = calculate_beta(aligned_returns[ticker], mkt)
            port_beta += asset_beta * norm_weights[ticker]
            
    return float(port_beta)

def assess_risk_score(volatility: float, hhi: float, beta: float, total_value: float) -> int:
    """
    Calculates an abstract "Risk Score" out of 100 based on calculated metrics.
    A higher score indicates higher risk.
    This is a heuristic function for the UI.
    """
    score = 0
    
    # Volatility impact (up to 40 points)
    # Assuming baseline market vol is ~0.15 (15%)
    vol_score = min(40, (volatility / 0.15) * 20)
    score += vol_score
    
    # Beta impact (up to 30 points)
    # Neutral is beta=1 (15 points). beta=2 is 30 points.
    beta_score = min(30, max(0, (beta - 0.5) * 20))
    score += beta_score
    
    # Concentration risk impact (up to 30 points)
    # HHI 10000 = 30 points. HHI < 1000 (well div) = ~3 points.
    hhi_score = min(30, (hhi / 10000) * 30)
    score += hhi_score
    
    return int(min(100, max(0, score)))
