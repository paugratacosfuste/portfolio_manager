"""
Chatbot tool schemas and executor functions for the agentic chatbot.
Each tool wraps existing utility functions so Claude can call them dynamically.
"""
import json
from typing import Dict, Any, List

from utils.data_fetcher import fetch_current_prices, fetch_historical_data, fetch_recent_news, fetch_asset_metadata
from utils.portfolio_metrics import (
    calculate_portfolio_volatility,
    calculate_portfolio_beta,
    calculate_hhi_index,
    assess_risk_score,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
)

# ── Anthropic-format tool schemas ─────────────────────────────────────────────

CHATBOT_TOOLS = [
    {
        "name": "get_portfolio_summary",
        "description": "Get a summary of the user's current portfolio including total value, per-holding values, weights, and asset metadata. Call this when the user asks about their portfolio overview or holdings.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_asset_price",
        "description": "Get the current market price of a specific asset by its ticker symbol. Use this when the user asks about the price of a specific stock, crypto, or ETF.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The ticker symbol (e.g. AAPL, BTC-USD, SPY)",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "calculate_portfolio_risk",
        "description": "Calculate comprehensive risk metrics for the user's portfolio: volatility, beta, HHI concentration, risk score, Sharpe ratio, and max drawdown. Call this when the user asks about risk, safety, or how risky their portfolio is.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "what_if_analysis",
        "description": "Simulate adding or removing an asset from the portfolio and calculate the resulting risk metrics. Use this when the user asks 'What if I buy/sell X?' or 'What would happen if I add/remove Y?'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "remove"],
                    "description": "Whether to simulate adding or removing the asset",
                },
                "ticker": {
                    "type": "string",
                    "description": "The ticker symbol to add or remove",
                },
                "quantity": {
                    "type": "number",
                    "description": "Number of shares/units to add or remove",
                },
            },
            "required": ["action", "ticker", "quantity"],
        },
    },
    {
        "name": "get_market_news",
        "description": "Fetch recent market news for a specific ticker. Use when the user asks about news, headlines, or recent events for a stock or crypto.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The ticker symbol to get news for",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_macro_prediction",
        "description": "Get the ML model's prediction for the probability of a market correction in the next month. Call this when the user asks about market outlook, macro conditions, or whether a crash is coming.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


# ── Tool executor functions ───────────────────────────────────────────────────

def _execute_get_portfolio_summary(holdings: Dict[str, float]) -> str:
    """Returns portfolio summary with values, weights, and metadata."""
    if not holdings:
        return json.dumps({"error": "No holdings in portfolio."})

    tickers = list(holdings.keys())
    prices = fetch_current_prices(tickers)
    metadata = fetch_asset_metadata(tickers)

    total_value = sum(holdings[t] * prices.get(t, 0) for t in tickers)
    summary = {
        "total_value": round(total_value, 2),
        "holdings": [],
    }

    for t in tickers:
        price = prices.get(t, 0)
        value = holdings[t] * price
        weight = (value / total_value * 100) if total_value > 0 else 0
        meta = metadata.get(t, {})
        summary["holdings"].append({
            "ticker": t,
            "quantity": holdings[t],
            "price": round(price, 2),
            "value": round(value, 2),
            "weight_pct": round(weight, 1),
            "asset_class": meta.get("asset_class", "Unknown"),
            "sector": meta.get("sector", "Unknown"),
        })

    return json.dumps(summary)


def _execute_get_asset_price(ticker: str) -> str:
    """Returns the current price for a single ticker."""
    prices = fetch_current_prices([ticker])
    price = prices.get(ticker, 0)
    return json.dumps({"ticker": ticker, "price": round(price, 2)})


def _execute_calculate_portfolio_risk(holdings: Dict[str, float]) -> str:
    """Calculates all risk metrics for the current portfolio."""
    if not holdings:
        return json.dumps({"error": "No holdings in portfolio."})

    tickers = list(holdings.keys())
    prices = fetch_current_prices(tickers)
    total_value = sum(holdings[t] * prices.get(t, 0) for t in tickers)
    weights = {}
    for t in tickers:
        weights[t] = (holdings[t] * prices.get(t, 0)) / total_value if total_value > 0 else 0

    hist = fetch_historical_data(tickers, period="1y")
    valid = [t for t in tickers if t in hist.columns]

    if len(valid) > 0 and len(hist) > 50:
        volatility = calculate_portfolio_volatility(hist[valid], weights)
        try:
            spy_df = fetch_historical_data(["SPY"], period="1y")
            mkt = spy_df["SPY"] if "SPY" in spy_df else hist.iloc[:, 0]
            beta = calculate_portfolio_beta(hist[valid], mkt, weights)
        except Exception:
            beta = 1.0
        hhi = calculate_hhi_index(weights)
        risk_score = assess_risk_score(volatility, hhi, beta, total_value)
        sharpe = calculate_sharpe_ratio(hist[valid], weights)
        max_dd = calculate_max_drawdown(hist[valid], weights)
    else:
        volatility, beta, hhi, risk_score, sharpe, max_dd = 0.0, 1.0, calculate_hhi_index(weights), 50, 0.0, 0.0

    return json.dumps({
        "total_value": round(total_value, 2),
        "volatility_pct": round(volatility * 100, 1),
        "beta": round(beta, 2),
        "hhi": round(hhi, 0),
        "risk_score": risk_score,
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd * 100, 1),
    })


def _execute_what_if_analysis(holdings: Dict[str, float], action: str, ticker: str, quantity: float) -> str:
    """Simulates adding/removing an asset and returns new risk metrics."""
    simulated = dict(holdings)

    if action == "add":
        simulated[ticker] = simulated.get(ticker, 0) + quantity
    elif action == "remove":
        if ticker in simulated:
            simulated[ticker] = max(0, simulated[ticker] - quantity)
            if simulated[ticker] == 0:
                del simulated[ticker]
        else:
            return json.dumps({"error": f"{ticker} not in portfolio, cannot remove."})

    if not simulated:
        return json.dumps({"error": "Portfolio would be empty after this change."})

    # Calculate new metrics
    tickers = list(simulated.keys())
    prices = fetch_current_prices(tickers)
    total_value = sum(simulated[t] * prices.get(t, 0) for t in tickers)
    weights = {}
    for t in tickers:
        weights[t] = (simulated[t] * prices.get(t, 0)) / total_value if total_value > 0 else 0

    hist = fetch_historical_data(tickers, period="1y")
    valid = [t for t in tickers if t in hist.columns]

    if len(valid) > 0 and len(hist) > 50:
        volatility = calculate_portfolio_volatility(hist[valid], weights)
        sharpe = calculate_sharpe_ratio(hist[valid], weights)
        max_dd = calculate_max_drawdown(hist[valid], weights)
    else:
        volatility, sharpe, max_dd = 0.0, 0.0, 0.0

    return json.dumps({
        "action": action,
        "ticker": ticker,
        "quantity": quantity,
        "new_total_value": round(total_value, 2),
        "new_volatility_pct": round(volatility * 100, 1),
        "new_sharpe_ratio": round(sharpe, 2),
        "new_max_drawdown_pct": round(max_dd * 100, 1),
        "new_weights": {t: round(w * 100, 1) for t, w in weights.items()},
    })


def _execute_get_market_news(ticker: str) -> str:
    """Fetches recent news for a ticker."""
    news = fetch_recent_news([ticker], limit=5)
    items = news.get(ticker, [])
    return json.dumps({"ticker": ticker, "news": items[:5]})


def _execute_get_macro_prediction() -> str:
    """Loads the macro risk model and returns a market correction prediction.
    Downloads each macro ticker individually for reliability."""
    try:
        import joblib
        import numpy as np
        import yfinance as yf
        import pandas as pd

        model = joblib.load("ml_pipeline/macro_risk_model.joblib")

        # Download each ticker individually to avoid batch download issues
        ticker_map = {'^GSPC': 'SP500', '^TNX': 'US10Y', '^VIX': 'VIX'}
        dxy_tickers = ['DX-Y.NYB', 'DX=F']  # fallback for Dollar Index
        frames = {}

        def _dl_close(yf_ticker):
            t_data = yf.download(yf_ticker, period="2y", progress=False)
            if t_data.empty:
                return None
            if isinstance(t_data.columns, pd.MultiIndex):
                return t_data['Close'].iloc[:, 0]
            elif 'Close' in t_data.columns:
                return t_data['Close']
            return t_data.iloc[:, 0]

        for yf_ticker, col_name in ticker_map.items():
            try:
                close = _dl_close(yf_ticker)
                if close is not None and len(close) > 0:
                    frames[col_name] = close
            except Exception:
                pass

        for dxy_ticker in dxy_tickers:
            try:
                close = _dl_close(dxy_ticker)
                if close is not None and len(close) > 0:
                    frames['DXY'] = close
                    break
            except Exception:
                pass

        missing = [k for k in ['SP500', 'US10Y', 'VIX', 'DXY'] if k not in frames]
        if missing:
            return json.dumps({"error": f"Could not fetch macro data for: {missing}"})

        df = pd.DataFrame(frames).dropna(how='all').ffill().dropna()

        df['SP500_Return'] = df['SP500'].pct_change()
        df['VIX_Change'] = df['VIX'].diff()
        df['US10Y_Change'] = df['US10Y'].diff()
        df['DXY_Return'] = df['DXY'].pct_change()

        df['SP500_20d_vol'] = df['SP500_Return'].rolling(20).std()
        df['SP500_200d_ma_diff'] = df['SP500'] / df['SP500'].rolling(200).mean() - 1
        df['VIX_zscore'] = (df['VIX'] - df['VIX'].rolling(252).mean()) / df['VIX'].rolling(252).std()
        df['US10Y_20d_std'] = df['US10Y_Change'].rolling(20).std()

        df = df.dropna()
        if df.empty:
            return json.dumps({"error": "Not enough historical data for macro features."})

        latest_data = df.iloc[-1:]

        prediction = int(model.predict(latest_data)[0])
        probability = float(model.predict_proba(latest_data)[0][1])

        return json.dumps({
            "prediction": "Market correction likely" if prediction == 1 else "Market stable",
            "correction_probability_pct": round(probability * 100, 1),
            "current_vix": round(float(latest_data['VIX'].iloc[0]), 1),
            "sp500_vs_200ma_pct": round(float(latest_data['SP500_200d_ma_diff'].iloc[0] * 100), 1),
        })
    except Exception as e:
        return json.dumps({"error": f"Could not run macro prediction: {str(e)}"})


# ── Dispatcher ────────────────────────────────────────────────────────────────

def execute_tool(tool_name: str, tool_input: Dict[str, Any], holdings: Dict[str, float]) -> str:
    """Dispatches a tool call to the appropriate executor function."""
    if tool_name == "get_portfolio_summary":
        return _execute_get_portfolio_summary(holdings)
    elif tool_name == "get_asset_price":
        return _execute_get_asset_price(tool_input["ticker"])
    elif tool_name == "calculate_portfolio_risk":
        return _execute_calculate_portfolio_risk(holdings)
    elif tool_name == "what_if_analysis":
        return _execute_what_if_analysis(
            holdings, tool_input["action"], tool_input["ticker"], tool_input["quantity"]
        )
    elif tool_name == "get_market_news":
        return _execute_get_market_news(tool_input["ticker"])
    elif tool_name == "get_macro_prediction":
        return _execute_get_macro_prediction()
    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
