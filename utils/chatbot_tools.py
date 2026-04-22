"""
Chatbot tool schemas and executor functions for the agentic chatbot.

Thin adapter layer: each executor builds an immutable `Portfolio` from the
chatbot's `{ticker: quantity}` dict (enriched with live metadata) and
delegates the actual math to `core/portfolio_ops.py`. Streamlit-specific
caching stays in `utils/data_fetcher.py`; this module is the bridge
between the chatbot interface and the framework-agnostic core.
"""
from typing import Any, Dict

from core.portfolio_ops import (
    compute_risk_metrics,
    portfolio_summary,
    simulate_trade,
)
from core.tool_schemas import (
    AssetPriceResponse,
    ErrorResponse,
    HoldingRow,
    MacroPredictionResponse,
    MarketNewsResponse,
    NewsItem,
    PortfolioSummaryResponse,
    RiskMetricsResponse,
    WhatIfResponse,
)
from core.types import Holding, Portfolio
from utils.data_fetcher import (
    fetch_asset_metadata,
    fetch_current_prices,
    fetch_historical_data,
    fetch_recent_news,
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


# ── Portfolio adapter ─────────────────────────────────────────────────────────


def _build_portfolio(
    holdings: Dict[str, float],
    metadata: Dict[str, Dict[str, Any]] | None = None,
) -> Portfolio:
    """Build an immutable Portfolio from chatbot `{ticker: qty}` + metadata."""
    meta = metadata or {}
    entries = tuple(
        Holding(
            ticker=ticker,
            quantity=qty,
            asset_class=(meta.get(ticker, {}).get("asset_class") or "Unknown"),
            sector=meta.get(ticker, {}).get("sector"),
            region=meta.get(ticker, {}).get("region"),
        )
        for ticker, qty in holdings.items()
        if qty > 0
    )
    return Portfolio(holdings=entries)


# ── Tool executor functions ───────────────────────────────────────────────────


def _execute_get_portfolio_summary(holdings: Dict[str, float]) -> str:
    """Returns portfolio summary with values, weights, and metadata."""
    if not holdings:
        return ErrorResponse(error="No holdings in portfolio.").model_dump_json()

    tickers = list(holdings.keys())
    prices = fetch_current_prices(tickers)
    metadata = fetch_asset_metadata(tickers)

    portfolio = _build_portfolio(holdings, metadata)
    summary = portfolio_summary(portfolio, prices)

    rows = tuple(
        HoldingRow(
            ticker=h["ticker"],
            quantity=h["quantity"],
            price=round(h["price"], 2),
            value=round(h["value"], 2),
            weight_pct=round(h["weight"] * 100, 1),
            asset_class=h["asset_class"] or "Unknown",
            sector=h["sector"] or "Unknown",
        )
        for h in summary["holdings"]
    )

    return PortfolioSummaryResponse(
        total_value=round(summary["total_value"], 2),
        holdings=rows,
    ).model_dump_json()


def _execute_get_asset_price(ticker: str) -> str:
    """Returns the current price for a single ticker."""
    prices = fetch_current_prices([ticker])
    price = prices.get(ticker, 0)
    return AssetPriceResponse(
        ticker=ticker, price=round(max(price, 0.0), 2)
    ).model_dump_json()


def _historical_frame(tickers: list[str]) -> tuple[Any, list[str]]:
    """Fetch 1-year historical prices and return (dataframe, valid_tickers)."""
    hist = fetch_historical_data(tickers, period="1y")
    valid = [t for t in tickers if t in hist.columns]
    return hist, valid


def _execute_calculate_portfolio_risk(holdings: Dict[str, float]) -> str:
    """Calculates all risk metrics for the current portfolio."""
    if not holdings:
        return ErrorResponse(error="No holdings in portfolio.").model_dump_json()

    tickers = list(holdings.keys())
    prices = fetch_current_prices(tickers)
    metadata = fetch_asset_metadata(tickers)
    portfolio = _build_portfolio(holdings, metadata)

    hist, valid = _historical_frame(tickers)
    has_data = len(valid) > 0 and len(hist) > 50

    market_returns = None
    if has_data:
        try:
            spy_df = fetch_historical_data(["SPY"], period="1y")
            market_returns = spy_df["SPY"] if "SPY" in spy_df else hist.iloc[:, 0]
        except Exception:
            market_returns = None

    hist_for_metrics = hist[valid] if has_data else hist.iloc[0:0]
    metrics = compute_risk_metrics(
        portfolio=portfolio,
        prices=prices,
        historical_prices=hist_for_metrics,
        market_returns=market_returns,
    )

    # Fallback for sparse-data case: keep the legacy "risk_score=50" default
    risk_score = metrics["risk_score"] if has_data else 50
    # Schema requires max_drawdown_pct ≤ 0 (it's a loss)
    max_dd_pct = min(round(metrics["max_drawdown"] * 100, 1), 0.0)
    # Schema caps hhi at 10000 (100% concentration in one asset)
    hhi = min(max(round(metrics["hhi"], 0), 0.0), 10000.0)

    return RiskMetricsResponse(
        total_value=round(metrics["total_value"], 2),
        volatility_pct=round(metrics["volatility"] * 100, 1),
        beta=round(metrics["beta"], 2),
        hhi=hhi,
        risk_score=int(max(0, min(100, risk_score))),
        sharpe_ratio=round(metrics["sharpe"], 2),
        max_drawdown_pct=max_dd_pct,
        cvar_95_pct=round(metrics["cvar"] * 100, 2),
    ).model_dump_json()


def _execute_what_if_analysis(
    holdings: Dict[str, float], action: str, ticker: str, quantity: float
) -> str:
    """Simulates adding/removing an asset and returns new risk metrics."""
    if action not in ("add", "remove"):
        return ErrorResponse(
            error=f"Unknown action '{action}'. Must be 'add' or 'remove'."
        ).model_dump_json()
    if quantity <= 0:
        return ErrorResponse(
            error="Quantity must be greater than zero."
        ).model_dump_json()

    metadata = fetch_asset_metadata(list(set(list(holdings.keys()) + [ticker])))
    current = _build_portfolio(holdings, metadata)

    qty_delta = quantity if action == "add" else -quantity
    try:
        new_portfolio = simulate_trade(
            current,
            ticker,
            qty_delta=qty_delta,
            asset_class=(metadata.get(ticker, {}).get("asset_class") or "Unknown"),
        )
    except ValueError:
        return ErrorResponse(
            error=f"{ticker} not in portfolio, cannot remove."
        ).model_dump_json()

    if not new_portfolio.holdings:
        return ErrorResponse(
            error="Portfolio would be empty after this change."
        ).model_dump_json()

    new_tickers = [h.ticker for h in new_portfolio.holdings]
    prices = fetch_current_prices(new_tickers)

    hist, valid = _historical_frame(new_tickers)
    has_data = len(valid) > 0 and len(hist) > 50
    hist_for_metrics = hist[valid] if has_data else hist.iloc[0:0]

    metrics = compute_risk_metrics(
        portfolio=new_portfolio,
        prices=prices,
        historical_prices=hist_for_metrics,
        market_returns=None,
    )

    summary = portfolio_summary(new_portfolio, prices)
    new_weights = {
        row["ticker"]: round(row["weight"] * 100, 1) for row in summary["holdings"]
    }

    return WhatIfResponse(
        action=action,  # type: ignore[arg-type]
        ticker=ticker,
        quantity=quantity,
        new_total_value=round(metrics["total_value"], 2),
        new_volatility_pct=round(metrics["volatility"] * 100, 1),
        new_sharpe_ratio=round(metrics["sharpe"], 2),
        new_max_drawdown_pct=round(metrics["max_drawdown"] * 100, 1),
        new_weights=new_weights,
    ).model_dump_json()


def _execute_get_market_news(ticker: str) -> str:
    """Fetches recent news for a ticker."""
    news = fetch_recent_news([ticker], limit=5)
    raw_items = news.get(ticker, [])[:5]
    items = tuple(
        NewsItem(
            title=(item.get("title") or "Untitled"),
            link=(item.get("link") or ""),
            publisher=(item.get("publisher") or ""),
            timestamp=(item.get("timestamp") or ""),
        )
        for item in raw_items
    )
    return MarketNewsResponse(ticker=ticker, news=items).model_dump_json()


def _execute_get_macro_prediction() -> str:
    """Loads the macro risk model and returns a market correction prediction.
    Uses centralized feature engineering from ml_pipeline.features."""
    try:
        import joblib

        from ml_pipeline.features import get_latest_macro_features

        model = joblib.load("ml_pipeline/macro_risk_model.joblib")

        latest_data = get_latest_macro_features(feature_version=1)

        prediction = int(model.predict(latest_data)[0])
        probability = float(model.predict_proba(latest_data)[0][1])

        return MacroPredictionResponse(
            prediction="Market correction likely" if prediction == 1 else "Market stable",
            correction_probability_pct=round(probability * 100, 1),
            current_vix=max(round(float(latest_data["VIX"].iloc[0]), 1), 0.0),
            sp500_vs_200ma_pct=round(
                float(latest_data["SP500_200d_ma_diff"].iloc[0] * 100), 1
            ),
        ).model_dump_json()
    except Exception as e:
        return ErrorResponse(
            error=f"Could not run macro prediction: {str(e)}"
        ).model_dump_json()


# ── Dispatcher ────────────────────────────────────────────────────────────────


def execute_tool(
    tool_name: str, tool_input: Dict[str, Any], holdings: Dict[str, float]
) -> str:
    """Dispatches a tool call to the appropriate executor function."""
    if tool_name == "get_portfolio_summary":
        return _execute_get_portfolio_summary(holdings)
    if tool_name == "get_asset_price":
        return _execute_get_asset_price(tool_input["ticker"])
    if tool_name == "calculate_portfolio_risk":
        return _execute_calculate_portfolio_risk(holdings)
    if tool_name == "what_if_analysis":
        return _execute_what_if_analysis(
            holdings, tool_input["action"], tool_input["ticker"], tool_input["quantity"]
        )
    if tool_name == "get_market_news":
        return _execute_get_market_news(tool_input["ticker"])
    if tool_name == "get_macro_prediction":
        return _execute_get_macro_prediction()
    return ErrorResponse(error=f"Unknown tool: {tool_name}").model_dump_json()
