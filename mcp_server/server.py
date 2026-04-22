"""MCP server exposing the portfolio toolkit over stdio.

Usage
-----
    python -m mcp_server.server

Claude Desktop config
---------------------
    {
      "mcpServers": {
        "portfolio-tracker": {
          "command": "/absolute/path/to/.venv/bin/python",
          "args": ["-m", "mcp_server.server"],
          "cwd": "/absolute/path/to/Portfolio_Tracker",
          "env": {"PORTFOLIO_TRACKER_CONFIG": "/absolute/path/to/portfolio.json"}
        }
      }
    }

Design notes
------------
- No Streamlit imports (runs in a separate process).
- Each tool builds a fresh `Portfolio` from the JSON config, so Claude
  Desktop always sees the latest edits on disk.
- All tool outputs reuse the same Pydantic schemas as the Streamlit
  chatbot (`core/tool_schemas.py`) — one source of truth for shape.
"""
from __future__ import annotations

import json
from typing import Literal

from mcp.server.fastmcp import FastMCP

from core.portfolio_ops import (
    compute_risk_metrics,
    portfolio_summary as core_portfolio_summary,
    simulate_trade,
)
from core.tool_schemas import (
    AssetPriceResponse,
    ErrorResponse,
    HoldingRow,
    MarketNewsResponse,
    NewsItem,
    PortfolioSummaryResponse,
    RiskMetricsResponse,
    WhatIfResponse,
)
from core.types import Holding, Portfolio
from mcp_server.portfolio_loader import load_portfolio_config


mcp = FastMCP("portfolio-tracker")


# ── Shared helpers ────────────────────────────────────────────────────────────


def _build_portfolio(
    holdings: dict[str, float], metadata: dict[str, dict]
) -> Portfolio:
    entries = tuple(
        Holding(
            ticker=t,
            quantity=float(q),
            asset_class=(metadata.get(t, {}).get("asset_class") or "Unknown"),
            sector=metadata.get(t, {}).get("sector"),
        )
        for t, q in holdings.items()
        if q > 0
    )
    return Portfolio(holdings=entries)


def _historical_frame(tickers: list[str]):
    """Defer-imported so tests can patch; yfinance is heavy."""
    from utils.data_fetcher import fetch_historical_data

    hist = fetch_historical_data(tickers, period="1y")
    valid = [t for t in tickers if t in hist.columns]
    return hist, valid


# ── Tools ─────────────────────────────────────────────────────────────────────


@mcp.tool()
def get_portfolio_summary() -> str:
    """Return total value, per-holding values, weights, and metadata for
    the portfolio defined in the JSON config."""
    cfg = load_portfolio_config()
    if not cfg.holdings:
        return ErrorResponse(
            error="No holdings in portfolio. Edit PORTFOLIO_TRACKER_CONFIG or ~/.portfolio_tracker/portfolio.json."
        ).model_dump_json()

    from utils.data_fetcher import fetch_asset_metadata, fetch_current_prices

    tickers = list(cfg.holdings.keys())
    prices = fetch_current_prices(tickers)
    metadata = fetch_asset_metadata(tickers)
    portfolio = _build_portfolio(cfg.holdings, metadata)
    summary = core_portfolio_summary(portfolio, prices)

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


@mcp.tool()
def get_asset_price(ticker: str) -> str:
    """Current market price for a single ticker (stock, ETF, crypto)."""
    from utils.data_fetcher import fetch_current_prices

    prices = fetch_current_prices([ticker])
    price = prices.get(ticker, 0)
    return AssetPriceResponse(
        ticker=ticker, price=round(max(price, 0.0), 2)
    ).model_dump_json()


@mcp.tool()
def calculate_portfolio_risk() -> str:
    """Compute full risk metrics for the current portfolio: volatility,
    beta, HHI, Sharpe, max drawdown, CVaR 95, and a 0–100 risk score."""
    cfg = load_portfolio_config()
    if not cfg.holdings:
        return ErrorResponse(error="No holdings in portfolio.").model_dump_json()

    from utils.data_fetcher import fetch_asset_metadata, fetch_current_prices

    tickers = list(cfg.holdings.keys())
    prices = fetch_current_prices(tickers)
    metadata = fetch_asset_metadata(tickers)
    portfolio = _build_portfolio(cfg.holdings, metadata)

    hist, valid = _historical_frame(tickers)
    has_data = len(valid) > 0 and len(hist) > 50

    market_returns = None
    if has_data:
        try:
            from utils.data_fetcher import fetch_historical_data
            spy_df = fetch_historical_data(["SPY"], period="1y")
            market_returns = (
                spy_df["SPY"] if "SPY" in spy_df else hist.iloc[:, 0]
            )
        except Exception:
            market_returns = None

    hist_for_metrics = hist[valid] if has_data else hist.iloc[0:0]
    metrics = compute_risk_metrics(
        portfolio=portfolio,
        prices=prices,
        historical_prices=hist_for_metrics,
        market_returns=market_returns,
    )
    risk_score = metrics["risk_score"] if has_data else 50
    max_dd_pct = min(round(metrics["max_drawdown"] * 100, 1), 0.0)
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


@mcp.tool()
def what_if_trade(
    action: Literal["add", "remove"], ticker: str, quantity: float
) -> str:
    """Simulate adding or removing an asset and return new risk metrics.
    Quantity must be > 0. Use action=remove to reduce an existing position."""
    if quantity <= 0:
        return ErrorResponse(error="Quantity must be > 0.").model_dump_json()

    cfg = load_portfolio_config()
    from utils.data_fetcher import fetch_asset_metadata, fetch_current_prices

    metadata = fetch_asset_metadata(list(set(list(cfg.holdings.keys()) + [ticker])))
    current = _build_portfolio(cfg.holdings, metadata)
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

    market_returns = None
    if has_data:
        try:
            from utils.data_fetcher import fetch_historical_data
            spy_df = fetch_historical_data(["SPY"], period="1y")
            market_returns = spy_df["SPY"] if "SPY" in spy_df else hist.iloc[:, 0]
        except Exception:
            market_returns = None

    new_metrics = compute_risk_metrics(
        portfolio=new_portfolio,
        prices=prices,
        historical_prices=hist_for_metrics,
        market_returns=market_returns,
    )
    new_risk_score = new_metrics["risk_score"] if has_data else 50
    max_dd = min(round(new_metrics["max_drawdown"] * 100, 1), 0.0)
    hhi_val = min(max(round(new_metrics["hhi"], 0), 0.0), 10000.0)

    new_risk = RiskMetricsResponse(
        total_value=round(new_metrics["total_value"], 2),
        volatility_pct=round(new_metrics["volatility"] * 100, 1),
        beta=round(new_metrics["beta"], 2),
        hhi=hhi_val,
        risk_score=int(max(0, min(100, new_risk_score))),
        sharpe_ratio=round(new_metrics["sharpe"], 2),
        max_drawdown_pct=max_dd,
        cvar_95_pct=round(new_metrics["cvar"] * 100, 2),
    )
    return WhatIfResponse(
        action=action, ticker=ticker, quantity=quantity, new_risk=new_risk,
    ).model_dump_json()


@mcp.tool()
def get_news(ticker: str, limit: int = 5) -> str:
    """Fetch the latest news headlines for a ticker (yfinance-backed).
    Returns up to `limit` items (capped at 10)."""
    from utils.data_fetcher import fetch_recent_news

    capped = max(1, min(10, int(limit)))
    try:
        news_items = fetch_recent_news([ticker], limit=capped).get(ticker, [])
    except Exception as ex:
        return ErrorResponse(error=f"News fetch failed: {ex}").model_dump_json()

    items = tuple(
        NewsItem(
            title=it.get("title", "") or "(untitled)",
            publisher=it.get("publisher", ""),
            link=it.get("link", ""),
            timestamp=it.get("timestamp", "Unknown"),
        )
        for it in news_items
    )
    return MarketNewsResponse(ticker=ticker, news=items).model_dump_json()


@mcp.tool()
def get_political_alpha_ledger(limit: int = 10) -> str:
    """Return the N most recent Political Alpha agent decisions (trade
    proposals + SKIPs). Useful for Claude Desktop to summarise what the
    news agent has been deciding. Returns JSON list."""
    from utils.signal_ledger import SignalLedger

    ledger = SignalLedger("data/political_alpha.db")
    ledger.initialize()
    capped = max(1, min(100, int(limit)))
    rows = ledger.list_recent(limit=capped)
    return json.dumps([
        {
            "entry_ts": r.entry_ts.isoformat(),
            "persona": r.persona,
            "side": r.trade_side,
            "ticker": r.trade_ticker,
            "size_pct": r.trade_size_pct,
            "horizon_hours": r.horizon_hours,
            "trigger_headline": r.trigger_headline,
            "hypothesis": r.hypothesis,
            "realized_pnl_pct": r.realized_pnl_pct,
        }
        for r in rows
    ])


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
