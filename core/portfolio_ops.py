"""Framework-agnostic portfolio operations.

Pure functions over immutable Portfolio/Holding values. No Streamlit,
no caching decorators — callers supply data (prices, historical frames)
explicitly so these functions are maximally testable and reusable across
the Streamlit app, MCP server, and cross-domain adapters.

Numerical primitives (volatility, beta, Sharpe, max_dd, CVaR, HHI,
risk_score) are imported from `utils.portfolio_metrics`, which is pure
NumPy/pandas. This module adds the Portfolio-shaped orchestration layer.
"""
from __future__ import annotations

import pandas as pd

from core.types import Holding, Portfolio
from utils.portfolio_metrics import (
    assess_risk_score,
    calculate_cvar,
    calculate_hhi_index,
    calculate_max_drawdown,
    calculate_portfolio_beta,
    calculate_portfolio_volatility,
    calculate_sharpe_ratio,
)


def portfolio_summary(
    portfolio: Portfolio,
    prices: dict[str, float],
    metadata: dict | None = None,
) -> dict:
    """Return a snapshot dict: total value, per-holding rows, asset-class breakdown.

    Missing prices resolve to 0.0 (never raises). Weights are computed
    against total portfolio value (holdings + cash); asset-class breakdown
    is computed against holdings-only value so cash doesn't dilute it.
    """
    holding_rows: list[dict] = []
    holdings_value = 0.0
    for h in portfolio.holdings:
        price = float(prices.get(h.ticker, 0.0))
        value = price * h.quantity
        holdings_value += value
        holding_rows.append(
            {
                "ticker": h.ticker,
                "quantity": h.quantity,
                "price": price,
                "value": value,
                "asset_class": h.asset_class,
                "sector": h.sector,
                "region": h.region,
            }
        )

    total_value = holdings_value + portfolio.cash_usd

    for row in holding_rows:
        row["weight"] = (row["value"] / total_value) if total_value > 0 else 0.0

    asset_class_breakdown: dict[str, float] = {}
    if holdings_value > 0:
        class_totals: dict[str, float] = {}
        for row in holding_rows:
            cls = row["asset_class"] or "Unknown"
            class_totals[cls] = class_totals.get(cls, 0.0) + row["value"]
        asset_class_breakdown = {
            cls: v / holdings_value for cls, v in class_totals.items()
        }

    return {
        "total_value": total_value,
        "total_cash": portfolio.cash_usd,
        "holdings_value": holdings_value,
        "holdings": holding_rows,
        "asset_class_breakdown": asset_class_breakdown,
        "metadata": metadata or {},
    }


def compute_weights(
    portfolio: Portfolio, prices: dict[str, float]
) -> dict[str, float]:
    """Return holdings-only weight map {ticker: weight}, sums to 1.0.

    Cash is NOT included in the denominator — callers that need a
    cash-inclusive view should compute it from `portfolio_summary`.
    Returns empty dict if holdings value is zero (all-cash or no prices).
    """
    values: dict[str, float] = {}
    total = 0.0
    for h in portfolio.holdings:
        price = float(prices.get(h.ticker, 0.0))
        v = price * h.quantity
        values[h.ticker] = v
        total += v

    if total <= 0.0:
        return {}
    return {t: v / total for t, v in values.items()}


def simulate_trade(
    portfolio: Portfolio,
    ticker: str,
    qty_delta: float,
    asset_class: str = "Unknown",
) -> Portfolio:
    """Return a new Portfolio with `qty_delta` added to `ticker`.

    Behavior:
      • qty_delta = 0 → returns the original portfolio unchanged.
      • Existing ticker + positive delta → quantity increases.
      • Existing ticker + negative delta → quantity decreases; if the
        result is ≤ 0 the holding is removed (position closed).
      • New ticker + positive delta → new Holding appended with the
        given asset_class.
      • New ticker + negative delta → ValueError (can't sell what you
        don't own).
    """
    if qty_delta == 0.0:
        return portfolio

    existing = {h.ticker: h for h in portfolio.holdings}
    if ticker not in existing and qty_delta < 0:
        raise ValueError(f"Ticker {ticker!r} not in portfolio; cannot sell.")

    new_holdings: list[Holding] = []
    matched = False
    for h in portfolio.holdings:
        if h.ticker == ticker:
            matched = True
            new_qty = h.quantity + qty_delta
            if new_qty > 0:
                new_holdings.append(
                    Holding(
                        ticker=h.ticker,
                        quantity=new_qty,
                        asset_class=h.asset_class,
                        sector=h.sector,
                        region=h.region,
                    )
                )
            # else: position closed — drop it
        else:
            new_holdings.append(h)

    if not matched:
        new_holdings.append(
            Holding(ticker=ticker, quantity=qty_delta, asset_class=asset_class)
        )

    return Portfolio(holdings=tuple(new_holdings), cash_usd=portfolio.cash_usd)


def apply_stress(
    portfolio: Portfolio,
    prices: dict[str, float],
    drawdowns: dict[str, float],
) -> dict:
    """Apply per-ticker price drawdowns and return before/after valuation.

    `drawdowns` maps ticker → decimal drop (e.g. -0.20 for -20%). Tickers
    absent from `drawdowns` are unchanged. Cash is never stressed.
    """
    before_total = portfolio.cash_usd
    after_total = portfolio.cash_usd
    rows: list[dict] = []

    for h in portfolio.holdings:
        price = float(prices.get(h.ticker, 0.0))
        before = price * h.quantity
        dd = float(drawdowns.get(h.ticker, 0.0))
        after = before * (1.0 + dd)
        before_total += before
        after_total += after
        rows.append(
            {
                "ticker": h.ticker,
                "before": before,
                "after": after,
                "drawdown": dd,
            }
        )

    change = after_total - before_total
    change_pct = (change / before_total) if before_total > 0 else 0.0

    return {
        "before_value": before_total,
        "after_value": after_total,
        "change": change,
        "change_pct": change_pct,
        "holdings": rows,
    }


def compute_risk_metrics(
    portfolio: Portfolio,
    prices: dict[str, float],
    historical_prices: pd.DataFrame,
    market_returns: pd.Series | None = None,
    risk_free_rate: float = 0.05,
) -> dict:
    """Compute the full risk dashboard for a portfolio.

    Returns a dict with volatility (annualized), beta (vs market, 1.0 if
    no market series), Sharpe, max_drawdown, CVaR (95%), HHI, and an
    0-100 aggregate risk_score. Empty portfolios return zeros everywhere.
    """
    zero = {
        "volatility": 0.0,
        "beta": 1.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "cvar": 0.0,
        "hhi": 0.0,
        "risk_score": 0,
        "total_value": 0.0,
    }

    if not portfolio.holdings:
        return zero

    weights = compute_weights(portfolio, prices)
    if not weights:
        return zero

    summary = portfolio_summary(portfolio, prices)
    total_value = summary["total_value"]

    volatility = calculate_portfolio_volatility(historical_prices, weights)
    hhi = calculate_hhi_index(weights)
    sharpe = calculate_sharpe_ratio(historical_prices, weights, risk_free_rate)
    max_dd = calculate_max_drawdown(historical_prices, weights)
    cvar = calculate_cvar(historical_prices, weights)

    if market_returns is not None and not market_returns.empty:
        beta = calculate_portfolio_beta(historical_prices, market_returns, weights)
    else:
        beta = 1.0

    risk_score = assess_risk_score(
        volatility=volatility, hhi=hhi, beta=beta, total_value=total_value
    )

    return {
        "volatility": float(volatility),
        "beta": float(beta),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "cvar": float(cvar),
        "hhi": float(hhi),
        "risk_score": int(risk_score),
        "total_value": float(total_value),
    }
