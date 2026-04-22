"""Tests for core/portfolio_ops.py — framework-agnostic portfolio operations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.portfolio_ops import (
    apply_stress,
    compute_risk_metrics,
    compute_weights,
    portfolio_summary,
    simulate_trade,
)
from core.types import Holding, Portfolio


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_portfolio() -> Portfolio:
    return Portfolio(
        holdings=(
            Holding(ticker="AAPL", quantity=10.0, asset_class="Stocks", sector="Technology"),
            Holding(ticker="MSFT", quantity=5.0, asset_class="Stocks", sector="Technology"),
            Holding(ticker="BTC-USD", quantity=0.5, asset_class="Crypto"),
        ),
        cash_usd=1000.0,
    )


@pytest.fixture
def sample_prices() -> dict[str, float]:
    return {"AAPL": 200.0, "MSFT": 400.0, "BTC-USD": 60000.0}


@pytest.fixture
def historical_prices() -> pd.DataFrame:
    """60 trading days of synthetic prices for AAPL, MSFT, BTC-USD."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-01-01", periods=60, freq="B")
    data = {
        "AAPL": 200.0 * (1 + rng.normal(0.0005, 0.02, 60)).cumprod(),
        "MSFT": 400.0 * (1 + rng.normal(0.0004, 0.018, 60)).cumprod(),
        "BTC-USD": 60000.0 * (1 + rng.normal(0.001, 0.04, 60)).cumprod(),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def market_returns(historical_prices: pd.DataFrame) -> pd.Series:
    rng = np.random.default_rng(7)
    return pd.Series(
        100.0 * (1 + rng.normal(0.0004, 0.012, 60)).cumprod(),
        index=historical_prices.index,
        name="SPY",
    )


# ── portfolio_summary ─────────────────────────────────────────────────────────


class TestPortfolioSummary:
    def test_total_value_includes_cash_and_holdings(self, sample_portfolio, sample_prices):
        s = portfolio_summary(sample_portfolio, sample_prices)
        # AAPL: 10 * 200 = 2000
        # MSFT: 5 * 400 = 2000
        # BTC-USD: 0.5 * 60000 = 30000
        # cash: 1000
        # total = 35000
        assert s["total_value"] == pytest.approx(35000.0)
        assert s["total_cash"] == pytest.approx(1000.0)

    def test_holdings_have_value_and_weight(self, sample_portfolio, sample_prices):
        s = portfolio_summary(sample_portfolio, sample_prices)
        holdings_by_ticker = {h["ticker"]: h for h in s["holdings"]}
        assert holdings_by_ticker["AAPL"]["value"] == pytest.approx(2000.0)
        assert holdings_by_ticker["AAPL"]["quantity"] == 10.0
        assert holdings_by_ticker["AAPL"]["price"] == 200.0
        # Weights should be fraction of total_value (including cash)
        assert holdings_by_ticker["AAPL"]["weight"] == pytest.approx(2000.0 / 35000.0)
        assert holdings_by_ticker["BTC-USD"]["weight"] == pytest.approx(30000.0 / 35000.0)

    def test_asset_class_breakdown(self, sample_portfolio, sample_prices):
        s = portfolio_summary(sample_portfolio, sample_prices)
        breakdown = s["asset_class_breakdown"]
        # Stocks: 4000 / 34000 (holdings-only denominator)
        # Crypto: 30000 / 34000
        assert breakdown["Stocks"] == pytest.approx(4000.0 / 34000.0)
        assert breakdown["Crypto"] == pytest.approx(30000.0 / 34000.0)

    def test_empty_portfolio(self):
        p = Portfolio(holdings=(), cash_usd=0.0)
        s = portfolio_summary(p, {})
        assert s["total_value"] == 0.0
        assert s["holdings"] == []
        assert s["asset_class_breakdown"] == {}

    def test_cash_only_portfolio(self):
        p = Portfolio(holdings=(), cash_usd=500.0)
        s = portfolio_summary(p, {})
        assert s["total_value"] == 500.0
        assert s["total_cash"] == 500.0
        assert s["holdings"] == []

    def test_missing_price_treated_as_zero(self, sample_portfolio):
        # Only AAPL has a price; MSFT and BTC-USD missing
        s = portfolio_summary(sample_portfolio, {"AAPL": 200.0})
        # AAPL 2000 + cash 1000 = 3000
        assert s["total_value"] == pytest.approx(3000.0)
        aapl = next(h for h in s["holdings"] if h["ticker"] == "AAPL")
        msft = next(h for h in s["holdings"] if h["ticker"] == "MSFT")
        assert aapl["price"] == 200.0
        assert msft["price"] == 0.0
        assert msft["value"] == 0.0


# ── compute_weights ───────────────────────────────────────────────────────────


class TestComputeWeights:
    def test_weights_sum_to_one_when_no_cash(self, sample_prices):
        p = Portfolio(
            holdings=(
                Holding(ticker="AAPL", quantity=10.0, asset_class="Stocks"),
                Holding(ticker="MSFT", quantity=5.0, asset_class="Stocks"),
            ),
            cash_usd=0.0,
        )
        w = compute_weights(p, sample_prices)
        assert sum(w.values()) == pytest.approx(1.0)
        assert w["AAPL"] == pytest.approx(2000.0 / 4000.0)
        assert w["MSFT"] == pytest.approx(2000.0 / 4000.0)

    def test_weights_exclude_cash_from_denominator(self, sample_portfolio, sample_prices):
        """Weights should be holdings-only (caller adds cash weight separately if needed)."""
        w = compute_weights(sample_portfolio, sample_prices)
        # 2000 + 2000 + 30000 = 34000 (holdings total, excludes cash)
        assert sum(w.values()) == pytest.approx(1.0)
        assert w["BTC-USD"] == pytest.approx(30000.0 / 34000.0)

    def test_empty_portfolio_returns_empty(self):
        p = Portfolio(holdings=(), cash_usd=100.0)
        assert compute_weights(p, {}) == {}

    def test_missing_price_weight_zero(self, sample_portfolio):
        w = compute_weights(sample_portfolio, {"AAPL": 200.0})
        # AAPL is the only priced holding; weight = 1.0
        assert w["AAPL"] == pytest.approx(1.0)
        assert w["MSFT"] == pytest.approx(0.0)

    def test_zero_total_value_returns_empty(self):
        p = Portfolio(
            holdings=(Holding(ticker="AAPL", quantity=10.0, asset_class="Stocks"),),
        )
        # Price is 0 → total value is 0 → no weights
        assert compute_weights(p, {"AAPL": 0.0}) == {}


# ── simulate_trade ────────────────────────────────────────────────────────────


class TestSimulateTrade:
    def test_add_to_existing_holding(self, sample_portfolio):
        p2 = simulate_trade(sample_portfolio, "AAPL", qty_delta=5.0)
        aapl = next(h for h in p2.holdings if h.ticker == "AAPL")
        assert aapl.quantity == 15.0

    def test_reduce_existing_holding(self, sample_portfolio):
        p2 = simulate_trade(sample_portfolio, "AAPL", qty_delta=-3.0)
        aapl = next(h for h in p2.holdings if h.ticker == "AAPL")
        assert aapl.quantity == 7.0

    def test_close_holding_removes_it(self, sample_portfolio):
        p2 = simulate_trade(sample_portfolio, "AAPL", qty_delta=-10.0)
        tickers = [h.ticker for h in p2.holdings]
        assert "AAPL" not in tickers
        assert len(p2.holdings) == 2

    def test_overshoot_close_removes_holding(self, sample_portfolio):
        """Selling more than owned closes the position at zero (not negative)."""
        p2 = simulate_trade(sample_portfolio, "AAPL", qty_delta=-50.0)
        tickers = [h.ticker for h in p2.holdings]
        assert "AAPL" not in tickers

    def test_add_new_holding(self, sample_portfolio):
        p2 = simulate_trade(sample_portfolio, "NVDA", qty_delta=10.0, asset_class="Stocks")
        nvda = next(h for h in p2.holdings if h.ticker == "NVDA")
        assert nvda.quantity == 10.0
        assert nvda.asset_class == "Stocks"
        assert len(p2.holdings) == 4

    def test_sell_nonexistent_holding_raises(self, sample_portfolio):
        with pytest.raises(ValueError, match="not in portfolio"):
            simulate_trade(sample_portfolio, "NVDA", qty_delta=-5.0)

    def test_zero_delta_is_noop(self, sample_portfolio):
        p2 = simulate_trade(sample_portfolio, "AAPL", qty_delta=0.0)
        assert p2 == sample_portfolio

    def test_original_portfolio_unchanged(self, sample_portfolio):
        """Immutability: original must not be mutated."""
        original_tickers = tuple(h.ticker for h in sample_portfolio.holdings)
        original_qtys = tuple(h.quantity for h in sample_portfolio.holdings)
        _ = simulate_trade(sample_portfolio, "AAPL", qty_delta=5.0)
        assert tuple(h.ticker for h in sample_portfolio.holdings) == original_tickers
        assert tuple(h.quantity for h in sample_portfolio.holdings) == original_qtys


# ── apply_stress ──────────────────────────────────────────────────────────────


class TestApplyStress:
    def test_single_ticker_drawdown(self, sample_portfolio, sample_prices):
        result = apply_stress(sample_portfolio, sample_prices, {"AAPL": -0.20})
        # AAPL value falls 20%: 2000 → 1600 (−400)
        assert result["before_value"] == pytest.approx(35000.0)
        assert result["after_value"] == pytest.approx(34600.0)
        assert result["change"] == pytest.approx(-400.0)
        assert result["change_pct"] == pytest.approx(-400.0 / 35000.0)

    def test_multi_ticker_drawdown(self, sample_portfolio, sample_prices):
        result = apply_stress(
            sample_portfolio,
            sample_prices,
            {"AAPL": -0.20, "BTC-USD": -0.50},
        )
        # AAPL: -400. BTC: -15000. Total: -15400
        assert result["after_value"] == pytest.approx(35000.0 - 15400.0)

    def test_no_drawdown_returns_unchanged(self, sample_portfolio, sample_prices):
        result = apply_stress(sample_portfolio, sample_prices, {})
        assert result["change"] == pytest.approx(0.0)
        assert result["before_value"] == pytest.approx(result["after_value"])

    def test_empty_portfolio(self):
        p = Portfolio(holdings=(), cash_usd=0.0)
        result = apply_stress(p, {}, {"AAPL": -0.20})
        assert result["before_value"] == 0.0
        assert result["after_value"] == 0.0
        assert result["change"] == 0.0

    def test_cash_is_not_stressed(self):
        p = Portfolio(
            holdings=(Holding(ticker="AAPL", quantity=10.0, asset_class="Stocks"),),
            cash_usd=1000.0,
        )
        result = apply_stress(p, {"AAPL": 200.0}, {"AAPL": -0.50})
        # Holdings: 2000 → 1000 (−1000). Cash unchanged. Total: 3000 → 2000
        assert result["before_value"] == pytest.approx(3000.0)
        assert result["after_value"] == pytest.approx(2000.0)

    def test_per_holding_detail(self, sample_portfolio, sample_prices):
        result = apply_stress(sample_portfolio, sample_prices, {"AAPL": -0.20})
        aapl = next(h for h in result["holdings"] if h["ticker"] == "AAPL")
        assert aapl["before"] == pytest.approx(2000.0)
        assert aapl["after"] == pytest.approx(1600.0)
        msft = next(h for h in result["holdings"] if h["ticker"] == "MSFT")
        assert msft["before"] == pytest.approx(msft["after"])  # untouched


# ── compute_risk_metrics ──────────────────────────────────────────────────────


class TestComputeRiskMetrics:
    def test_returns_all_expected_keys(self, sample_portfolio, sample_prices, historical_prices, market_returns):
        m = compute_risk_metrics(
            sample_portfolio, sample_prices, historical_prices, market_returns
        )
        for key in (
            "volatility",
            "beta",
            "sharpe",
            "max_drawdown",
            "cvar",
            "hhi",
            "risk_score",
        ):
            assert key in m

    def test_volatility_is_nonneg_float(self, sample_portfolio, sample_prices, historical_prices):
        m = compute_risk_metrics(sample_portfolio, sample_prices, historical_prices)
        assert isinstance(m["volatility"], float)
        assert m["volatility"] >= 0.0

    def test_beta_defaults_to_one_without_market(
        self, sample_portfolio, sample_prices, historical_prices
    ):
        m = compute_risk_metrics(sample_portfolio, sample_prices, historical_prices, market_returns=None)
        assert m["beta"] == pytest.approx(1.0)

    def test_hhi_range(self, sample_portfolio, sample_prices, historical_prices):
        m = compute_risk_metrics(sample_portfolio, sample_prices, historical_prices)
        assert 0.0 <= m["hhi"] <= 10000.0

    def test_max_drawdown_nonpositive(self, sample_portfolio, sample_prices, historical_prices):
        m = compute_risk_metrics(sample_portfolio, sample_prices, historical_prices)
        assert m["max_drawdown"] <= 0.0

    def test_risk_score_bounded(self, sample_portfolio, sample_prices, historical_prices, market_returns):
        m = compute_risk_metrics(sample_portfolio, sample_prices, historical_prices, market_returns)
        assert 0 <= m["risk_score"] <= 100

    def test_empty_portfolio_returns_zeros(self):
        p = Portfolio(holdings=(), cash_usd=0.0)
        m = compute_risk_metrics(p, {}, pd.DataFrame())
        assert m["volatility"] == 0.0
        assert m["hhi"] == 0.0
        assert m["max_drawdown"] == 0.0

    def test_single_holding_high_concentration(self, sample_prices, historical_prices):
        p = Portfolio(
            holdings=(Holding(ticker="AAPL", quantity=10.0, asset_class="Stocks"),),
        )
        m = compute_risk_metrics(p, sample_prices, historical_prices)
        # Single holding → HHI = 10000 (max concentration)
        assert m["hhi"] == pytest.approx(10000.0)
