"""Tests for core/tool_schemas.py — Pydantic response schemas for LLM tools.

These schemas pin the output contract between our tool executors and any
LLM (chatbot, MCP server, future Vercel adapter). Frozen + validated, so
a typo or missing field fails loudly instead of silently breaking Claude.
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

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


# ── ErrorResponse ─────────────────────────────────────────────────────────────


class TestErrorResponse:
    def test_construct(self):
        e = ErrorResponse(error="No holdings in portfolio.")
        assert e.error == "No holdings in portfolio."

    def test_is_frozen(self):
        e = ErrorResponse(error="x")
        with pytest.raises(ValidationError):
            e.error = "y"  # type: ignore[misc]

    def test_error_cannot_be_empty(self):
        with pytest.raises(ValidationError):
            ErrorResponse(error="")

    def test_roundtrip(self):
        e = ErrorResponse(error="Portfolio empty")
        restored = ErrorResponse.model_validate_json(e.model_dump_json())
        assert restored == e


# ── AssetPriceResponse ────────────────────────────────────────────────────────


class TestAssetPriceResponse:
    def test_construct(self):
        r = AssetPriceResponse(ticker="AAPL", price=150.0)
        assert r.ticker == "AAPL"
        assert r.price == 150.0

    def test_ticker_not_empty(self):
        with pytest.raises(ValidationError):
            AssetPriceResponse(ticker="", price=150.0)

    def test_price_nonneg(self):
        with pytest.raises(ValidationError):
            AssetPriceResponse(ticker="AAPL", price=-1.0)
        # Zero is allowed (missing-price fallback)
        assert AssetPriceResponse(ticker="AAPL", price=0.0).price == 0.0

    def test_is_frozen(self):
        r = AssetPriceResponse(ticker="AAPL", price=150.0)
        with pytest.raises(ValidationError):
            r.price = 200.0  # type: ignore[misc]


# ── HoldingRow + PortfolioSummaryResponse ─────────────────────────────────────


class TestHoldingRow:
    def _valid(self, **overrides):
        base = dict(
            ticker="AAPL",
            quantity=10.0,
            price=150.0,
            value=1500.0,
            weight_pct=37.5,
            asset_class="Stocks",
            sector="Technology",
        )
        base.update(overrides)
        return base

    def test_construct(self):
        row = HoldingRow(**self._valid())
        assert row.ticker == "AAPL"
        assert row.weight_pct == 37.5

    def test_quantity_nonneg(self):
        with pytest.raises(ValidationError):
            HoldingRow(**self._valid(quantity=-1.0))

    def test_weight_pct_range(self):
        # Allow 0..100 inclusive
        HoldingRow(**self._valid(weight_pct=0.0))
        HoldingRow(**self._valid(weight_pct=100.0))
        with pytest.raises(ValidationError):
            HoldingRow(**self._valid(weight_pct=-0.1))
        with pytest.raises(ValidationError):
            HoldingRow(**self._valid(weight_pct=100.01))


class TestPortfolioSummaryResponse:
    def _row(self, ticker="AAPL"):
        return HoldingRow(
            ticker=ticker, quantity=10.0, price=150.0, value=1500.0,
            weight_pct=50.0, asset_class="Stocks", sector="Technology",
        )

    def test_construct_with_holdings(self):
        r = PortfolioSummaryResponse(
            total_value=3000.0,
            holdings=(self._row("AAPL"), self._row("MSFT")),
        )
        assert r.total_value == 3000.0
        assert len(r.holdings) == 2

    def test_construct_empty(self):
        r = PortfolioSummaryResponse(total_value=0.0, holdings=())
        assert r.holdings == ()

    def test_total_value_nonneg(self):
        with pytest.raises(ValidationError):
            PortfolioSummaryResponse(total_value=-1.0, holdings=())

    def test_roundtrip(self):
        r = PortfolioSummaryResponse(
            total_value=1500.0, holdings=(self._row(),),
        )
        restored = PortfolioSummaryResponse.model_validate_json(r.model_dump_json())
        assert restored == r


# ── RiskMetricsResponse ───────────────────────────────────────────────────────


class TestRiskMetricsResponse:
    def _valid(self, **overrides):
        base = dict(
            total_value=10000.0,
            volatility_pct=15.0,
            beta=1.1,
            hhi=5000.0,
            risk_score=50,
            sharpe_ratio=0.8,
            max_drawdown_pct=-12.5,
            cvar_95_pct=-3.2,
        )
        base.update(overrides)
        return base

    def test_construct(self):
        r = RiskMetricsResponse(**self._valid())
        assert r.risk_score == 50
        assert r.beta == 1.1

    def test_risk_score_range(self):
        with pytest.raises(ValidationError):
            RiskMetricsResponse(**self._valid(risk_score=-1))
        with pytest.raises(ValidationError):
            RiskMetricsResponse(**self._valid(risk_score=101))

    def test_hhi_range(self):
        with pytest.raises(ValidationError):
            RiskMetricsResponse(**self._valid(hhi=-1.0))
        with pytest.raises(ValidationError):
            RiskMetricsResponse(**self._valid(hhi=10001.0))

    def test_max_dd_nonpositive(self):
        # Drawdown is a loss — must be ≤ 0
        RiskMetricsResponse(**self._valid(max_drawdown_pct=0.0))
        RiskMetricsResponse(**self._valid(max_drawdown_pct=-50.0))
        with pytest.raises(ValidationError):
            RiskMetricsResponse(**self._valid(max_drawdown_pct=5.0))

    def test_roundtrip(self):
        r = RiskMetricsResponse(**self._valid())
        restored = RiskMetricsResponse.model_validate_json(r.model_dump_json())
        assert restored == r


# ── WhatIfResponse ────────────────────────────────────────────────────────────


class TestWhatIfResponse:
    def _valid(self, **overrides):
        base = dict(
            action="add",
            ticker="TSLA",
            quantity=5.0,
            new_total_value=5000.0,
            new_volatility_pct=18.0,
            new_sharpe_ratio=0.6,
            new_max_drawdown_pct=-15.0,
            new_weights={"AAPL": 54.5, "TSLA": 45.5},
        )
        base.update(overrides)
        return base

    def test_construct(self):
        r = WhatIfResponse(**self._valid())
        assert r.action == "add"
        assert r.new_weights["TSLA"] == 45.5

    def test_action_enum(self):
        with pytest.raises(ValidationError):
            WhatIfResponse(**self._valid(action="swap"))

    def test_quantity_positive(self):
        with pytest.raises(ValidationError):
            WhatIfResponse(**self._valid(quantity=0.0))
        with pytest.raises(ValidationError):
            WhatIfResponse(**self._valid(quantity=-5.0))

    def test_new_total_value_nonneg(self):
        with pytest.raises(ValidationError):
            WhatIfResponse(**self._valid(new_total_value=-1.0))


# ── MarketNewsResponse ────────────────────────────────────────────────────────


class TestMarketNewsResponse:
    def test_construct_with_items(self):
        item = NewsItem(
            title="Apple launches iPhone",
            link="https://example.com/a",
            publisher="Reuters",
            timestamp="2026-04-22 10:00",
        )
        r = MarketNewsResponse(ticker="AAPL", news=(item,))
        assert r.ticker == "AAPL"
        assert len(r.news) == 1

    def test_construct_empty_news(self):
        r = MarketNewsResponse(ticker="AAPL", news=())
        assert r.news == ()

    def test_news_item_title_required(self):
        with pytest.raises(ValidationError):
            NewsItem(title="", link="x", publisher="y", timestamp="z")

    def test_roundtrip(self):
        item = NewsItem(title="T", link="L", publisher="P", timestamp="2026-04-22")
        r = MarketNewsResponse(ticker="AAPL", news=(item,))
        restored = MarketNewsResponse.model_validate_json(r.model_dump_json())
        assert restored == r


# ── MacroPredictionResponse ───────────────────────────────────────────────────


class TestMacroPredictionResponse:
    def _valid(self, **overrides):
        base = dict(
            prediction="Market correction likely",
            correction_probability_pct=72.5,
            current_vix=22.3,
            sp500_vs_200ma_pct=-3.1,
        )
        base.update(overrides)
        return base

    def test_construct(self):
        r = MacroPredictionResponse(**self._valid())
        assert r.prediction.startswith("Market")

    def test_probability_range(self):
        with pytest.raises(ValidationError):
            MacroPredictionResponse(**self._valid(correction_probability_pct=-0.1))
        with pytest.raises(ValidationError):
            MacroPredictionResponse(**self._valid(correction_probability_pct=100.01))

    def test_vix_nonneg(self):
        with pytest.raises(ValidationError):
            MacroPredictionResponse(**self._valid(current_vix=-1.0))

    def test_roundtrip(self):
        r = MacroPredictionResponse(**self._valid())
        restored = MacroPredictionResponse.model_validate_json(r.model_dump_json())
        assert restored == r


# ── End-to-end: schemas serialize to expected JSON keys ───────────────────────


class TestJsonShape:
    """Verify the JSON shape matches what existing chatbot consumers expect.

    These are contract checks. If a schema change would break the downstream
    chatbot prompt, one of these assertions will fail first.
    """

    def test_portfolio_summary_keys(self):
        r = PortfolioSummaryResponse(total_value=100.0, holdings=())
        payload = json.loads(r.model_dump_json())
        assert set(payload.keys()) == {"total_value", "holdings"}

    def test_risk_metrics_keys(self):
        r = RiskMetricsResponse(
            total_value=0.0, volatility_pct=0.0, beta=1.0, hhi=0.0,
            risk_score=50, sharpe_ratio=0.0, max_drawdown_pct=0.0, cvar_95_pct=0.0,
        )
        keys = set(json.loads(r.model_dump_json()).keys())
        assert keys == {
            "total_value", "volatility_pct", "beta", "hhi",
            "risk_score", "sharpe_ratio", "max_drawdown_pct", "cvar_95_pct",
        }

    def test_what_if_keys(self):
        r = WhatIfResponse(
            action="add", ticker="TSLA", quantity=5.0,
            new_total_value=1.0, new_volatility_pct=0.0,
            new_sharpe_ratio=0.0, new_max_drawdown_pct=0.0, new_weights={},
        )
        keys = set(json.loads(r.model_dump_json()).keys())
        assert keys == {
            "action", "ticker", "quantity", "new_total_value",
            "new_volatility_pct", "new_sharpe_ratio",
            "new_max_drawdown_pct", "new_weights",
        }
