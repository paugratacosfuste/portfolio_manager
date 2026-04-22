"""Tests for core/types.py — immutable data types for v3."""
import json

import pytest
from pydantic import ValidationError

from core.types import (
    DebateVerdict,
    Holding,
    LedgerEntry,
    Portfolio,
)


# ── Holding ───────────────────────────────────────────────────────────────────


class TestHolding:
    def test_construct_with_all_fields(self):
        h = Holding(
            ticker="AAPL",
            quantity=10.0,
            asset_class="Stocks",
            sector="Technology",
            region="United States",
        )
        assert h.ticker == "AAPL"
        assert h.quantity == 10.0

    def test_construct_with_optional_fields_none(self):
        h = Holding(ticker="BTC-USD", quantity=0.5, asset_class="Crypto")
        assert h.sector is None
        assert h.region is None

    def test_is_frozen(self):
        h = Holding(ticker="AAPL", quantity=10.0, asset_class="Stocks")
        with pytest.raises(ValidationError):
            h.quantity = 99.0  # type: ignore[misc]

    def test_quantity_must_be_nonnegative(self):
        with pytest.raises(ValidationError):
            Holding(ticker="AAPL", quantity=-1.0, asset_class="Stocks")

    def test_ticker_must_not_be_empty(self):
        with pytest.raises(ValidationError):
            Holding(ticker="", quantity=10.0, asset_class="Stocks")

    def test_json_roundtrip(self):
        h = Holding(ticker="AAPL", quantity=10.0, asset_class="Stocks", sector="Tech")
        payload = h.model_dump_json()
        restored = Holding.model_validate_json(payload)
        assert restored == h


# ── Portfolio ─────────────────────────────────────────────────────────────────


class TestPortfolio:
    def test_construct_empty(self):
        p = Portfolio(holdings=())
        assert p.holdings == ()
        assert p.cash_usd == 0.0

    def test_construct_with_holdings(self):
        h1 = Holding(ticker="AAPL", quantity=10.0, asset_class="Stocks")
        h2 = Holding(ticker="MSFT", quantity=5.0, asset_class="Stocks")
        p = Portfolio(holdings=(h1, h2), cash_usd=1000.0)
        assert len(p.holdings) == 2
        assert p.cash_usd == 1000.0

    def test_is_frozen(self):
        p = Portfolio(holdings=())
        with pytest.raises(ValidationError):
            p.cash_usd = 99.0  # type: ignore[misc]

    def test_cash_cannot_be_negative(self):
        with pytest.raises(ValidationError):
            Portfolio(holdings=(), cash_usd=-1.0)

    def test_from_dict_builds_portfolio(self):
        holdings_dict = {"AAPL": 10.0, "MSFT": 5.0, "BTC-USD": 0.1}
        p = Portfolio.from_dict(holdings_dict)
        tickers = sorted(h.ticker for h in p.holdings)
        assert tickers == ["AAPL", "BTC-USD", "MSFT"]
        assert all(h.asset_class for h in p.holdings)

    def test_from_dict_empty(self):
        p = Portfolio.from_dict({})
        assert p.holdings == ()

    def test_from_dict_skips_zero_quantities(self):
        p = Portfolio.from_dict({"AAPL": 10.0, "MSFT": 0.0})
        assert len(p.holdings) == 1
        assert p.holdings[0].ticker == "AAPL"

    def test_tickers_helper(self):
        h1 = Holding(ticker="AAPL", quantity=10.0, asset_class="Stocks")
        h2 = Holding(ticker="MSFT", quantity=5.0, asset_class="Stocks")
        p = Portfolio(holdings=(h1, h2))
        assert p.tickers() == ("AAPL", "MSFT")

    def test_quantity_map(self):
        h1 = Holding(ticker="AAPL", quantity=10.0, asset_class="Stocks")
        h2 = Holding(ticker="MSFT", quantity=5.0, asset_class="Stocks")
        p = Portfolio(holdings=(h1, h2))
        assert p.quantity_map() == {"AAPL": 10.0, "MSFT": 5.0}


# ── LedgerEntry ───────────────────────────────────────────────────────────────


class TestLedgerEntry:
    def _valid_kwargs(self, **overrides):
        base = dict(
            entry_ts="2026-04-22T10:00:00Z",
            trigger_headline="Trump announces 25% tariff on Chinese imports",
            trigger_source="truth_social",
            persona="trump",
            sentiment=-0.72,
            trade_ticker="SPY",
            trade_side="SHORT",
            trade_size_pct=0.05,
            horizon_hours=24,
            hypothesis="Tariff announcement historically drives SPY down 1-2% over 24h as traders price in supply-chain hit to multinationals.",
        )
        base.update(overrides)
        return base

    def test_construct_open_position(self):
        e = LedgerEntry(**self._valid_kwargs())
        assert e.trade_side == "SHORT"
        assert e.close_ts is None
        assert e.realized_pnl_pct is None

    def test_construct_closed_position(self):
        e = LedgerEntry(
            **self._valid_kwargs(),
            close_ts="2026-04-23T10:00:00Z",
            realized_pnl_pct=-0.015,
        )
        assert e.close_ts == "2026-04-23T10:00:00Z"
        assert e.realized_pnl_pct == -0.015

    def test_is_frozen(self):
        e = LedgerEntry(**self._valid_kwargs())
        with pytest.raises(ValidationError):
            e.sentiment = 0.0  # type: ignore[misc]

    def test_trade_side_must_be_enum(self):
        with pytest.raises(ValidationError):
            LedgerEntry(**self._valid_kwargs(trade_side="HOLD"))

    def test_sentiment_range(self):
        with pytest.raises(ValidationError):
            LedgerEntry(**self._valid_kwargs(sentiment=2.0))
        with pytest.raises(ValidationError):
            LedgerEntry(**self._valid_kwargs(sentiment=-2.0))

    def test_size_pct_cap(self):
        with pytest.raises(ValidationError):
            LedgerEntry(**self._valid_kwargs(trade_size_pct=0.15))  # > 10% cap

    def test_horizon_hours_range(self):
        with pytest.raises(ValidationError):
            LedgerEntry(**self._valid_kwargs(horizon_hours=0))
        with pytest.raises(ValidationError):
            LedgerEntry(**self._valid_kwargs(horizon_hours=200))  # > 1 week

    def test_hypothesis_min_length(self):
        with pytest.raises(ValidationError):
            LedgerEntry(**self._valid_kwargs(hypothesis="too short"))

    def test_json_roundtrip(self):
        e = LedgerEntry(**self._valid_kwargs())
        restored = LedgerEntry.model_validate_json(e.model_dump_json())
        assert restored == e


# ── DebateVerdict ─────────────────────────────────────────────────────────────


class TestDebateVerdict:
    def _valid_kwargs(self, **overrides):
        base = dict(
            decision="BUY",
            conviction=4,
            disagreements_resolved=("Bull/Bear disagreed on growth durability — sided with Bull based on earnings trend",),
            dissenting_view_acknowledged="Risk Officer flagged concentration risk, accepted but deemed below threshold",
            decision_change_condition="Would flip to SELL if next earnings miss by >5% or if tech sector correlation spikes above 0.9",
            summary="BUY with 4/5 conviction after weighing disagreement.",
            transcript_json="[]",
        )
        base.update(overrides)
        return base

    def test_construct_high_conviction(self):
        v = DebateVerdict(**self._valid_kwargs())
        assert v.decision == "BUY"
        assert v.conviction == 4

    def test_is_frozen(self):
        v = DebateVerdict(**self._valid_kwargs())
        with pytest.raises(ValidationError):
            v.conviction = 1  # type: ignore[misc]

    def test_decision_must_be_enum(self):
        with pytest.raises(ValidationError):
            DebateVerdict(**self._valid_kwargs(decision="MAYBE"))

    def test_conviction_range(self):
        with pytest.raises(ValidationError):
            DebateVerdict(**self._valid_kwargs(conviction=0))
        with pytest.raises(ValidationError):
            DebateVerdict(**self._valid_kwargs(conviction=6))

    def test_disagreements_resolved_not_empty(self):
        with pytest.raises(ValidationError):
            DebateVerdict(**self._valid_kwargs(disagreements_resolved=()))

    def test_dissenting_view_min_length(self):
        with pytest.raises(ValidationError):
            DebateVerdict(**self._valid_kwargs(dissenting_view_acknowledged="too short"))

    def test_decision_change_condition_min_length(self):
        with pytest.raises(ValidationError):
            DebateVerdict(**self._valid_kwargs(decision_change_condition="short"))

    def test_low_conviction_requires_detailed_condition(self):
        """Conviction < 3 requires a LONGER change condition (>=80 chars)."""
        short_condition = "x" * 40
        with pytest.raises(ValidationError):
            DebateVerdict(**self._valid_kwargs(conviction=2, decision_change_condition=short_condition))

    def test_low_conviction_with_detailed_condition_passes(self):
        long_condition = "Would flip if " + "x" * 80
        v = DebateVerdict(**self._valid_kwargs(conviction=2, decision_change_condition=long_condition))
        assert v.conviction == 2

    def test_json_roundtrip(self):
        v = DebateVerdict(**self._valid_kwargs())
        restored = DebateVerdict.model_validate_json(v.model_dump_json())
        assert restored == v

    def test_transcript_json_is_parseable(self):
        v = DebateVerdict(**self._valid_kwargs(transcript_json=json.dumps([{"round": 1, "agent": "bull"}])))
        parsed = json.loads(v.transcript_json)
        assert parsed[0]["agent"] == "bull"
