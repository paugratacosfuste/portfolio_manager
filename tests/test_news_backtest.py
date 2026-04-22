"""Tests for utils/news_backtest.py — offline replay engine + scorecard.

Replay takes a chronological CSV of curated headlines, runs each through
a policy function, and mark-to-markets each proposed trade. The engine
is fully deterministic — the caller injects a `price_fn` so tests don't
hit yfinance.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

import pytest

from core.news_types import AssetImpact, LedgerEntry, NewsEvent, Persona, hash_headline
from utils.news_backtest import (
    BacktestReport,
    PersonaScorecard,
    compute_scorecard,
    realized_pnl_pct,
    replay_headlines,
)


UTC = timezone.utc


@pytest.fixture
def headlines_csv(tmp_path: Path) -> Path:
    p = tmp_path / "timeline.csv"
    p.write_text(
        dedent(
            """\
            ts,persona,headline,source_url,expected_tickers,notes
            2024-01-02T15:00:00+00:00,trump,Trump threatens 60% tariff on Chinese EVs,http://x/1,SPY,
            2024-02-10T14:30:00+00:00,powell,Powell signals hawkish pivot at FOMC,http://x/2,TLT,
            2024-03-05T12:00:00+00:00,trump,Trump announces Truth Social deal,http://x/3,DJT,off-topic
            """
        )
    )
    return p


@pytest.fixture
def trump(monkeypatch) -> Persona:
    return Persona(
        id="trump", name="Donald Trump", role="US President", tier=1,
        watched_sources=("gdelt", "truth_social"),
        keywords=("Trump",),
        high_salience_terms=("tariff",),
        primary_asset_impacts=(
            AssetImpact(asset="SPY", direction_on_positive="LONG"),
        ),
        historical_reaction_horizon_hours=24,
    )


# ── realized_pnl_pct helper ───────────────────────────────────────────────────


class TestRealizedPnl:
    def test_long_profit(self):
        assert realized_pnl_pct(side="LONG", entry=100.0, exit_=105.0) == pytest.approx(5.0)

    def test_long_loss(self):
        assert realized_pnl_pct(side="LONG", entry=100.0, exit_=95.0) == pytest.approx(-5.0)

    def test_short_profit(self):
        assert realized_pnl_pct(side="SHORT", entry=100.0, exit_=95.0) == pytest.approx(5.0)

    def test_short_loss(self):
        assert realized_pnl_pct(side="SHORT", entry=100.0, exit_=105.0) == pytest.approx(-5.0)

    def test_zero_entry_safe(self):
        # Degenerate: return 0 rather than blow up
        assert realized_pnl_pct(side="LONG", entry=0.0, exit_=10.0) == 0.0


# ── compute_scorecard ─────────────────────────────────────────────────────────


def _row(persona="trump", side="LONG", ticker="SPY", pnl=1.0, horizon=24):
    return LedgerEntry(
        entry_ts=datetime(2024, 1, 2, 10, 0, tzinfo=UTC),
        trigger_headline="h",
        trigger_source="gdelt",
        persona=persona,
        sentiment=0.5,
        trade_ticker=ticker,
        trade_side=side,
        trade_size_pct=0.05,
        horizon_hours=horizon,
        hypothesis="x" * 60,
        close_ts=datetime(2024, 1, 3, 10, 0, tzinfo=UTC),
        realized_pnl_pct=pnl,
    )


class TestComputeScorecard:
    def test_basic_counts(self):
        rows = [_row(pnl=2.0), _row(pnl=-1.0), _row(pnl=3.0)]
        card = compute_scorecard(rows, persona="trump")
        assert card.total_trades == 3
        assert card.wins == 2
        assert card.losses == 1
        assert card.hit_rate == pytest.approx(2 / 3)
        assert card.avg_pnl_pct == pytest.approx((2.0 - 1.0 + 3.0) / 3)

    def test_profit_factor(self):
        rows = [_row(pnl=3.0), _row(pnl=-1.0), _row(pnl=-0.5)]
        card = compute_scorecard(rows, persona="trump")
        # sum(wins) / |sum(losses)| = 3 / 1.5 = 2.0
        assert card.profit_factor == pytest.approx(2.0)

    def test_all_wins_profit_factor_infinite(self):
        rows = [_row(pnl=1.0), _row(pnl=2.0)]
        card = compute_scorecard(rows, persona="trump")
        assert card.profit_factor == float("inf")

    def test_empty_returns_zero_card(self):
        card = compute_scorecard([], persona="trump")
        assert card.total_trades == 0
        assert card.hit_rate == 0.0
        assert card.avg_pnl_pct == 0.0

    def test_skips_excluded_from_trade_stats(self):
        rows = [
            _row(pnl=1.0),
            LedgerEntry(
                entry_ts=datetime(2024, 1, 2, 10, 0, tzinfo=UTC),
                trigger_headline="h", trigger_source="gdelt", persona="trump",
                sentiment=0.0, trade_ticker="NONE", trade_side="SKIP",
                trade_size_pct=0.0, horizon_hours=1, hypothesis="SKIP: x" * 10,
            ),
        ]
        card = compute_scorecard(rows, persona="trump")
        assert card.total_trades == 1
        assert card.skips == 1


# ── replay_headlines ──────────────────────────────────────────────────────────


class TestReplay:
    def test_replay_generates_ledger_entries(
        self, headlines_csv: Path, trump: Persona, tmp_path: Path
    ):
        # Fake policy: always propose LONG SPY 5% for tier-1 events, skip others
        def policy(event: NewsEvent, ledger, portfolio) -> LedgerEntry | None:
            if event.persona_id == "trump" and "tariff" in event.headline.lower():
                entry = LedgerEntry(
                    entry_ts=event.ts,
                    trigger_headline=event.headline,
                    trigger_source=event.source,
                    persona=event.persona_id,
                    sentiment=0.5,
                    trade_ticker="SPY",
                    trade_side="SHORT",
                    trade_size_pct=0.05,
                    horizon_hours=24,
                    hypothesis="tariff -> risk-off. base rate ~60%. falsifier: fed pivot.",
                )
            else:
                entry = LedgerEntry(
                    entry_ts=event.ts,
                    trigger_headline=event.headline,
                    trigger_source=event.source,
                    persona=event.persona_id or "unknown",
                    sentiment=0.0, trade_ticker="NONE", trade_side="SKIP",
                    trade_size_pct=0.0, horizon_hours=1,
                    hypothesis="SKIP: off-topic",
                )
            ledger.append(entry)
            return entry

        # Fake price_fn: SPY rallies 2% so our SHORT loses 2%
        def price_fn(ticker: str, entry_ts, close_ts) -> float:
            return -2.0  # LONG loss by 2% -> short would be +2% but replay flips it

        report = replay_headlines(
            timeline_csv=headlines_csv,
            personas=(trump,),
            policy_fn=policy,
            ledger_path=tmp_path / "replay.db",
            price_fn=price_fn,
        )

        assert isinstance(report, BacktestReport)
        assert report.total_events == 3
        # At least one trade was proposed (the tariff one)
        assert report.total_trades >= 1
        # Per-persona scorecard present
        assert "trump" in report.per_persona

    def test_replay_skips_rows_without_persona_match(
        self, tmp_path: Path, trump: Persona
    ):
        csv_path = tmp_path / "weird.csv"
        csv_path.write_text(
            "ts,persona,headline,source_url,expected_tickers,notes\n"
            "2024-01-02T15:00:00+00:00,unknown,Some news,http://x,,\n"
        )

        calls: list[NewsEvent] = []

        def policy(event, ledger, portfolio) -> LedgerEntry | None:
            calls.append(event)
            return None

        report = replay_headlines(
            timeline_csv=csv_path, personas=(trump,),
            policy_fn=policy, ledger_path=tmp_path / "w.db",
            price_fn=lambda *a, **kw: 0.0,
        )
        # Unknown persona rows still reach the policy (agent decides)
        assert report.total_events == 1


class TestPersonaScorecardFrozen:
    def test_is_immutable(self):
        card = PersonaScorecard(
            persona="trump", total_trades=1, skips=0, wins=1, losses=0,
            hit_rate=1.0, avg_pnl_pct=1.0, median_holding_hours=24.0,
            sharpe_annualized=0.0, max_drawdown_pct=0.0, profit_factor=float("inf"),
        )
        with pytest.raises(Exception):
            card.hit_rate = 0.5  # type: ignore[misc]
