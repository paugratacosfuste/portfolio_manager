"""Tests for utils/political_alpha_agent.py — the 5th LLM pattern.

The agent consumes one NewsEvent at a time and must either (a) emit a
propose_trade tool call with a TradeProposal or (b) emit a skip. Anthropic
is mocked — we fake the two-turn tool-use dance (first message has a
tool_use block, second returns `end_turn`).
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core.news_types import (
    AssetImpact,
    LedgerEntry,
    NewsEvent,
    Persona,
    hash_headline,
)
from core.tool_schemas import TradeProposal
from utils.political_alpha_agent import (
    AGENT_SYSTEM_PROMPT,
    AGENT_TOOLS,
    run_news_agent_tick,
)
from utils.signal_ledger import SignalLedger


UTC = timezone.utc


@pytest.fixture
def trump_persona() -> Persona:
    return Persona(
        id="trump",
        name="Donald Trump",
        role="US President",
        tier=1,
        watched_sources=("gdelt", "truth_social"),
        keywords=("Trump",),
        high_salience_terms=("tariff", "China"),
        primary_asset_impacts=(
            AssetImpact(asset="SPY", direction_on_positive="LONG"),
            AssetImpact(asset="BTC-USD", direction_on_positive="LONG"),
        ),
        historical_reaction_horizon_hours=24,
    )


@pytest.fixture
def sample_event(trump_persona) -> NewsEvent:
    headline = "Trump announces 50% tariff on Chinese EVs"
    return NewsEvent(
        event_id=hash_headline(headline),
        ts=datetime(2026, 4, 22, 10, 0, tzinfo=UTC),
        source="truth_social",
        persona_id="trump",
        headline=headline,
        url="http://example.com/1",
        raw_text=None,
        salience=0.85,
    )


# ── TradeProposal schema ──────────────────────────────────────────────────────


class TestTradeProposal:
    def test_construct(self):
        tp = TradeProposal(
            ticker="SPY",
            side="SHORT",
            size_pct=0.05,
            horizon_hours=24,
            confidence=0.7,
            hypothesis=(
                "Tariff escalation tends to precede SPY drawdown within 24h "
                "via risk-off flow. Falsifier: Fed dovish pivot."
            ),
            stop_loss_pct=0.02,
        )
        assert tp.side == "SHORT"

    def test_size_cap(self):
        with pytest.raises(Exception):
            TradeProposal(
                ticker="SPY", side="LONG", size_pct=0.11, horizon_hours=24,
                confidence=0.5,
                hypothesis="x" * 50,
            )

    def test_horizon_cap(self):
        with pytest.raises(Exception):
            TradeProposal(
                ticker="SPY", side="LONG", size_pct=0.05, horizon_hours=200,
                confidence=0.5,
                hypothesis="x" * 50,
            )

    def test_hypothesis_minlength(self):
        with pytest.raises(Exception):
            TradeProposal(
                ticker="SPY", side="LONG", size_pct=0.05, horizon_hours=24,
                confidence=0.5,
                hypothesis="too short",
            )


# ── Agent tool schemas shape ──────────────────────────────────────────────────


class TestAgentTools:
    def test_tools_list_present(self):
        names = {t["name"] for t in AGENT_TOOLS}
        assert {
            "score_headline_sentiment",
            "lookup_historical_reaction",
            "check_open_positions",
            "get_current_price",
            "propose_trade",
            "skip",
        }.issubset(names)

    def test_propose_trade_input_schema_enforces_caps(self):
        schema = next(t for t in AGENT_TOOLS if t["name"] == "propose_trade")[
            "input_schema"
        ]
        props = schema["properties"]
        assert props["size_pct"]["maximum"] == 0.10
        assert props["horizon_hours"]["maximum"] == 168

    def test_system_prompt_mentions_hard_rules(self):
        assert "10%" in AGENT_SYSTEM_PROMPT
        assert "168" in AGENT_SYSTEM_PROMPT
        assert "SKIP" in AGENT_SYSTEM_PROMPT


# ── run_news_agent_tick: happy-path propose_trade ─────────────────────────────


def _mk_tool_use_block(tool_name: str, tool_input: dict, tool_id: str = "tu_1"):
    return SimpleNamespace(type="tool_use", name=tool_name, input=tool_input, id=tool_id)


def _mk_text_block(text: str):
    return SimpleNamespace(type="text", text=text)


class TestRunNewsAgentTick:
    def test_propose_trade_writes_to_ledger(
        self, tmp_path, sample_event, trump_persona
    ):
        ledger = SignalLedger(tmp_path / "ledger.db")
        ledger.initialize()

        client = MagicMock()
        # First response: agent calls propose_trade
        first = SimpleNamespace(
            stop_reason="tool_use",
            content=[
                _mk_tool_use_block(
                    "propose_trade",
                    {
                        "ticker": "SPY",
                        "side": "SHORT",
                        "size_pct": 0.04,
                        "horizon_hours": 24,
                        "confidence": 0.7,
                        "hypothesis": (
                            "Tariff escalation historically precedes SPY "
                            "drawdown within 24h via risk-off flow. Falsifier: "
                            "Fed dovish pivot."
                        ),
                        "stop_loss_pct": 0.02,
                    },
                )
            ],
            usage=SimpleNamespace(input_tokens=100, output_tokens=50),
        )
        # Second response (after tool_result fed back): end_turn
        second = SimpleNamespace(
            stop_reason="end_turn",
            content=[_mk_text_block("Proposal submitted.")],
            usage=SimpleNamespace(input_tokens=120, output_tokens=20),
        )
        client.messages.create.side_effect = [first, second]

        entry = run_news_agent_tick(
            event=sample_event,
            persona=trump_persona,
            ledger=ledger,
            client=client,
        )

        assert entry is not None
        assert entry.trade_side == "SHORT"
        assert entry.trade_ticker == "SPY"
        assert entry.trade_size_pct == pytest.approx(0.04)
        assert entry.persona == "trump"
        # Ledger row written
        all_rows = ledger.list_recent(limit=10)
        assert len(all_rows) == 1
        assert all_rows[0].trade_ticker == "SPY"

    def test_skip_writes_skip_row(self, tmp_path, sample_event, trump_persona):
        ledger = SignalLedger(tmp_path / "ledger.db")
        ledger.initialize()

        client = MagicMock()
        first = SimpleNamespace(
            stop_reason="tool_use",
            content=[
                _mk_tool_use_block(
                    "skip",
                    {"reason": "sentiment |score| < 0.15; no conviction."},
                )
            ],
            usage=SimpleNamespace(input_tokens=80, output_tokens=10),
        )
        second = SimpleNamespace(
            stop_reason="end_turn",
            content=[_mk_text_block("Skipped.")],
            usage=SimpleNamespace(input_tokens=90, output_tokens=5),
        )
        client.messages.create.side_effect = [first, second]

        entry = run_news_agent_tick(
            event=sample_event,
            persona=trump_persona,
            ledger=ledger,
            client=client,
        )

        assert entry is not None
        assert entry.trade_side == "SKIP"
        assert entry.trade_size_pct == 0.0
        rows = ledger.list_recent(limit=10)
        assert len(rows) == 1
        assert rows[0].trade_side == "SKIP"

    def test_invalid_proposal_is_rejected_and_skipped(
        self, tmp_path, sample_event, trump_persona
    ):
        """If Claude emits a proposal that violates the Pydantic schema
        (e.g. size_pct too large), the agent must not write it — instead
        record a SKIP with reason."""
        ledger = SignalLedger(tmp_path / "ledger.db")
        ledger.initialize()

        client = MagicMock()
        bad = SimpleNamespace(
            stop_reason="tool_use",
            content=[
                _mk_tool_use_block(
                    "propose_trade",
                    {
                        "ticker": "SPY",
                        "side": "SHORT",
                        "size_pct": 0.99,  # violates cap
                        "horizon_hours": 24,
                        "confidence": 0.7,
                        "hypothesis": "A" * 80,
                    },
                )
            ],
            usage=SimpleNamespace(input_tokens=100, output_tokens=50),
        )
        second = SimpleNamespace(
            stop_reason="end_turn",
            content=[_mk_text_block("ok")],
            usage=SimpleNamespace(input_tokens=90, output_tokens=5),
        )
        client.messages.create.side_effect = [bad, second]

        entry = run_news_agent_tick(
            event=sample_event,
            persona=trump_persona,
            ledger=ledger,
            client=client,
        )
        assert entry is not None
        assert entry.trade_side == "SKIP"
        assert "invalid" in entry.hypothesis.lower() or "violat" in entry.hypothesis.lower()
