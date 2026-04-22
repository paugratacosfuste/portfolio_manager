"""Tests for core/news_types.py — immutable news & persona value objects.

NewsEvent and Persona live in core/ (not utils/) so the MCP server can
reuse them without pulling in yfinance or Streamlit. Frozen dataclasses
keep the event log safe from in-place mutation across the agent loop.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.news_types import (
    AssetImpact,
    NewsEvent,
    NewsSource,
    Persona,
    hash_headline,
)


# ── hash_headline ─────────────────────────────────────────────────────────────


class TestHashHeadline:
    def test_deterministic(self):
        h1 = hash_headline("Trump announces 50% tariff on China")
        h2 = hash_headline("Trump announces 50% tariff on China")
        assert h1 == h2

    def test_normalizes_whitespace(self):
        h1 = hash_headline("Trump announces 50% tariff")
        h2 = hash_headline("  Trump announces   50%   tariff  ")
        assert h1 == h2

    def test_case_insensitive(self):
        h1 = hash_headline("Trump Tariff")
        h2 = hash_headline("TRUMP TARIFF")
        assert h1 == h2

    def test_distinct_for_distinct_headlines(self):
        assert hash_headline("Trump tariffs") != hash_headline("Xi tariffs")

    def test_hex_string(self):
        h = hash_headline("anything")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex


# ── AssetImpact ───────────────────────────────────────────────────────────────


class TestAssetImpact:
    def test_construct(self):
        ai = AssetImpact(asset="SPY", direction_on_positive="LONG")
        assert ai.asset == "SPY"
        assert ai.direction_on_positive == "LONG"

    def test_direction_enum(self):
        with pytest.raises((TypeError, ValueError)):
            # SHORT/LONG only
            AssetImpact(asset="SPY", direction_on_positive="SIDEWAYS")  # type: ignore[arg-type]


# ── Persona ───────────────────────────────────────────────────────────────────


class TestPersona:
    def _valid(self, **overrides):
        base = dict(
            id="trump",
            name="Donald Trump",
            role="US President",
            tier=1,
            watched_sources=("truth_social", "gdelt"),
            keywords=("Trump", "President Trump"),
            high_salience_terms=("tariff", "sanctions"),
            primary_asset_impacts=(
                AssetImpact(asset="SPY", direction_on_positive="LONG"),
            ),
            historical_reaction_horizon_hours=24,
        )
        base.update(overrides)
        return base

    def test_construct(self):
        p = Persona(**self._valid())
        assert p.id == "trump"
        assert p.tier == 1
        assert "Trump" in p.keywords

    def test_tier_range(self):
        with pytest.raises(ValueError):
            Persona(**self._valid(tier=0))
        with pytest.raises(ValueError):
            Persona(**self._valid(tier=4))

    def test_frozen(self):
        p = Persona(**self._valid())
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            p.tier = 2  # type: ignore[misc]

    def test_horizon_positive(self):
        with pytest.raises(ValueError):
            Persona(**self._valid(historical_reaction_horizon_hours=0))

    def test_matches_headline(self):
        p = Persona(**self._valid())
        assert p.matches("President Trump announces tariff on imports") is True
        assert p.matches("Xi Jinping holds meeting") is False

    def test_matches_case_insensitive(self):
        p = Persona(**self._valid())
        assert p.matches("trump signs order") is True


# ── NewsEvent ─────────────────────────────────────────────────────────────────


class TestNewsEvent:
    def _valid(self, **overrides):
        base = dict(
            event_id=hash_headline("Trump signs exec order on tariffs"),
            ts=datetime(2026, 4, 22, 10, 0, tzinfo=timezone.utc),
            source="gdelt",
            persona_id="trump",
            headline="Trump signs exec order on tariffs",
            url="https://example.com/a",
            raw_text=None,
            salience=0.72,
        )
        base.update(overrides)
        return base

    def test_construct(self):
        e = NewsEvent(**self._valid())
        assert e.persona_id == "trump"
        assert 0.0 <= e.salience <= 1.0

    def test_frozen(self):
        e = NewsEvent(**self._valid())
        with pytest.raises(Exception):
            e.salience = 0.99  # type: ignore[misc]

    def test_source_enum(self):
        # Accept valid sources
        NewsEvent(**self._valid(source="gdelt"))
        NewsEvent(**self._valid(source="truth_social"))
        NewsEvent(**self._valid(source="rss"))
        # Reject invalid
        with pytest.raises(ValueError):
            NewsEvent(**self._valid(source="twitter"))

    def test_salience_range(self):
        with pytest.raises(ValueError):
            NewsEvent(**self._valid(salience=-0.1))
        with pytest.raises(ValueError):
            NewsEvent(**self._valid(salience=1.01))

    def test_ts_must_be_timezone_aware(self):
        with pytest.raises(ValueError):
            NewsEvent(**self._valid(ts=datetime(2026, 4, 22, 10, 0)))  # naive

    def test_headline_not_empty(self):
        with pytest.raises(ValueError):
            NewsEvent(**self._valid(headline=""))

    def test_persona_id_optional(self):
        e = NewsEvent(**self._valid(persona_id=None))
        assert e.persona_id is None


# ── NewsSource sanity (type alias / Literal check) ────────────────────────────


class TestNewsSource:
    def test_accepts_expected_sources(self):
        # NewsSource is a Literal; runtime check via NewsEvent construction
        for src in ("gdelt", "truth_social", "rss"):
            NewsEvent(
                event_id=hash_headline(src),
                ts=datetime(2026, 4, 22, tzinfo=timezone.utc),
                source=src,  # type: ignore[arg-type]
                persona_id=None,
                headline="x",
                url="y",
                raw_text=None,
                salience=0.5,
            )
