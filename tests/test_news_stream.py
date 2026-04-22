"""Tests for utils/news_stream.py — NewsWatcher + salience scorer.

The watcher is the event source for the agentic loop. It must:
  - Deduplicate across sources (same Trump tariff headline from RSS +
    Truth Social archive should collapse to one NewsEvent).
  - Attach a persona_id by matching the headline against persona keywords.
  - Score salience so the agent only spends tokens on events that matter.
  - Cache seen event_ids to SQLite so restarts don't flood the agent.

Network is fully mocked — no real GDELT / Truth Social hits in CI.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.news_types import AssetImpact, NewsEvent, Persona, hash_headline
from utils.news_stream import (
    NewsWatcher,
    SalienceInputs,
    compute_salience,
    gdelt_search_url,
)


UTC = timezone.utc


@pytest.fixture
def personas() -> tuple[Persona, ...]:
    return (
        Persona(
            id="trump",
            name="Donald Trump",
            role="US President",
            tier=1,
            watched_sources=("gdelt", "rss"),
            keywords=("Trump", "President Trump"),
            high_salience_terms=("tariff", "sanctions", "China"),
            primary_asset_impacts=(
                AssetImpact(asset="SPY", direction_on_positive="LONG"),
            ),
            historical_reaction_horizon_hours=24,
        ),
        Persona(
            id="lagarde",
            name="Christine Lagarde",
            role="ECB President",
            tier=1,
            watched_sources=("gdelt",),
            keywords=("Lagarde",),
            high_salience_terms=("hawkish", "dovish"),
            primary_asset_impacts=(
                AssetImpact(asset="EURUSD=X", direction_on_positive="LONG"),
            ),
            historical_reaction_horizon_hours=4,
        ),
        Persona(
            id="starmer",
            name="Keir Starmer",
            role="UK PM",
            tier=2,
            watched_sources=("gdelt",),
            keywords=("Starmer",),
            high_salience_terms=("budget",),
            primary_asset_impacts=(
                AssetImpact(asset="EWU", direction_on_positive="LONG"),
            ),
            historical_reaction_horizon_hours=24,
        ),
    )


# ── compute_salience ──────────────────────────────────────────────────────────


class TestSalience:
    def test_tier1_with_hot_term_recent_is_max(self, personas):
        trump = personas[0]
        now = datetime.now(UTC)
        inputs = SalienceInputs(
            headline="Trump announces tariff on China imports",
            persona=trump,
            ts=now,
            now=now,
        )
        score = compute_salience(inputs)
        assert 0.8 <= score <= 1.0

    def test_tier2_no_hot_term_older_is_low(self, personas):
        starmer = personas[2]
        now = datetime.now(UTC)
        inputs = SalienceInputs(
            headline="Starmer meets local mayor at ribbon cutting",
            persona=starmer,
            ts=now - timedelta(hours=30),
            now=now,
        )
        score = compute_salience(inputs)
        assert score < 0.4

    def test_score_in_range(self, personas):
        trump = personas[0]
        now = datetime.now(UTC)
        inputs = SalienceInputs(
            headline="Trump tariff", persona=trump, ts=now, now=now
        )
        assert 0.0 <= compute_salience(inputs) <= 1.0


# ── NewsWatcher: persona matching + dedupe ────────────────────────────────────


class TestNewsWatcherPersonaMatching:
    def test_matches_single_persona(self, personas, tmp_path):
        watcher = NewsWatcher(personas=personas, cache_db=tmp_path / "cache.db")
        pid = watcher.match_persona("Trump slaps 50% tariff on Chinese EVs")
        assert pid == "trump"

    def test_no_match_returns_none(self, personas, tmp_path):
        watcher = NewsWatcher(personas=personas, cache_db=tmp_path / "cache.db")
        assert watcher.match_persona("Local school wins spelling bee") is None

    def test_multiple_keyword_candidates_picks_tier1(self, personas, tmp_path):
        # Tier 1 should win over Tier 2 on a tie
        p1 = Persona(
            id="a", name="A", role="X", tier=1, watched_sources=("gdelt",),
            keywords=("Summit",), high_salience_terms=(),
            primary_asset_impacts=(AssetImpact(asset="SPY", direction_on_positive="LONG"),),
            historical_reaction_horizon_hours=1,
        )
        p2 = Persona(
            id="b", name="B", role="Y", tier=2, watched_sources=("gdelt",),
            keywords=("Summit",), high_salience_terms=(),
            primary_asset_impacts=(AssetImpact(asset="SPY", direction_on_positive="LONG"),),
            historical_reaction_horizon_hours=1,
        )
        watcher = NewsWatcher(personas=(p1, p2), cache_db=tmp_path / "c.db")
        assert watcher.match_persona("Leaders hold Summit") == "a"


# ── NewsWatcher: dedupe cache ─────────────────────────────────────────────────


class TestNewsWatcherCache:
    def test_cache_filters_seen_events(self, personas, tmp_path):
        watcher = NewsWatcher(personas=personas, cache_db=tmp_path / "cache.db")
        watcher.initialize()

        headline = "Trump announces tariff on China"
        event = NewsEvent(
            event_id=hash_headline(headline),
            ts=datetime.now(UTC),
            source="gdelt",
            persona_id="trump",
            headline=headline,
            url="http://x/1",
            raw_text=None,
            salience=0.8,
        )

        first = watcher.filter_new((event,))
        assert len(first) == 1
        watcher.mark_seen(first)

        second = watcher.filter_new((event,))
        assert len(second) == 0  # already seen


# ── NewsWatcher: poll() with mocked HTTP ──────────────────────────────────────


class TestNewsWatcherPoll:
    @patch("utils.news_stream._fetch_gdelt")
    @patch("utils.news_stream._fetch_truth_social")
    @patch("utils.news_stream._fetch_rss")
    def test_poll_merges_sources_and_dedupes(
        self, mock_rss, mock_truth, mock_gdelt, personas, tmp_path
    ):
        now = datetime.now(UTC)

        gdelt_evt = NewsEvent(
            event_id=hash_headline("Trump tariff China"),
            ts=now,
            source="gdelt",
            persona_id=None,
            headline="Trump tariff China",
            url="http://gdelt/1",
            raw_text=None,
            salience=0.5,
        )
        rss_dup = NewsEvent(
            event_id=hash_headline("Trump tariff China"),  # same hash
            ts=now,
            source="rss",
            persona_id=None,
            headline="Trump tariff China",
            url="http://rss/1",
            raw_text=None,
            salience=0.5,
        )
        truth_evt = NewsEvent(
            event_id=hash_headline("I signed the order"),
            ts=now,
            source="truth_social",
            persona_id=None,
            headline="I signed the order",
            url="http://truth/1",
            raw_text=None,
            salience=0.5,
        )

        mock_gdelt.return_value = [gdelt_evt]
        mock_rss.return_value = [rss_dup]
        mock_truth.return_value = [truth_evt]

        watcher = NewsWatcher(personas=personas, cache_db=tmp_path / "c.db")
        watcher.initialize()

        events = watcher.poll()
        assert len(events) == 2  # one duplicate collapsed
        ids = {e.event_id for e in events}
        assert hash_headline("Trump tariff China") in ids
        assert hash_headline("I signed the order") in ids


# ── gdelt_search_url helper ───────────────────────────────────────────────────


class TestGdeltSearchUrl:
    def test_contains_doc_api_endpoint(self):
        url = gdelt_search_url(query="Trump tariff", max_records=10)
        assert "gdeltproject.org/api/v2/doc/doc" in url
        assert "query=" in url
        assert "maxrecords=10" in url

    def test_quote_escapes_multiword_query(self):
        url = gdelt_search_url(query='Trump tariff')
        # spaces should be url-encoded
        assert " " not in url
