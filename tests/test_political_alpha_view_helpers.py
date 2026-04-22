"""Tests for political_alpha_view_helpers.

The view file itself is Streamlit-bound and hard to unit-test; this file
covers the deterministic helpers it depends on.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.news_types import LedgerEntry
from core.personas import load_personas
from utils.political_alpha_view_helpers import (
    CuratedEvent,
    entries_to_csv_text,
    entries_to_equity_curve,
    load_curated_events,
    seeded_price_fn,
)


@pytest.fixture
def personas():
    return load_personas(Path("data/personas.yaml"))


@pytest.fixture
def curated(personas):
    return load_curated_events(personas)


# ── load_curated_events ───────────────────────────────────────────────────────


def test_load_curated_events_returns_nonempty(curated):
    assert len(curated) > 0


def test_load_curated_events_is_chronological(curated):
    ts = [c.event.ts for c in curated]
    assert ts == sorted(ts)


def test_load_curated_events_every_row_has_persona(curated):
    for c in curated:
        assert c.persona_id
        assert c.event.persona_id == c.persona_id


def test_load_curated_events_every_event_has_hash(curated):
    for c in curated:
        assert len(c.event.event_id) == 64  # sha256 hex


def test_load_curated_events_ignores_missing_dir(tmp_path, personas):
    assert load_curated_events(personas, headlines_dir=tmp_path / "nope") == ()


def test_load_curated_events_ignores_unknown_persona_csv(tmp_path, personas):
    bogus = tmp_path / "headlines"
    bogus.mkdir()
    (bogus / "not_a_persona.csv").write_text(
        "ts,persona,headline,source_url,expected_tickers,notes\n"
        "2020-01-01T00:00:00+00:00,not_a_persona,test,http://x,SPY,note\n"
    )
    assert load_curated_events(personas, headlines_dir=bogus) == ()


def test_load_curated_events_skips_bad_timestamps(tmp_path, personas):
    # Pick a valid persona id dynamically so this test survives yaml edits
    pid = personas[0].id
    d = tmp_path / "h"
    d.mkdir()
    (d / f"{pid}.csv").write_text(
        "ts,persona,headline,source_url,expected_tickers,notes\n"
        "not-a-date,x,bad row,http://x,SPY,\n"
        "2020-01-01T00:00:00+00:00,x,good row,http://x,SPY,\n"
    )
    events = load_curated_events(personas, headlines_dir=d)
    assert len(events) == 1
    assert events[0].event.headline == "good row"


# ── seeded_price_fn ───────────────────────────────────────────────────────────


def test_seeded_price_fn_is_deterministic():
    fn = seeded_price_fn(seed=7)
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    t1 = t0 + timedelta(hours=24)
    assert fn("SPY", t0, t1) == fn("SPY", t0, t1)


def test_seeded_price_fn_varies_across_tickers():
    fn = seeded_price_fn(seed=7)
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    t1 = t0 + timedelta(hours=24)
    assert fn("SPY", t0, t1) != fn("TLT", t0, t1)


def test_seeded_price_fn_returns_reasonable_magnitude():
    fn = seeded_price_fn(seed=7)
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    results = [fn(f"T{i}", t0, t0 + timedelta(hours=24)) for i in range(50)]
    # All within ±15% is plenty loose for a demo distribution
    assert all(-15 <= r <= 15 for r in results)


# ── entries_to_equity_curve ───────────────────────────────────────────────────


def _make_entry(
    entry_ts: datetime, side: str = "LONG", pnl: float | None = 1.0
) -> LedgerEntry:
    return LedgerEntry(
        entry_ts=entry_ts,
        trigger_headline="h",
        trigger_source="gdelt",
        persona="powell",
        sentiment=0.4,
        trade_ticker="SPY",
        trade_side=side,
        trade_size_pct=0.02,
        horizon_hours=24,
        hypothesis="x" * 50,
        close_ts=entry_ts + timedelta(hours=24) if pnl is not None else None,
        realized_pnl_pct=pnl,
    )


def test_equity_curve_skips_skips_and_opens():
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    entries = [
        _make_entry(t0, pnl=1.0),
        _make_entry(t0 + timedelta(hours=1), pnl=2.0),
        _make_entry(t0 + timedelta(hours=2), side="SKIP", pnl=None),
        _make_entry(t0 + timedelta(hours=3), pnl=None),  # open
    ]
    curve = entries_to_equity_curve(entries)
    assert len(curve) == 2
    assert curve[0][1] == 1.0
    assert curve[1][1] == 3.0  # cumulative


def test_equity_curve_sorts_by_entry_ts():
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    entries = [
        _make_entry(t0 + timedelta(hours=5), pnl=2.0),
        _make_entry(t0 + timedelta(hours=1), pnl=1.0),
    ]
    curve = entries_to_equity_curve(entries)
    assert curve[0][0] < curve[1][0]


# ── entries_to_csv_text ──────────────────────────────────────────────────────


def test_csv_export_has_header_row():
    csv_text = entries_to_csv_text([])
    header = csv_text.splitlines()[0]
    assert "entry_ts" in header
    assert "hypothesis" in header


def test_csv_export_serializes_entries():
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    entries = [_make_entry(t0, pnl=1.5)]
    csv_text = entries_to_csv_text(entries)
    lines = csv_text.splitlines()
    assert len(lines) == 2
    # entry row contains the persona + side
    assert "powell" in lines[1]
    assert "LONG" in lines[1]
