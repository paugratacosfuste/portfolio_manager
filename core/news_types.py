"""Immutable value objects for the Political Alpha news pipeline.

Lives in `core/` (framework-agnostic) so the Streamlit view, the MCP
server, and the backtest harness all share one type system. No Streamlit,
no yfinance, no HTTP clients — pure data + pure validation.

- `hash_headline` normalizes a headline and returns a deterministic
  SHA-256 hex. Used as the dedupe key in the news ledger.
- `AssetImpact` maps a persona to a ticker and the expected direction
  on a positive-sentiment event.
- `Persona` is a frozen dataclass describing one tracked figure.
- `NewsEvent` is the frozen record passed between the watcher, the
  agent, and the ledger.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

NewsSource = Literal["gdelt", "truth_social", "rss", "yfinance"]
Direction = Literal["LONG", "SHORT"]
TradeSide = Literal["LONG", "SHORT", "SKIP"]

_VALID_SOURCES = {"gdelt", "truth_social", "rss", "yfinance"}
_VALID_DIRECTIONS = {"LONG", "SHORT"}
_VALID_TRADE_SIDES = {"LONG", "SHORT", "SKIP"}
_WHITESPACE_RE = re.compile(r"\s+")


def hash_headline(headline: str) -> str:
    """Deterministic 64-char SHA-256 hex of a normalized headline.

    Normalization: lowercase + collapse whitespace. This is deliberately
    loose — near-duplicates from different sources should dedupe.
    """
    normalized = _WHITESPACE_RE.sub(" ", headline.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AssetImpact:
    asset: str
    direction_on_positive: Direction

    def __post_init__(self) -> None:
        if self.direction_on_positive not in _VALID_DIRECTIONS:
            raise ValueError(
                f"direction_on_positive must be LONG or SHORT, "
                f"got {self.direction_on_positive!r}"
            )


@dataclass(frozen=True)
class Persona:
    id: str
    name: str
    role: str
    tier: int
    watched_sources: tuple[NewsSource, ...]
    keywords: tuple[str, ...]
    high_salience_terms: tuple[str, ...]
    primary_asset_impacts: tuple[AssetImpact, ...]
    historical_reaction_horizon_hours: int

    def __post_init__(self) -> None:
        if self.tier not in (1, 2, 3):
            raise ValueError(f"tier must be 1, 2 or 3; got {self.tier}")
        if self.historical_reaction_horizon_hours <= 0:
            raise ValueError("historical_reaction_horizon_hours must be > 0")
        for src in self.watched_sources:
            if src not in _VALID_SOURCES:
                raise ValueError(f"unknown watched_source: {src!r}")

    def matches(self, text: str) -> bool:
        """True if any persona keyword appears in `text` (case-insensitive)."""
        lowered = text.lower()
        return any(kw.lower() in lowered for kw in self.keywords)


@dataclass(frozen=True)
class NewsEvent:
    event_id: str
    ts: datetime
    source: NewsSource
    persona_id: str | None
    headline: str
    url: str
    raw_text: str | None
    salience: float

    def __post_init__(self) -> None:
        if self.source not in _VALID_SOURCES:
            raise ValueError(f"unknown source: {self.source!r}")
        if not (0.0 <= self.salience <= 1.0):
            raise ValueError(f"salience must be in [0, 1]; got {self.salience}")
        if self.ts.tzinfo is None:
            raise ValueError("ts must be timezone-aware (use datetime.now(timezone.utc))")
        if not self.headline.strip():
            raise ValueError("headline cannot be empty")


@dataclass(frozen=True)
class LedgerEntry:
    """One row in the signal ledger — a single agent decision + PnL slot.

    Immutable: after the agent writes, only the close_ts / realized_pnl_pct
    cells are filled in via a write that returns a *new* entry (see
    `with_close`). The append-only guarantee makes the ledger auditable.
    """
    entry_ts: datetime
    trigger_headline: str
    trigger_source: NewsSource
    persona: str
    sentiment: float
    trade_ticker: str
    trade_side: TradeSide
    trade_size_pct: float
    horizon_hours: int
    hypothesis: str
    close_ts: datetime | None = None
    realized_pnl_pct: float | None = None

    def __post_init__(self) -> None:
        if self.trade_side not in _VALID_TRADE_SIDES:
            raise ValueError(
                f"trade_side must be LONG/SHORT/SKIP; got {self.trade_side!r}"
            )
        if self.trigger_source not in _VALID_SOURCES:
            raise ValueError(f"unknown trigger_source: {self.trigger_source!r}")
        if not (-1.0 <= self.sentiment <= 1.0):
            raise ValueError(f"sentiment must be in [-1, 1]; got {self.sentiment}")
        if not (0.0 <= self.trade_size_pct <= 0.10):
            raise ValueError(
                f"trade_size_pct must be in [0, 0.10]; got {self.trade_size_pct}"
            )
        if not (1 <= self.horizon_hours <= 168):
            raise ValueError(
                f"horizon_hours must be in [1, 168]; got {self.horizon_hours}"
            )
        if self.entry_ts.tzinfo is None:
            raise ValueError("entry_ts must be timezone-aware")
        if self.close_ts is not None and self.close_ts.tzinfo is None:
            raise ValueError("close_ts must be timezone-aware")

    def with_close(
        self, close_ts: datetime, realized_pnl_pct: float
    ) -> "LedgerEntry":
        return LedgerEntry(
            entry_ts=self.entry_ts,
            trigger_headline=self.trigger_headline,
            trigger_source=self.trigger_source,
            persona=self.persona,
            sentiment=self.sentiment,
            trade_ticker=self.trade_ticker,
            trade_side=self.trade_side,
            trade_size_pct=self.trade_size_pct,
            horizon_hours=self.horizon_hours,
            hypothesis=self.hypothesis,
            close_ts=close_ts,
            realized_pnl_pct=realized_pnl_pct,
        )
