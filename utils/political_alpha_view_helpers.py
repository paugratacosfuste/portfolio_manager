"""Helpers for the Political Alpha Streamlit view.

All non-UI logic lives here so the view file stays presentational and
easy to read. Everything in this module is deterministic / unit-testable.
"""
from __future__ import annotations

import csv
import hashlib
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from core.news_types import LedgerEntry, NewsEvent, Persona, hash_headline


HEADLINES_DIR = Path("data/historical_headlines")


@dataclass(frozen=True)
class CuratedEvent:
    """A curated historical headline, resolved to a NewsEvent + persona id."""
    event: NewsEvent
    persona_id: str
    expected_tickers: tuple[str, ...]
    notes: str


def _parse_ts(raw: str) -> datetime:
    ts = datetime.fromisoformat(raw)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def load_curated_events(
    personas: tuple[Persona, ...],
    headlines_dir: Path = HEADLINES_DIR,
) -> tuple[CuratedEvent, ...]:
    """Read every `<persona_id>.csv` in data/historical_headlines/.

    Returned in chronological order. Missing CSVs are silently ignored —
    the view handles that case with a friendly notice.
    """
    valid_ids = {p.id for p in personas}
    events: list[CuratedEvent] = []
    if not headlines_dir.exists():
        return ()
    for path in sorted(headlines_dir.glob("*.csv")):
        persona_id = path.stem
        if persona_id not in valid_ids:
            continue
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    ts = _parse_ts(row["ts"])
                except Exception:
                    continue
                tickers = tuple(
                    t.strip() for t in row.get("expected_tickers", "").split() if t.strip()
                )
                event = NewsEvent(
                    event_id=hash_headline(row["headline"]),
                    ts=ts,
                    source="gdelt",
                    persona_id=persona_id,
                    headline=row["headline"],
                    url=row.get("source_url", ""),
                    raw_text=None,
                    salience=1.0,
                )
                events.append(
                    CuratedEvent(
                        event=event,
                        persona_id=persona_id,
                        expected_tickers=tickers,
                        notes=row.get("notes", ""),
                    )
                )
    events.sort(key=lambda e: e.event.ts)
    return tuple(events)


def seeded_price_fn(seed: int = 42):
    """Return a deterministic PriceFn for offline backtest demos.

    The curve is a cheap Gaussian-ish draw seeded by (ticker, entry_ts).
    This is NOT market data — it exists so the Scorecard tab is fast and
    reproducible during a demo. For real PnL, use the 'Reconcile with
    yfinance' button, which uses data_fetcher.
    """
    def _fn(ticker: str, entry_ts: datetime, close_ts: datetime) -> float:
        key = f"{ticker}|{entry_ts.isoformat()}|{seed}".encode()
        digest = hashlib.sha256(key).digest()
        # Map first 4 bytes to [-1, 1], then scale to a sane % range
        uniform = int.from_bytes(digest[:4], "big") / 2**32  # [0, 1)
        centered = (uniform - 0.5) * 2.0                     # [-1, 1)
        # Mild skew toward 0 so scorecard shows a realistic mix
        rng = random.Random(key)
        noise = rng.gauss(0.0, 1.5)
        return round(centered * 2.5 + noise, 4)  # roughly [-6%, +6%]
    return _fn


def yfinance_price_fn(horizon_cap_hours: int = 168):
    """Return a PriceFn that reads real prices via utils.data_fetcher.

    Yields 0.0 if the ticker / window is unavailable (so the ledger row
    is marked closed with 0% — which the scorecard flags as a neutral
    trade rather than blocking the UI).
    """
    def _fn(ticker: str, entry_ts: datetime, close_ts: datetime) -> float:
        try:
            import yfinance as yf

            # Cap absurd windows; news agent caps at 168h anyway
            if (close_ts - entry_ts).total_seconds() / 3600 > horizon_cap_hours:
                close_ts = entry_ts
            hist = yf.Ticker(ticker).history(
                start=entry_ts.date(),
                end=close_ts.date(),
                interval="1h",
            )
            if hist.empty or len(hist) < 2:
                return 0.0
            entry_px = float(hist["Close"].iloc[0])
            exit_px = float(hist["Close"].iloc[-1])
            if entry_px == 0:
                return 0.0
            return round((exit_px - entry_px) / entry_px * 100.0, 4)
        except Exception:
            return 0.0
    return _fn


def entries_to_equity_curve(
    entries: Iterable[LedgerEntry],
) -> list[tuple[datetime, float]]:
    """Cumulative PnL% curve, in entry_ts order, excluding SKIPs and opens."""
    closed = sorted(
        (e for e in entries if e.trade_side != "SKIP" and e.realized_pnl_pct is not None),
        key=lambda e: e.entry_ts,
    )
    curve: list[tuple[datetime, float]] = []
    running = 0.0
    for e in closed:
        running += float(e.realized_pnl_pct or 0.0)
        curve.append((e.entry_ts, round(running, 4)))
    return curve


def entries_to_csv_text(entries: Iterable[LedgerEntry]) -> str:
    """Serialize ledger entries to CSV text for the Download button."""
    import io

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "entry_ts", "persona", "trade_side", "trade_ticker", "trade_size_pct",
        "horizon_hours", "sentiment", "trigger_headline", "trigger_source",
        "close_ts", "realized_pnl_pct", "hypothesis",
    ])
    for e in entries:
        writer.writerow([
            e.entry_ts.isoformat(),
            e.persona,
            e.trade_side,
            e.trade_ticker,
            e.trade_size_pct,
            e.horizon_hours,
            e.sentiment,
            e.trigger_headline,
            e.trigger_source,
            e.close_ts.isoformat() if e.close_ts else "",
            "" if e.realized_pnl_pct is None else e.realized_pnl_pct,
            e.hypothesis,
        ])
    return buf.getvalue()
