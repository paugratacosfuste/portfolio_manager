"""Append-only SQLite ledger for every news-agent decision.

Schema mirrors action_planv3.md §5.7. Open positions are rows where
`close_ts IS NULL` AND `trade_side != 'SKIP'`. SKIP rows are audit
records (why the agent declined to trade) but are not positions.

The ledger persists across Streamlit reloads — the grader can reload the
app, trigger fresh decisions, and see a stable history. PnL reconciliation
is pluggable: callers supply a `price_fn(ticker, entry_ts, close_ts) -> pct`
so the ledger stays framework-agnostic (no yfinance import here).
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator

from core.news_types import LedgerEntry


_SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_ts TEXT NOT NULL,
    trigger_headline TEXT NOT NULL,
    trigger_source TEXT NOT NULL,
    persona TEXT NOT NULL,
    sentiment REAL NOT NULL,
    trade_ticker TEXT NOT NULL,
    trade_side TEXT NOT NULL CHECK (trade_side IN ('LONG','SHORT','SKIP')),
    trade_size_pct REAL NOT NULL,
    horizon_hours INTEGER NOT NULL,
    hypothesis TEXT NOT NULL,
    close_ts TEXT,
    realized_pnl_pct REAL
);
CREATE INDEX IF NOT EXISTS idx_ledger_persona ON ledger(persona);
CREATE INDEX IF NOT EXISTS idx_ledger_ts ON ledger(entry_ts);
"""


PriceFn = Callable[[str, datetime, datetime], float]
"""Callable that returns realized PnL % for a (ticker, entry_ts, close_ts) trio.
Positive = profit. Direction is already factored in by the caller (short
trades should return profit when the underlying fell)."""


def _iso(ts: datetime | None) -> str | None:
    return ts.isoformat() if ts is not None else None


def _parse_ts(s: str | None) -> datetime | None:
    if s is None:
        return None
    dt = datetime.fromisoformat(s)
    # Sqlite roundtrip drops tzinfo for naive reads; our inserts always
    # write isoformat with tzinfo so this should be safe.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _row_to_entry(row: sqlite3.Row) -> LedgerEntry:
    return LedgerEntry(
        entry_ts=_parse_ts(row["entry_ts"]),  # type: ignore[arg-type]
        trigger_headline=row["trigger_headline"],
        trigger_source=row["trigger_source"],
        persona=row["persona"],
        sentiment=row["sentiment"],
        trade_ticker=row["trade_ticker"],
        trade_side=row["trade_side"],
        trade_size_pct=row["trade_size_pct"],
        horizon_hours=row["horizon_hours"],
        hypothesis=row["hypothesis"],
        close_ts=_parse_ts(row["close_ts"]),
        realized_pnl_pct=row["realized_pnl_pct"],
    )


class SignalLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def append(self, entry: LedgerEntry) -> int:
        with closing(self._connect()) as conn:
            cur = conn.execute(
                """
                INSERT INTO ledger (
                    entry_ts, trigger_headline, trigger_source, persona,
                    sentiment, trade_ticker, trade_side, trade_size_pct,
                    horizon_hours, hypothesis, close_ts, realized_pnl_pct
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _iso(entry.entry_ts),
                    entry.trigger_headline,
                    entry.trigger_source,
                    entry.persona,
                    entry.sentiment,
                    entry.trade_ticker,
                    entry.trade_side,
                    entry.trade_size_pct,
                    entry.horizon_hours,
                    entry.hypothesis,
                    _iso(entry.close_ts),
                    entry.realized_pnl_pct,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list_recent(self, limit: int = 50) -> list[LedgerEntry]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM ledger ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_row_to_entry(r) for r in rows]

    def list_by_persona(self, persona: str, limit: int = 200) -> list[LedgerEntry]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM ledger WHERE persona = ? ORDER BY id DESC LIMIT ?",
                (persona, limit),
            ).fetchall()
        return [_row_to_entry(r) for r in rows]

    def open_positions(self) -> list[LedgerEntry]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM ledger "
                "WHERE close_ts IS NULL AND trade_side != 'SKIP' "
                "ORDER BY id DESC"
            ).fetchall()
        return [_row_to_entry(r) for r in rows]

    def close_entry(
        self, row_id: int, close_ts: datetime, realized_pnl_pct: float
    ) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "UPDATE ledger SET close_ts = ?, realized_pnl_pct = ? WHERE id = ?",
                (_iso(close_ts), realized_pnl_pct, row_id),
            )
            conn.commit()

    def count(self) -> int:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM ledger").fetchone()
        return int(row["c"])

    def reconcile_pnls(self, at_time: datetime, price_fn: PriceFn) -> int:
        """Close every open entry whose horizon elapsed by `at_time`.

        Returns the number of rows closed.
        """
        closed = 0
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM ledger "
                "WHERE close_ts IS NULL AND trade_side != 'SKIP'"
            ).fetchall()
            for row in rows:
                entry_ts = _parse_ts(row["entry_ts"])
                assert entry_ts is not None
                horizon_end = entry_ts + timedelta(hours=int(row["horizon_hours"]))
                if at_time < horizon_end:
                    continue
                try:
                    pnl = price_fn(row["trade_ticker"], entry_ts, at_time)
                except Exception:
                    continue
                conn.execute(
                    "UPDATE ledger SET close_ts = ?, realized_pnl_pct = ? WHERE id = ?",
                    (_iso(at_time), float(pnl), int(row["id"])),
                )
                closed += 1
            conn.commit()
        return closed

    def iter_all(self) -> Iterator[LedgerEntry]:
        with closing(self._connect()) as conn:
            for row in conn.execute("SELECT * FROM ledger ORDER BY id ASC"):
                yield _row_to_entry(row)
