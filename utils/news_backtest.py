"""News-driven replay engine + per-persona scorecard.

`replay_headlines` reads a curated CSV of historical events, resolves
each to a `NewsEvent`, runs it through a policy function (typically the
live agent), and mark-to-markets every open position at T+horizon using
an injected `price_fn` (unit-testable; in prod, backed by yfinance).

All outputs are immutable frozen dataclasses so downstream views can
cache them safely across Streamlit reruns.
"""
from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from core.news_types import (
    LedgerEntry,
    NewsEvent,
    Persona,
    hash_headline,
)
from core.types import Portfolio
from utils.signal_ledger import PriceFn, SignalLedger


@dataclass(frozen=True)
class PersonaScorecard:
    persona: str
    total_trades: int
    skips: int
    wins: int
    losses: int
    hit_rate: float
    avg_pnl_pct: float
    median_holding_hours: float
    sharpe_annualized: float
    max_drawdown_pct: float
    profit_factor: float


@dataclass(frozen=True)
class BacktestReport:
    total_events: int
    total_trades: int
    total_skips: int
    overall_pnl_pct: float
    per_persona: dict[str, PersonaScorecard] = field(default_factory=dict)


# ── Helpers ───────────────────────────────────────────────────────────────────


def realized_pnl_pct(side: str, entry: float, exit_: float) -> float:
    """Return directional PnL%. SHORT inverts the sign. Safe on zero entry."""
    if entry == 0:
        return 0.0
    raw = (exit_ - entry) / entry * 100.0
    return raw if side == "LONG" else -raw


def _read_timeline(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _row_to_event(row: dict, personas: tuple[Persona, ...]) -> NewsEvent:
    ts = datetime.fromisoformat(row["ts"])
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    persona_id = row.get("persona") or None
    if persona_id and not any(p.id == persona_id for p in personas):
        persona_id = None
    return NewsEvent(
        event_id=hash_headline(row["headline"]),
        ts=ts,
        source="gdelt",  # uniform for curated timelines
        persona_id=persona_id,
        headline=row["headline"],
        url=row.get("source_url", ""),
        raw_text=None,
        salience=1.0,
    )


def compute_scorecard(
    entries: Iterable[LedgerEntry], persona: str
) -> PersonaScorecard:
    rows = [e for e in entries if e.persona == persona]
    trades = [e for e in rows if e.trade_side != "SKIP"]
    skips = [e for e in rows if e.trade_side == "SKIP"]

    pnls = [e.realized_pnl_pct for e in trades if e.realized_pnl_pct is not None]

    if not trades:
        return PersonaScorecard(
            persona=persona, total_trades=0, skips=len(skips), wins=0, losses=0,
            hit_rate=0.0, avg_pnl_pct=0.0, median_holding_hours=0.0,
            sharpe_annualized=0.0, max_drawdown_pct=0.0, profit_factor=0.0,
        )

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    avg_pnl = (sum(pnls) / len(pnls)) if pnls else 0.0

    holdings = [
        (e.close_ts - e.entry_ts).total_seconds() / 3600.0
        for e in trades
        if e.close_ts is not None
    ]
    median_holding = statistics.median(holdings) if holdings else 0.0

    if len(pnls) >= 2:
        stdev = statistics.pstdev(pnls)
        sharpe = (avg_pnl / stdev) * (252 ** 0.5) if stdev > 0 else 0.0
    else:
        sharpe = 0.0

    max_dd = _max_drawdown_pct(pnls)

    if not losses:
        profit_factor = float("inf") if wins else 0.0
    else:
        profit_factor = sum(wins) / abs(sum(losses)) if wins else 0.0

    return PersonaScorecard(
        persona=persona,
        total_trades=len(trades),
        skips=len(skips),
        wins=len(wins),
        losses=len(losses),
        hit_rate=len(wins) / len(trades) if trades else 0.0,
        avg_pnl_pct=avg_pnl,
        median_holding_hours=median_holding,
        sharpe_annualized=sharpe,
        max_drawdown_pct=max_dd,
        profit_factor=profit_factor,
    )


def _max_drawdown_pct(pnls: list[float]) -> float:
    """Peak-to-trough on the cumulative PnL curve (in %)."""
    if not pnls:
        return 0.0
    curve, peak, max_dd = 0.0, 0.0, 0.0
    for p in pnls:
        curve += p
        peak = max(peak, curve)
        max_dd = min(max_dd, curve - peak)
    return max_dd


# ── replay_headlines ──────────────────────────────────────────────────────────


PolicyFn = Callable[[NewsEvent, SignalLedger, Portfolio], LedgerEntry | None]
ProgressFn = Callable[[int, int, str], None]


def replay_headlines(
    timeline_csv: str | Path,
    personas: tuple[Persona, ...],
    policy_fn: PolicyFn,
    ledger_path: str | Path,
    price_fn: PriceFn,
    starting_portfolio: Portfolio | None = None,
    max_events: int | None = None,
    progress_fn: ProgressFn | None = None,
) -> BacktestReport:
    rows = _read_timeline(Path(timeline_csv))
    if max_events is not None and max_events > 0:
        rows = rows[:max_events]
    ledger = SignalLedger(ledger_path)
    ledger.initialize()

    portfolio = starting_portfolio or Portfolio(holdings=())

    total_events = len(rows)
    total_trades = 0
    total_skips = 0

    for i, row in enumerate(rows, 1):
        if progress_fn is not None:
            progress_fn(i, total_events, row.get("headline", ""))
        try:
            event = _row_to_event(row, personas)
        except Exception:
            continue
        entry = policy_fn(event, ledger, portfolio)
        if entry is None:
            continue
        if entry.trade_side == "SKIP":
            total_skips += 1
        else:
            total_trades += 1

    # Mark-to-market every open trade at now
    closed = ledger.reconcile_pnls(
        at_time=datetime.now(timezone.utc), price_fn=price_fn
    )

    entries = list(ledger.iter_all())
    unique_personas = sorted({e.persona for e in entries if e.persona != "unknown"})
    per_persona = {p: compute_scorecard(entries, p) for p in unique_personas}

    overall_pnl = sum(
        e.realized_pnl_pct or 0.0
        for e in entries
        if e.trade_side != "SKIP" and e.realized_pnl_pct is not None
    )

    return BacktestReport(
        total_events=total_events,
        total_trades=total_trades,
        total_skips=total_skips,
        overall_pnl_pct=round(overall_pnl, 4),
        per_persona=per_persona,
    )
