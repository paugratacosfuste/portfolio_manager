"""Immutable data types shared across the v3 codebase.

All types are Pydantic v2 BaseModel with `frozen=True` — giving us:
  • immutability (protects the session-state signal bus from accidental mutation)
  • JSON round-trips (for SQLite ledger persistence and LLM structured output)
  • field validation (conviction ranges, hypothesis lengths, enum constraints)
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Holding(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    ticker: str = Field(min_length=1, max_length=20)
    quantity: float = Field(ge=0.0)
    asset_class: str = Field(min_length=1)
    sector: str | None = None
    region: str | None = None


class Portfolio(BaseModel):
    model_config = ConfigDict(frozen=True)

    holdings: tuple[Holding, ...] = ()
    cash_usd: float = Field(default=0.0, ge=0.0)

    @classmethod
    def from_dict(cls, holdings: dict[str, float], asset_class: str = "Unknown") -> Portfolio:
        """Build a Portfolio from a simple {ticker: quantity} dict.

        Zero-quantity entries are skipped. asset_class defaults to "Unknown";
        callers that need richer metadata should construct Holdings directly.
        """
        entries = tuple(
            Holding(ticker=t, quantity=q, asset_class=asset_class)
            for t, q in holdings.items()
            if q > 0
        )
        return cls(holdings=entries)

    def tickers(self) -> tuple[str, ...]:
        return tuple(h.ticker for h in self.holdings)

    def quantity_map(self) -> dict[str, float]:
        return {h.ticker: h.quantity for h in self.holdings}


class LedgerEntry(BaseModel):
    """One row of the Political Alpha signal ledger.

    An entry is opened when the agent proposes a trade. `close_ts` and
    `realized_pnl_pct` are filled later by the reconciliation job.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    entry_ts: str = Field(min_length=1)
    trigger_headline: str = Field(min_length=1)
    trigger_source: str = Field(min_length=1)
    persona: str = Field(min_length=1)
    sentiment: float = Field(ge=-1.0, le=1.0)
    trade_ticker: str = Field(min_length=1)
    trade_side: Literal["LONG", "SHORT", "SKIP"]
    trade_size_pct: float = Field(ge=0.0, le=0.10)
    horizon_hours: int = Field(ge=1, le=168)
    hypothesis: str = Field(min_length=40, max_length=400)
    close_ts: str | None = None
    realized_pnl_pct: float | None = None


class DebateVerdict(BaseModel):
    """Final ruling from the CIO judge after a multi-agent debate.

    Schema is designed to prevent averaging: conviction must be 1-5,
    dissenting views must be acknowledged substantively, and low-conviction
    verdicts require a detailed change condition.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    decision: Literal["BUY", "HOLD", "SELL"]
    conviction: int = Field(ge=1, le=5)
    disagreements_resolved: tuple[str, ...] = Field(min_length=1)
    dissenting_view_acknowledged: str = Field(min_length=50)
    decision_change_condition: str = Field(min_length=30)
    summary: str = Field(min_length=1)
    transcript_json: str = Field(default="[]")

    @model_validator(mode="after")
    def _low_conviction_requires_detailed_condition(self) -> DebateVerdict:
        if self.conviction < 3 and len(self.decision_change_condition) < 80:
            raise ValueError(
                "Low conviction (<3) requires a decision_change_condition of "
                f"at least 80 characters; got {len(self.decision_change_condition)}."
            )
        return self
