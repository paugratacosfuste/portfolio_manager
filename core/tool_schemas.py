"""Pydantic response schemas for LLM tool outputs.

These models pin the output contract between tool executors and any LLM
consumer (chatbot, MCP server, future Vercel adapter). Frozen + validated,
so a typo or missing field fails loudly instead of silently shipping a
malformed payload to Claude.

Serialize via `.model_dump_json()` — callers receive a JSON string with
the exact key shape downstream prompts expect.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


_FROZEN = ConfigDict(frozen=True)


class ErrorResponse(BaseModel):
    model_config = _FROZEN

    error: str = Field(min_length=1)


class AssetPriceResponse(BaseModel):
    model_config = _FROZEN

    ticker: str = Field(min_length=1)
    price: float = Field(ge=0)


class HoldingRow(BaseModel):
    model_config = _FROZEN

    ticker: str = Field(min_length=1)
    quantity: float = Field(ge=0)
    price: float = Field(ge=0)
    value: float = Field(ge=0)
    weight_pct: float = Field(ge=0, le=100)
    asset_class: str
    sector: str


class PortfolioSummaryResponse(BaseModel):
    model_config = _FROZEN

    total_value: float = Field(ge=0)
    holdings: tuple[HoldingRow, ...]


class RiskMetricsResponse(BaseModel):
    model_config = _FROZEN

    total_value: float = Field(ge=0)
    volatility_pct: float
    beta: float
    hhi: float = Field(ge=0, le=10000)
    risk_score: int = Field(ge=0, le=100)
    sharpe_ratio: float
    max_drawdown_pct: float = Field(le=0)
    cvar_95_pct: float


class WhatIfResponse(BaseModel):
    model_config = _FROZEN

    action: Literal["add", "remove"]
    ticker: str = Field(min_length=1)
    quantity: float = Field(gt=0)
    new_total_value: float = Field(ge=0)
    new_volatility_pct: float
    new_sharpe_ratio: float
    new_max_drawdown_pct: float
    new_weights: dict[str, float]


class NewsItem(BaseModel):
    model_config = _FROZEN

    title: str = Field(min_length=1)
    link: str
    publisher: str
    timestamp: str


class MarketNewsResponse(BaseModel):
    model_config = _FROZEN

    ticker: str = Field(min_length=1)
    news: tuple[NewsItem, ...]


class MacroPredictionResponse(BaseModel):
    model_config = _FROZEN

    prediction: str = Field(min_length=1)
    correction_probability_pct: float = Field(ge=0, le=100)
    current_vix: float = Field(ge=0)
    sp500_vs_200ma_pct: float


# ── Political Alpha / Pattern 5 ───────────────────────────────────────────────


class TradeProposal(BaseModel):
    """Output of the news agent's `propose_trade` tool call.

    Caps here mirror AGENT_SYSTEM_PROMPT hard rules — any violation blocks
    the proposal from being written to the ledger.
    """
    model_config = _FROZEN

    ticker: str = Field(min_length=1)
    side: Literal["LONG", "SHORT"]
    size_pct: float = Field(gt=0.0, le=0.10)
    horizon_hours: int = Field(ge=1, le=168)
    confidence: float = Field(ge=0.0, le=1.0)
    hypothesis: str = Field(min_length=40, max_length=400)
    stop_loss_pct: float | None = Field(default=None, ge=0.005, le=0.20)
