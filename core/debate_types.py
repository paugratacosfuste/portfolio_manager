"""Immutable schemas for the multi-agent debate pattern.

Three analyst agents each produce an `AnalystArgument`. A CIO judge
(Opus with extended thinking) reads all three and emits a `CIOVerdict`.

All types are frozen Pydantic models so downstream views can cache them
across Streamlit reruns.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


_FROZEN = ConfigDict(frozen=True)


class AnalystArgument(BaseModel):
    """Structured argument from one analyst role."""
    model_config = _FROZEN

    role: Literal["bull", "bear", "macro"]
    claim: str = Field(min_length=20, max_length=500)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=5)
    risks: tuple[str, ...] = Field(min_length=1, max_length=5)
    conviction: float = Field(ge=0.0, le=1.0)


class CIOVerdict(BaseModel):
    """The judge's final ruling after weighing all three analysts."""
    model_config = _FROZEN

    verdict: Literal["BULL", "BEAR", "NEUTRAL"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=40, max_length=2000)
    key_risks: tuple[str, ...] = Field(min_length=1, max_length=5)
    # Actionable next step, one sentence. E.g. "Reduce equity exposure by 5pp"
    recommended_action: str = Field(min_length=10, max_length=300)


class DebateResult(BaseModel):
    """Full transcript of a single debate: topic + arguments + verdict + optional thinking."""
    model_config = _FROZEN

    topic: str = Field(min_length=5, max_length=1000)
    bull: AnalystArgument
    bear: AnalystArgument
    macro: AnalystArgument
    verdict: CIOVerdict
    # Opus extended-thinking summary text (optional; may be omitted if
    # thinking was disabled or returned no summarized blocks).
    judge_thinking: str | None = None
    # Token & cost accounting for the "Model Cost" card
    total_input_tokens: int = Field(ge=0, default=0)
    total_output_tokens: int = Field(ge=0, default=0)
    total_thinking_tokens: int = Field(ge=0, default=0)
