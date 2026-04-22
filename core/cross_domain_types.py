"""Frozen Pydantic schemas for the cross-domain pattern gallery.

Three domains, three different LLM patterns, each with a validated
output shape:

  healthcare -> TriageAssessment     (structured JSON generation)
  legal      -> ContractReview       (single-call summarisation + flags)
  travel     -> ItineraryDay/Plan    (agentic tool-use loop)

All types are immutable so views can memoize them safely across
Streamlit reruns.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


_FROZEN = ConfigDict(frozen=True)


# ── Healthcare: structured JSON generation (CDSS-lite triage) ────────────────


class TriageRecommendation(BaseModel):
    """One recommended test/action inside a triage assessment."""
    model_config = _FROZEN

    kind: Literal["test", "specialist_referral", "medication_review", "observation"]
    label: str = Field(min_length=3, max_length=120)
    reason: str = Field(min_length=10, max_length=400)


class TriageAssessment(BaseModel):
    """Structured output from the CDSS-lite triage model.

    Educational prototype only. Not a medical device, not CE/FDA-marked.
    """
    model_config = _FROZEN

    severity: Literal["LOW", "MODERATE", "HIGH", "EMERGENCY"]
    confidence: float = Field(ge=0.0, le=1.0)
    differential: tuple[str, ...] = Field(min_length=1, max_length=5)
    recommendations: tuple[TriageRecommendation, ...] = Field(
        min_length=1, max_length=6
    )
    red_flags: tuple[str, ...] = Field(max_length=5, default=())
    plain_language_summary: str = Field(min_length=20, max_length=600)


# ── Legal: single-call summarisation (contract clause review) ────────────────


class ContractRisk(BaseModel):
    model_config = _FROZEN

    severity: Literal["LOW", "MEDIUM", "HIGH"]
    label: str = Field(min_length=3, max_length=120)
    description: str = Field(min_length=10, max_length=500)


class ContractReview(BaseModel):
    """Plain-English summary of a contract clause + risk flags.

    Educational prototype only. Not legal advice.
    """
    model_config = _FROZEN

    plain_summary: str = Field(min_length=40, max_length=1500)
    parties: tuple[str, ...] = Field(max_length=6, default=())
    key_obligations: tuple[str, ...] = Field(min_length=1, max_length=8)
    risks: tuple[ContractRisk, ...] = Field(max_length=6, default=())
    negotiation_suggestions: tuple[str, ...] = Field(max_length=5, default=())


# ── Travel: agentic tool-use loop (itinerary planner) ────────────────────────


class ItineraryActivity(BaseModel):
    model_config = _FROZEN

    time_of_day: Literal["morning", "afternoon", "evening"]
    title: str = Field(min_length=3, max_length=120)
    location: str = Field(min_length=1, max_length=120)
    notes: str = Field(max_length=400, default="")


class ItineraryDay(BaseModel):
    model_config = _FROZEN

    day_index: int = Field(ge=1, le=30)
    date_label: str = Field(min_length=1, max_length=40)
    weather_summary: str = Field(max_length=200, default="")
    activities: tuple[ItineraryActivity, ...] = Field(min_length=1, max_length=6)


class TravelPlan(BaseModel):
    """Final agentic-loop output: a multi-day itinerary.

    Educational prototype only. Prices/weather/availability are mocked.
    """
    model_config = _FROZEN

    destination: str = Field(min_length=2, max_length=120)
    traveller_profile: str = Field(min_length=3, max_length=200)
    days: tuple[ItineraryDay, ...] = Field(min_length=1, max_length=14)
    estimated_total_cost_usd: float = Field(ge=0.0, le=100_000.0)
    packing_tips: tuple[str, ...] = Field(max_length=6, default=())
