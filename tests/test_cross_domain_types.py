"""Schema-level tests for the cross-domain pattern gallery types."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.cross_domain_types import (
    ContractRisk,
    ContractReview,
    ItineraryActivity,
    ItineraryDay,
    TravelPlan,
    TriageAssessment,
    TriageRecommendation,
)


# ── Healthcare / triage ──────────────────────────────────────────────────────


def _valid_rec(kind="test"):
    return TriageRecommendation(
        kind=kind,
        label="CBC with differential",
        reason="Rule out infection given fever + elevated WBC.",
    )


class TestTriageAssessment:
    def test_valid(self):
        a = TriageAssessment(
            severity="MODERATE",
            confidence=0.8,
            differential=("viral URI", "bacterial pneumonia"),
            recommendations=(_valid_rec(),),
            red_flags=("persistent fever > 3d",),
            plain_language_summary=(
                "Likely a moderate respiratory infection. "
                "Further workup recommended."
            ),
        )
        assert a.severity == "MODERATE"
        assert len(a.recommendations) == 1

    def test_severity_enum(self):
        with pytest.raises(ValidationError):
            TriageAssessment(
                severity="CRITICAL",  # not in enum
                confidence=0.8,
                differential=("x",),
                recommendations=(_valid_rec(),),
                plain_language_summary="A" * 30,
            )

    def test_confidence_range(self):
        with pytest.raises(ValidationError):
            TriageAssessment(
                severity="LOW",
                confidence=1.5,
                differential=("x",),
                recommendations=(_valid_rec(),),
                plain_language_summary="A" * 30,
            )

    def test_recommendations_required(self):
        with pytest.raises(ValidationError):
            TriageAssessment(
                severity="LOW",
                confidence=0.5,
                differential=("x",),
                recommendations=(),
                plain_language_summary="A" * 30,
            )

    def test_summary_too_short(self):
        with pytest.raises(ValidationError):
            TriageAssessment(
                severity="LOW",
                confidence=0.5,
                differential=("x",),
                recommendations=(_valid_rec(),),
                plain_language_summary="short",
            )

    def test_frozen(self):
        a = TriageAssessment(
            severity="LOW",
            confidence=0.5,
            differential=("x",),
            recommendations=(_valid_rec(),),
            plain_language_summary="A" * 30,
        )
        with pytest.raises(ValidationError):
            a.severity = "HIGH"  # type: ignore[misc]


# ── Legal / contract ─────────────────────────────────────────────────────────


class TestContractReview:
    def test_valid(self):
        r = ContractReview(
            plain_summary=(
                "This agreement obliges the SaaS vendor to deliver "
                "monthly uptime of 99.9% or credit the customer."
            ),
            parties=("Acme Inc.", "GlobalCorp"),
            key_obligations=("SLA 99.9%", "Refund within 30 days"),
            risks=(
                ContractRisk(
                    severity="HIGH",
                    label="Auto-renewal clause",
                    description=(
                        "The contract auto-renews for 12 months unless "
                        "cancelled 90 days prior. Risk of lock-in."
                    ),
                ),
            ),
            negotiation_suggestions=("Replace auto-renewal with opt-in",),
        )
        assert r.risks[0].severity == "HIGH"
        assert len(r.key_obligations) == 2

    def test_summary_too_short(self):
        with pytest.raises(ValidationError):
            ContractReview(
                plain_summary="short",
                key_obligations=("obligation",),
            )

    def test_key_obligations_required(self):
        with pytest.raises(ValidationError):
            ContractReview(
                plain_summary="A" * 60,
                key_obligations=(),
            )

    def test_risk_severity_enum(self):
        with pytest.raises(ValidationError):
            ContractRisk(
                severity="CRITICAL",
                label="x",
                description="A" * 20,
            )


# ── Travel / itinerary ───────────────────────────────────────────────────────


def _activity():
    return ItineraryActivity(
        time_of_day="morning",
        title="Sagrada Família visit",
        location="Barcelona",
        notes="Book tickets in advance.",
    )


class TestTravelPlan:
    def test_valid(self):
        plan = TravelPlan(
            destination="Barcelona",
            traveller_profile="Couple, 30s, enjoys food and architecture",
            days=(
                ItineraryDay(
                    day_index=1,
                    date_label="Day 1",
                    weather_summary="Sunny, 24°C",
                    activities=(_activity(),),
                ),
            ),
            estimated_total_cost_usd=1500.0,
            packing_tips=("Comfortable walking shoes",),
        )
        assert plan.destination == "Barcelona"

    def test_cost_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            TravelPlan(
                destination="x",
                traveller_profile="xyz",
                days=(
                    ItineraryDay(
                        day_index=1,
                        date_label="Day 1",
                        activities=(_activity(),),
                    ),
                ),
                estimated_total_cost_usd=-5.0,
            )

    def test_day_index_range(self):
        with pytest.raises(ValidationError):
            ItineraryDay(
                day_index=31,
                date_label="Day 31",
                activities=(_activity(),),
            )

    def test_time_of_day_enum(self):
        with pytest.raises(ValidationError):
            ItineraryActivity(
                time_of_day="night",
                title="xxxxx",
                location="xxx",
            )
