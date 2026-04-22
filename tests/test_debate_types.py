"""Schema-level tests for the debate types."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.debate_types import AnalystArgument, CIOVerdict, DebateResult


def _valid_analyst(role="bull"):
    return AnalystArgument(
        role=role,
        claim="A" * 25,
        evidence=("point 1", "point 2"),
        risks=("risk 1",),
        conviction=0.7,
    )


def _valid_verdict():
    return CIOVerdict(
        verdict="BULL",
        confidence=0.8,
        reasoning="r" * 50,
        key_risks=("main risk",),
        recommended_action="Add 5% SPY exposure",
    )


class TestAnalystArgument:
    def test_valid(self):
        a = _valid_analyst()
        assert a.role == "bull"
        assert len(a.evidence) == 2

    def test_role_must_be_one_of_three(self):
        with pytest.raises(ValidationError):
            _valid_analyst(role="neutral")

    def test_short_claim_rejected(self):
        with pytest.raises(ValidationError):
            AnalystArgument(
                role="bull", claim="too short", evidence=("e",),
                risks=("r",), conviction=0.5,
            )

    def test_empty_evidence_rejected(self):
        with pytest.raises(ValidationError):
            AnalystArgument(
                role="bull", claim="A" * 25, evidence=(),
                risks=("r",), conviction=0.5,
            )

    def test_conviction_range(self):
        with pytest.raises(ValidationError):
            AnalystArgument(
                role="bull", claim="A" * 25, evidence=("e",),
                risks=("r",), conviction=1.5,
            )

    def test_frozen(self):
        a = _valid_analyst()
        with pytest.raises(ValidationError):
            a.conviction = 0.1  # type: ignore[misc]


class TestCIOVerdict:
    def test_valid(self):
        v = _valid_verdict()
        assert v.verdict == "BULL"

    def test_verdict_enum(self):
        with pytest.raises(ValidationError):
            CIOVerdict(
                verdict="MAYBE", confidence=0.5, reasoning="r" * 50,
                key_risks=("r",), recommended_action="Do something now",
            )

    def test_short_reasoning_rejected(self):
        with pytest.raises(ValidationError):
            CIOVerdict(
                verdict="BULL", confidence=0.5, reasoning="short",
                key_risks=("r",), recommended_action="Do something now",
            )


class TestDebateResult:
    def test_valid_full_result(self):
        r = DebateResult(
            topic="Is NVDA a buy here given AI capex?",
            bull=_valid_analyst("bull"),
            bear=_valid_analyst("bear"),
            macro=_valid_analyst("macro"),
            verdict=_valid_verdict(),
            judge_thinking="Weighing macro vs bottom-up…",
            total_input_tokens=1000,
            total_output_tokens=500,
            total_thinking_tokens=2000,
        )
        assert r.verdict.verdict == "BULL"

    def test_thinking_optional(self):
        r = DebateResult(
            topic="A valid topic for the debate",
            bull=_valid_analyst("bull"),
            bear=_valid_analyst("bear"),
            macro=_valid_analyst("macro"),
            verdict=_valid_verdict(),
        )
        assert r.judge_thinking is None
