"""Tests for the eval harness (runner + judge + cases).

We mock the Anthropic client — no real API calls. These tests verify:
  - Case registry is coherent (non-empty, unique IDs, known patterns)
  - Judge parses well-formed JSON and enforces the schema
  - Runner executes end-to-end with a mocked client
  - Hard constraints (must_contain / must_reject) can flip pass -> fail
  - Dry run works without any client
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from evals.cases import EVAL_CASES, EvalCase
from evals.judge import JudgeScore, judge_output
from evals.runner import (
    CaseResult,
    _check_hard_constraints,
    _summarize,
    run_all,
)


def _text_block(t):
    return SimpleNamespace(type="text", text=t)


def _usage(inp=100, out=50):
    return SimpleNamespace(input_tokens=inp, output_tokens=out)


# ── Case registry sanity ─────────────────────────────────────────────────────


class TestCaseRegistry:
    def test_non_empty(self):
        assert len(EVAL_CASES) > 0

    def test_unique_ids(self):
        ids = [c.id for c in EVAL_CASES]
        assert len(ids) == len(set(ids)), f"Duplicate case IDs: {ids}"

    def test_known_patterns(self):
        valid = {"debate", "triage", "contract", "travel"}
        for c in EVAL_CASES:
            assert c.pattern in valid, f"Unknown pattern: {c.pattern}"

    def test_all_have_rubric(self):
        for c in EVAL_CASES:
            assert len(c.expects) > 20, f"Case {c.id} has thin rubric"


# ── Judge schema ─────────────────────────────────────────────────────────────


_VALID_JUDGE_JSON = """\
{
  "score": 4,
  "verdict": "pass",
  "strengths": ["clear severity assignment", "actionable recommendations"],
  "weaknesses": ["differential a bit narrow"],
  "reasoning": "The output hits the main rubric criteria — severity is EMERGENCY, recommendations lead with calling emergency services, and differential includes ACS. Minor weakness: only two differential entries."
}"""


class TestJudgeScore:
    def test_valid(self):
        s = JudgeScore(
            score=5,
            verdict="pass",
            strengths=("a",),
            weaknesses=(),
            reasoning="x" * 40,
        )
        assert s.score == 5

    def test_score_range(self):
        with pytest.raises(ValidationError):
            JudgeScore(score=6, verdict="pass", reasoning="x" * 40)

    def test_verdict_enum(self):
        with pytest.raises(ValidationError):
            JudgeScore(score=3, verdict="maybe", reasoning="x" * 40)


class TestJudgeOutput:
    def test_happy_path(self):
        client = MagicMock()
        client.messages.create.return_value = SimpleNamespace(
            content=[_text_block(_VALID_JUDGE_JSON)],
            usage=_usage(),
        )

        score = judge_output(
            rubric="The output must be EMERGENCY.",
            pattern_output="SEVERITY: EMERGENCY...",
            client=client,
        )
        assert isinstance(score, JudgeScore)
        assert score.verdict == "pass"
        assert score.score == 4

    def test_cache_control_on_judge_prompt(self):
        client = MagicMock()
        client.messages.create.return_value = SimpleNamespace(
            content=[_text_block(_VALID_JUDGE_JSON)],
            usage=_usage(),
        )
        judge_output("rubric", "output", client)
        system = client.messages.create.call_args.kwargs["system"]
        assert system[-1]["cache_control"] == {"type": "ephemeral"}


# ── Runner — hard constraints ────────────────────────────────────────────────


class TestHardConstraints:
    def test_all_ok(self):
        case = EvalCase(
            id="x", pattern="triage", inputs={},
            expects="x" * 30,
            must_contain=("EMERGENCY",),
            must_reject=("prescription",),
        )
        ok_contain, ok_reject = _check_hard_constraints(
            "Severity: EMERGENCY", case
        )
        assert ok_contain and ok_reject

    def test_missing_required(self):
        case = EvalCase(
            id="x", pattern="triage", inputs={},
            expects="x" * 30,
            must_contain=("EMERGENCY",),
        )
        ok_contain, _ = _check_hard_constraints("Severity: LOW", case)
        assert not ok_contain

    def test_reject_triggered(self):
        case = EvalCase(
            id="x", pattern="triage", inputs={},
            expects="x" * 30,
            must_reject=("prescription",),
        )
        _, ok_reject = _check_hard_constraints(
            "Recommended prescription: 500mg.", case
        )
        assert not ok_reject


# ── Runner — dry run end-to-end ──────────────────────────────────────────────


class TestRunner:
    def test_dry_run_produces_result_per_case(self):
        cases = list(EVAL_CASES[:3])
        results = run_all(cases, client=None, dry_run=True)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, CaseResult)
            assert r.verdict == "dry_run"
            assert r.score is None

    def test_summarize_counts(self):
        results = [
            CaseResult(case_id="a", pattern="debate", latency_sec=1.0,
                       verdict="pass", score=5,
                       must_contain_ok=True, must_reject_ok=True,
                       output_preview="", judge_reasoning="", error=None),
            CaseResult(case_id="b", pattern="triage", latency_sec=1.0,
                       verdict="fail", score=1,
                       must_contain_ok=False, must_reject_ok=True,
                       output_preview="", judge_reasoning="", error=None),
            CaseResult(case_id="c", pattern="contract", latency_sec=1.0,
                       verdict="partial", score=3,
                       must_contain_ok=True, must_reject_ok=True,
                       output_preview="", judge_reasoning="", error=None),
        ]
        s = _summarize(results)
        assert s["total"] == 3
        assert s["pass"] == 1
        assert s["fail"] == 1
        assert s["partial"] == 1

    def test_case_pipeline_error_becomes_error_result(self):
        """If a case's pipeline raises, we record error and continue."""
        bad_case = EvalCase(
            id="broken",
            pattern="nonexistent_pattern",
            inputs={},
            expects="x" * 30,
        )
        client = MagicMock()
        results = run_all([bad_case], client=client, dry_run=False)
        assert len(results) == 1
        assert results[0].verdict == "error"
        assert results[0].error is not None
        assert "Unknown pattern" in results[0].error
