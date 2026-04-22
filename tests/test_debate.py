"""Tests for utils/debate.py — multi-agent debate orchestrator.

The Anthropic client is mocked. Three analyst calls (Haiku) and one judge
call (Opus with extended thinking) are emulated with SimpleNamespace
responses. No network I/O.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from core.debate_types import AnalystArgument, CIOVerdict, DebateResult
from utils.debate import (
    ANALYST_MODEL,
    JUDGE_MODEL,
    JUDGE_SYSTEM_PROMPT,
    _analyst_system_prompt,
    _cached_system,
    _extract_json,
    _text_from_response,
    _thinking_summary,
    _token_sum,
    run_debate,
)


# ── JSON extraction ───────────────────────────────────────────────────────────


class TestExtractJSON:
    def test_plain_object(self):
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_leading_prose_stripped(self):
        text = 'Sure! Here is the JSON:\n{"role": "bull", "x": 2}\n'
        assert _extract_json(text) == {"role": "bull", "x": 2}

    def test_trailing_prose_stripped(self):
        text = '{"k": "v"}\n\nLet me know if you need more detail.'
        assert _extract_json(text) == {"k": "v"}

    def test_nested_object(self):
        text = 'prefix {"outer": {"inner": [1, 2, 3]}} suffix'
        assert _extract_json(text) == {"outer": {"inner": [1, 2, 3]}}

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="no JSON"):
            _extract_json("nothing but prose here")

    def test_unbalanced_raises(self):
        with pytest.raises(ValueError, match="unbalanced"):
            _extract_json('{"a": 1')


# ── System prompt + cache_control ────────────────────────────────────────────


class TestSystemPrompts:
    def test_all_three_roles_produce_prompts(self):
        for role in ("bull", "bear", "macro"):
            prompt = _analyst_system_prompt(role)
            assert role.upper() in prompt
            assert "JSON" in prompt

    def test_unknown_role_raises(self):
        with pytest.raises(KeyError):
            _analyst_system_prompt("neutral")

    def test_cached_system_attaches_ephemeral(self):
        blocks = _cached_system("hello")
        assert blocks == [
            {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}
        ]

    def test_judge_prompt_includes_neutral_option(self):
        assert "NEUTRAL" in JUDGE_SYSTEM_PROMPT
        assert "ACTIONABLE" in JUDGE_SYSTEM_PROMPT


# ── Response parsing helpers ─────────────────────────────────────────────────


def _text_block(t: str):
    return SimpleNamespace(type="text", text=t)


def _thinking_block(t: str):
    return SimpleNamespace(type="thinking", thinking=t)


class TestResponseHelpers:
    def test_text_from_response_concatenates(self):
        resp = SimpleNamespace(
            content=[_text_block("a"), _text_block("b")]
        )
        assert _text_from_response(resp) == "a\nb"

    def test_text_from_response_skips_thinking(self):
        resp = SimpleNamespace(
            content=[_thinking_block("reasoning"), _text_block("final")]
        )
        assert _text_from_response(resp) == "final"

    def test_thinking_summary_collects(self):
        resp = SimpleNamespace(
            content=[
                _thinking_block("step 1"),
                _text_block("irrelevant"),
                _thinking_block("step 2"),
            ]
        )
        assert _thinking_summary(resp) == "step 1\n\nstep 2"

    def test_thinking_summary_none_when_empty(self):
        resp = SimpleNamespace(content=[_text_block("no thinking here")])
        assert _thinking_summary(resp) is None

    def test_token_sum_aggregates(self):
        r1 = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=100, output_tokens=50)
        )
        r2 = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=200, output_tokens=80, thinking_tokens=1000
            )
        )
        tin, tout, tthink = _token_sum(r1, r2)
        assert tin == 300
        assert tout == 130
        assert tthink == 1000

    def test_token_sum_handles_missing_usage(self):
        r1 = SimpleNamespace(usage=None)
        r2 = SimpleNamespace()
        assert _token_sum(r1, r2) == (0, 0, 0)


# ── Full orchestration: run_debate ────────────────────────────────────────────


def _analyst_json(role: str, conviction: float = 0.7) -> str:
    return (
        '{"role": "' + role + '", '
        '"claim": "' + ("A" * 30) + '", '
        '"evidence": ["first driver", "second driver"], '
        '"risks": ["primary downside"], '
        '"conviction": ' + str(conviction) + "}"
    )


def _verdict_json(verdict: str = "BULL") -> str:
    return (
        '{"verdict": "' + verdict + '", '
        '"confidence": 0.75, '
        '"reasoning": "' + ("r" * 80) + '", '
        '"key_risks": ["tail risk A"], '
        '"recommended_action": "Add 3% position in the underlying"}'
    )


def _analyst_response(role: str, in_tok: int = 300, out_tok: int = 200):
    return SimpleNamespace(
        content=[_text_block(_analyst_json(role))],
        usage=SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok),
    )


def _judge_response(verdict: str = "BULL"):
    return SimpleNamespace(
        content=[
            _thinking_block("Weighing bull vs bear first..."),
            _thinking_block("Macro analyst tips the balance."),
            _text_block(_verdict_json(verdict)),
        ],
        usage=SimpleNamespace(
            input_tokens=800, output_tokens=350, thinking_tokens=4200
        ),
    )


class TestRunDebate:
    def test_happy_path_full_result(self):
        client = MagicMock()
        client.messages.create.side_effect = [
            _analyst_response("bull"),
            _analyst_response("bear"),
            _analyst_response("macro"),
            _judge_response("BULL"),
        ]

        result = run_debate(
            topic="Should we add 5% NVDA to the portfolio here?",
            context="Current holdings: AAPL 10, MSFT 5. Risk 62.",
            client=client,
        )

        assert isinstance(result, DebateResult)
        assert result.bull.role == "bull"
        assert result.bear.role == "bear"
        assert result.macro.role == "macro"
        assert result.verdict.verdict == "BULL"
        assert result.judge_thinking is not None
        assert "Weighing" in result.judge_thinking
        # Token aggregation across 3 analysts + 1 judge
        assert result.total_input_tokens == 300 * 3 + 800
        assert result.total_output_tokens == 200 * 3 + 350
        assert result.total_thinking_tokens == 4200

    def test_uses_correct_models(self):
        client = MagicMock()
        client.messages.create.side_effect = [
            _analyst_response("bull"),
            _analyst_response("bear"),
            _analyst_response("macro"),
            _judge_response("NEUTRAL"),
        ]

        run_debate("A valid debate topic", "context", client)

        calls = client.messages.create.call_args_list
        assert len(calls) == 4
        # First three = analysts on Haiku
        for i in range(3):
            assert calls[i].kwargs["model"] == ANALYST_MODEL
            assert "thinking" not in calls[i].kwargs
        # Fourth = judge on Opus with thinking enabled
        assert calls[3].kwargs["model"] == JUDGE_MODEL
        assert calls[3].kwargs["thinking"]["type"] == "enabled"
        assert calls[3].kwargs["thinking"]["budget_tokens"] > 0

    def test_cache_control_on_system_prompts(self):
        client = MagicMock()
        client.messages.create.side_effect = [
            _analyst_response("bull"),
            _analyst_response("bear"),
            _analyst_response("macro"),
            _judge_response("BEAR"),
        ]

        run_debate("A valid debate topic", "context", client)

        for call in client.messages.create.call_args_list:
            system = call.kwargs["system"]
            assert isinstance(system, list)
            assert system[-1]["cache_control"] == {"type": "ephemeral"}

    def test_role_field_is_forced(self):
        """If the model returns the wrong role in its JSON, the orchestrator
        overrides it with the role we asked for."""
        client = MagicMock()
        # Bull analyst mis-echoes its role as "bear"
        bad_bull = SimpleNamespace(
            content=[_text_block(_analyst_json("bear"))],
            usage=SimpleNamespace(input_tokens=100, output_tokens=50),
        )
        client.messages.create.side_effect = [
            bad_bull,
            _analyst_response("bear"),
            _analyst_response("macro"),
            _judge_response("NEUTRAL"),
        ]

        result = run_debate("A valid debate topic", "context", client)
        assert result.bull.role == "bull"

    def test_prose_wrapped_json_still_parses(self):
        """Claude sometimes emits a leading sentence — extractor must cope."""
        client = MagicMock()
        wrapped = SimpleNamespace(
            content=[
                _text_block(
                    "Sure, here is the argument:\n" + _analyst_json("bull")
                )
            ],
            usage=SimpleNamespace(input_tokens=100, output_tokens=50),
        )
        client.messages.create.side_effect = [
            wrapped,
            _analyst_response("bear"),
            _analyst_response("macro"),
            _judge_response("BULL"),
        ]

        result = run_debate("A valid debate topic", "context", client)
        assert result.bull.conviction == pytest.approx(0.7)

    def test_schema_violation_propagates(self):
        """A malformed analyst response should raise ValidationError
        rather than silently pass through."""
        client = MagicMock()
        bad = SimpleNamespace(
            content=[
                _text_block(
                    '{"role": "bull", "claim": "too short", '
                    '"evidence": ["e"], "risks": ["r"], "conviction": 0.5}'
                )
            ],
            usage=SimpleNamespace(input_tokens=100, output_tokens=50),
        )
        client.messages.create.side_effect = [bad]

        with pytest.raises(ValidationError):
            run_debate("A valid debate topic", "context", client)

    def test_judge_without_thinking_still_returns_verdict(self):
        """If the judge returns no thinking blocks, judge_thinking is None
        but the verdict still parses."""
        client = MagicMock()
        judge_plain = SimpleNamespace(
            content=[_text_block(_verdict_json("NEUTRAL"))],
            usage=SimpleNamespace(input_tokens=800, output_tokens=300),
        )
        client.messages.create.side_effect = [
            _analyst_response("bull"),
            _analyst_response("bear"),
            _analyst_response("macro"),
            judge_plain,
        ]

        result = run_debate("A valid debate topic", "context", client)
        assert result.judge_thinking is None
        assert result.verdict.verdict == "NEUTRAL"
