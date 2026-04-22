"""Haiku judge for the eval harness.

Given a rubric + pattern output, score it on a 0-5 scale and return a
structured `JudgeScore`. Uses the same JSON-schema-constrained single-
call pattern the app itself uses — eating our own dog food.
"""
from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


JUDGE_MODEL = "claude-haiku-4-5-20251001"
JUDGE_MAX_TOKENS = 700


class JudgeScore(BaseModel):
    """Judge's assessment of one pattern output against a rubric."""
    model_config = ConfigDict(frozen=True)

    score: int = Field(ge=0, le=5)
    verdict: Literal["pass", "partial", "fail"]
    strengths: tuple[str, ...] = Field(max_length=5, default=())
    weaknesses: tuple[str, ...] = Field(max_length=5, default=())
    reasoning: str = Field(min_length=20, max_length=1000)


JUDGE_SYSTEM_PROMPT = """\
You are a rigorous evaluator. You will be given:
  - A RUBRIC describing what a good output looks like
  - The actual OUTPUT from an LLM pipeline

Your job is to score the output 0-5 against the rubric:
  5 = hits every rubric criterion clearly
  4 = hits most criteria, minor gaps
  3 = hits the main idea but misses a criterion
  2 = significant gaps or incorrect direction
  1 = largely off-topic or wrong
  0 = totally wrong, dangerous, or failed to produce output

Map to verdict: 4-5 -> pass, 3 -> partial, 0-2 -> fail.

Output ONLY this JSON object (no prose before or after):
{
  "score": 0-5,
  "verdict": "pass" | "partial" | "fail",
  "strengths": ["..."],
  "weaknesses": ["..."],
  "reasoning": "<2-5 sentences walking through your judgement>"
}
"""


def _extract_json(text: str) -> dict:
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in judge response")
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unbalanced JSON object in judge response")


def _cached_system(prompt_text: str) -> list[dict[str, Any]]:
    return [{
        "type": "text",
        "text": prompt_text,
        "cache_control": {"type": "ephemeral"},
    }]


def judge_output(
    rubric: str,
    pattern_output: str,
    client: Any,
) -> JudgeScore:
    user_msg = (
        "RUBRIC:\n" + rubric + "\n\n"
        "OUTPUT:\n" + pattern_output + "\n\n"
        "Score the output now."
    )
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=JUDGE_MAX_TOKENS,
        system=_cached_system(JUDGE_SYSTEM_PROMPT),
        messages=[{"role": "user", "content": user_msg}],
    )
    text = "\n".join(
        b.text for b in response.content
        if getattr(b, "type", None) == "text"
    )
    return JudgeScore(**_extract_json(text))
