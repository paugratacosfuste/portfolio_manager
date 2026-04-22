"""Multi-agent debate orchestrator.

Three analyst roles (bull / bear / macro) each receive the same topic and
portfolio context, then produce a structured `AnalystArgument` via
JSON-schema-constrained generation. A CIO judge (Opus with extended
thinking enabled) reads all three and emits a `CIOVerdict`.

Cost discipline
---------------
- Analysts run on Haiku 4.5 (cheap, three calls per debate).
- CIO judge runs on Opus 4.x with extended thinking budget_tokens=6000.
- System prompts use cache_control: ephemeral to drive >50% cache hit
  across debate runs that share the same portfolio context.

The module is async-free for testability — Streamlit drives it
synchronously and shows a spinner per analyst.
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from core.debate_types import AnalystArgument, CIOVerdict, DebateResult


ANALYST_MODEL = "claude-haiku-4-5-20251001"
JUDGE_MODEL = "claude-opus-4-7"
JUDGE_THINKING_BUDGET = 6000
JUDGE_MAX_TOKENS = 10000
ANALYST_MAX_TOKENS = 900


# ── System prompts ────────────────────────────────────────────────────────────


def _analyst_system_prompt(role: str) -> str:
    personas = {
        "bull": (
            "You are the BULL analyst. You argue why the position makes "
            "money. You must be specific, cite concrete drivers, and "
            "acknowledge at least one real risk. Do not straw-man the "
            "bear case."
        ),
        "bear": (
            "You are the BEAR analyst. You argue why the position loses "
            "money, gets called in, or drags the portfolio. Specific "
            "transmission channels only — no 'general market risk' "
            "platitudes."
        ),
        "macro": (
            "You are the MACRO analyst. You argue from rates, FX, "
            "liquidity, geopolitics, and policy — not bottom-up company "
            "fundamentals. You must cite one historical base rate or "
            "analog event."
        ),
    }
    body = personas[role]
    return (
        f"{body}\n\n"
        "You MUST output ONLY a JSON object matching this schema "
        "(no prose before or after):\n"
        "{\n"
        '  "role": "' + role + '",\n'
        '  "claim": "<20-500 char one-paragraph thesis>",\n'
        '  "evidence": ["<point 1>", "<point 2>", ...],\n'
        '  "risks": ["<risk 1>", ...],\n'
        '  "conviction": 0.0-1.0\n'
        "}\n"
        "Evidence and risks: 1-5 items each. Conviction: your honest "
        "weight on the thesis. Do NOT fabricate numbers or URLs."
    )


JUDGE_SYSTEM_PROMPT = """\
You are the Chief Investment Officer. Three analysts (BULL, BEAR, MACRO) \
have each submitted a structured argument. Your job is to weigh them \
and issue a verdict for the portfolio.

Rules:
  - Do not simply average the three — explicitly resolve disagreements.
  - If two analysts align and one dissents, explain why the dissent \
does or doesn't change your mind.
  - NEUTRAL is a real option when evidence is balanced or uncertain.
  - Your recommended_action must be ACTIONABLE (one sentence, concrete, \
executable in the Portfolio Tracker app).

Output ONLY this JSON object (no prose before or after):
{
  "verdict": "BULL" | "BEAR" | "NEUTRAL",
  "confidence": 0.0-1.0,
  "reasoning": "<40-2000 char paragraph weighing the three analysts>",
  "key_risks": ["<top risk 1>", ...],
  "recommended_action": "<one concrete executable action>"
}
"""


# ── Helpers ───────────────────────────────────────────────────────────────────


def _cached_system(prompt_text: str) -> list[dict[str, Any]]:
    """System prompt list with ephemeral cache_control at the final block."""
    return [{
        "type": "text",
        "text": prompt_text,
        "cache_control": {"type": "ephemeral"},
    }]


def _extract_json(text: str) -> dict:
    """Strip prose before/after the first balanced JSON object and parse.

    Claude usually complies with 'JSON only' but sometimes emits a
    leading sentence. We locate the first '{' and matching '}' and feed
    that through json.loads.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in response")
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unbalanced JSON object in response")


def _text_from_response(response: Any) -> str:
    """Concatenate all text blocks (skipping thinking blocks)."""
    chunks: list[str] = []
    for block in response.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            chunks.append(block.text)
    return "\n".join(chunks)


def _thinking_summary(response: Any) -> str | None:
    """Return concatenated summarized-thinking text, or None."""
    parts: list[str] = []
    for block in response.content:
        btype = getattr(block, "type", None)
        if btype == "thinking":
            # Opus returns thinking blocks with .thinking text
            parts.append(getattr(block, "thinking", "") or "")
    result = "\n\n".join(p for p in parts if p)
    return result or None


def _track(response: Any, model: str) -> None:
    try:
        from utils.ai_advisor import track_llm_usage
        track_llm_usage(response, model, 0.0)
    except Exception:
        pass


# ── Analyst + judge calls ────────────────────────────────────────────────────


def _run_analyst(
    role: str, topic: str, context: str, client: Any
) -> tuple[AnalystArgument, Any]:
    system = _cached_system(_analyst_system_prompt(role))
    user_msg = (
        f"Topic: {topic}\n\n"
        f"Portfolio context:\n{context}\n\n"
        f"Produce your {role.upper()} argument now."
    )
    response = client.messages.create(
        model=ANALYST_MODEL,
        max_tokens=ANALYST_MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    _track(response, ANALYST_MODEL)
    payload = _extract_json(_text_from_response(response))
    # Force the role field (model sometimes mis-echoes it)
    payload["role"] = role
    return AnalystArgument(**payload), response


def _run_judge(
    topic: str,
    context: str,
    bull: AnalystArgument,
    bear: AnalystArgument,
    macro: AnalystArgument,
    client: Any,
) -> tuple[CIOVerdict, str | None, Any]:
    system = _cached_system(JUDGE_SYSTEM_PROMPT)
    user_msg = (
        f"Topic: {topic}\n\n"
        f"Portfolio context:\n{context}\n\n"
        "BULL ARGUMENT:\n" + bull.model_dump_json(indent=2) + "\n\n"
        "BEAR ARGUMENT:\n" + bear.model_dump_json(indent=2) + "\n\n"
        "MACRO ARGUMENT:\n" + macro.model_dump_json(indent=2) + "\n\n"
        "Issue your verdict now."
    )
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=JUDGE_MAX_TOKENS,
        thinking={"type": "enabled", "budget_tokens": JUDGE_THINKING_BUDGET},
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    _track(response, JUDGE_MODEL)
    payload = _extract_json(_text_from_response(response))
    verdict = CIOVerdict(**payload)
    return verdict, _thinking_summary(response), response


def _token_sum(*responses: Any) -> tuple[int, int, int]:
    """Return (input, output, thinking) token totals across responses."""
    tin = tout = tthink = 0
    for r in responses:
        usage = getattr(r, "usage", None)
        if usage is None:
            continue
        tin += int(getattr(usage, "input_tokens", 0) or 0)
        tout += int(getattr(usage, "output_tokens", 0) or 0)
        # Anthropic exposes a separate "thinking_tokens" field on usage
        # for thinking-enabled models; fall back to 0 if not present.
        tthink += int(getattr(usage, "thinking_tokens", 0) or 0)
    return tin, tout, tthink


# ── Public entry point ───────────────────────────────────────────────────────


def run_debate(topic: str, context: str, client: Any) -> DebateResult:
    """Run a full 3-analyst → CIO-judge debate. Synchronous.

    Parameters
    ----------
    topic : str
        The question under debate. 5–1000 chars. E.g.
        "Should we add 5% NVDA to the portfolio here?"
    context : str
        Portfolio context to give to all analysts and the judge.
        Typically "Current holdings: AAPL 10, MSFT 5. Risk score 62.
        Macro prediction: risk-off with 0.7 confidence."
    client : anthropic.Anthropic
        Live Anthropic client.
    """
    bull, r_bull = _run_analyst("bull", topic, context, client)
    bear, r_bear = _run_analyst("bear", topic, context, client)
    macro, r_macro = _run_analyst("macro", topic, context, client)
    verdict, thinking, r_judge = _run_judge(
        topic, context, bull, bear, macro, client,
    )
    tin, tout, tthink = _token_sum(r_bull, r_bear, r_macro, r_judge)
    try:
        return DebateResult(
            topic=topic,
            bull=bull,
            bear=bear,
            macro=macro,
            verdict=verdict,
            judge_thinking=thinking,
            total_input_tokens=tin,
            total_output_tokens=tout,
            total_thinking_tokens=tthink,
        )
    except ValidationError as ve:
        raise ValueError(f"Debate result failed validation: {ve}") from ve
