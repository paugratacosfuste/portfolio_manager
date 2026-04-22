"""Cross-domain pattern gallery — 3 domains, 3 LLM patterns.

The point is to show the portfolio-tracker patterns transfer. Every
function here is a thin prompt/schema wrapper around the Anthropic
`messages.create` endpoint; the heavy lifting is the *shape* of the call,
not the domain.

  - run_triage()       -> pattern #3: structured JSON generation (Haiku)
  - run_contract()     -> pattern #1: single-call summarisation (Haiku)
  - run_travel_agent() -> pattern #2: agentic tool-use loop  (Sonnet)

All three use `cache_control: ephemeral` on the final system prompt
content block. Educational prototype — no medical/legal/travel advice.
"""
from __future__ import annotations

import json
from typing import Any

from core.cross_domain_types import (
    ContractReview,
    ItineraryActivity,
    ItineraryDay,
    TravelPlan,
    TriageAssessment,
)


HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"
MAX_TOKENS_TRIAGE = 1100
MAX_TOKENS_CONTRACT = 1400
MAX_TOKENS_TRAVEL_STEP = 1600
TRAVEL_AGENT_MAX_TURNS = 6


# ── System prompts ────────────────────────────────────────────────────────────


TRIAGE_SYSTEM_PROMPT = """\
You are a clinical decision-support assistant for an educational prototype, \
NOT a medical device. Triage presenting symptoms into one of four severity \
levels and suggest a differential + non-invasive next steps.

Hard rules:
  - You are not providing medical advice. Your output will be reviewed by \
a clinician before any patient action.
  - NEVER recommend specific prescription dosages.
  - If input suggests immediate life threat (chest pain + diaphoresis, \
stroke signs, anaphylaxis, severe bleeding), severity must be EMERGENCY \
and recommendations must lead with "call emergency services".

Output ONLY this JSON object (no prose):
{
  "severity": "LOW" | "MODERATE" | "HIGH" | "EMERGENCY",
  "confidence": 0.0-1.0,
  "differential": ["diagnosis 1", "diagnosis 2", ...],
  "recommendations": [
    {"kind": "test" | "specialist_referral" | "medication_review" | "observation",
     "label": "short name", "reason": "1-2 sentence rationale"}
  ],
  "red_flags": ["flag 1", ...],
  "plain_language_summary": "<2-3 sentences a non-clinician can read>"
}
"""


CONTRACT_SYSTEM_PROMPT = """\
You are a paralegal assistant for an educational prototype — NOT a \
licensed attorney. Summarize a contract clause in plain English, list \
the key obligations on both sides, flag risky language, and suggest \
negotiation tweaks.

Hard rules:
  - You are not providing legal advice.
  - Call out ambiguous or one-sided language explicitly; do not soften it.
  - If a clause is genuinely balanced, say so — do not invent risks.

Output ONLY this JSON object (no prose):
{
  "plain_summary": "<2-4 sentence plain-English summary>",
  "parties": ["Party A", "Party B", ...],
  "key_obligations": ["obligation 1", ...],
  "risks": [
    {"severity": "LOW" | "MEDIUM" | "HIGH",
     "label": "short name",
     "description": "1-3 sentence explanation"}
  ],
  "negotiation_suggestions": ["suggestion 1", ...]
}
"""


TRAVEL_SYSTEM_PROMPT = """\
You are a travel planning agent for an educational prototype. You have \
tools for weather, flights (MOCKED), and hotels (MOCKED). Use tools to \
gather information THEN call `emit_itinerary` ONCE with the final plan.

Hard rules:
  - All prices and availability are MOCKED for demo purposes.
  - Use at least one tool call before emitting the itinerary.
  - Keep the itinerary short: 1-5 days.
  - Respect traveller preferences (dietary, accessibility, budget).

When you are ready, call `emit_itinerary` with a TravelPlan JSON object.
"""


# ── Travel agent: tool schemas ───────────────────────────────────────────────


TRAVEL_TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_destination_weather",
        "description": "Get the mocked weather forecast for a destination.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer", "minimum": 1, "maximum": 14},
            },
            "required": ["city", "days"],
        },
    },
    {
        "name": "search_flights_mock",
        "description": "Mock flight search. Returns a cheap flight price.",
        "input_schema": {
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
            },
            "required": ["origin", "destination"],
        },
    },
    {
        "name": "search_hotels_mock",
        "description": "Mock hotel search. Returns a recommended hotel + nightly rate.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "budget_tier": {
                    "type": "string",
                    "enum": ["budget", "mid", "luxury"],
                },
            },
            "required": ["city", "budget_tier"],
        },
    },
    {
        "name": "emit_itinerary",
        "description": (
            "Emit the final TravelPlan and end the conversation. "
            "Call this ONCE at the end."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "destination": {"type": "string"},
                "traveller_profile": {"type": "string"},
                "days": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "day_index": {"type": "integer"},
                            "date_label": {"type": "string"},
                            "weather_summary": {"type": "string"},
                            "activities": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "time_of_day": {
                                            "type": "string",
                                            "enum": ["morning", "afternoon", "evening"],
                                        },
                                        "title": {"type": "string"},
                                        "location": {"type": "string"},
                                        "notes": {"type": "string"},
                                    },
                                    "required": ["time_of_day", "title", "location"],
                                },
                            },
                        },
                        "required": ["day_index", "date_label", "activities"],
                    },
                },
                "estimated_total_cost_usd": {"type": "number"},
                "packing_tips": {
                    "type": "array", "items": {"type": "string"},
                },
            },
            "required": [
                "destination",
                "traveller_profile",
                "days",
                "estimated_total_cost_usd",
            ],
        },
    },
]


# ── Travel: mock tool executors ──────────────────────────────────────────────


def _mock_weather(city: str, days: int) -> dict:
    return {
        "city": city,
        "forecast": [
            {"day": i + 1, "summary": f"Partly sunny, 20–24°C"}
            for i in range(max(1, min(days, 14)))
        ],
    }


def _mock_flights(origin: str, destination: str) -> dict:
    return {
        "origin": origin,
        "destination": destination,
        "best_price_usd": 420,
        "carrier": "MockAir",
        "note": "All prices mocked for demo.",
    }


def _mock_hotels(city: str, budget_tier: str) -> dict:
    nightly = {"budget": 70, "mid": 150, "luxury": 380}.get(budget_tier, 150)
    return {
        "city": city,
        "hotel": f"{budget_tier.title()} stay — Demo Hotel",
        "nightly_rate_usd": nightly,
    }


def execute_travel_tool(name: str, args: dict) -> str:
    if name == "get_destination_weather":
        payload = _mock_weather(args["city"], int(args.get("days", 3)))
    elif name == "search_flights_mock":
        payload = _mock_flights(args["origin"], args["destination"])
    elif name == "search_hotels_mock":
        payload = _mock_hotels(args["city"], args.get("budget_tier", "mid"))
    else:
        payload = {"error": f"unknown tool: {name}"}
    return json.dumps(payload)


# ── Shared helpers ───────────────────────────────────────────────────────────


def _cached_system(prompt_text: str) -> list[dict[str, Any]]:
    return [{
        "type": "text",
        "text": prompt_text,
        "cache_control": {"type": "ephemeral"},
    }]


def _extract_json(text: str) -> dict:
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
    parts: list[str] = []
    for block in response.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            parts.append(block.text)
    return "\n".join(parts)


def _track(response: Any, model: str) -> None:
    try:
        from utils.ai_advisor import track_llm_usage
        track_llm_usage(response, model, 0.0)
    except Exception:
        pass


# ── Public: triage (structured JSON) ─────────────────────────────────────────


def run_triage(
    presentation: str,
    patient_context: str,
    client: Any,
) -> TriageAssessment:
    user_msg = (
        f"Patient context:\n{patient_context}\n\n"
        f"Presenting complaint / symptoms:\n{presentation}\n\n"
        "Produce the triage JSON now."
    )
    response = client.messages.create(
        model=HAIKU,
        max_tokens=MAX_TOKENS_TRIAGE,
        system=_cached_system(TRIAGE_SYSTEM_PROMPT),
        messages=[{"role": "user", "content": user_msg}],
    )
    _track(response, HAIKU)
    return TriageAssessment(**_extract_json(_text_from_response(response)))


# ── Public: contract review (single-call summarisation) ─────────────────────


def run_contract_review(
    clause_text: str,
    client: Any,
    perspective: str = "buyer",
) -> ContractReview:
    user_msg = (
        f"Review from the perspective of the {perspective}.\n\n"
        f"Clause text:\n---\n{clause_text}\n---\n\n"
        "Produce the review JSON now."
    )
    response = client.messages.create(
        model=HAIKU,
        max_tokens=MAX_TOKENS_CONTRACT,
        system=_cached_system(CONTRACT_SYSTEM_PROMPT),
        messages=[{"role": "user", "content": user_msg}],
    )
    _track(response, HAIKU)
    return ContractReview(**_extract_json(_text_from_response(response)))


# ── Public: travel agent (agentic tool-use loop) ────────────────────────────


def _coerce_travel_plan(args: dict) -> TravelPlan:
    """Normalize emit_itinerary args into a TravelPlan (tuples + models)."""
    raw_days = args.get("days") or []
    days: list[ItineraryDay] = []
    for i, d in enumerate(raw_days):
        raw_acts = d.get("activities") or []
        acts = tuple(
            ItineraryActivity(
                time_of_day=a["time_of_day"],
                title=a["title"],
                location=a["location"],
                notes=a.get("notes", ""),
            )
            for a in raw_acts
        )
        days.append(ItineraryDay(
            day_index=d.get("day_index", i + 1),
            date_label=d.get("date_label", f"Day {i + 1}"),
            weather_summary=d.get("weather_summary", ""),
            activities=acts,
        ))
    return TravelPlan(
        destination=args["destination"],
        traveller_profile=args["traveller_profile"],
        days=tuple(days),
        estimated_total_cost_usd=float(args["estimated_total_cost_usd"]),
        packing_tips=tuple(args.get("packing_tips") or ()),
    )


def run_travel_agent(
    destination: str,
    days: int,
    traveller_profile: str,
    client: Any,
    max_turns: int = TRAVEL_AGENT_MAX_TURNS,
) -> tuple[TravelPlan, list[str]]:
    """Run the agentic tool-use loop until the model emits an itinerary.

    Returns (plan, transcript) where transcript is a human-readable list
    of tool-call lines suitable for a debug expander.
    """
    system = _cached_system(TRAVEL_SYSTEM_PROMPT)
    messages: list[dict[str, Any]] = [{
        "role": "user",
        "content": (
            f"Plan a {days}-day trip to {destination}.\n"
            f"Traveller: {traveller_profile}.\n\n"
            "Use the tools to gather info, then emit the itinerary."
        ),
    }]
    transcript: list[str] = []

    for _turn in range(max_turns):
        response = client.messages.create(
            model=SONNET,
            max_tokens=MAX_TOKENS_TRAVEL_STEP,
            system=system,
            tools=TRAVEL_TOOLS,
            messages=messages,
        )
        _track(response, SONNET)

        tool_use_blocks = [
            b for b in response.content
            if getattr(b, "type", None) == "tool_use"
        ]

        if not tool_use_blocks:
            raise ValueError(
                "Travel agent ended without calling emit_itinerary."
            )

        # Check if emit_itinerary is being called — if so, return immediately
        for block in tool_use_blocks:
            if block.name == "emit_itinerary":
                transcript.append(f"emit_itinerary(...)")
                return _coerce_travel_plan(block.input), transcript

        # Otherwise, execute non-terminal tools and loop
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in tool_use_blocks:
            result = execute_travel_tool(block.name, block.input)
            transcript.append(f"{block.name}({block.input})")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })
        messages.append({"role": "user", "content": tool_results})

    raise ValueError(
        f"Travel agent exceeded {max_turns} turns without emitting itinerary."
    )
