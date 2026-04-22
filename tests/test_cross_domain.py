"""Tests for utils/cross_domain.py — 3 patterns × 3 domains gallery.

Anthropic client is mocked via SimpleNamespace. No network I/O.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from core.cross_domain_types import (
    ContractReview,
    TravelPlan,
    TriageAssessment,
)
from utils.cross_domain import (
    HAIKU,
    SONNET,
    TRAVEL_TOOLS,
    _coerce_travel_plan,
    _extract_json,
    execute_travel_tool,
    run_contract_review,
    run_travel_agent,
    run_triage,
)


def _text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(name: str, args: dict, tool_id: str = "tu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=args, id=tool_id)


def _usage(inp=100, out=50):
    return SimpleNamespace(input_tokens=inp, output_tokens=out)


# ── Shared helpers ────────────────────────────────────────────────────────────


class TestExtractJSON:
    def test_plain(self):
        assert _extract_json('{"k": 1}') == {"k": 1}

    def test_prose_prefixed(self):
        assert _extract_json('Here you go: {"k": 1} cheers') == {"k": 1}

    def test_unbalanced_raises(self):
        with pytest.raises(ValueError):
            _extract_json('{"a":')


# ── Healthcare triage (pattern #3) ───────────────────────────────────────────


_TRIAGE_JSON = """\
{
  "severity": "MODERATE",
  "confidence": 0.75,
  "differential": ["viral URI", "bacterial pneumonia"],
  "recommendations": [
    {"kind": "test", "label": "CBC with diff",
     "reason": "Differentiate viral vs bacterial given elevated WBC."}
  ],
  "red_flags": ["persistent fever > 3 days"],
  "plain_language_summary": "Likely a moderate respiratory infection. Further workup recommended before antibiotics."
}"""


class TestRunTriage:
    def test_happy_path(self):
        client = MagicMock()
        client.messages.create.return_value = SimpleNamespace(
            content=[_text_block(_TRIAGE_JSON)],
            usage=_usage(),
        )

        result = run_triage(
            presentation="Fever 39°C, productive cough 4 days, elevated WBC.",
            patient_context="42F, no significant PMH, non-smoker.",
            client=client,
        )

        assert isinstance(result, TriageAssessment)
        assert result.severity == "MODERATE"
        assert result.confidence == 0.75
        assert result.recommendations[0].kind == "test"

    def test_uses_haiku_and_cached_system(self):
        client = MagicMock()
        client.messages.create.return_value = SimpleNamespace(
            content=[_text_block(_TRIAGE_JSON)],
            usage=_usage(),
        )

        run_triage("symptoms", "context", client)

        call = client.messages.create.call_args
        assert call.kwargs["model"] == HAIKU
        system = call.kwargs["system"]
        assert system[-1]["cache_control"] == {"type": "ephemeral"}

    def test_prose_wrapped_json_parses(self):
        client = MagicMock()
        client.messages.create.return_value = SimpleNamespace(
            content=[_text_block("Here is my assessment:\n" + _TRIAGE_JSON)],
            usage=_usage(),
        )

        result = run_triage("s", "c", client)
        assert result.severity == "MODERATE"

    def test_schema_violation_raises(self):
        bad = '{"severity": "TOTALLY_FINE", "confidence": 0.5, "differential": ["x"], "recommendations": [{"kind": "test", "label": "abc", "reason": "abcdefghij"}], "plain_language_summary": "' + "z" * 30 + '"}'
        client = MagicMock()
        client.messages.create.return_value = SimpleNamespace(
            content=[_text_block(bad)],
            usage=_usage(),
        )

        with pytest.raises(ValidationError):
            run_triage("s", "c", client)


# ── Legal contract review (pattern #1) ───────────────────────────────────────


_CONTRACT_JSON = """\
{
  "plain_summary": "This SaaS agreement forces the customer to pay even if the vendor's service is degraded below 99.0% uptime, unless they cancel with 90 days' notice.",
  "parties": ["Acme SaaS", "Customer Inc."],
  "key_obligations": ["Monthly payment", "SLA 99.9%", "90-day notice to cancel"],
  "risks": [
    {"severity": "HIGH", "label": "Auto-renewal trap",
     "description": "Contract auto-renews for 12 months unless cancelled 90 days prior. Risk of lock-in and billing surprise."}
  ],
  "negotiation_suggestions": ["Replace auto-renewal with explicit opt-in."]
}"""


class TestRunContractReview:
    def test_happy_path(self):
        client = MagicMock()
        client.messages.create.return_value = SimpleNamespace(
            content=[_text_block(_CONTRACT_JSON)],
            usage=_usage(),
        )

        review = run_contract_review("Clause text here...", client)

        assert isinstance(review, ContractReview)
        assert review.risks[0].severity == "HIGH"
        assert "Acme SaaS" in review.parties

    def test_uses_haiku(self):
        client = MagicMock()
        client.messages.create.return_value = SimpleNamespace(
            content=[_text_block(_CONTRACT_JSON)],
            usage=_usage(),
        )

        run_contract_review("Clause text", client)
        assert client.messages.create.call_args.kwargs["model"] == HAIKU

    def test_perspective_reaches_prompt(self):
        client = MagicMock()
        client.messages.create.return_value = SimpleNamespace(
            content=[_text_block(_CONTRACT_JSON)],
            usage=_usage(),
        )

        run_contract_review("Clause", client, perspective="vendor")
        user_content = client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "vendor" in user_content


# ── Travel agent (pattern #2: agentic tool-use loop) ─────────────────────────


class TestTravelTools:
    def test_all_four_tools_present(self):
        names = {t["name"] for t in TRAVEL_TOOLS}
        assert names == {
            "get_destination_weather",
            "search_flights_mock",
            "search_hotels_mock",
            "emit_itinerary",
        }


class TestExecuteTravelTool:
    def test_weather(self):
        import json as _json
        result = _json.loads(
            execute_travel_tool("get_destination_weather",
                                {"city": "Barcelona", "days": 3})
        )
        assert result["city"] == "Barcelona"
        assert len(result["forecast"]) == 3

    def test_flights_mocked_price(self):
        import json as _json
        result = _json.loads(
            execute_travel_tool("search_flights_mock",
                                {"origin": "NYC", "destination": "BCN"})
        )
        assert result["best_price_usd"] == 420

    def test_hotels_budget_tier(self):
        import json as _json
        mid = _json.loads(
            execute_travel_tool("search_hotels_mock",
                                {"city": "BCN", "budget_tier": "mid"})
        )
        luxury = _json.loads(
            execute_travel_tool("search_hotels_mock",
                                {"city": "BCN", "budget_tier": "luxury"})
        )
        assert luxury["nightly_rate_usd"] > mid["nightly_rate_usd"]

    def test_unknown_tool_returns_error(self):
        import json as _json
        result = _json.loads(execute_travel_tool("unknown", {}))
        assert "error" in result


# ── Travel agent loop ────────────────────────────────────────────────────────


_VALID_ITINERARY = {
    "destination": "Barcelona",
    "traveller_profile": "Couple, 30s, foodie",
    "days": [
        {
            "day_index": 1,
            "date_label": "Day 1",
            "weather_summary": "Sunny, 24°C",
            "activities": [
                {
                    "time_of_day": "morning",
                    "title": "Sagrada Família",
                    "location": "Barcelona",
                    "notes": "Book tickets.",
                },
                {
                    "time_of_day": "evening",
                    "title": "Tapas at El Xampanyet",
                    "location": "Barcelona",
                },
            ],
        }
    ],
    "estimated_total_cost_usd": 1500.0,
    "packing_tips": ["Walking shoes", "Light jacket"],
}


class TestCoerceTravelPlan:
    def test_basic(self):
        plan = _coerce_travel_plan(_VALID_ITINERARY)
        assert isinstance(plan, TravelPlan)
        assert plan.days[0].activities[1].time_of_day == "evening"

    def test_defaults_missing_optional_fields(self):
        args = {
            "destination": "Lisbon",
            "traveller_profile": "Solo hiker, fit",
            "days": [{
                "day_index": 1,
                "date_label": "Day 1",
                "activities": [{
                    "time_of_day": "morning",
                    "title": "Alfama walk",
                    "location": "Lisbon",
                }],
            }],
            "estimated_total_cost_usd": 500,
        }
        plan = _coerce_travel_plan(args)
        assert plan.days[0].weather_summary == ""
        assert plan.packing_tips == ()


class TestRunTravelAgent:
    def test_emits_itinerary_after_tool_calls(self):
        """Turn 1 -> agent calls weather + flights (parallel).
        Turn 2 -> after tool_results, agent calls emit_itinerary."""
        client = MagicMock()
        turn1 = SimpleNamespace(
            content=[
                _tool_use_block(
                    "get_destination_weather",
                    {"city": "Barcelona", "days": 1},
                    tool_id="tu_w",
                ),
                _tool_use_block(
                    "search_flights_mock",
                    {"origin": "NYC", "destination": "BCN"},
                    tool_id="tu_f",
                ),
            ],
            usage=_usage(200, 80),
        )
        turn2 = SimpleNamespace(
            content=[
                _tool_use_block(
                    "emit_itinerary", _VALID_ITINERARY, tool_id="tu_e",
                ),
            ],
            usage=_usage(500, 150),
        )
        client.messages.create.side_effect = [turn1, turn2]

        plan, transcript = run_travel_agent(
            destination="Barcelona",
            days=1,
            traveller_profile="Couple, 30s",
            client=client,
        )

        assert isinstance(plan, TravelPlan)
        assert plan.destination == "Barcelona"
        assert "get_destination_weather" in transcript[0]
        assert "search_flights_mock" in transcript[1]
        assert "emit_itinerary" in transcript[-1]

    def test_uses_sonnet(self):
        client = MagicMock()
        client.messages.create.return_value = SimpleNamespace(
            content=[_tool_use_block("emit_itinerary", _VALID_ITINERARY)],
            usage=_usage(),
        )

        run_travel_agent("Lisbon", 1, "Solo", client)
        assert client.messages.create.call_args.kwargs["model"] == SONNET

    def test_no_tool_calls_raises(self):
        """If agent goes silent without calling emit, raise."""
        client = MagicMock()
        client.messages.create.return_value = SimpleNamespace(
            content=[_text_block("I have no idea where to send you.")],
            usage=_usage(),
        )

        with pytest.raises(ValueError, match="ended without"):
            run_travel_agent("Barcelona", 2, "Couple", client)

    def test_turn_budget_exhaustion_raises(self):
        """Agent keeps calling non-terminal tools forever."""
        client = MagicMock()
        never_emits = SimpleNamespace(
            content=[
                _tool_use_block(
                    "get_destination_weather",
                    {"city": "BCN", "days": 1},
                )
            ],
            usage=_usage(),
        )
        client.messages.create.side_effect = [never_emits] * 10

        with pytest.raises(ValueError, match="exceeded"):
            run_travel_agent("Barcelona", 1, "Solo", client, max_turns=3)
