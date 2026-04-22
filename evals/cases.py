"""Golden-answer eval cases for each LLM pattern.

Each case has:
  - id          : stable identifier
  - pattern     : one of debate / triage / contract / travel
  - inputs      : kwargs fed to the pipeline
  - expects     : natural-language rubric the judge checks
  - must_contain: hard substrings the output must include (if any)
  - must_reject : hard substrings the output must NOT include
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    id: str
    pattern: str
    inputs: dict
    expects: str
    must_contain: tuple[str, ...] = ()
    must_reject: tuple[str, ...] = ()


EVAL_CASES: tuple[EvalCase, ...] = (
    # ── Debate (WS4) — multi-agent + extended thinking ───────────────────────
    EvalCase(
        id="debate_bull_vs_bear_nvda",
        pattern="debate",
        inputs={
            "topic": "Should we add 5% NVDA here given AI capex intensity?",
            "context": (
                "Holdings: AAPL 10, MSFT 5, SPY 3. Risk profile: Moderate. "
                "Macro risk model: P(risk-off)=0.45, label=0."
            ),
        },
        expects=(
            "The BULL analyst argues for NVDA, the BEAR argues against, "
            "the MACRO analyst grounds the call in rates / liquidity, "
            "and the CIO verdict is BULL / BEAR / NEUTRAL with a concrete "
            "recommended_action (e.g. 'Add 3% NVDA…' or 'Hold, revisit if…')."
        ),
    ),

    # ── Triage (WS5.1 healthcare) — structured JSON ──────────────────────────
    EvalCase(
        id="triage_emergency_chest_pain",
        pattern="triage",
        inputs={
            "presentation": (
                "58M acute central chest pain radiating to left arm, "
                "diaphoresis, nausea. Onset 20 min ago at rest."
            ),
            "patient_context": (
                "HTN, hyperlipidaemia, ex-smoker 10y, on atorvastatin + amlodipine."
            ),
        },
        expects=(
            "Severity is EMERGENCY. Differential includes acute coronary "
            "syndrome / MI. Recommendations lead with emergency services / "
            "immediate ECG. Red flags include chest pain + diaphoresis."
        ),
        must_contain=("EMERGENCY",),
        must_reject=("prescription dose",),
    ),
    EvalCase(
        id="triage_moderate_fever_cough",
        pattern="triage",
        inputs={
            "presentation": (
                "42F, fever 39°C for 4 days, productive yellow sputum, "
                "dyspnoea on exertion."
            ),
            "patient_context": "No PMH. Non-smoker. BMI 24.",
        },
        expects=(
            "Severity is MODERATE or HIGH (not EMERGENCY or LOW). "
            "Differential includes bacterial pneumonia and viral URI. "
            "Recommends tests (CBC, CXR) before any antibiotic decision."
        ),
    ),

    # ── Contract review (WS5.1 legal) — single-call summarisation ───────────
    EvalCase(
        id="contract_auto_renewal_risk",
        pattern="contract",
        inputs={
            "clause_text": (
                "This Service Agreement auto-renews for successive 12-month "
                "terms unless written notice of non-renewal is given at least "
                "90 days prior. Vendor may raise fees by up to 15% per annum "
                "on renewal without further notice."
            ),
            "perspective": "buyer",
        },
        expects=(
            "Plain summary mentions auto-renewal and 90-day notice. "
            "Key obligations include the notice period and fee acceptance. "
            "At least one HIGH-severity risk (auto-renewal lock-in or "
            "price escalation). Negotiation suggestions include making "
            "renewal opt-in or capping fee increases."
        ),
        must_contain=("HIGH",),
    ),

    # ── Travel planner (WS5.1 travel) — agentic tool-use loop ───────────────
    EvalCase(
        id="travel_barcelona_weekend",
        pattern="travel",
        inputs={
            "destination": "Barcelona",
            "days": 2,
            "traveller_profile": (
                "Couple, mid-30s, love food markets and architecture. "
                "Mid-range budget. No dietary restrictions."
            ),
        },
        expects=(
            "Itinerary covers 2 days with morning/afternoon/evening "
            "activities. Each day has a weather summary. Activities include "
            "recognisable Barcelona landmarks (Sagrada Familia, Boqueria, "
            "Park Güell, Gothic Quarter, or similar). Estimated cost > 0."
        ),
        must_contain=("Barcelona",),
    ),
)
