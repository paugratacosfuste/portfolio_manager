"""Cross-Domain Pattern Gallery.

The same LLM integration patterns that power the portfolio tracker,
applied to three unrelated domains to demonstrate transferability:

  - Healthcare triage   -> structured JSON generation (Haiku)
  - Legal contract      -> single-call summarisation   (Haiku)
  - Travel planning     -> agentic tool-use loop       (Sonnet)

Every domain view renders a domain-specific disclaimer banner.
No real advice is given.
"""
from __future__ import annotations

import streamlit as st

from core.cross_domain_types import (
    ContractReview,
    TravelPlan,
    TriageAssessment,
)
from utils.ai_advisor import client as anthropic_client
from utils.cross_domain import (
    run_contract_review,
    run_travel_agent,
    run_triage,
)


# ── Sample inputs (grader-friendly one-click demos) ──────────────────────────


TRIAGE_PRESETS = {
    "Adult fever + productive cough": (
        "42F. Fever 39°C for 4 days, productive cough with yellow sputum, "
        "dyspnoea on exertion. Recent travel: none. Vaccinations up to date.",
        "No significant past medical history. Non-smoker. BMI 24.",
    ),
    "Chest pain + diaphoresis": (
        "58M, acute-onset central chest pain radiating to left arm, "
        "profuse sweating, nausea. Pain started 20 minutes ago at rest.",
        "Hypertension, hyperlipidaemia. Ex-smoker 10 years. "
        "On atorvastatin and amlodipine.",
    ),
    "Child rash + low-grade fever": (
        "4yo, 3 days of diffuse maculopapular rash starting on face, now on trunk. "
        "Low-grade fever 37.8°C. No neck stiffness.",
        "Fully immunised per national schedule. No sick contacts at nursery.",
    ),
}


CONTRACT_PRESETS = {
    "SaaS auto-renewal": """\
This Service Agreement shall commence on the Effective Date and continue for an
initial term of twelve (12) months ("Initial Term"). Thereafter, the Agreement
shall automatically renew for successive twelve-month terms unless either
party provides written notice of non-renewal at least ninety (90) days prior
to the end of the then-current term. Upon renewal, Vendor may increase fees
by up to fifteen percent (15%) per annum without further notice. Customer's
continued use of the Services after the renewal date constitutes acceptance
of the renewed terms and any fee increases.""",
    "Limitation of liability": """\
In no event shall Vendor's aggregate liability arising out of or related to
this Agreement, whether in contract, tort, or otherwise, exceed the lesser
of (a) the fees paid by Customer to Vendor in the six (6) months preceding
the event giving rise to the claim, or (b) one hundred U.S. dollars ($100).
Vendor shall not be liable for any indirect, incidental, special,
consequential, or punitive damages, including loss of profits, loss of data,
or business interruption, regardless of whether Vendor has been advised of
the possibility of such damages.""",
    "IP assignment (employment)": """\
Employee hereby assigns to Company all right, title, and interest, including
all intellectual property rights, in and to any inventions, works of authorship,
or other creations conceived, developed, or reduced to practice by Employee,
whether alone or with others, during the term of employment OR within twelve
(12) months thereafter, whether or not such creations relate to Company's
business and whether or not developed on Company time or using Company
resources. Employee waives all moral rights in such creations to the maximum
extent permitted by applicable law.""",
}


TRAVEL_PRESETS = {
    "Long weekend Barcelona, foodie couple": (
        "Barcelona",
        3,
        "Couple, mid-30s. Love food markets, modern architecture, "
        "and short walks. No dietary restrictions. Mid-range budget.",
    ),
    "Solo 5-day Lisbon": (
        "Lisbon",
        5,
        "Solo traveller, fit, age 28. Interested in history, viewpoints, "
        "and local music. Budget tier.",
    ),
    "Family 4-day Tokyo": (
        "Tokyo",
        4,
        "Family of four with two kids (ages 7 and 10). Need kid-friendly "
        "activities and an accessible base location. Mid-range budget.",
    ),
}


# ── Disclaimers ──────────────────────────────────────────────────────────────


_DISCLAIMER_STYLE = (
    "background:#4a1e1e;color:#f7d4d4;padding:10px 14px;"
    "border-radius:6px;border-left:4px solid #e04646;"
    "margin-bottom:14px;font-size:0.9rem;"
)


def _banner(text: str) -> None:
    st.markdown(
        f"<div style='{_DISCLAIMER_STYLE}'>{text}</div>",
        unsafe_allow_html=True,
    )


# ── Domain: Healthcare (pattern #3 — structured JSON) ────────────────────────


_SEVERITY_COLOR = {
    "LOW": "#1fbf6a",
    "MODERATE": "#c7a83e",
    "HIGH": "#e08640",
    "EMERGENCY": "#e04646",
}


def _render_triage_result(assessment: TriageAssessment) -> None:
    color = _SEVERITY_COLOR[assessment.severity]
    st.markdown(
        f"<div style='border:2px solid {color};padding:14px 18px;"
        f"border-radius:8px;background:#141414;margin:14px 0;'>"
        f"<div style='color:{color};font-size:1.2rem;font-weight:700;"
        f"letter-spacing:1px;'>SEVERITY: {assessment.severity}</div>"
        f"<div style='color:#aaa;font-size:0.85rem;margin-top:4px;'>"
        f"Confidence: <b style='color:{color};'>"
        f"{int(assessment.confidence * 100)}%</b></div>"
        f"<div style='color:#e8e8e8;margin-top:10px;line-height:1.5;'>"
        f"{assessment.plain_language_summary}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Differential**")
        for dx in assessment.differential:
            st.markdown(f"- {dx}")
    with col_b:
        if assessment.red_flags:
            st.markdown("**Red flags**")
            for rf in assessment.red_flags:
                st.markdown(f"- {rf}")

    st.markdown("**Recommended next steps**")
    for rec in assessment.recommendations:
        st.markdown(f"- **[{rec.kind}]** {rec.label} — {rec.reason}")


def _render_healthcare_tab() -> None:
    _banner(
        "<b>Educational prototype — not a medical device.</b> "
        "This tool is for demonstration only. It is not CE-marked, not "
        "FDA-cleared, and must not be used for real clinical decisions."
    )
    st.markdown(
        "### Structured JSON generation (pattern #3) — "
        "Clinical decision support"
    )
    st.markdown(
        "<p style='color:#aaa;'>Claude Haiku produces a "
        "JSON-schema-validated <code>TriageAssessment</code> with "
        "severity, differential, and recommended next steps.</p>",
        unsafe_allow_html=True,
    )

    preset = st.selectbox(
        "Sample presentation", list(TRIAGE_PRESETS.keys()), key="triage_preset"
    )
    default_present, default_context = TRIAGE_PRESETS[preset]
    presentation = st.text_area(
        "Presenting complaint / symptoms",
        value=default_present,
        height=120,
        key="triage_present",
    )
    context = st.text_area(
        "Patient context (PMH, meds, relevant history)",
        value=default_context,
        height=80,
        key="triage_context",
    )

    if st.button("Run triage", type="primary", key="triage_run"):
        if not anthropic_client:
            st.error("Anthropic client not initialized.")
            return
        if len(presentation.strip()) < 10:
            st.warning("Enter a presentation of at least 10 characters.")
            return
        try:
            with st.spinner("Claude is assessing…"):
                assessment = run_triage(
                    presentation.strip(),
                    context.strip() or "No additional context.",
                    anthropic_client,
                )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Triage failed: {exc}")
            return
        _render_triage_result(assessment)


# ── Domain: Legal (pattern #1 — single-call summarisation) ──────────────────


_RISK_COLOR = {
    "LOW": "#1fbf6a",
    "MEDIUM": "#c7a83e",
    "HIGH": "#e04646",
}


def _render_contract_result(review: ContractReview) -> None:
    st.markdown(
        f"<div style='padding:12px 16px;border-radius:8px;"
        f"background:#141414;margin:12px 0;'>"
        f"<div style='color:#ccc;font-size:0.8rem;"
        f"text-transform:uppercase;letter-spacing:0.5px;'>"
        f"Plain-English summary</div>"
        f"<div style='color:#e8e8e8;margin-top:8px;line-height:1.55;'>"
        f"{review.plain_summary}</div></div>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if review.parties:
            st.markdown("**Parties**")
            for p in review.parties:
                st.markdown(f"- {p}")
        st.markdown("**Key obligations**")
        for ob in review.key_obligations:
            st.markdown(f"- {ob}")
    with col_b:
        if review.risks:
            st.markdown("**Risk flags**")
            for risk in review.risks:
                color = _RISK_COLOR[risk.severity]
                st.markdown(
                    f"<div style='border-left:3px solid {color};"
                    f"padding:6px 10px;background:#1b1b1b;"
                    f"border-radius:4px;margin-bottom:6px;'>"
                    f"<span style='color:{color};font-weight:700;'>"
                    f"[{risk.severity}]</span> "
                    f"<b>{risk.label}</b><br>"
                    f"<span style='color:#ccc;font-size:0.9rem;'>"
                    f"{risk.description}</span></div>",
                    unsafe_allow_html=True,
                )
        if review.negotiation_suggestions:
            st.markdown("**Suggested pushback**")
            for s in review.negotiation_suggestions:
                st.markdown(f"- {s}")


def _render_legal_tab() -> None:
    _banner(
        "<b>Educational prototype — not legal advice.</b> "
        "This tool does not create a lawyer-client relationship and "
        "outputs may be inaccurate or incomplete. Consult a licensed "
        "attorney for real contract review."
    )
    st.markdown(
        "### Single-call summarisation (pattern #1) — Contract review"
    )
    st.markdown(
        "<p style='color:#aaa;'>Claude Haiku produces a plain-English "
        "summary, extracts obligations, and flags risky clauses as a "
        "<code>ContractReview</code> JSON object.</p>",
        unsafe_allow_html=True,
    )

    preset = st.selectbox(
        "Sample clause", list(CONTRACT_PRESETS.keys()), key="contract_preset"
    )
    clause_text = st.text_area(
        "Contract clause",
        value=CONTRACT_PRESETS[preset],
        height=220,
        key="contract_clause",
    )
    perspective = st.radio(
        "Review from whose perspective?",
        ["buyer", "seller", "neutral"],
        horizontal=True,
        key="contract_perspective",
    )

    if st.button("Review clause", type="primary", key="contract_run"):
        if not anthropic_client:
            st.error("Anthropic client not initialized.")
            return
        if len(clause_text.strip()) < 30:
            st.warning("Paste a clause of at least 30 characters.")
            return
        try:
            with st.spinner("Claude is reviewing…"):
                review = run_contract_review(
                    clause_text.strip(),
                    anthropic_client,
                    perspective=perspective,
                )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Review failed: {exc}")
            return
        _render_contract_result(review)


# ── Domain: Travel (pattern #2 — agentic tool-use loop) ─────────────────────


def _render_travel_result(plan: TravelPlan, transcript: list[str]) -> None:
    st.markdown(
        f"<div style='padding:12px 16px;border-radius:8px;"
        f"background:#141414;margin:12px 0;'>"
        f"<div style='color:#ccc;font-size:0.8rem;"
        f"text-transform:uppercase;letter-spacing:0.5px;'>"
        f"Itinerary — {plan.destination}</div>"
        f"<div style='color:#e8e8e8;margin-top:6px;'>"
        f"{plan.traveller_profile}</div>"
        f"<div style='color:#1fbf6a;margin-top:10px;font-size:1.1rem;'>"
        f"Estimated total: ${plan.estimated_total_cost_usd:,.0f} USD "
        f"<span style='color:#888;font-size:0.8rem;'>(mocked prices)</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    for day in plan.days:
        st.markdown(
            f"#### {day.date_label}"
            f" — <span style='color:#aaa;font-size:0.9rem;font-weight:400;'>"
            f"{day.weather_summary}</span>",
            unsafe_allow_html=True,
        )
        for act in day.activities:
            st.markdown(
                f"- **{act.time_of_day.title()}** · {act.title} "
                f"<span style='color:#888;'>({act.location})</span>"
                + (f"<br><span style='color:#bbb;font-size:0.9rem;'>"
                   f"{act.notes}</span>" if act.notes else ""),
                unsafe_allow_html=True,
            )

    if plan.packing_tips:
        with st.expander("Packing tips", expanded=False):
            for tip in plan.packing_tips:
                st.markdown(f"- {tip}")

    with st.expander("Agent tool-call transcript", expanded=False):
        st.markdown(
            "<div style='color:#999;font-size:0.85rem;'>Every line is a "
            "tool call Claude autonomously made before emitting the "
            "itinerary.</div>",
            unsafe_allow_html=True,
        )
        for line in transcript:
            st.code(line, language="text")


def _render_travel_tab() -> None:
    _banner(
        "<b>Educational prototype — not travel advice.</b> "
        "Prices, availability, and weather are <b>mocked</b>. "
        "Do not book trips based on this output."
    )
    st.markdown(
        "### Agentic tool-use loop (pattern #2) — Itinerary planner"
    )
    st.markdown(
        "<p style='color:#aaa;'>Claude Sonnet autonomously calls "
        "<code>get_destination_weather</code>, <code>search_flights_mock</code>, "
        "<code>search_hotels_mock</code> across turns, then emits a "
        "validated <code>TravelPlan</code> via <code>emit_itinerary</code>.</p>",
        unsafe_allow_html=True,
    )

    preset = st.selectbox(
        "Sample trip", list(TRAVEL_PRESETS.keys()), key="travel_preset"
    )
    default_dest, default_days, default_profile = TRAVEL_PRESETS[preset]
    col_a, col_b = st.columns([2, 1])
    with col_a:
        destination = st.text_input(
            "Destination", value=default_dest, key="travel_dest"
        )
    with col_b:
        days = st.number_input(
            "Days", min_value=1, max_value=7,
            value=default_days, key="travel_days",
        )
    profile = st.text_area(
        "Traveller profile",
        value=default_profile,
        height=100,
        key="travel_profile",
    )

    if st.button("Plan my trip", type="primary", key="travel_run"):
        if not anthropic_client:
            st.error("Anthropic client not initialized.")
            return
        if len(destination.strip()) < 2:
            st.warning("Enter a destination.")
            return
        try:
            with st.spinner("Claude is researching and planning…"):
                plan, transcript = run_travel_agent(
                    destination.strip(),
                    int(days),
                    profile.strip() or "No specific preferences.",
                    anthropic_client,
                )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Travel planner failed: {exc}")
            return
        _render_travel_result(plan, transcript)


# ── Main entry point ─────────────────────────────────────────────────────────


def render_cross_domain_gallery() -> None:
    st.markdown(
        "<h1>Cross-Domain Pattern Gallery</h1>", unsafe_allow_html=True
    )
    st.markdown(
        "<p>The same three LLM integration patterns that power the "
        "portfolio tracker, applied to three unrelated domains. The point "
        "is that these patterns are not about finance — they're about how "
        "you structure Claude calls.</p>",
        unsafe_allow_html=True,
    )

    with st.expander("What's shared vs. what's domain-specific", expanded=False):
        st.markdown(
            "- **Shared code:** the `run_*` orchestrators in "
            "`utils/cross_domain.py`, the frozen Pydantic schemas in "
            "`core/cross_domain_types.py`, the JSON extractor, "
            "the `cache_control: ephemeral` system prompt wrapper, "
            "and the `track_llm_usage` cost accounting.\n"
            "- **Domain-specific:** the system prompt, the Pydantic schema, "
            "and the rendering of the result. Each domain is ~90 lines of "
            "adapter — the pattern plumbing is reused."
        )

    tab_hc, tab_legal, tab_travel = st.tabs(
        ["Healthcare (triage)", "Legal (contract)", "Travel (itinerary)"]
    )
    with tab_hc:
        _render_healthcare_tab()
    with tab_legal:
        _render_legal_tab()
    with tab_travel:
        _render_travel_tab()
