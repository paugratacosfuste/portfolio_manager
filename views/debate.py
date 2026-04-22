"""Debate (CIO) — Multi-agent debate view (LLM pattern #6).

Three Haiku analysts (BULL / BEAR / MACRO) each produce a structured
`AnalystArgument`. An Opus CIO judge with extended thinking reads all
three and issues a `CIOVerdict`. This demonstrates:

  - Multi-agent coordination (three specialized personas)
  - JSON-schema-constrained structured output
  - Extended thinking (Opus 4.x with budget_tokens)
  - Prompt caching via cache_control: ephemeral

All trade-suggesting output is prefaced with a disclaimer banner.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from core.debate_types import AnalystArgument, CIOVerdict, DebateResult
from utils.ai_advisor import client as anthropic_client
from utils.debate import run_debate


TOPIC_TEMPLATES: dict[str, str] = {
    "Custom…": "",
    "Add 5% NVDA here?": (
        "Should we add a 5% NVDA position to the portfolio given current "
        "AI capex intensity and the recent pullback?"
    ),
    "Rotate growth → value?": (
        "Should we rotate 10% of our portfolio out of growth equities "
        "into value/defensives at this point in the cycle?"
    ),
    "Cut BTC exposure?": (
        "Should we reduce our crypto exposure ahead of potential "
        "macro turbulence, or hold the current weight?"
    ),
    "Raise cash to 20%?": (
        "Should we raise cash to 20% of the portfolio as a tactical "
        "defensive move given current macro signals?"
    ),
}


# ── Context builder ───────────────────────────────────────────────────────────


def _portfolio_context() -> str:
    """Summarize the current portfolio + latest signals for the analysts.

    Pulls from st.session_state — the shared signal registry already
    populated by Macro Radar, News, Efficient Frontier, etc.
    """
    holdings = st.session_state.get("holdings", {}) or {}
    profile = st.session_state.get("profile", {}) or {}
    macro = st.session_state.get("macro_prediction") or {}
    sent = st.session_state.get("portfolio_sentiment") or {}
    stress = st.session_state.get("last_stress_test") or {}

    lines: list[str] = []
    if holdings:
        holdings_str = ", ".join(f"{t} {q}" for t, q in holdings.items())
        lines.append(f"Holdings: {holdings_str}")
    else:
        lines.append("Holdings: (none — grader demo mode)")
    if profile:
        risk_label = profile.get("risk_tolerance") or profile.get("risk") or "unknown"
        horizon = profile.get("time_horizon") or profile.get("horizon") or "unknown"
        lines.append(f"Risk profile: {risk_label}. Horizon: {horizon}.")
    if macro:
        prob = macro.get("probability")
        pred = macro.get("prediction")
        if prob is not None:
            lines.append(
                f"Macro risk model: P(risk-off)={prob:.2f}, label={pred}"
            )
    if sent:
        score = sent.get("score")
        if score is not None:
            lines.append(f"Portfolio-weighted news sentiment: {score:+.2f}")
    if stress:
        name = stress.get("scenario_name") or stress.get("name")
        impact = stress.get("portfolio_impact_pct")
        if name and impact is not None:
            lines.append(f"Last stress test: {name} → {impact:+.1f}%")

    return "\n".join(lines) if lines else "No portfolio context available."


# ── Render helpers ────────────────────────────────────────────────────────────


_ROLE_STYLES = {
    "bull": {"emoji": "", "color": "#1fbf6a", "label": "BULL"},
    "bear": {"emoji": "", "color": "#e04646", "label": "BEAR"},
    "macro": {"emoji": "", "color": "#3a86ff", "label": "MACRO"},
}

_VERDICT_COLORS = {
    "BULL": "#1fbf6a",
    "BEAR": "#e04646",
    "NEUTRAL": "#8a8a8a",
}


def _render_disclaimer() -> None:
    st.markdown(
        "<div style='background:#4a1e1e;color:#f7d4d4;padding:10px 14px;"
        "border-radius:6px;border-left:4px solid #e04646;margin-bottom:14px;"
        "font-size:0.9rem;'>"
        "<b>Educational prototype — not financial advice.</b> "
        "The CIO verdict is a synthesized LLM output for demonstrating "
        "multi-agent debate. Do not trade on it."
        "</div>",
        unsafe_allow_html=True,
    )


def _render_analyst_card(arg: AnalystArgument) -> None:
    style = _ROLE_STYLES[arg.role]
    conviction_pct = int(arg.conviction * 100)
    st.markdown(
        f"<div style='border-left:4px solid {style['color']};"
        f"padding:10px 14px;background:#1b1b1b;border-radius:6px;"
        f"margin-bottom:8px;'>"
        f"<div style='color:{style['color']};font-weight:700;font-size:1.05rem;'>"
        f"{style['label']}</div>"
        f"<div style='color:#e8e8e8;margin-top:6px;'>{arg.claim}</div>"
        f"<div style='color:#999;font-size:0.85rem;margin-top:8px;'>"
        f"Conviction: <b style='color:{style['color']};'>{conviction_pct}%</b>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    with st.expander(f"{style['label']} — evidence & risks", expanded=False):
        st.markdown("**Evidence**")
        for e in arg.evidence:
            st.markdown(f"- {e}")
        st.markdown("**Risks acknowledged**")
        for r in arg.risks:
            st.markdown(f"- {r}")


def _render_verdict(verdict: CIOVerdict) -> None:
    color = _VERDICT_COLORS[verdict.verdict]
    confidence_pct = int(verdict.confidence * 100)
    st.markdown(
        f"<div style='border:2px solid {color};padding:16px 20px;"
        f"border-radius:8px;background:#141414;margin-bottom:14px;'>"
        f"<div style='color:{color};font-size:1.3rem;font-weight:800;"
        f"letter-spacing:1px;'>CIO VERDICT: {verdict.verdict}</div>"
        f"<div style='color:#aaa;font-size:0.9rem;margin-top:4px;'>"
        f"Confidence: <b style='color:{color};'>{confidence_pct}%</b></div>"
        f"<div style='color:#e8e8e8;margin-top:12px;line-height:1.5;'>"
        f"{verdict.reasoning}</div>"
        f"<div style='margin-top:14px;padding:10px;background:#1f1f1f;"
        f"border-radius:4px;'>"
        f"<div style='color:#ccc;font-size:0.8rem;text-transform:uppercase;"
        f"letter-spacing:0.5px;'>Recommended action</div>"
        f"<div style='color:#fff;font-weight:600;margin-top:4px;'>"
        f"{verdict.recommended_action}</div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Key risks the CIO flagged", expanded=False):
        for risk in verdict.key_risks:
            st.markdown(f"- {risk}")


def _render_cost_card(result: DebateResult) -> None:
    cols = st.columns(3)
    cols[0].metric("Input tokens", f"{result.total_input_tokens:,}")
    cols[1].metric("Output tokens", f"{result.total_output_tokens:,}")
    cols[2].metric(
        "Thinking tokens",
        f"{result.total_thinking_tokens:,}",
        help=(
            "Opus extended-thinking budget used by the CIO judge. "
            "Billed as output tokens at the Opus rate."
        ),
    )


# ── Main entry point ─────────────────────────────────────────────────────────


def render_debate() -> None:
    st.markdown(
        "<h1>Debate (CIO) — Multi-Agent Analyst Panel</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#aaa;'>Three analysts (BULL / BEAR / MACRO) argue "
        "a topic. An Opus CIO judge with extended thinking weighs them "
        "and issues an actionable verdict.</p>",
        unsafe_allow_html=True,
    )
    _render_disclaimer()

    with st.expander("About this pattern (LLM integration #6)", expanded=False):
        st.markdown(
            "- **Analysts** run on `claude-haiku-4-5-20251001` — three parallel "
            "structured-output calls, JSON-schema-constrained.\n"
            "- **Judge** runs on `claude-opus-4-7` with "
            "`thinking={'type': 'enabled', 'budget_tokens': 6000}`. "
            "The extended-thinking trace is surfaced in the collapsible "
            "panel below the verdict.\n"
            "- **Caching** — every system prompt is sent with "
            "`cache_control: ephemeral` at the final content block, so "
            "repeated debates on the same portfolio context hit the cache.\n"
            "- **Why multi-agent:** forcing specialization (BULL can't "
            "hedge, BEAR can't handwave 'market risk') produces sharper "
            "arguments than a single polymath prompt."
        )

    if anthropic_client is None:
        st.error(
            "Anthropic client not initialized. Set `ANTHROPIC_API_KEY` "
            "in your `.env` and restart."
        )
        return

    col_a, col_b = st.columns([3, 2])
    with col_a:
        template_name = st.selectbox(
            "Preset or custom topic",
            list(TOPIC_TEMPLATES.keys()),
            index=1,
        )
    with col_b:
        st.write("")
        st.write("")
        run_clicked = st.button(
            "Run debate",
            type="primary",
            use_container_width=True,
        )

    default_topic = TOPIC_TEMPLATES[template_name]
    topic = st.text_area(
        "Debate topic",
        value=default_topic,
        height=80,
        placeholder="e.g. 'Should we add 5% NVDA to the portfolio here?'",
    )

    context = _portfolio_context()
    with st.expander("Context passed to all analysts + judge", expanded=False):
        st.code(context, language="text")

    if run_clicked:
        if len(topic.strip()) < 5:
            st.warning("Enter a topic of at least 5 characters.")
            return
        _execute_debate(topic.strip(), context)

    cached = st.session_state.get("debate_verdict")
    if cached is not None and not run_clicked:
        st.markdown("### Last debate")
        _render_debate_result(cached)


def _execute_debate(topic: str, context: str) -> None:
    progress = st.empty()
    with st.spinner("BULL analyst arguing…"):
        progress.info("Step 1/4 — BULL analyst")
    try:
        with st.spinner("Running debate (3 analysts + 1 judge with extended thinking)…"):
            result = run_debate(topic, context, anthropic_client)
    except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
        progress.empty()
        st.error(f"Debate failed: {exc}")
        return
    progress.empty()
    st.session_state["debate_verdict"] = result
    _render_debate_result(result)


def _render_debate_result(result: DebateResult) -> None:
    st.markdown("### Verdict")
    _render_verdict(result.verdict)

    if result.judge_thinking:
        with st.expander("CIO extended-thinking trace", expanded=False):
            st.markdown(
                "<div style='color:#999;font-size:0.85rem;margin-bottom:6px;'>"
                "Summarized reasoning from the Opus judge. This is what the "
                "model 'thought' before committing to a verdict."
                "</div>",
                unsafe_allow_html=True,
            )
            st.code(result.judge_thinking, language="text")

    st.markdown("### Analyst arguments")
    col_bull, col_bear, col_macro = st.columns(3)
    with col_bull:
        _render_analyst_card(result.bull)
    with col_bear:
        _render_analyst_card(result.bear)
    with col_macro:
        _render_analyst_card(result.macro)

    st.markdown("### Token usage (this debate)")
    _render_cost_card(result)
