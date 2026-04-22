"""Political Alpha — 5th LLM pattern view.

Three tabs:
  1. Live Tape — fire the agent on curated events, show streaming tool-use
  2. Backtest Scorecard — per-persona PnL stats + equity curve
  3. Ledger Explorer — filter & export the SQLite ledger

No financial advice. Educational prototype only — disclaimer banner is
rendered at the top of every tab.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from core.personas import load_personas
from utils.ai_advisor import client as anthropic_client
from utils.news_backtest import compute_scorecard, replay_headlines
from utils.news_stream import NewsWatcher
from utils.political_alpha_agent import run_news_agent_tick
from utils.political_alpha_view_helpers import (
    entries_to_csv_text,
    entries_to_equity_curve,
    load_curated_events,
    seeded_price_fn,
    yfinance_price_fn,
)
from utils.signal_ledger import SignalLedger


LEDGER_PATH = Path("data/political_alpha.db")
PERSONAS_PATH = Path("data/personas.yaml")
HEADLINES_DIR = Path("data/historical_headlines")
WATCHER_CACHE_DB = Path("data/news_watcher_cache.db")

MAX_EVENTS_PER_POLL = 5
MIN_SALIENCE = 0.35


# ── Shared loaders ────────────────────────────────────────────────────────────


@st.cache_resource
def _load_personas_cached():
    return load_personas(PERSONAS_PATH)


@st.cache_resource
def _ledger() -> SignalLedger:
    ledger = SignalLedger(LEDGER_PATH)
    ledger.initialize()
    return ledger


@st.cache_data(ttl=300)
def _curated_events_cached():
    personas = _load_personas_cached()
    return load_curated_events(personas, HEADLINES_DIR)


@st.cache_resource
def _news_watcher() -> NewsWatcher:
    personas = _load_personas_cached()
    watcher = NewsWatcher(personas=personas, cache_db=WATCHER_CACHE_DB)
    watcher.initialize()
    return watcher


# ── Disclaimer ────────────────────────────────────────────────────────────────


def _render_disclaimer(*, mode: str = "live") -> None:
    if mode == "live":
        body = (
            "Educational prototype — not financial advice. The Political "
            "Alpha agent fires autonomous trade proposals against a live "
            "news feed (GDELT, Truth Social, RSS). Do not use for real trading."
        )
    else:
        body = (
            "Educational prototype — not financial advice. The backtest "
            "replays a curated 2018-2021 archive with real historical "
            "headlines and ground-truth outcomes. For evaluation only."
        )
    st.markdown(
        f"<div style='background:#C44536;color:white;padding:10px 16px;"
        f"border-radius:6px;font-weight:600;margin-bottom:12px;'>"
        f"⚠️ {body}</div>",
        unsafe_allow_html=True,
    )


# ── Tab 1: Live Tape ──────────────────────────────────────────────────────────


def _render_live_tape() -> None:
    _render_disclaimer(mode="live")
    st.markdown("### Live Tape")
    st.caption(
        "Real-time news agent. GDELT 2.0 + Truth Social + RSS (Reuters/CNBC) "
        "are polled on each refresh. Claude Sonnet 4.6 auto-fires on every "
        "unseen, salient headline — calls its tools, checks historical "
        "context, then proposes a trade or skips."
    )

    watcher = _news_watcher()
    personas = _load_personas_cached()
    ledger = _ledger()

    col_a, col_b, col_c = st.columns([2, 2, 2])
    with col_a:
        auto_poll = st.toggle(
            "Auto-poll every 2 min",
            value=False,
            help="When on, the page re-runs every 120s and the agent fires "
                 "on new headlines automatically. Off by default to avoid "
                 "runaway LLM cost.",
        )
    with col_b:
        poll_clicked = st.button(
            "Poll live feed now",
            type="primary",
            use_container_width=True,
        )
    with col_c:
        st.metric("Ledger rows", ledger.count())

    if auto_poll:
        st_autorefresh(interval=120_000, key="pa_live_refresh")

    triggered = poll_clicked or auto_poll

    status_box = st.empty()
    new_decisions: list = []

    if triggered:
        if anthropic_client is None:
            st.error("Anthropic client not initialized — set ANTHROPIC_API_KEY in .env.")
            return

        holdings = st.session_state.get("holdings") or {}
        tickers = tuple(str(t).upper() for t in holdings.keys() if t)
        status_box.info(
            f"Polling GDELT + Truth Social + RSS + yfinance "
            f"({len(tickers)} portfolio tickers)…"
        )
        try:
            fresh_events = watcher.poll(
                min_salience=MIN_SALIENCE,
                portfolio_tickers=tickers,
            )
        except Exception as ex:
            status_box.error(f"Feed poll failed: {ex}")
            return

        if not fresh_events:
            status_box.info(
                "No new salient headlines this cycle. "
                "(Try lowering `MIN_SALIENCE` or wait for the next poll.)"
            )
        else:
            to_process = fresh_events[:MAX_EVENTS_PER_POLL]
            status_box.info(
                f"{len(fresh_events)} new event(s) found — firing agent on "
                f"top {len(to_process)} (capped to protect LLM cost)."
            )

            prog = st.progress(0.0)
            for i, event in enumerate(to_process, 1):
                persona = next(
                    (p for p in personas if p.id == event.persona_id), None
                )
                if persona is None:
                    prog.progress(i / len(to_process))
                    continue
                try:
                    entry = run_news_agent_tick(
                        event=event,
                        persona=persona,
                        ledger=ledger,
                        client=anthropic_client,
                    )
                    if entry is not None:
                        new_decisions.append(entry)
                except Exception as ex:
                    st.warning(f"Agent raised on '{event.headline[:60]}…': {ex}")
                prog.progress(i / len(to_process))

            watcher.mark_seen(to_process)
            prog.empty()
            status_box.success(
                f"Processed {len(to_process)} headline(s) → "
                f"{len(new_decisions)} ledger entries written."
            )

    if new_decisions:
        st.markdown("#### This cycle")
        for e in new_decisions:
            _render_decision_card(e)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("#### Last 10 decisions")
    recent = ledger.list_recent(limit=10)
    if not recent:
        st.info("Ledger is empty. Click **Poll live feed now** above.")
        return
    for e in recent:
        _render_decision_card(e, compact=True)


def _render_decision_card(entry, *, compact: bool = False) -> None:
    """Render a single LedgerEntry as a Streamlit card."""
    is_skip = entry.trade_side == "SKIP"
    color = "#666" if is_skip else ("#1F8A70" if entry.trade_side == "LONG" else "#C44536")
    label = "SKIP" if is_skip else f"{entry.trade_side} {entry.trade_ticker}"
    pnl_text = (
        f"PnL: {entry.realized_pnl_pct:+.2f}%"
        if entry.realized_pnl_pct is not None
        else "Open"
    )
    pnl_text = "—" if is_skip else pnl_text

    header = (
        f"<div style='border-left:4px solid {color};padding:8px 12px;"
        f"margin:6px 0;background:rgba(255,255,255,0.02);'>"
        f"<div style='display:flex;justify-content:space-between;'>"
        f"<span><b>{label}</b> · {entry.persona} · "
        f"{entry.entry_ts.strftime('%Y-%m-%d %H:%M')}</span>"
        f"<span style='color:#888;'>{pnl_text}</span></div>"
        f"<div style='color:#aaa;font-size:0.9em;margin-top:4px;'>"
        f"{entry.trigger_headline}</div>"
    )
    if not compact:
        header += (
            f"<div style='color:#ddd;font-size:0.9em;margin-top:6px;"
            f"font-style:italic;'>{entry.hypothesis}</div>"
        )
        if not is_skip:
            header += (
                f"<div style='color:#888;font-size:0.8em;margin-top:4px;'>"
                f"size={entry.trade_size_pct:.1%} · horizon={entry.horizon_hours}h"
                f"</div>"
            )
    header += "</div>"
    st.markdown(header, unsafe_allow_html=True)


# ── Tab 2: Backtest Scorecard ─────────────────────────────────────────────────


def _render_scorecard() -> None:
    _render_disclaimer(mode="historical")
    st.markdown("### Historical Evaluation")
    st.caption(
        "Replay the curated 2018-2021 archive (real Reuters / Fed / ECB / "
        "Twitter headlines with source URLs) through the agent policy, "
        "then mark-to-market via an injected price function. Ground-truth "
        "outcomes are known — this is how we score the agent."
    )

    personas = _load_personas_cached()
    ledger = _ledger()

    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])
    with col1:
        price_mode = st.radio(
            "Price function",
            ["Seeded demo", "yfinance (real)"],
            horizontal=True,
        )
    with col2:
        backtest_model = st.radio(
            "Agent model",
            ["Haiku (fast)", "Sonnet (accurate)"],
            horizontal=True,
            help="Haiku is ~3x faster and cheap. Sonnet matches the live-tape model exactly.",
        )
    with col3:
        max_events = st.slider(
            "Max events per persona",
            min_value=3,
            max_value=30,
            value=8,
            help="Each event costs one LLM call. Lower = faster demo.",
        )
    with col4:
        run_backtest = st.button(
            "Run backtest on curated events",
            use_container_width=True,
            type="primary",
        )
    with col5:
        st.metric("Ledger rows", ledger.count())

    if run_backtest:
        if anthropic_client is None:
            st.error("Set ANTHROPIC_API_KEY in .env to run the backtest.")
            return

        price_fn = (
            yfinance_price_fn() if price_mode == "yfinance (real)" else seeded_price_fn()
        )
        csvs = sorted(HEADLINES_DIR.glob("*.csv"))
        if not csvs:
            st.warning("No curated headline CSVs found.")
            return

        total_expected = len(csvs) * max_events
        progress = st.progress(0.0, text="Replaying events…")
        status = st.empty()
        all_reports = []
        processed = 0

        model_override = (
            "claude-haiku-4-5-20251001" if backtest_model == "Haiku (fast)" else None
        )

        def _policy(event, led, portfolio):
            persona = next((p for p in personas if p.id == event.persona_id), None)
            if persona is None:
                return None
            return run_news_agent_tick(
                event=event,
                persona=persona,
                ledger=led,
                client=anthropic_client,
                model_override=model_override,
            )

        for csv_path in csvs:
            persona_name = csv_path.stem

            def _tick(i: int, total: int, headline: str, _p=persona_name) -> None:
                nonlocal processed
                processed += 1
                frac = min(1.0, processed / max(1, total_expected))
                snippet = headline[:80] + ("…" if len(headline) > 80 else "")
                progress.progress(frac, text=f"{_p} {i}/{total} · {snippet}")

            try:
                report = replay_headlines(
                    timeline_csv=csv_path,
                    personas=personas,
                    policy_fn=_policy,
                    ledger_path=LEDGER_PATH,
                    price_fn=price_fn,
                    max_events=max_events,
                    progress_fn=_tick,
                )
                all_reports.append((persona_name, report))
                status.info(
                    f"Finished **{persona_name}** — "
                    f"{report.total_trades} trades, {report.total_skips} skips."
                )
            except Exception as ex:
                st.warning(f"Replay failed for {persona_name}: {ex}")

        progress.empty()
        status.empty()
        st.success(
            f"Replayed {len(all_reports)} persona timelines "
            f"({sum(r.total_events for _, r in all_reports)} events)."
        )

    # Per-persona scorecard table
    entries = list(ledger.iter_all())
    if not entries:
        st.info("No ledger entries yet — run a backtest or fire events in the Live Tape tab.")
        return

    persona_ids = sorted({e.persona for e in entries if e.persona != "unknown"})
    if not persona_ids:
        return

    st.markdown("#### Per-persona metrics")
    scorecard_rows = []
    for pid in persona_ids:
        sc = compute_scorecard(entries, pid)
        scorecard_rows.append({
            "Persona": pid,
            "Trades": sc.total_trades,
            "Skips": sc.skips,
            "Hit rate": f"{sc.hit_rate:.1%}",
            "Avg PnL %": f"{sc.avg_pnl_pct:+.2f}",
            "Sharpe (ann.)": f"{sc.sharpe_annualized:.2f}",
            "Max DD %": f"{sc.max_drawdown_pct:.2f}",
            "Profit factor": (
                "∞" if sc.profit_factor == float("inf") else f"{sc.profit_factor:.2f}"
            ),
            "Median hold (h)": f"{sc.median_holding_hours:.1f}",
        })
    st.dataframe(pd.DataFrame(scorecard_rows), hide_index=True, use_container_width=True)

    st.markdown("#### Equity curve (cumulative PnL %)")
    selected = st.selectbox("Focus persona", ["All"] + persona_ids)
    subset = (
        entries if selected == "All"
        else [e for e in entries if e.persona == selected]
    )
    curve = entries_to_equity_curve(subset)
    if not curve:
        st.info("No closed trades for this selection.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[ts for ts, _ in curve],
        y=[v for _, v in curve],
        mode="lines+markers",
        line=dict(color="#1F8A70", width=2),
        fill="tozeroy",
        fillcolor="rgba(31,138,112,0.1)",
    ))
    fig.update_layout(
        xaxis_title="Entry time",
        yaxis_title="Cumulative PnL %",
        margin=dict(t=10, b=40, l=60, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#888")
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 3: Ledger Explorer ────────────────────────────────────────────────────


def _render_ledger_explorer() -> None:
    _render_disclaimer(mode="live")
    st.markdown("### Ledger Explorer")
    st.caption("Every agent decision is persisted in SQLite. Filter, inspect, export.")

    ledger = _ledger()
    entries = list(ledger.iter_all())
    if not entries:
        st.info("Ledger is empty. Visit Live Tape or run a backtest.")
        return

    personas_present = sorted({e.persona for e in entries})

    c1, c2, c3 = st.columns(3)
    with c1:
        persona_pick = st.multiselect("Persona", personas_present, default=personas_present)
    with c2:
        side_pick = st.multiselect("Side", ["LONG", "SHORT", "SKIP"], default=["LONG", "SHORT", "SKIP"])
    with c3:
        outcome_pick = st.radio(
            "Outcome",
            ["All", "Wins only", "Losses only", "Open", "Skips only"],
            horizontal=False,
        )

    def _keep(e) -> bool:
        if e.persona not in persona_pick:
            return False
        if e.trade_side not in side_pick:
            return False
        if outcome_pick == "Wins only":
            return e.realized_pnl_pct is not None and e.realized_pnl_pct > 0
        if outcome_pick == "Losses only":
            return e.realized_pnl_pct is not None and e.realized_pnl_pct < 0
        if outcome_pick == "Open":
            return e.close_ts is None and e.trade_side != "SKIP"
        if outcome_pick == "Skips only":
            return e.trade_side == "SKIP"
        return True

    filtered = [e for e in entries if _keep(e)]
    st.markdown(f"**{len(filtered)} / {len(entries)} rows match.**")

    if not filtered:
        return

    df = pd.DataFrame([
        {
            "Entry time": e.entry_ts.strftime("%Y-%m-%d %H:%M"),
            "Persona": e.persona,
            "Side": e.trade_side,
            "Ticker": e.trade_ticker,
            "Size %": f"{e.trade_size_pct:.1%}" if e.trade_side != "SKIP" else "—",
            "Horizon h": e.horizon_hours,
            "PnL %": (
                f"{e.realized_pnl_pct:+.2f}" if e.realized_pnl_pct is not None else "—"
            ),
            "Headline": e.trigger_headline[:70],
        }
        for e in filtered
    ])
    st.dataframe(df, hide_index=True, use_container_width=True, height=400)

    st.download_button(
        "Download filtered ledger as CSV",
        data=entries_to_csv_text(filtered),
        file_name=f"political_alpha_ledger_{datetime.now(timezone.utc).date()}.csv",
        mime="text/csv",
    )

    with st.expander("Inspect a single entry (full hypothesis)"):
        idx = st.number_input(
            "Row", min_value=0, max_value=len(filtered) - 1, value=0, step=1,
        )
        e = filtered[int(idx)]
        st.json({
            "entry_ts": e.entry_ts.isoformat(),
            "persona": e.persona,
            "side": e.trade_side,
            "ticker": e.trade_ticker,
            "size_pct": e.trade_size_pct,
            "horizon_hours": e.horizon_hours,
            "trigger_headline": e.trigger_headline,
            "trigger_source": e.trigger_source,
            "hypothesis": e.hypothesis,
            "close_ts": e.close_ts.isoformat() if e.close_ts else None,
            "realized_pnl_pct": e.realized_pnl_pct,
        })


# ── Entry point ───────────────────────────────────────────────────────────────


def render_political_alpha() -> None:
    st.markdown("<h1>Political Alpha — Autonomous News Agent</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p>A Sonnet 4.6 tool-use loop pulls <b>real-time</b> news from "
        "GDELT 2.0, the Truth Social archive, and RSS feeds, reacts to "
        "headlines from tracked political and financial personas, proposes "
        "or skips a short-horizon trade, and logs every decision to an "
        "append-only ledger.</p>",
        unsafe_allow_html=True,
    )

    with st.expander("About this pattern (LLM integration #5)"):
        st.markdown("""
**Pattern:** Agentic tool-use loop driven by a *live* event stream.

**News sources (polled on each refresh):**
- **GDELT 2.0 DOC API** — 15-min cadence global news, queried per persona keyword set.
- **CNN Truth Social archive** — Trump posts, ~5-min cadence.
- **RSS** — Reuters topNews, CNBC markets (fallback + broad coverage).

**Tools exposed to Claude:** `score_headline_sentiment` (project's own
TF-IDF model), `lookup_historical_reaction`, `check_open_positions`,
`get_current_price` (yfinance-backed), `propose_trade`, `skip`.

**Guardrails:** Size ≤ 10%, horizon ≤ 168h, hypothesis ≥ 40 chars.
Violations fail Pydantic validation → auto-rewritten as a SKIP with a
machine-readable reason. The prompt describes the rules; Python enforces
them. Polling is capped at **5 events per cycle** to protect LLM cost.

**Why this is different from the chatbot (LLM #2):** the chatbot is
user-initiated and conversational. This agent is *event-initiated*,
autonomous, and writes to an append-only ledger. It's the closest
analogue in the codebase to an always-on trading policy.

**Historical evaluation:** the second tab replays a curated 2018-2021
archive (real Reuters/Fed/ECB/Twitter headlines with known outcomes) so
we can score agent accuracy with ground-truth PnL.
""")

    tab_live, tab_scorecard, tab_ledger = st.tabs(
        ["Live Tape", "Historical Evaluation", "Ledger Explorer"]
    )
    with tab_live:
        _render_live_tape()
    with tab_scorecard:
        _render_scorecard()
    with tab_ledger:
        _render_ledger_explorer()
