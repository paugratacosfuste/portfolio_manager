"""Political Alpha agent — the 5th LLM pattern.

One NewsEvent in, one LedgerEntry out. The agent is given six tools and
must terminate with either `propose_trade` or `skip`. All hard policy
caps (size ≤10%, horizon ≤168h, schema validation) are enforced in
Python, not in the prompt — if the model violates them, we SKIP with a
machine-readable reason rather than trust it.

Runs inside Streamlit via `client.messages.create` (streaming is the
job of the view layer — this module stays synchronous for testability).
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List

from pydantic import ValidationError

from core.news_types import LedgerEntry, NewsEvent, Persona
from core.tool_schemas import TradeProposal
from utils.signal_ledger import SignalLedger


MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1200
MAX_ITERATIONS = 4


AGENT_SYSTEM_PROMPT = """\
You are a short-horizon geopolitical trading policy. You react to a single \
news event at a time. You must either:
  (a) emit EXACTLY ONE trade proposal via the `propose_trade` tool, or
  (b) emit `skip` with a reason.

Hard rules (violations are rejected by Python validation):
  - Never propose a trade with size > 10% of the portfolio.
  - Never propose a horizon > 168 hours (1 week).
  - If the event contradicts an open position in the ledger for the same \
ticker, emit SKIP with reason "conflicting_open_position" — do not stack.
  - Your hypothesis must be ≥ 40 characters. Cite three things:
      (i)   the transmission channel,
      (ii)  the historical base rate you are pattern-matching to,
      (iii) the main falsifier.
  - If sentiment |score| < 0.15, prefer SKIP.

Tool usage policy:
  - First call `score_headline_sentiment` on the headline.
  - Then call `lookup_historical_reaction` for the persona (base rate).
  - Check `check_open_positions` if your proposed ticker might already be open.
  - Use `get_current_price` only if the price matters to your hypothesis.
  - Finally, call EXACTLY ONE of `propose_trade` or `skip`.

You are an educational prototype. Do not fabricate citations or pretend to \
have access to data outside the provided tools.
"""


AGENT_TOOLS = [
    {
        "name": "score_headline_sentiment",
        "description": (
            "Score a headline with the project's TF-IDF sentiment model. "
            "Returns P(pos)-P(neg) in [-1, 1]. Call this first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "headline": {"type": "string"},
            },
            "required": ["headline"],
        },
    },
    {
        "name": "lookup_historical_reaction",
        "description": (
            "Look up historical market reactions to similar events by this "
            "persona. Returns up to 5 prior events with realized 24h asset "
            "moves, or an empty list if no CSV is curated for the persona."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "persona_id": {"type": "string"},
                "query_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["persona_id"],
        },
    },
    {
        "name": "check_open_positions",
        "description": (
            "List current open (un-closed) ledger entries, optionally "
            "filtered by ticker."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
        },
    },
    {
        "name": "get_current_price",
        "description": "Get the latest close price for a ticker.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "propose_trade",
        "description": (
            "Emit a structured trade proposal. This terminates the tick. "
            "Will be REJECTED if size_pct > 0.10 or horizon_hours > 168."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "side": {"type": "string", "enum": ["LONG", "SHORT"]},
                "size_pct": {
                    "type": "number", "minimum": 0.001, "maximum": 0.10,
                },
                "horizon_hours": {
                    "type": "integer", "minimum": 1, "maximum": 168,
                },
                "confidence": {
                    "type": "number", "minimum": 0.0, "maximum": 1.0,
                },
                "hypothesis": {
                    "type": "string", "minLength": 40, "maxLength": 400,
                },
                "stop_loss_pct": {
                    "type": "number", "minimum": 0.005, "maximum": 0.20,
                },
            },
            "required": [
                "ticker", "side", "size_pct", "horizon_hours",
                "confidence", "hypothesis",
            ],
        },
    },
    {
        "name": "skip",
        "description": (
            "Emit a structured SKIP. Use when sentiment is too weak, the "
            "event is off-topic, or a conflicting position exists."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string", "minLength": 1}},
            "required": ["reason"],
        },
    },
]


# ── Tool implementations (registered per tick) ────────────────────────────────


def _load_sentiment_model():
    """Lazy-load the TF-IDF sentiment pipeline."""
    import joblib
    return joblib.load("ml_pipeline/sentiment_pipeline.joblib")


def _score_sentiment(headline: str) -> float:
    try:
        model = _load_sentiment_model()
        proba = model.predict_proba([headline])[0]
        classes = list(model.classes_)
        pos_idx = classes.index(1)
        neg_idx = classes.index(-1)
        return float(proba[pos_idx] - proba[neg_idx])
    except Exception:
        return 0.0


def _historical_reaction(persona_id: str, query_keywords: list[str] | None) -> list[dict]:
    from pathlib import Path

    path = Path("data/historical_headlines") / f"{persona_id}.csv"
    if not path.exists():
        return []
    try:
        import csv

        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return []

    if query_keywords:
        terms = [k.lower() for k in query_keywords]
        rows = [r for r in rows if any(t in r.get("headline", "").lower() for t in terms)]
    return rows[:5]


def _open_positions(ledger: SignalLedger, ticker: str | None) -> list[dict]:
    opens = ledger.open_positions()
    if ticker:
        opens = [e for e in opens if e.trade_ticker == ticker]
    return [
        {
            "ticker": e.trade_ticker,
            "side": e.trade_side,
            "size_pct": e.trade_size_pct,
            "entry_ts": e.entry_ts.isoformat(),
            "horizon_hours": e.horizon_hours,
            "persona": e.persona,
        }
        for e in opens
    ]


def _current_price(ticker: str) -> float:
    try:
        from utils.data_fetcher import fetch_current_prices
        prices = fetch_current_prices([ticker])
        return float(prices.get(ticker, 0.0))
    except Exception:
        return 0.0


def _execute_agent_tool(
    name: str, input_data: Dict[str, Any], ledger: SignalLedger
) -> str:
    if name == "score_headline_sentiment":
        score = _score_sentiment(input_data.get("headline", ""))
        return json.dumps({"score": round(score, 4)})
    if name == "lookup_historical_reaction":
        return json.dumps(
            {
                "events": _historical_reaction(
                    input_data.get("persona_id", ""),
                    input_data.get("query_keywords"),
                )
            }
        )
    if name == "check_open_positions":
        return json.dumps(
            {"open_positions": _open_positions(ledger, input_data.get("ticker"))}
        )
    if name == "get_current_price":
        return json.dumps(
            {
                "ticker": input_data["ticker"],
                "price": round(_current_price(input_data["ticker"]), 4),
            }
        )
    return json.dumps({"error": f"Unknown tool: {name}"})


# ── Agent loop ────────────────────────────────────────────────────────────────


def _proposal_to_entry(
    event: NewsEvent, persona_id: str, proposal: TradeProposal
) -> LedgerEntry:
    return LedgerEntry(
        entry_ts=event.ts,
        trigger_headline=event.headline,
        trigger_source=event.source,
        persona=persona_id,
        sentiment=0.0,  # sentiment recorded in hypothesis body; kept simple here
        trade_ticker=proposal.ticker,
        trade_side=proposal.side,
        trade_size_pct=proposal.size_pct,
        horizon_hours=proposal.horizon_hours,
        hypothesis=proposal.hypothesis,
    )


def _skip_to_entry(event: NewsEvent, persona_id: str, reason: str) -> LedgerEntry:
    return LedgerEntry(
        entry_ts=event.ts,
        trigger_headline=event.headline,
        trigger_source=event.source,
        persona=persona_id,
        sentiment=0.0,
        trade_ticker="NONE",
        trade_side="SKIP",
        trade_size_pct=0.0,
        horizon_hours=1,
        hypothesis=f"SKIP: {reason[:380]}",
    )


def _track_cost(client_response: Any) -> None:
    try:
        from utils.ai_advisor import track_llm_usage
        track_llm_usage(client_response, MODEL, 0.0)
    except Exception:
        pass


def run_news_agent_tick(
    event: NewsEvent,
    persona: Persona,
    ledger: SignalLedger,
    client: Any,
    stream_handler: Callable[[str], None] | None = None,
    model_override: str | None = None,
) -> LedgerEntry | None:
    """One tick of the Political Alpha policy loop.

    Returns the LedgerEntry written (trade proposal or SKIP), or None if
    the agent exhausts iterations without terminating.

    `model_override` swaps the default Sonnet model for a faster one
    (e.g. Haiku) during backtests. Tracked cost still uses the real model.
    """
    active_model = model_override or MODEL
    user_msg = (
        f"Event: [{event.source}] {event.headline}\n"
        f"Persona: {persona.name} ({persona.role}, tier {persona.tier})\n"
        f"Primary asset impacts: "
        + ", ".join(
            f"{ai.asset} ({ai.direction_on_positive} on positive)"
            for ai in persona.primary_asset_impacts
        )
        + "\n"
        f"Timestamp: {event.ts.isoformat()}\n\n"
        "Follow the tool usage policy and terminate with propose_trade or skip."
    )

    messages: List[Dict[str, Any]] = [{"role": "user", "content": user_msg}]

    for _ in range(MAX_ITERATIONS):
        response = client.messages.create(
            model=active_model,
            max_tokens=MAX_TOKENS,
            system=[{
                "type": "text",
                "text": AGENT_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            tools=AGENT_TOOLS,
            messages=messages,
        )
        _track_cost(response)

        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        if response.stop_reason != "tool_use":
            # Model gave up without calling a terminal tool — record a SKIP
            entry = _skip_to_entry(
                event, persona.id, "agent stopped without terminal tool call"
            )
            ledger.append(entry)
            return entry

        tool_results: list[dict] = []
        terminal_entry: LedgerEntry | None = None

        for block in assistant_content:
            if getattr(block, "type", None) != "tool_use":
                continue

            name = block.name
            tool_input = dict(block.input)
            tool_id = block.id

            if name == "propose_trade":
                try:
                    proposal = TradeProposal(**tool_input)
                    terminal_entry = _proposal_to_entry(event, persona.id, proposal)
                except ValidationError as ve:
                    terminal_entry = _skip_to_entry(
                        event, persona.id,
                        f"invalid proposal (schema violated): {ve.errors()[:1]}",
                    )
                break

            if name == "skip":
                reason = str(tool_input.get("reason", "no reason given"))
                terminal_entry = _skip_to_entry(event, persona.id, reason)
                break

            # Non-terminal tool: execute, collect result, continue loop
            result = _execute_agent_tool(name, tool_input, ledger)
            if stream_handler:
                stream_handler(f"[{name}] {result[:200]}")
            tool_results.append(
                {"type": "tool_result", "tool_use_id": tool_id, "content": result}
            )

        if terminal_entry is not None:
            ledger.append(terminal_entry)
            return terminal_entry

        if not tool_results:
            # Agent emitted tool_use blocks but none were ours — defensive fallback
            entry = _skip_to_entry(event, persona.id, "no recognizable tool call")
            ledger.append(entry)
            return entry

        messages.append({"role": "user", "content": tool_results})

    # Ran out of iterations
    entry = _skip_to_entry(event, persona.id, "max iterations reached")
    ledger.append(entry)
    return entry
