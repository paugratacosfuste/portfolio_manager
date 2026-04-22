# Assignment 3: AI Portfolio Advisor — Third Submission

**Pau Gratacós Fusté | PDAI | ESADE 2026**

## 1. Main Questions Answered in This Prototype

Assignment 2 delivered 4 LLM patterns (single-call, agentic chatbot, structured JSON, orchestrated chain) and scored 8.75/10. The explicit feedback asked for a **5th LLM pattern** — an agentic loop consuming real-time news/tweets with sentiment scoring to propose short-term trades (political leaders' quotes angle) — and for the patterns to be shown to **transfer outside the portfolio domain**. Two new questions drive this submission:

- **Q4: Can an LLM act as a *live* autonomous trading agent reacting to real-time political news?** I built **Political Alpha**, an event-driven tool-use loop on Sonnet that polls four live feeds every 2 min (GDELT 2.0 DOC API, CNN's Truth Social archive, Reuters/CNBC RSS, and per-holding yfinance news), filters on a registry of 31 tracked personas (Trump, Powell, Lagarde, Musk, Xi, etc.), scores sentiment with our own TF-IDF model, consults historical base rates, and either emits a schema-validated trade proposal (ticker, LONG/SHORT, size ≤ 10%, horizon ≤ 168h, hypothesis) or a reasoned SKIP. Every decision is appended to a SQLite ledger, and a separate Historical Evaluation tab replays a curated 2018-2021 archive with known outcomes to score per-persona hit rate, Sharpe, max drawdown, and profit factor.

- **Q5: Can one CIO-grade verdict be synthesized from specialist analysts arguing against each other?** The **Debate (CIO)** view runs three Haiku analysts in parallel — BULL, BEAR, MACRO — each producing a JSON-schema-constrained `AnalystArgument` (claim, evidence[], risks[], conviction). An **Opus 4.7 judge with `thinking={"type": "enabled", "budget_tokens": 6000}`** reads all three and issues a `BULL / BEAR / NEUTRAL` verdict with a recommended_action. Forcing role specialization (BULL can't hedge, BEAR can't handwave) produces sharper arguments than a single polymath prompt, and the extended-thinking trace is surfaced to the user.

Combined with Assignment 2's patterns, the prototype now covers **6 distinct LLM integration patterns** plus a **cross-domain gallery** proving transferability.

## 2. LLM Integration Patterns (6 Distinct Patterns)

| # | Pattern | Model | View | Non-Trivial Aspect |
|---|---|---|---|---|
| 1 | Single-call summarization | Haiku 4.5 | Dashboard, News, Macro | Prompt enriched with risk metrics + multi-source signals |
| 2 | Agentic tool-use loop | Sonnet 4.6 | AI Chat | 6 tools, multi-turn, Claude controls flow autonomously |
| 3 | Structured JSON generation | Sonnet 4.6 | Stress Test | Schema-validated JSON; Python applies params quantitatively |
| 4 | Orchestrated multi-step chain | Sonnet 4.6 | Suggestions (Autopilot) | 3-step: LLM proposes → Python simulates → LLM synthesizes |
| **5** | **Event-driven autonomous news agent** | **Sonnet 4.6** | **Political Alpha** | **Live feeds (GDELT/Truth Social/RSS/yfinance) → persona filter → sentiment (own TF-IDF) → agentic tool loop → SQLite ledger** |
| **6** | **Multi-agent debate + extended thinking** | **Haiku ×3 + Opus 4.7** | **Debate (CIO)** | **3 role-specialized analysts → Opus judge with 6k-token thinking budget** |

## 3. Additional Improvements (Non-LLM)

- **Cross-Domain Pattern Gallery**: The same LLM patterns applied outside finance — **healthcare triage** (pattern #3, urgency JSON), **legal contract clause review** (pattern #1, risk summary), **travel itinerary planner with mocked tools** (pattern #2). Every domain renders a domain-specific disclaimer banner.
- **MCP Integration (Model Context Protocol)**: A separate `mcp_server/` process exposes the portfolio toolkit over stdio so Claude Desktop can call `get_portfolio_summary`, `calculate_portfolio_risk`, `what_if_trade`, `get_news`, and `get_political_alpha_ledger` autonomously — the prototype becomes a reusable tool set, not just a Streamlit app.
- **Framework-agnostic `core/` package**: Pure immutable dataclasses (`NewsEvent`, `LedgerEntry`, `Persona`, `AnalystArgument`, `CIOVerdict`) with `__post_init__` validation. Enforced by a test that `core/` contains no Streamlit imports — so the Streamlit view, the MCP server, and the backtest harness all share one type system.
- **Eval harness (`/evals`)**: CLI runner (`python -m evals.runner`) replays curated cases through every pattern, checks hard constraints (must_contain / must_reject), and uses a Haiku judge for soft grading. Dry-run mode supports offline CI.
- **Prompt caching everywhere**: System prompts and tool definitions sent with `cache_control: {"type": "ephemeral"}` at the final content block. Target ≥ 50% cache hit across repeat invocations — tracked live in the cost dashboard.
- **SignalLedger (SQLite, append-only)**: Every agent decision is immutable — `LedgerEntry.with_close()` returns a *new* row instead of mutating, making the ledger auditable. The Ledger Explorer tab filters, inspects, and exports CSV.
- **Comprehensive test suite**: **331 tests passing** (`pytest --cov`). Includes a core-purity test, 23 debate tests with mocked clients, ledger replay tests, eval harness tests, and a core-package isolation test.
- **Deploy readiness**: `runtime.txt` pins Python 3.11; `requirements.txt` pinned; `.env.example` for key rotation.

## 4. Main Difficulties Encountered

- **Live news autonomy vs. API cost**: The first cut of Political Alpha replayed a CSV archive with a manual "Fire" button — technically tool-use, but not autonomous and the news was 4 years stale. Pivoted to real-time polling with `streamlit-autorefresh` (120s interval), an LRU dedup cache with 48h TTL, a `MIN_SALIENCE=0.35` threshold, and `MAX_EVENTS_PER_POLL=5` to bound LLM cost per cycle.
- **GDELT 2.0 flakiness**: Per-persona serial GDELT queries against 31 personas timed out at ~620 s worst case. Rewrote with `ThreadPoolExecutor(max_workers=8)` + `as_completed`, 8 s per-persona timeout, and tier-1-only filtering — caps total GDELT latency at ~8 s. RSS, Truth Social, and yfinance carry the rest of the load when GDELT is empty.
- **Historical backtest UX**: Running the agent over 4 persona CSVs × ~30 headlines was ~10 min of sequential Sonnet calls with a per-CSV progress bar that looked frozen. Added a per-event progress callback, a max-events-per-persona slider, and a Haiku/Sonnet model toggle — backtest now finishes in ~90 s on Haiku with live headline-by-headline progress.
- **Extended-thinking integration**: Opus `thinking` blocks arrive interleaved with text blocks. Implemented `_thinking_summary()` that concatenates `thinking`-typed blocks separately from text, so the verdict JSON parse isn't polluted by the reasoning trace.
- **MCP + Streamlit coexistence**: MCP runs as a separate stdio process, not inside Streamlit. Shared `core/` means both read the exact same `Portfolio` / `SignalLedger` types with zero drift.
- **Pydantic 2 strict validation**: `NewsEvent.__post_init__` rejected `source="yfinance"` because the `Literal` enum didn't include it — surfaced as cryptic `ValueError` during live polling. Fixed by adding `"yfinance"` to both the `Literal` and the runtime `_VALID_SOURCES` set.

## 5. How I Leveraged AI

- **Claude Code as a pair programmer**: Used Claude Code to drive TDD across the v3 refactor — tests written first (RED), minimal code to pass (GREEN), refactor. Every new module shipped with a matching `tests/test_<module>.py`. Final suite: 331 tests.
- **Architecture via `/plan`**: The framework-agnostic `core/` split was designed in a planning session before any code was written — the planner agent surfaced risks (Streamlit import leakage, thread-unsafe SQLite) and produced a day-by-day workstream plan (`action_planv3.md`).
- **Cross-agent validation**: The debate feature itself was validated using the same multi-agent pattern during design — a security-reviewer agent and a python-reviewer agent ran in parallel on the `utils/debate.py` module to catch schema drift and token-accounting bugs.
- **Prompt engineering iteration**: The Political Alpha system prompt went through 5 revisions — initial version hallucinated tickers; adding explicit "only trade on the persona's primary_asset_impacts list" and "SKIP is a first-class decision, not a failure mode" dropped invalid-ticker rate from ~15% to 0% on the eval harness.
- **Documentation + structured reporting**: This 2-pager was drafted with AI assistance, grounded in `git log`, the actual test output, and the v3 action plan — all claims are traceable to code.

## 6. System Architecture

| Layer | Technologies |
|---|---|
| Frontend | Streamlit (13 interactive views, `streamlit-autorefresh` for live polling, custom CSS, disclaimer banners) |
| Core package | `core/` — framework-agnostic `@dataclass(frozen=True)` types with validation; no Streamlit imports |
| Live news | GDELT 2.0 DOC API (parallel fetches), CNN Truth Social archive, Reuters/CNBC RSS via `feedparser`, yfinance per-holding news |
| Visualizations | Plotly (equity curves, scorecard tables, persona cards, verdict panels) |
| ML / NLP | scikit-learn + XGBoost (GridSearchCV, TimeSeriesSplit), own TF-IDF sentiment model |
| LLM API | Anthropic Claude SDK — Haiku 4.5 (analysts, summaries, evals) · Sonnet 4.6 (agents, structured) · **Opus 4.7 with extended thinking** (CIO judge) |
| Prompt caching | `cache_control: ephemeral` at final content block across all patterns |
| Persistence | SQLite `data/political_alpha.db` (append-only `SignalLedger`), 48h dedup cache |
| MCP | Official `mcp` PyPI SDK, stdio transport, 5 tools exposed to Claude Desktop |
| Eval harness | `evals/runner.py` CLI with judge (Haiku) + hard constraint checker |
| Testing | `pytest` + `pytest-asyncio` + `pytest-cov`; 331 tests, 80 %+ coverage on new code |

## 7. Addressing Assignment 2 Feedback

The 8.75/10 feedback highlighted two concrete gaps: only 4 LLM patterns (could push further) and no demonstration that patterns transfer outside the single portfolio use case. Both are addressed:

| Feedback Point | How Addressed in Assignment 3 |
|---|---|
| "Add a 5th LLM pattern — agentic loop on live news/tweets with sentiment → trade proposals, political leaders' quotes" | **Pattern #5 (Political Alpha)** — live GDELT/Truth-Social/RSS/yfinance polling → 31-persona registry → TF-IDF sentiment → Sonnet agentic tool loop → schema-validated trade or reasoned SKIP → SQLite ledger. Historical Evaluation tab scores the policy with known outcomes. |
| "Apply patterns beyond the portfolio domain" | **Cross-Domain Gallery** — healthcare triage (pattern #3), legal contract review (pattern #1), travel planner with mocked tools (pattern #2). Each domain renders its own disclaimer banner. |
| "Colleagues demonstrated MCP" | **MCP server** (`mcp_server/`) — stdio transport exposes 5 portfolio tools to Claude Desktop; same `core/` types as the Streamlit view → zero drift. |
| "Push LLM usage further" | **Pattern #6 (Debate)** — BULL/BEAR/MACRO Haiku analysts + **Opus 4.7 judge with 6 000-token extended-thinking budget**, multi-agent coordination with JSON-schema-constrained arguments. |
| Evaluation rigor | **`/evals` harness** + 331-test suite + append-only ledger for auditability. |

---

**Repository:** `github.com/paugratacosfuste/portfolio_manager` · **Live Demo:** `portfolio-manager-pau.streamlit.app` · **Built with:** Streamlit, Plotly, scikit-learn, XGBoost, Anthropic Claude API (Haiku 4.5 + Sonnet 4.6 + Opus 4.7), MCP SDK, SQLite, GDELT 2.0, feedparser
