# CLAUDE.md — Working Agreement for v3 Development

> **Read this first on every new session.** It complements `action_planv3.md` (the *what* and *why*) with the *how-we-work*.

---

## Project Context

**Portfolio Tracker v3** — a Streamlit app built for ESADE's Prototyping Products course. The goal of v3 is to turn a graded 8.75/10 project into a 10/10 third submission. Current branch: `v3`. Graded snapshot lives on `main`, safety mirror on `version-2`.

The full v3 spec is in `action_planv3.md` (gitignored — dev doc only). Read it before starting any workstream.

---

## The Working Agreement (strict)

### Git — user-only
- **Claude NEVER runs `git add`, `git commit`, `git push`, `git stash`, `git rebase`, `git reset`, `git checkout`, or any other state-changing git command.**
- Claude may *suggest* git commands in a code block for the user to copy-paste.
- Read-only git commands (`git status`, `git diff`, `git log`, `git show`) are fine for inspection.
- The user commits on their own cadence.

### TDD — mandatory
- Tests are written **before** the implementation, for every new function/module.
- Workflow per unit: RED (write failing test) → GREEN (minimal code to pass) → IMPROVE (refactor).
- Every new module must ship with a matching `tests/test_<module>.py`.
- Coverage target on new code: **≥ 80%**. Aim for 85–90% on core logic.
- Use `pytest` + `pytest-asyncio` (for async agent code) + `pytest-cov`.
- Mock the `anthropic` client in tests — no real API calls in the test suite.

### Decision-making — user has final say on judgment calls
- **Code choices** (how to name a variable, whether to use a dict or dataclass, which sklearn estimator): Claude decides.
- **Logical choices** (which personas to track, what salience threshold to default to, whether to skip a tier of a spec, trade-offs that have product consequences): Claude **asks** the user. No silent assumptions.
- When in doubt: ask. Cost of pausing is low.

### Scope — build to the plan, not beyond
- Stay inside `action_planv3.md`. If a new idea surfaces mid-build, flag it to the user and ask whether to add it to the plan or defer.
- Never add backwards-compat shims or "just-in-case" abstractions. Delete unused code confidently.
- Error handling only where the plan specifies it or where a system boundary demands it.

### Testing flow for user
- The user tests features by running the Streamlit server locally and exercising the UI.
- Claude's job: deliver a feature end-to-end (tests + code + wiring), report "ready to test", list the exact commands to run and what to look for.
- User reports back: works / bug / change. Then iterate.

---

## Tech Stack (don't drift)

| Layer | Choice | Notes |
|---|---|---|
| UI framework | Streamlit | Do not migrate to Next.js/Vercel until WS5 optional stretch |
| LLM | Anthropic (`anthropic` SDK) | Haiku for single-call, Sonnet for agents/tools, Opus for judge + extended thinking |
| ML | sklearn + xgboost | Existing models stay as-is |
| Data | yfinance, GDELT 2.0, CNN Truth Social archive | No paid APIs |
| Storage | SQLite (`data/political_alpha.db`) + CSV | No Postgres, no cloud DB |
| MCP | official `mcp` PyPI SDK, stdio transport | Separate process from Streamlit |
| Deployment | Streamlit Community Cloud | Vercel deferred |
| Python | 3.11 | `runtime.txt` pins this |

---

## Repo Conventions

### Framework-agnostic `core/` package
- `core/` contains **pure functions** and **immutable dataclasses** (`@dataclass(frozen=True)`).
- **No Streamlit imports** anywhere in `core/`. Enforced by a test (`tests/test_core_purity.py`).
- Both `utils/chatbot_tools.py` and `mcp_server/server.py` are thin adapters over `core/`.

### Session-state contract
Extend, never rename or remove. Current keys in use:
```
portfolio_sentiment, macro_prediction, optimal_weights,
last_stress_test, backtest_metrics
```
New v3 keys:
```
news_ledger, news_agent_status, last_news_trigger,
news_backtest_results, debate_verdict, eval_last_run, cache_savings
```

### File-size discipline
- Typical: 200–400 lines.
- Hard max: 800 lines.
- Split by domain, not by layer.

### Immutability
- Prefer `@dataclass(frozen=True)` or `NamedTuple`.
- Never mutate function arguments.
- Returning copies costs nothing compared to a concurrency bug.

### Comments
- Default: no comments.
- Write a comment **only** when the *why* is non-obvious (hidden constraint, tricky invariant, workaround for a known bug).
- Never explain what well-named code does.

### Disclaimers
- Every trade-suggesting view: red banner at the top — "Educational prototype, not financial advice."
- Every cross-domain demo view (healthcare, legal, travel): analogous banner.
- Exact wording in `action_planv3.md` section 11.

---

## LLM Cost Discipline

- **Haiku 4.5** for: single-call synthesis, news summarization, eval judges, simple classifiers.
- **Sonnet 4.6** for: agentic tool-use loops, structured JSON generation, debate analysts.
- **Opus 4.x** for: extended-thinking CIO judge (budget 16k tokens), rare "deep analysis" modes.
- All LLM calls go through `utils.ai_advisor.track_llm_usage` so the cost dashboard stays accurate.
- System prompts and tool definitions use `cache_control: {"type": "ephemeral"}` at the final content block. Target cache hit ≥ 50%.

---

## Common Commands

```bash
# Activate env
source .venv/bin/activate   # or equivalent

# Run app
streamlit run app.py

# Run tests with coverage
pytest --cov=core --cov=utils --cov=mcp_server --cov-report=term-missing

# Run a single test file
pytest tests/test_core_portfolio_ops.py -v

# Lint + format (user's rules require these)
ruff check .
black .
isort .

# Run MCP server standalone (for Claude Desktop testing)
python -m mcp_server.server

# Run eval harness (once implemented)
python -m evals.runner
```

---

## Day-by-Day Status

> Updated at the end of each work block.

- [ ] **Day 1 AM** — WS1 refactor: `core/` package extracted; existing tests still green.
- [ ] **Day 1 PM** — WS2 scaffolding: `news_stream`, `signal_ledger`, personas YAML, headline CSVs, tests.
- [ ] **Day 2** — WS2 agent + live-tape view; streaming; disclaimer banner.
- [ ] **Day 3** — WS2 backtest harness + per-persona scorecard UI.
- [ ] **Day 4** — WS3 MCP server + Claude Desktop screencast; prompt caching wired.
- [ ] **Day 5** — WS4 multi-agent debate + Opus extended-thinking judge.
- [ ] **Day 6** — WS5 pattern gallery (5 patterns × 3 domains) + eval harness.
- [ ] **Day 7** — Polish: streaming everywhere, CLAUDE.md final, README, Mermaid, deploy to Streamlit Cloud.

---

## When Stuck

1. Re-read `action_planv3.md` for the relevant workstream.
2. If the plan is ambiguous, **ask the user** — don't guess.
3. If a library/API detail is unclear, use `docs-lookup` agent or WebSearch.
4. If a refactor decision is non-obvious, use `architect` agent.
5. If tests fail and the cause isn't immediate, use `tdd-guide` agent.

---

## Never Do These

- Run any state-changing git command.
- Add code without a test.
- Introduce a third-party paid API.
- Migrate off Streamlit.
- Skip the disclaimer banner on any trade-suggesting or cross-domain view.
- Put Streamlit imports inside `core/`.
- Use `--dangerously-skip-permissions` or bypass any hook.
- Assume a logical choice without asking.

---

*Last updated: Day 0 — project setup complete, on branch `v3`, ready for WS1.*
