# Assignment 2 Implementation Plan — Interactive Portfolio Chatbot with Tool Use

## Context

Assignment 2 requires adding a **new non-straightforward LLM feature** to the existing Portfolio Tracker prototype. The current app already uses Claude for 3 simple single-call features (portfolio advice, news summary, benchmark comparison) — all "straightforward" (simple prompt in, text displayed). The assignment explicitly requires the new LLM feature to be **multi-call, use tool calling, or involve complex output processing**.

---

## New Feature: Interactive Portfolio Chatbot with Tool Use

A conversational AI chat tab where users ask natural-language questions about their portfolio. Claude autonomously decides which tools to call (real-time prices, risk calculations, what-if scenarios, news, macro predictions), Python executes them, and results are fed back to Claude for a grounded final answer.

**Why this is non-straightforward (hits all 3 criteria):**
- **Multi-call chatbot**: Conversation history maintained across turns
- **Tool use**: Claude decides when to call 6 Python functions via Anthropic's tool-use API
- **Complex output processing**: Agentic loop parses tool_use blocks, executes functions, serializes results as JSON, feeds back to Claude

### Tools the chatbot will have:
| Tool | Description | Wraps |
|------|-------------|-------|
| `get_portfolio_summary` | Current holdings, values, weights | `fetch_current_prices` |
| `get_asset_price` | Real-time price for any ticker | `fetch_current_prices` |
| `calculate_portfolio_risk` | Volatility, beta, HHI, Sharpe, max drawdown | `portfolio_metrics.*` |
| `what_if_analysis` | Simulate adding/removing assets, show before/after risk | `portfolio_metrics.*` |
| `get_market_news` | Latest headlines for a ticker | `fetch_recent_news` |
| `get_macro_prediction` | ML model market correction probability | `macro_risk_model.joblib` |

---

## Implementation Steps

### Phase 1: New portfolio metrics (extend existing)

**File: `utils/portfolio_metrics.py`** — ADD 2 functions:
- `calculate_sharpe_ratio(historical_prices, weights, risk_free_rate=0.05)` — annualized Sharpe ratio
- `calculate_max_drawdown(historical_prices, weights)` — largest peak-to-trough decline

**File: `views/suggestions.py`** — ADD Sharpe Ratio and Max Drawdown to the metric cards row (alongside volatility/beta/HHI)

### Phase 2: Chatbot backend

**File: `utils/chatbot_tools.py`** — CREATE
- `CHATBOT_TOOLS` list: 6 tool schemas in Anthropic API format
- Executor functions for each tool (wrap existing `data_fetcher` and `portfolio_metrics` functions)
- `execute_tool(tool_name, tool_input, holdings)` dispatcher

**File: `utils/chatbot_engine.py`** — CREATE
- `run_chatbot_turn(user_message, conversation_history, holdings, profile, eli10_mode)` — the core agentic loop:
  1. Append user message to history
  2. Call Claude with tools
  3. If `stop_reason == "tool_use"`: execute each tool, append tool_result, loop back to step 2
  4. If `stop_reason == "end_turn"`: extract final text, return
  5. Max 5 iterations safety cap
- System prompt adapts based on ELI10 mode
- Reuses the existing `client` from `ai_advisor.py`

### Phase 3: Chatbot UI

**File: `views/ai_chat.py`** — CREATE
- `render_ai_chat()` function using Streamlit's `st.chat_message` and `st.chat_input`
- Session state: `chat_history` (API format) + `chat_display` (UI format)
- `st.status` expander showing which tools Claude calls (transparency for graders)
- Example prompt suggestion chips
- Clear conversation button

**File: `app.py`** — MODIFY
- Import `render_ai_chat`
- Add "AI Chat" to navigation radio (place second, after Dashboard)
- Add routing case

### Phase 4: Polish and caching

**File: `utils/data_fetcher.py`** — ADD `@st.cache_data(ttl=300)` to `fetch_current_prices` and `fetch_historical_data`

**File: `styles/main.css`** — ADD chat-specific styling

**File: `requirements.txt`** — Verify `anthropic>=0.18.0` (tool use support)

### Phase 5: Tests

**File: `tests/test_chatbot_tools.py`** — CREATE (mock yfinance, test each executor)

**File: `tests/test_chatbot_engine.py`** — CREATE (mock Claude, test agentic loop)

**File: `tests/test_portfolio_metrics.py`** — MODIFY (add Sharpe/drawdown tests)

---

## Critical Files Reference

| File | Action | Purpose |
|------|--------|---------|
| `utils/chatbot_tools.py` | CREATE | Tool schemas + executor functions |
| `utils/chatbot_engine.py` | CREATE | Agentic tool-use loop |
| `views/ai_chat.py` | CREATE | Chat UI with Streamlit |
| `app.py` | MODIFY (3 lines) | Add AI Chat tab |
| `utils/portfolio_metrics.py` | MODIFY | Add Sharpe ratio + max drawdown |
| `views/suggestions.py` | MODIFY | Display new metrics |
| `utils/data_fetcher.py` | MODIFY | Add caching decorators |
| `styles/main.css` | MODIFY | Chat styling |
| `tests/test_chatbot_tools.py` | CREATE | Tool executor tests |
| `tests/test_chatbot_engine.py` | CREATE | Agentic loop tests |
| `tests/test_portfolio_metrics.py` | MODIFY | New metric tests |

## Verification

1. Run `streamlit run app.py` and navigate to "AI Chat" tab
2. Test conversations that trigger each tool
3. Verify multi-turn context retention
4. Toggle ELI10 mode and verify adaptation
5. Run `pytest` — all tests pass
6. Check Suggestions tab shows Sharpe Ratio and Max Drawdown
