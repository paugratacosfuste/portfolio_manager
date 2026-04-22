# Portfolio Tracker — Claude Desktop MCP integration

Expose this project's tools (`get_portfolio_summary`, `calculate_portfolio_risk`,
`what_if_trade`, `get_news`, `get_asset_price`, `get_political_alpha_ledger`)
to Claude Desktop over MCP stdio.

## 1. Create the portfolio config

Claude Desktop runs the server in a separate process from Streamlit, so
it reads your holdings from a JSON file. Create it once:

```bash
mkdir -p ~/.portfolio_tracker
cat > ~/.portfolio_tracker/portfolio.json <<'EOF'
{
  "holdings": {
    "AAPL": 10,
    "MSFT": 5,
    "BTC-USD": 0.25,
    "SPY": 3
  },
  "risk_profile": "Moderate"
}
EOF
```

## 2. Register the server with Claude Desktop

Open Claude Desktop → Settings → Developer → "Edit config" and add this
entry under `mcpServers`. Replace `{{ABSOLUTE_PATH}}` with the full path
to your Portfolio_Tracker checkout.

```json
{
  "mcpServers": {
    "portfolio-tracker": {
      "command": "{{ABSOLUTE_PATH}}/Portfolio_Tracker/.venv/bin/python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "{{ABSOLUTE_PATH}}/Portfolio_Tracker",
      "env": {
        "PORTFOLIO_TRACKER_CONFIG": "~/.portfolio_tracker/portfolio.json"
      }
    }
  }
}
```

Quit and reopen Claude Desktop. A hammer/🛠️ icon should appear in the
composer — clicking it lists the six portfolio-tracker tools.

## 3. Verify the server manually

Before wiring into Claude Desktop, run the server once to catch config
errors:

```bash
cd {{ABSOLUTE_PATH}}/Portfolio_Tracker
source .venv/bin/activate
python -m mcp_server.server
```

The process will block on stdin (that's the stdio transport waiting for
JSON-RPC). Ctrl-C to exit. If it blocks cleanly it's ready to be driven
by Claude Desktop.

## 4. Example prompts in Claude Desktop

Once registered, try:

- *"Use the portfolio-tracker tools to show me my current portfolio and
  compute its risk metrics."*
- *"What would happen to my Sharpe ratio if I added 5 shares of NVDA?"*
- *"Summarise the last 10 decisions from the Political Alpha agent."*

Claude will autonomously call the MCP tools and stream the results back
to you — this is the same agentic tool-use loop that powers the
in-Streamlit chatbot, but driven by Claude Desktop over MCP.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "No holdings in portfolio" error | Check `PORTFOLIO_TRACKER_CONFIG` path & JSON validity. |
| Server silently fails to register | Inspect Claude Desktop logs: `~/Library/Logs/Claude/` on macOS. |
| yfinance rate-limit errors | Seen intermittently on first call; retry the prompt. |
| `ModuleNotFoundError: mcp_server` | `cwd` in the config must point to the Portfolio_Tracker root. |
