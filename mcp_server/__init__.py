"""MCP server exposing the portfolio toolkit to Claude Desktop and other MCP clients.

Separate process from Streamlit — consumes the same `core/` functions but
reads its portfolio from a JSON config instead of Streamlit session state.

Run standalone:
    python -m mcp_server.server

Claude Desktop config snippet lives in mcp_server/CLAUDE_DESKTOP_CONFIG.md.
"""
