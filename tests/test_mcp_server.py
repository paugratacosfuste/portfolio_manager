"""Integration tests for mcp_server.server.

Tests drive the FastMCP tool functions directly (not over stdio) and mock
the data_fetcher layer so no network is required.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pandas as pd
import pytest


@pytest.fixture
def empty_config(tmp_path, monkeypatch):
    monkeypatch.setenv("PORTFOLIO_TRACKER_CONFIG", str(tmp_path / "empty.json"))
    return tmp_path


@pytest.fixture
def sample_config(tmp_path, monkeypatch):
    path = tmp_path / "portfolio.json"
    path.write_text(json.dumps({
        "holdings": {"AAPL": 10, "MSFT": 5},
        "risk_profile": "Moderate",
    }))
    monkeypatch.setenv("PORTFOLIO_TRACKER_CONFIG", str(path))
    return path


def _call_tool_sync(tool_name: str, arguments: dict | None = None):
    """Invoke a FastMCP tool by name via the server's async dispatch."""
    from mcp_server.server import mcp

    async def _run():
        result = await mcp.call_tool(tool_name, arguments or {})
        # FastMCP returns (content_blocks, structured_output) — we want the
        # text content for all six tools (they return JSON strings).
        content, _structured = result
        if not content:
            return ""
        # content is a list of TextContent-like blocks; take the first
        first = content[0]
        return first.text if hasattr(first, "text") else str(first)

    return asyncio.run(_run())


# ── Registration ──────────────────────────────────────────────────────────────


def test_six_tools_registered():
    from mcp_server.server import mcp

    tools = asyncio.run(mcp.list_tools())
    assert len(tools) == 6
    names = {t.name for t in tools}
    assert {
        "get_portfolio_summary",
        "get_asset_price",
        "calculate_portfolio_risk",
        "what_if_trade",
        "get_news",
        "get_political_alpha_ledger",
    } == names


# ── get_portfolio_summary ─────────────────────────────────────────────────────


def test_summary_returns_error_when_no_holdings(empty_config):
    out = _call_tool_sync("get_portfolio_summary")
    payload = json.loads(out)
    assert "error" in payload
    assert "No holdings" in payload["error"]


def test_summary_happy_path(sample_config):
    with patch("utils.data_fetcher.fetch_current_prices") as mock_prices, \
         patch("utils.data_fetcher.fetch_asset_metadata") as mock_meta:
        mock_prices.return_value = {"AAPL": 200.0, "MSFT": 400.0}
        mock_meta.return_value = {
            "AAPL": {"asset_class": "Equity", "sector": "Technology"},
            "MSFT": {"asset_class": "Equity", "sector": "Technology"},
        }
        out = _call_tool_sync("get_portfolio_summary")
    payload = json.loads(out)
    assert payload["total_value"] == pytest.approx(10 * 200 + 5 * 400)
    assert len(payload["holdings"]) == 2
    tickers = {h["ticker"] for h in payload["holdings"]}
    assert tickers == {"AAPL", "MSFT"}


# ── get_asset_price ──────────────────────────────────────────────────────────


def test_asset_price_returns_schema(empty_config):
    with patch("utils.data_fetcher.fetch_current_prices") as m:
        m.return_value = {"AAPL": 199.95}
        out = _call_tool_sync("get_asset_price", {"ticker": "AAPL"})
    payload = json.loads(out)
    assert payload["ticker"] == "AAPL"
    assert payload["price"] == 199.95


def test_asset_price_zero_on_missing(empty_config):
    with patch("utils.data_fetcher.fetch_current_prices") as m:
        m.return_value = {}
        out = _call_tool_sync("get_asset_price", {"ticker": "NOPE"})
    assert json.loads(out)["price"] == 0.0


# ── what_if_trade ────────────────────────────────────────────────────────────


def test_what_if_rejects_non_positive_quantity(sample_config):
    out = _call_tool_sync(
        "what_if_trade", {"action": "add", "ticker": "SPY", "quantity": 0}
    )
    assert "must be > 0" in json.loads(out)["error"]


def test_what_if_remove_nonexistent_asset(sample_config):
    with patch("utils.data_fetcher.fetch_asset_metadata") as mock_meta:
        mock_meta.return_value = {"AAPL": {}, "MSFT": {}, "GOOG": {}}
        out = _call_tool_sync(
            "what_if_trade", {"action": "remove", "ticker": "GOOG", "quantity": 1}
        )
    assert "cannot remove" in json.loads(out)["error"]


# ── get_news ─────────────────────────────────────────────────────────────────


def test_news_caps_limit(empty_config):
    captured = {}

    def fake_news(tickers, limit):
        captured["limit"] = limit
        return {tickers[0]: [
            {
                "title": "t",
                "publisher": "p",
                "link": "http://x",
                "timestamp": "2026-04-01 12:00",
            }
        ]}

    with patch("utils.data_fetcher.fetch_recent_news", side_effect=fake_news):
        out = _call_tool_sync("get_news", {"ticker": "AAPL", "limit": 999})
    assert captured["limit"] == 10  # capped
    payload = json.loads(out)
    assert payload["ticker"] == "AAPL"
    assert len(payload["news"]) == 1


def test_news_handles_exceptions_gracefully(empty_config):
    with patch(
        "utils.data_fetcher.fetch_recent_news",
        side_effect=RuntimeError("rate limited"),
    ):
        out = _call_tool_sync("get_news", {"ticker": "AAPL"})
    assert "error" in json.loads(out)


# ── get_political_alpha_ledger ───────────────────────────────────────────────


def test_political_alpha_ledger_returns_json_list(tmp_path, monkeypatch):
    # Redirect ledger path into tmp and seed one entry
    from datetime import datetime, timezone

    from core.news_types import LedgerEntry
    from utils.signal_ledger import SignalLedger

    db = tmp_path / "pa.db"
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()

    ledger = SignalLedger(tmp_path / "data" / "political_alpha.db")
    ledger.initialize()
    ledger.append(LedgerEntry(
        entry_ts=datetime(2025, 1, 1, tzinfo=timezone.utc),
        trigger_headline="Test headline",
        trigger_source="gdelt",
        persona="powell",
        sentiment=0.4,
        trade_ticker="SPY",
        trade_side="LONG",
        trade_size_pct=0.02,
        horizon_hours=24,
        hypothesis="x" * 50,
    ))

    out = _call_tool_sync("get_political_alpha_ledger", {"limit": 5})
    payload = json.loads(out)
    assert isinstance(payload, list)
    assert len(payload) == 1
    assert payload[0]["persona"] == "powell"
    assert payload[0]["side"] == "LONG"


def test_political_alpha_ledger_empty_is_empty_list(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    out = _call_tool_sync("get_political_alpha_ledger")
    assert json.loads(out) == []
