"""Tests for mcp_server.portfolio_loader."""
from __future__ import annotations

import json
import os

import pytest

from mcp_server.portfolio_loader import (
    PortfolioConfig,
    load_portfolio_config,
)


@pytest.fixture
def env_config(tmp_path, monkeypatch):
    """Point PORTFOLIO_TRACKER_CONFIG at a writeable tmp file."""
    path = tmp_path / "portfolio.json"
    monkeypatch.setenv("PORTFOLIO_TRACKER_CONFIG", str(path))
    return path


def test_missing_file_returns_empty_config(tmp_path, monkeypatch):
    monkeypatch.setenv("PORTFOLIO_TRACKER_CONFIG", str(tmp_path / "nope.json"))
    cfg = load_portfolio_config()
    assert cfg == PortfolioConfig(holdings={}, risk_profile="Moderate")


def test_loads_happy_path(env_config):
    env_config.write_text(json.dumps({
        "holdings": {"AAPL": 10, "BTC-USD": 0.25},
        "risk_profile": "Aggressive",
    }))
    cfg = load_portfolio_config()
    assert cfg.holdings == {"AAPL": 10.0, "BTC-USD": 0.25}
    assert cfg.risk_profile == "Aggressive"


def test_defaults_risk_profile(env_config):
    env_config.write_text(json.dumps({"holdings": {"AAPL": 1}}))
    assert load_portfolio_config().risk_profile == "Moderate"


def test_rejects_non_dict_holdings(env_config):
    env_config.write_text(json.dumps({"holdings": [1, 2, 3]}))
    with pytest.raises(ValueError, match="must be a dict"):
        load_portfolio_config()


def test_rejects_non_numeric_quantity(env_config):
    env_config.write_text(json.dumps({"holdings": {"AAPL": "ten"}}))
    with pytest.raises(ValueError, match="not numeric"):
        load_portfolio_config()


def test_rejects_negative_quantity(env_config):
    env_config.write_text(json.dumps({"holdings": {"AAPL": -1}}))
    with pytest.raises(ValueError, match="negative"):
        load_portfolio_config()


def test_rejects_invalid_json(env_config):
    env_config.write_text("{not json")
    with pytest.raises(ValueError, match="Invalid portfolio config"):
        load_portfolio_config()


def test_coerces_integer_and_float_quantities(env_config):
    env_config.write_text(json.dumps({"holdings": {"A": 10, "B": 2.5}}))
    cfg = load_portfolio_config()
    assert isinstance(cfg.holdings["A"], float)
    assert cfg.holdings["A"] == 10.0
    assert cfg.holdings["B"] == 2.5
