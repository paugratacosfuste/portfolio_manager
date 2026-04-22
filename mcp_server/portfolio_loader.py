"""Load a portfolio for the MCP server from a JSON config file.

The Streamlit app keeps holdings in session state. The MCP server is a
separate process so it reads a JSON file whose path defaults to
`~/.portfolio_tracker/portfolio.json` and is overridable via the
`PORTFOLIO_TRACKER_CONFIG` environment variable.

File format:
    {
        "holdings": {"AAPL": 10, "MSFT": 5.5, "BTC-USD": 0.25},
        "risk_profile": "Moderate"    // optional
    }
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CONFIG_PATH = Path.home() / ".portfolio_tracker" / "portfolio.json"


@dataclass(frozen=True)
class PortfolioConfig:
    holdings: dict[str, float]
    risk_profile: str


def _config_path() -> Path:
    override = os.environ.get("PORTFOLIO_TRACKER_CONFIG")
    return Path(override) if override else DEFAULT_CONFIG_PATH


def load_portfolio_config() -> PortfolioConfig:
    """Read the JSON config. Missing file → empty portfolio (MCP server still serves)."""
    path = _config_path()
    if not path.exists():
        return PortfolioConfig(holdings={}, risk_profile="Moderate")

    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as ex:
        raise ValueError(
            f"Invalid portfolio config at {path}: {ex}. "
            f"Expected {{'holdings': {{'TICKER': quantity}}}}."
        ) from ex

    holdings_raw = raw.get("holdings", {})
    if not isinstance(holdings_raw, dict):
        raise ValueError(f"{path}: 'holdings' must be a dict")

    clean: dict[str, float] = {}
    for ticker, qty in holdings_raw.items():
        try:
            q = float(qty)
        except (TypeError, ValueError):
            raise ValueError(
                f"{path}: holding quantity for {ticker!r} is not numeric"
            )
        if q < 0:
            raise ValueError(f"{path}: negative quantity for {ticker!r}")
        clean[str(ticker)] = q

    return PortfolioConfig(
        holdings=clean,
        risk_profile=str(raw.get("risk_profile", "Moderate")),
    )
