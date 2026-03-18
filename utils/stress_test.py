"""
AI-powered portfolio stress testing engine.
Uses Claude to generate structured scenario parameters, then applies them
quantitatively to the portfolio.
"""
import json
import time
from typing import Dict, Any, List
from utils.ai_advisor import client, track_llm_usage


def generate_stress_scenario(
    scenario_description: str,
    holdings_with_metadata: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Calls Claude with a sophisticated prompt to generate structured JSON
    stress-test parameters for each holding.

    Returns parsed dict with scenario details and per-holding impacts.
    """
    if not client:
        return {"error": "Anthropic API key is missing or invalid."}

    holdings_str = "\n".join(
        [
            f"- {h['ticker']}: sector={h.get('sector','Unknown')}, "
            f"asset_class={h.get('asset_class','Unknown')}, "
            f"weight={h.get('weight_pct', 0):.1f}%"
            for h in holdings_with_metadata
        ]
    )

    system_prompt = (
        "You are a quantitative risk analyst. You must respond with ONLY valid JSON, "
        "no markdown fences, no commentary outside the JSON. "
        "Generate a realistic stress-test scenario with per-holding drawdown estimates."
    )

    user_prompt = f"""Given this stress scenario: "{scenario_description}"

And these portfolio holdings:
{holdings_str}

Generate a JSON object with this EXACT schema:
{{
  "scenario_name": "short name",
  "scenario_summary": "2-3 sentence description of what happens in this scenario",
  "overall_market_impact_pct": <negative number, e.g. -15.0>,
  "correlation_spike": <number 0-1, how much correlations increase>,
  "impacts": [
    {{
      "ticker": "AAPL",
      "sector": "Technology",
      "drawdown_pct": <negative number, e.g. -25.0>,
      "rationale": "1-2 sentence explanation of why this asset is affected this much"
    }}
  ]
}}

Rules:
- drawdown_pct must be NEGATIVE (representing loss)
- Be realistic: tech stocks drop more in tech crashes, bonds may rise in equity crashes
- Crypto should be most volatile in most scenarios
- Defensive sectors (healthcare, utilities) should drop less
- Include ALL holdings listed above
- overall_market_impact_pct should reflect the S&P 500 impact"""

    try:
        _model = "claude-sonnet-4-6"
        t0 = time.time()
        response = client.messages.create(
            model=_model,
            max_tokens=1500,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        track_llm_usage(response, _model, time.time() - t0)

        raw = response.content[0].text.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]

        scenario = json.loads(raw)
        return scenario

    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse Claude's JSON response: {e}", "raw": raw}
    except Exception as e:
        return {"error": f"Error generating stress scenario: {e}"}


def apply_stress_to_portfolio(
    holdings: Dict[str, float],
    current_prices: Dict[str, float],
    scenario_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Applies the stress scenario drawdowns to the portfolio and computes
    stressed values, losses, and impact breakdowns.
    """
    impacts_by_ticker = {}
    for impact in scenario_data.get("impacts", []):
        impacts_by_ticker[impact["ticker"]] = impact

    results = {
        "scenario_name": scenario_data.get("scenario_name", "Unknown"),
        "scenario_summary": scenario_data.get("scenario_summary", ""),
        "overall_market_impact_pct": scenario_data.get("overall_market_impact_pct", 0),
        "correlation_spike": scenario_data.get("correlation_spike", 0),
        "holdings_impact": [],
        "current_total": 0.0,
        "stressed_total": 0.0,
    }

    current_total = 0.0
    stressed_total = 0.0

    for ticker, qty in holdings.items():
        price = current_prices.get(ticker, 0)
        current_value = qty * price
        current_total += current_value

        impact = impacts_by_ticker.get(ticker, {})
        drawdown_pct = impact.get("drawdown_pct", scenario_data.get("overall_market_impact_pct", -10))
        stressed_value = current_value * (1 + drawdown_pct / 100)
        loss = stressed_value - current_value

        stressed_total += stressed_value

        results["holdings_impact"].append({
            "ticker": ticker,
            "current_value": round(current_value, 2),
            "stressed_value": round(stressed_value, 2),
            "loss": round(loss, 2),
            "drawdown_pct": drawdown_pct,
            "rationale": impact.get("rationale", "General market impact applied."),
        })

    results["current_total"] = round(current_total, 2)
    results["stressed_total"] = round(stressed_total, 2)
    results["total_loss"] = round(stressed_total - current_total, 2)
    results["total_loss_pct"] = round(
        ((stressed_total - current_total) / current_total * 100) if current_total > 0 else 0, 1
    )

    return results
