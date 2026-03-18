import os
import time
import streamlit as st
from anthropic import Anthropic
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Anthropic client
# Make sure to set ANTHROPIC_API_KEY in your .env file
try:
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
except Exception as e:
    client = None
    print(f"Warning: Could not initialize Anthropic client: {e}")

# ── LLM Usage Tracking ───────────────────────────────────────────────────────

# Pricing per million tokens (input / output)
LLM_PRICING = {
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
}

# Persistent store — set once from app.py via set_llm_stats_store() using
# st.cache_resource so it survives Streamlit's module reloading.
_llm_stats: dict = None  # type: ignore

def set_llm_stats_store(store: dict):
    """Called from app.py to inject the persistent stats dict."""
    global _llm_stats
    _llm_stats = store

def get_llm_stats() -> dict:
    """Return the current LLM usage stats."""
    if _llm_stats is None:
        return {"total_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_cost": 0.0, "total_latency": 0.0}
    return dict(_llm_stats)

def track_llm_usage(response, model_name: str, latency: float):
    """Accumulate token / cost / latency stats from an Anthropic response."""
    if _llm_stats is None:
        return
    try:
        inp = response.usage.input_tokens
        out = response.usage.output_tokens
        pricing = LLM_PRICING.get(model_name, {"input": 3.00, "output": 15.00})
        cost = (inp * pricing["input"] + out * pricing["output"]) / 1_000_000

        _llm_stats["total_calls"] += 1
        _llm_stats["input_tokens"] += inp
        _llm_stats["output_tokens"] += out
        _llm_stats["total_cost"] += cost
        _llm_stats["total_latency"] += latency
    except Exception as e:
        print(f"Warning: LLM usage tracking failed: {e}")

def generate_portfolio_advice(
    portfolio_data: Dict[str, Any],
    risk_metrics: Dict[str, Any],
    macro_prediction: Dict[str, Any],
    user_profile: Dict[str, Any],
    eli10_mode: bool = False
) -> str:
    """
    Generates personalized portfolio advice using Claude.
    """
    if not client:
        return "Anthropic API key is missing or invalid. Please check your .env file."
        
    # Format the data into a readable string for the prompt
    holdings_str = "\n".join([f"- {ticker}: ${val:,.2f} ({portfolio_data['weights'].get(ticker, 0)*100:.1f}%)" 
                             for ticker, val in portfolio_data['holdings'].items()])
                             
    # Base prompt construction
    prompt = f"""
I am an investor asking for portfolio advice. Here is my profile and data:

USER PROFILE:
- Name: {user_profile.get('name', 'Investor')}
- Stated Risk Tolerance: {user_profile.get('risk_tolerance', 'Moderate')}
- Investment Horizon: {user_profile.get('horizon', 'Medium-term (3-7 years)')}

CURRENT PORTFOLIO (${portfolio_data.get('total_value', 0):,.2f}):
{holdings_str}

CALCULATED RISK METRICS:
- Annualized Volatility: {risk_metrics.get('volatility', 0)*100:.1f}%
- Portfolio Beta (vs S&P 500): {risk_metrics.get('beta', 1.0):.2f}
- Concentration Risk (HHI): {risk_metrics.get('hhi', 0):.1f} / 10000
- Overall Risk Score: {risk_metrics.get('risk_score', 0)} / 100

MACRO ENVIRONMENT (ML Model Prediction):
- Probability of Market Correction (next month): {macro_prediction.get('probability', 0)*100:.1f}%
- Predicted Scenario: {"Correction expected" if macro_prediction.get('prediction') == 1 else "Market stable"}

YOUR TASK:
Act as an expert, fiduciary financial advisor. Analyze the gap between my 'Stated Risk Tolerance' and the actual math of my 'Calculated Risk Metrics'.
Also consider the 'Macro Environment' prediction.
Provide 3 concrete, actionable rebalancing suggestions to optimize my portfolio for my goals and the current macro climate.
    """
    
    if eli10_mode:
        system_prompt = """You are a friendly, patient financial advisor who explains everything as if speaking to a 10-year-old ("Explain Like I'm 10" mode). 
Avoid all complex financial jargon. If you must use a financial concept, formulate an easy-to-understand analogy (like using buckets of water, slices of pizza, or a roller coaster). 
Keep your tone encouraging, simple, and very clear."""
    else:
        system_prompt = """You are a precise, data-driven, premium financial advisor. 
You speak to your clients with professional, sophisticated, and objective language. 
Focus on empirical data, risk-adjusted returns, and modern portfolio theory. Avoid hype or emotional language.
Use markdown formatting (bullet points, bold text) to make your points clear and structured."""

    try:
        _model = "claude-haiku-4-5-20251001"
        t0 = time.time()
        response = client.messages.create(
            model=_model,
            max_tokens=1000,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        track_llm_usage(response, _model, time.time() - t0)
        return response.content[0].text
    except Exception as e:
        return f"Error communicating with Claude: {e}"

def generate_news_summary(
    news_items: List[Dict],
    holdings: List[str],
    eli10_mode: bool = False
) -> str:
    """
    Summarizes recent news contextually relevant to the user's portfolio.
    """
    if not client:
        return "Anthropic API key is missing or invalid. Please check your .env file."
        
    if not news_items:
        return "No recent news found for your holdings."
        
    news_text = "\n\n".join([f"Headline: {n.get('Title', n.get('title'))}\nSource: {n.get('Publisher', n.get('publisher'))}" for n in news_items])
    
    prompt = f"""
Here are some recent news headlines affecting my portfolio holdings ({', '.join(holdings)}):

{news_text}

Provide a brief, synthesized summary of what this means for my portfolio.
    """
    
    if eli10_mode:
        system_prompt = "You are a friendly explainer. Summarize this news so a 10-year-old can understand how it might affect their piggy bank investments. Use simple analogies."
    else:
        system_prompt = "You are a concise financial analyst. Provide a professional, objective summary of how these news events interrelate and impact the specific holdings mentioned. Focus on fundamental impact."

    try:
        _model = "claude-haiku-4-5-20251001"
        t0 = time.time()
        response = client.messages.create(
            model=_model,
            max_tokens=500,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        track_llm_usage(response, _model, time.time() - t0)
        return response.content[0].text
    except Exception as e:
        return f"Error communicating with Claude: {e}"

def compare_portfolio_with_standard(
    user_portfolio_data: Dict[str, Any],
    user_risk_score: int,
    standard_portfolio_name: str,
    standard_portfolio_desc: str,
    eli10_mode: bool = False
) -> str:
    """
    Compares the user's custom portfolio to a well-known standard portfolio (like All-Weather).
    """
    if not client:
        return "Anthropic API key is missing or invalid."
        
    holdings_str = "\n".join([f"- {ticker}: ({user_portfolio_data['weights'].get(ticker, 0)*100:.1f}%)" 
                             for ticker in user_portfolio_data['holdings'].keys()])
                             
    prompt = f"""
I want to compare my custom portfolio to the classic "{standard_portfolio_name}".

MY PORTFOLIO (Risk Score: {user_risk_score}/100):
{holdings_str}

STANDARD PORTFOLIO: "{standard_portfolio_name}"
Description/Logic: {standard_portfolio_desc}

Provide a brief compare-and-contrast analysis. What are the trade-offs I'm making by choosing my custom allocation over the {standard_portfolio_name}?
    """
    
    if eli10_mode:
        system_prompt = "Explain the difference between these two ways of saving money as if you were talking to a 10-year-old. Keep it simple and use a fun analogy."
    else:
        system_prompt = "You are a quantitative portfolio manager. Give an objective, professional breakdown of the differing risk exposures and structural differences between the two portfolios."

    try:
        _model = "claude-haiku-4-5-20251001"
        t0 = time.time()
        response = client.messages.create(
            model=_model,
            max_tokens=600,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        track_llm_usage(response, _model, time.time() - t0)
        return response.content[0].text
    except Exception as e:
        return f"Error communicating with Claude: {e}"


# ── Portfolio Autopilot — Orchestrated Multi-Step LLM Chain ───────────────

import json

def generate_autopilot_recommendations(
    holdings: Dict[str, float],
    prices: Dict[str, float],
    risk_metrics: Dict[str, Any],
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    4-step orchestrated LLM chain:
      1. Claude proposes 3 trades as structured JSON
      2. Python simulates each via what_if_analysis
      3. Claude synthesises a final recommendation
      4. Returns everything for display
    """
    if not client:
        return {"error": "Anthropic API key is missing or invalid."}

    from utils.chatbot_tools import _execute_what_if_analysis, _execute_calculate_portfolio_risk

    total_value = sum(holdings[t] * prices.get(t, 0) for t in holdings)
    weights = {t: (holdings[t] * prices.get(t, 0)) / total_value if total_value > 0 else 0 for t in holdings}
    holdings_str = "\n".join(
        f"- {t}: {holdings[t]} shares @ ${prices.get(t,0):,.2f} = ${holdings[t]*prices.get(t,0):,.2f} ({weights[t]*100:.1f}%)"
        for t in holdings
    )

    # ── STEP 1: Claude proposes 3 trades ──────────────────────────────────
    step1_prompt = f"""You are an expert portfolio strategist. Analyze this portfolio and propose exactly 3 trade options to improve it.

INVESTOR PROFILE:
- Risk Tolerance: {profile.get('risk_tolerance', 'Moderate')}
- Horizon: {profile.get('horizon', 'Medium-term')}

CURRENT PORTFOLIO (${total_value:,.2f}):
{holdings_str}

RISK METRICS:
- Volatility: {risk_metrics.get('volatility', 0)*100:.1f}%
- Beta: {risk_metrics.get('beta', 1.0):.2f}
- Sharpe: {risk_metrics.get('sharpe', 0):.2f}
- HHI: {risk_metrics.get('hhi', 0):.0f}
- Max Drawdown: {risk_metrics.get('max_drawdown', 0)*100:.1f}%

Respond with ONLY a JSON array of exactly 3 trade proposals. No markdown fences, no commentary:
[
  {{"action": "buy", "ticker": "...", "quantity": <int>, "rationale": "..."}},
  {{"action": "sell", "ticker": "...", "quantity": <int>, "rationale": "..."}},
  {{"action": "buy", "ticker": "...", "quantity": <int>, "rationale": "..."}}
]

Rules:
- "sell" tickers MUST be in the current portfolio
- "buy" can be new or existing tickers
- quantities should be realistic (small adjustments, not the whole portfolio)
- each proposal should target a DIFFERENT improvement (diversification, risk reduction, return enhancement)"""

    try:
        _model = "claude-sonnet-4-6"
        t0 = time.time()
        resp1 = client.messages.create(
            model=_model,
            max_tokens=800,
            system="You are a quantitative portfolio strategist. Respond ONLY with valid JSON, no markdown fences.",
            messages=[{"role": "user", "content": step1_prompt}],
        )
        track_llm_usage(resp1, _model, time.time() - t0)

        raw = resp1.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
        proposals = json.loads(raw)
    except (json.JSONDecodeError, Exception) as e:
        return {"error": f"Step 1 failed — could not parse trade proposals: {e}"}

    # ── STEP 2: Simulate each trade with existing what_if_analysis ────────
    current_risk_json = _execute_calculate_portfolio_risk(holdings)
    current_risk = json.loads(current_risk_json)

    simulations = []
    for i, trade in enumerate(proposals[:3]):
        action_map = {"buy": "add", "sell": "remove"}
        action = action_map.get(trade.get("action", "buy"), "add")
        result_json = _execute_what_if_analysis(
            holdings, action, trade["ticker"], trade["quantity"]
        )
        result = json.loads(result_json)
        result["proposal_label"] = f"Proposal {chr(65+i)}"
        result["rationale"] = trade.get("rationale", "")
        result["trade_description"] = f"{trade['action'].upper()} {trade['quantity']} {trade['ticker']}"
        simulations.append(result)

    # ── STEP 3: Claude synthesises a final recommendation ─────────────────
    sim_summary = "\n\n".join(
        f"{s['proposal_label']} — {s['trade_description']}:\n"
        f"  Rationale: {s['rationale']}\n"
        f"  New Volatility: {s.get('new_volatility_pct', 'N/A')}% | Sharpe: {s.get('new_sharpe_ratio', 'N/A')} | Max DD: {s.get('new_max_drawdown_pct', 'N/A')}%"
        for s in simulations
    )

    step3_prompt = f"""You are an expert portfolio strategist. I ran 3 simulated trades on my portfolio.

CURRENT METRICS:
- Volatility: {current_risk.get('volatility_pct', 'N/A')}%
- Sharpe: {current_risk.get('sharpe_ratio', 'N/A')}
- Max Drawdown: {current_risk.get('max_drawdown_pct', 'N/A')}%
- HHI: {current_risk.get('hhi', 'N/A')}

SIMULATION RESULTS:
{sim_summary}

INVESTOR: {profile.get('risk_tolerance', 'Moderate')} risk tolerance, {profile.get('horizon', 'Medium-term')} horizon.

Provide a clear, structured final recommendation:
1. Rank the 3 proposals from best to worst for this investor
2. Explain the tradeoffs of each
3. Give a final verdict: which single proposal should they execute and why?

Use markdown formatting. Be concise but thorough."""

    try:
        t0 = time.time()
        resp3 = client.messages.create(
            model=_model,
            max_tokens=1000,
            system="You are a precise, data-driven financial advisor. Use markdown formatting.",
            messages=[{"role": "user", "content": step3_prompt}],
        )
        track_llm_usage(resp3, _model, time.time() - t0)
        synthesis = resp3.content[0].text
    except Exception as e:
        synthesis = f"Error in synthesis step: {e}"

    return {
        "proposals": proposals[:3],
        "simulations": simulations,
        "current_risk": current_risk,
        "synthesis": synthesis,
    }
