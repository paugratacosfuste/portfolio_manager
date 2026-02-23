import os
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
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
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
        
    news_text = "\n\n".join([f"Headline: {n.get('title')}\nSource: {n.get('publisher')}" for n in news_items])
    
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
        response = client.messages.create(
            model="claude-3-haiku-20240307", # Use Haiku for faster, cheaper summarization
            max_tokens=500,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
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
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=600,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    except Exception as e:
        return f"Error communicating with Claude: {e}"
