from dotenv import load_dotenv
load_dotenv()
import os
from utils.ai_advisor import generate_portfolio_advice

if not os.getenv("ANTHROPIC_API_KEY"):
    print("No Anthropic API key found, skipping API test.")
else:
    dummy_portfolio = {'holdings': {'AAPL': 1000, 'MSFT': 1000}, 'weights': {'AAPL': 0.5, 'MSFT': 0.5}, 'total_value': 2000}
    dummy_metrics = {'volatility': 0.15, 'beta': 1.1, 'hhi': 5000, 'risk_score': 65}
    dummy_prediction = {'probability': 0.65, 'prediction': 1}
    dummy_profile = {'name': 'Test User', 'risk_tolerance': 'High', 'horizon': 'Long-term'}
    
    print("Testing generate_portfolio_advice (Normal Mode)...")
    res = generate_portfolio_advice(dummy_portfolio, dummy_metrics, dummy_prediction, dummy_profile, eli10_mode=False)
    print(res[:150] + "...\n")
    
    print("Testing generate_portfolio_advice (ELI10 Mode)...")
    res = generate_portfolio_advice(dummy_portfolio, dummy_metrics, dummy_prediction, dummy_profile, eli10_mode=True)
    print(res[:150] + "...")
