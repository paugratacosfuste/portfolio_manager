import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_fetcher import fetch_historical_data, fetch_current_prices
from utils.portfolio_metrics import calculate_portfolio_volatility, calculate_portfolio_beta, calculate_hhi_index, assess_risk_score

def render_suggestions():
    st.title("Risk Analysis & Suggestions")
    st.write("Analyzing the gap between your stated risk tolerance and mathematical portfolio risk.")
    
    holdings = st.session_state.holdings
    profile = st.session_state.profile
    
    if not holdings:
        st.warning("No holdings found.")
        return
        
    tickers = list(holdings.keys())
    
    with st.spinner("Calculating Risk Metrics..."):
        # Fetch 1y data for risk calculations
        prices_df = fetch_historical_data(tickers, period='1y')
        current_prices = fetch_current_prices(tickers)
        
        # Calculate weights
        weights = {}
        total_value = sum(qty * current_prices.get(t, 0) for t, qty in holdings.items())
        for t, qty in holdings.items():
            if total_value > 0:
                weights[t] = (qty * current_prices.get(t, 0)) / total_value
            else:
                weights[t] = 0
                
        # Handle cases where not all tickers have historical data
        valid_tickers_for_risk = [t for t in tickers if t in prices_df.columns]
        
        if len(valid_tickers_for_risk) > 0 and len(prices_df) > 50:
            volatility = calculate_portfolio_volatility(prices_df[valid_tickers_for_risk], weights)
            try:
                spy_df = fetch_historical_data(['SPY'], period='1y')
                mkt = spy_df['SPY'] if 'SPY' in spy_df else prices_df.iloc[:, 0]
                beta = calculate_portfolio_beta(prices_df[valid_tickers_for_risk], mkt, weights)
            except Exception:
                beta = 1.0
            hhi = calculate_hhi_index(weights)
            risk_score = assess_risk_score(volatility, hhi, beta, total_value)
        else:
            volatility = 0.0
            beta = 1.0
            hhi = calculate_hhi_index(weights)
            risk_score = 50 # Default middle
            st.warning("Not enough historical data to calculate full risk metrics.")
            
    # Display Metrics
    st.markdown("### Risk Snapshot")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Risk Score (0-100)", f"{risk_score}", help="Higher means riskier. Based on Volatility, Beta, and Concentration.")
    with col2:
        st.metric("Ann. Volatility", f"{volatility*100:.1f}%", help="Expected annual swing in portfolio value.")
    with col3:
        st.metric("Portfolio Beta", f"{beta:.2f}", help=">1 is more volatile than market, <1 is less.")
    with col4:
        st.metric("Concentration (HHI)", f"{hhi:.0f}", help=">2500 is highly concentrated.")
        
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    # The Actionable Advice Section
    st.markdown("### AI Rebalancing Suggestions")
    
    # Collect data for prompt
    portfolio_data = {
        'holdings': holdings,
        'weights': weights,
        'total_value': total_value
    }
    
    metrics_data = {
        'volatility': volatility,
        'beta': beta,
        'hhi': hhi,
        'risk_score': risk_score
    }
    
    # Import here to avoid circular dependencies if any
    from utils.ai_advisor import generate_portfolio_advice
    
    # Dummy macro for this page (will do real one in radar)
    dummy_macro = {'probability': 0.5, 'prediction': 0}
    
    if st.button("Generate Personalized Advice (Claude)"):
        with st.spinner("Claude is analyzing your portfolio..."):
            advice = generate_portfolio_advice(
                portfolio_data=portfolio_data,
                risk_metrics=metrics_data,
                macro_prediction=dummy_macro,
                user_profile=profile,
                eli10_mode=st.session_state.get('eli10_mode', False)
            )

            st.info(f"**Claude's Analysis:**\n\n{advice}")
