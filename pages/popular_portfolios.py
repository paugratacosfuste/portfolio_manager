import streamlit as st
import pandas as pd
from utils.ai_advisor import compare_portfolio_with_standard

def render_popular_portfolios():
    st.markdown("<h1>Standard Portfolios Gallery</h1>", unsafe_allow_html=True)
    st.markdown("<p>Compare your custom allocation against time-tested investment strategies.</p>", unsafe_allow_html=True)
    
    holdings = st.session_state.holdings
    
    if not holdings:
        st.warning("No holdings found to compare.")
        return
        
    # Dictionary of standard portfolios
    standard_portfolios = {
        "Ray Dalio's All-Weather Portfolio": {
            "desc": "Designed to perform well regardless of the economic environment (inflation, deflation, growth, depression).",
            "weights": {
                "Stocks (VTI)": "30%",
                "Long-term Treasuries (TLT)": "40%",
                "Intermediate Treasuries (IEF)": "15%",
                "Gold (GLD)": "7.5%",
                "Commodities (DBC)": "7.5%"
            }
        },
        "The Classic 60/40 Portfolio": {
            "desc": "The traditional balanced approach for moderate risk tolerance.",
            "weights": {
                "Broad US Stocks (SPY)": "60%",
                "US Aggregate Bonds (BND)": "40%"
            }
        },
        "The Swensen Model (Yale Endowment)": {
            "desc": "Heavily diversified into alternative assets to reduce reliance on domestic equity.",
            "weights": {
                "Domestic Equity": "30%",
                "Foreign Equity": "15%",
                "Emerging Markets": "5%",
                "Real Estate (REITs)": "20%",
                "US Treasury Bonds": "15%",
                "TIPS": "15%"
            }
        }
    }
    
    st.markdown("### Select a Benchmark")
    selected_name = st.selectbox("Benchmark Portfolio", list(standard_portfolios.keys()))
    selected_portfolio = standard_portfolios[selected_name]
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown(f"#### {selected_name}")
        st.markdown(f"*{selected_portfolio['desc']}*")
        
        df_standard = pd.DataFrame(list(selected_portfolio['weights'].items()), columns=["Asset Class", "Weight"])
        st.table(df_standard)
        
    with col2:
        st.markdown("#### AI Compare & Contrast")
        st.info("Curious how your custom portfolio stacks up against the " + selected_name + " structurally?")
        
        if st.button("Run Claude Comparison"):
            # Mock portfolio data and risk score (In a real app, pass the calculated weights and risk score from suggestions.py)
            user_portfolio_data = {'holdings': holdings, 'weights': {k: 1/len(holdings) for k in holdings.keys()}}
            user_risk_score = 65 
            
            with st.spinner("Claude is analyzing the structural differences..."):
                analysis = compare_portfolio_with_standard(
                    user_portfolio_data=user_portfolio_data,
                    user_risk_score=user_risk_score,
                    standard_portfolio_name=selected_name,
                    standard_portfolio_desc=selected_portfolio['desc'],
                    eli10_mode=st.session_state.get('eli10_mode', False)
                )
                
                st.markdown("<div style='background-color:#FFFFFF; padding:1.5rem; border-radius:10px; border: 1px solid #D9D9D9;'>", unsafe_allow_html=True)
                st.markdown(analysis)
                st.markdown("</div>", unsafe_allow_html=True)
