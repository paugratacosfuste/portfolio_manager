import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_fetcher import fetch_current_prices, fetch_historical_data

def render_dashboard():
    st.title("Dashboard: Reality Check")
    st.write("Welcome to your portfolio overview. Let's see how your current allocation is performing.")
    
    holdings = st.session_state.holdings
    profile = st.session_state.profile
    
    if not holdings:
        st.warning("No holdings found.")
        return
        
    tickers = list(holdings.keys())
    
    with st.spinner("Fetching current market data..."):
        prices = fetch_current_prices(tickers)
        
    # Calculate Total Value & Weights
    total_value = 0.0
    portfolio_data = []
    
    for ticker, qty in holdings.items():
        price = prices.get(ticker, 0.0)
        value = qty * price
        total_value += value
        portfolio_data.append({"Ticker": ticker, "Quantity": qty, "Price": price, "Value": value})
        
    df_portfolio = pd.DataFrame(portfolio_data)
    if total_value > 0:
        df_portfolio['Weight (%)'] = (df_portfolio['Value'] / total_value) * 100
    else:
        df_portfolio['Weight (%)'] = 0.0
        
    # Top Level Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Portfolio Value", f"${total_value:,.2f}")
    with col2:
        top_asset = df_portfolio.loc[df_portfolio['Value'].idxmax(), 'Ticker'] if not df_portfolio.empty and total_value > 0 else 'N/A'
        st.metric("Top Asset", top_asset)
    with col3:
        st.metric("Stated Risk Tolerance", profile['risk_tolerance'])

    st.markdown("")
    
    # Visualizations
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("### Asset Allocation")
        if total_value > 0:
            fig = px.pie(
                df_portfolio, 
                values='Value', 
                names='Ticker', 
                hole=0.4,
                color_discrete_sequence=['#3A6EA5', '#1F8A70', '#0B1F3A', '#C44536', '#F4A261', '#2A9D8F']
            )
            fig.update_layout(
                margin=dict(t=0, b=0, l=0, r=0), 
                showlegend=True,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Portfolio value is zero.")
            
    with col2:
        st.markdown("### 1-Year Historical Performance (Normalized)")
        with st.spinner("Fetching historical data..."):
            hist_data = fetch_historical_data(tickers, period='1y')
            
        if not hist_data.empty:
            # Normalize to 100 at start
            normalized = (hist_data / hist_data.iloc[0]) * 100
            
            # Simple portfolio baseline (equal weight for visualization purposes if actual weights get complex)
            # Better: Calculate weighted portfolio value over time
            port_history = pd.Series(0.0, index=hist_data.index)
            for ticker in tickers:
                if ticker in hist_data.columns:
                    # Value of that holding over time
                    qty = holdings[ticker]
                    port_history += hist_data[ticker] * qty
                    
            normalized['Your Portfolio'] = (port_history / port_history.iloc[0]) * 100 if port_history.iloc[0] > 0 else 0
            
            fig = px.line(
                normalized, 
                y='Your Portfolio',
                color_discrete_sequence=['#0B1F3A']
            )
            # Add benchmark (e.g., S&P 500 equivalent if fetched, otherwise just show portfolio)
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Normalized Value (Baseline 100)",
                margin=dict(t=0, b=0, l=0, r=0),
                legend_title_text='',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Could not fetch historical data for charts.")
            
    st.markdown("### Holdings Details")
    st.dataframe(
        df_portfolio.style.format({
            'Price': '${:.2f}',
            'Value': '${:.2f}',
            'Weight (%)': '{:.1f}%'
        }),
        use_container_width=True,
        hide_index=True
    )
