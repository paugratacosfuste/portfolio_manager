import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.data_fetcher import fetch_current_prices, fetch_historical_data

def render_dashboard():
    st.markdown("<h1>Portfolio Overview</h1>", unsafe_allow_html=True)
    st.markdown("<p>Welcome to your portfolio overview. Let's see how your current allocation is performing.</p>", unsafe_allow_html=True)
    
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
        st.markdown(f"<p>Total Portfolio Value</p><p class='big-number'>${total_value:,.2f}</p>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<p>Top Asset</p><p class='medium-number'>{df_portfolio.loc[df_portfolio['Value'].idxmax(), 'Ticker'] if not df_portfolio.empty and total_value > 0 else 'N/A'}</p>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<p>Stated Risk Tolerance</p><p class='medium-number'>{profile['risk_tolerance']}</p>", unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
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
        st.markdown("### Historical Value")
        with st.spinner("Fetching historical data..."):
            hist_data = fetch_historical_data(tickers, period='1y')
            
        if not hist_data.empty:
            # Calculate total portfolio value over time
            port_history = pd.Series(0.0, index=hist_data.index)
            for ticker in tickers:
                if ticker in hist_data.columns:
                    # Value of that holding over time
                    qty = holdings[ticker]
                    # fill missing prices with previous day to avoid zero-drops
                    filled_history = hist_data[ticker].ffill().fillna(0)
                    port_history += filled_history * qty
                    
            fig = px.area(
                x=port_history.index, 
                y=port_history.values,
                color_discrete_sequence=['#3A6EA5']
            )
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Total Value ($)",
                margin=dict(t=0, b=0, l=0, r=0),
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Could not fetch historical data for charts.")
            
    st.markdown("### Holdings Details")
    st.dataframe(
        df_portfolio.style.format({
            'Quantity': '{:.2f}',
            'Price': '${:.2f}',
            'Value': '${:.2f}',
            'Weight (%)': '{:.1f}%'
        }),
        use_container_width=True,
        hide_index=True
    )

    # ── Correlation Heatmap ─────────────────────────────────────────────
    if hist_data is not None and not hist_data.empty and len(tickers) > 1:
        st.markdown("### Correlation Matrix")
        st.markdown(
            "<p style='color:#666; font-size:0.9rem;'>Daily return correlations between your holdings. "
            "Low or negative correlations indicate good diversification.</p>",
            unsafe_allow_html=True,
        )
        valid_cols = [t for t in tickers if t in hist_data.columns]
        if len(valid_cols) > 1:
            returns = hist_data[valid_cols].pct_change().dropna()
            corr_matrix = returns.corr()

            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns.tolist(),
                y=corr_matrix.index.tolist(),
                colorscale=[[0, "#3A6EA5"], [0.5, "#FFFFFF"], [1, "#C44536"]],
                zmin=-1,
                zmax=1,
                text=corr_matrix.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 11},
                hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.2f}<extra></extra>",
            ))
            fig_corr.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=max(350, len(valid_cols) * 50),
            )
            st.plotly_chart(fig_corr, use_container_width=True)
