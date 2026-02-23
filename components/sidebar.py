import streamlit as st
import pandas as pd
from utils.data_fetcher import fetch_current_prices

def render_sidebar():
    """
    Renders the sidebar for user onboarding and portfolio input.
    Returns user profile dict and holdings dict.
    """
    
    with st.sidebar:
        st.markdown("<h2 style='color: #0B1F3A; margin-top: 0;'>Portfolio Setup</h2>", unsafe_allow_html=True)
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        
        # User Profile Information
        st.markdown("### Profile")
        name = st.text_input("Name", value="Investor")
        risk_tolerance = st.select_slider(
            "Risk Tolerance",
            options=["Conservative", "Moderate", "Growth", "Aggressive"],
            value="Moderate",
            help="Higher risk tolerance allows for more volatility in exchange for potential higher returns."
        )
        horizon = st.radio(
            "Investment Horizon",
            ["Trading (< 1 year)", "Short-term (1-5 years)", "Mid-term (5-10 years)", "Long-term (10+ years)"],
            index=2,
            help="How long do you plan to hold these investments before needing the cash?"
        )
        
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        
        # Portfolio Holdings Input
        st.markdown("### Current Holdings")
        st.markdown("<p style='font-size: 0.9rem; color: #1E1E1E;'>Enter Ticker and Shares/Coins owned.</p>", unsafe_allow_html=True)
        
        # Default starting data
        if 'holdings_df' not in st.session_state:
            st.session_state.holdings_df = pd.DataFrame(
                [
                    # 4 Stocks
                    {"Ticker": "AAPL", "Quantity": 50.0},
                    {"Ticker": "MSFT", "Quantity": 30.0},
                    {"Ticker": "JNJ", "Quantity": 100.0},
                    {"Ticker": "NVDA", "Quantity": 20.0},
                    # 2 ETFs
                    {"Ticker": "SPY", "Quantity": 15.0},
                    {"Ticker": "QQQ", "Quantity": 15.0},
                    # 2 Cryptos
                    {"Ticker": "BTC-USD", "Quantity": 0.5},
                    {"Ticker": "ETH-USD", "Quantity": 5.0},
                    # 1 Bond
                    {"Ticker": "BND", "Quantity": 40.0}
                ]
            )
            
        # Data Editor
        edited_df = st.data_editor(
            st.session_state.holdings_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", required=True),
                "Quantity": st.column_config.NumberColumn("Quantity", min_value=0.0001, required=True, format="%.2f")
            },
            hide_index=True
        )
        
        # Save state
        st.session_state.holdings_df = edited_df

        # Price preview — gives immediate feedback for crypto/ETH tickers
        preview_rows = edited_df.dropna(subset=["Ticker", "Quantity"])
        if not preview_rows.empty:
            crypto_map_preview = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "XRP": "XRP-USD", "ADA": "ADA-USD"}
            preview_tickers = []
            for _, row in preview_rows.iterrows():
                if row['Ticker'] and row['Quantity'] > 0:
                    t = str(row['Ticker']).upper().strip()
                    preview_tickers.append(crypto_map_preview.get(t, t))
            if preview_tickers:
                prices_preview = fetch_current_prices(preview_tickers)
                if prices_preview:
                    st.caption("Current Prices:")
                    price_df = pd.DataFrame(
                        [{"Ticker": t, "Price (USD)": f"${p:,.2f}" if p > 0 else "N/A"}
                         for t, p in prices_preview.items()]
                    )
                    st.dataframe(price_df, hide_index=True, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Global ELI10 Toggle
        st.markdown("### AI Settings")
        eli10_mode = st.toggle("🤖 Enable 'ELI10' Mode", value=False, help="Explain Like I'm 10: Simplifies all AI advice and terminology.")
        st.session_state.eli10_mode = eli10_mode
        
        # Convert df to dict for backend processing
        valid_holdings = edited_df.dropna(subset=["Ticker", "Quantity"])
        
        # Crypto Mapping
        crypto_map = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "XRP": "XRP-USD", "ADA": "ADA-USD"}
        
        holdings_dict = {}
        for _, row in valid_holdings.iterrows():
            if row['Ticker'] and row['Quantity'] > 0:
                ticker = str(row['Ticker']).upper().strip()
                if ticker in crypto_map:
                    ticker = crypto_map[ticker]
                holdings_dict[ticker] = float(row['Quantity'])
        
        profile_dict = {
            "name": name,
            "risk_tolerance": risk_tolerance,
            "horizon": horizon
        }
        
        return profile_dict, holdings_dict
