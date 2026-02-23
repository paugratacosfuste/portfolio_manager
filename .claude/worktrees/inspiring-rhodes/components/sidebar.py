import streamlit as st
import pandas as pd

# Common crypto symbols that need -USD suffix for Yahoo Finance
CRYPTO_TICKERS = {
    "BTC", "ETH", "SOL", "ADA", "DOT", "LINK", "AVAX", "MATIC", "XRP",
    "DOGE", "SHIB", "UNI", "AAVE", "LTC", "BCH", "ATOM", "NEAR", "FTM",
    "ALGO", "XLM", "VET", "MANA", "SAND", "AXS", "CRO", "FIL", "APE",
    "ICP", "EOS", "XTZ", "THETA", "HNT", "ZEC", "DASH", "NEO", "COMP",
    "MKR", "SNX", "SUSHI", "YFI", "BAT", "ENJ", "GRT", "ONE", "RUNE",
    "CAKE", "LUNA", "TRX", "BNB", "POL", "ARB", "OP", "SUI", "APT",
    "SEI", "TIA", "JUP", "PEPE", "WIF", "BONK", "RENDER", "FET", "TAO",
}

def normalize_ticker(ticker: str) -> str:
    """Normalizes a ticker symbol. Appends -USD for known crypto symbols."""
    ticker = ticker.strip().upper()
    # If already has -USD suffix, return as is
    if ticker.endswith("-USD"):
        return ticker
    # If it's a known crypto ticker, append -USD
    if ticker in CRYPTO_TICKERS:
        return f"{ticker}-USD"
    return ticker

def render_sidebar():
    """
    Renders the sidebar for user onboarding and portfolio input.
    Returns user profile dict and holdings dict.
    """
    
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/line-chart.png", width=60)
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
        horizon = st.selectbox(
            "Investment Horizon",
            ["Short-term (1-3 years)", "Medium-term (3-7 years)", "Long-term (7+ years)"],
            index=1,
            help="How long to you plan to hold these investments before needing the cash?"
        )
        
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        
        # Portfolio Holdings Input
        st.markdown("### Current Holdings")
        st.markdown("<p style='font-size: 0.9rem; color: #1E1E1E;'>Enter Ticker and Shares/Coins owned.</p>", unsafe_allow_html=True)
        
        # Default starting data
        if 'holdings_df' not in st.session_state:
            st.session_state.holdings_df = pd.DataFrame(
                [
                    {"Ticker": "AAPL", "Quantity": 50.0},
                    {"Ticker": "MSFT", "Quantity": 30.0},
                    {"Ticker": "JNJ", "Quantity": 100.0},
                    {"Ticker": "BTC-USD", "Quantity": 0.5}
                ]
            )
            
        # Data Editor
        edited_df = st.data_editor(
            st.session_state.holdings_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", required=True),
                "Quantity": st.column_config.NumberColumn("Quantity", min_value=0.0001, required=True)
            },
            hide_index=True
        )
        
        # Save state
        st.session_state.holdings_df = edited_df
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Global ELI10 Toggle
        st.markdown("### AI Settings")
        eli10_mode = st.toggle("🤖 Enable 'ELI10' Mode", value=False, help="Explain Like I'm 10: Simplifies all AI advice and terminology.")
        st.session_state.eli10_mode = eli10_mode
        
        # Convert df to dict for backend processing
        valid_holdings = edited_df.dropna(subset=["Ticker", "Quantity"])
        # Format: {'AAPL': 50.0, 'ETH-USD': 2.0}
        # Normalizes crypto tickers (e.g., ETH -> ETH-USD)
        holdings_dict = {normalize_ticker(row['Ticker']): float(row['Quantity']) for _, row in valid_holdings.iterrows() if row['Ticker'] and row['Quantity'] > 0}
        
        profile_dict = {
            "name": name,
            "risk_tolerance": risk_tolerance,
            "horizon": horizon
        }
        
        return profile_dict, holdings_dict
