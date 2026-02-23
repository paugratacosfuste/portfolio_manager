import streamlit as st
import os

# Set absolute page configuration first (Must be first Streamlit command)
st.set_page_config(
    page_title="AI Portfolio Advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css("styles/main.css")

from components.sidebar import render_sidebar
from views.dashboard import render_dashboard
from views.suggestions import render_suggestions
from views.macro_radar import render_macro_radar
from views.news import render_news
from views.popular_portfolios import render_popular_portfolios

def main():
    # Render Sidebar and get user inputs
    profile, holdings = render_sidebar()
    
    # Store globally in session state
    st.session_state.profile = profile
    st.session_state.holdings = holdings
    
    selected_page = st.radio(
        "Navigation",
        ["Dashboard", "Suggestions", "Macro Radar", "News & Sentiment", "Standard Portfolios"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    if len(holdings) == 0:
        st.warning("Please add at least one holding in the sidebar to view your portfolio analytics.")
        st.stop()
        
    # Page Routing
    if selected_page == "Dashboard":
        render_dashboard()
    elif selected_page == "Suggestions":
        render_suggestions()
    elif selected_page == "Macro Radar":
        render_macro_radar()
    elif selected_page == "News & Sentiment":
        render_news()
    elif selected_page == "Standard Portfolios":
        render_popular_portfolios()

if __name__ == "__main__":
    main()
