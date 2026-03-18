import streamlit as st

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

# Persistent LLM stats store — survives Streamlit reruns & module reloading
@st.cache_resource
def _create_llm_stats_store():
    return {
        "total_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_cost": 0.0,
        "total_latency": 0.0,
    }

from utils.ai_advisor import set_llm_stats_store
set_llm_stats_store(_create_llm_stats_store())

from components.sidebar import render_sidebar
from views.dashboard import render_dashboard
from views.ai_chat import render_ai_chat
from views.stress_test import render_stress_test
from views.suggestions import render_suggestions
from views.macro_radar import render_macro_radar
from views.news import render_news
from views.popular_portfolios import render_popular_portfolios
from views.ml_transparency import render_ml_transparency
from views.efficient_frontier import render_efficient_frontier
from views.backtest import render_backtest

def main():
    # Render Sidebar and get user inputs
    profile, holdings = render_sidebar()

    # Store globally in session state
    st.session_state.profile = profile
    st.session_state.holdings = holdings

    selected_page = st.radio(
        "Navigation",
        ["Dashboard", "AI Chat", "Stress Test", "Suggestions", "Macro Radar", "News & Sentiment", "Efficient Frontier", "Backtest", "Standard Portfolios", "ML Transparency"],
        horizontal=True,
        label_visibility="collapsed"
    )

    with st.expander("About This App — Technical Overview"):
        st.markdown("""
**AI Portfolio Advisor** is a full-stack portfolio analytics platform built with Streamlit, combining real-time market data with multiple AI/ML integration patterns and a unified cross-component signal pipeline.

**4 LLM Integration Patterns:**
1. **Single-call summarisation** (Haiku) — Portfolio advice, news summaries, macro analysis
2. **Agentic tool-use loop** (Sonnet) — Chatbot with 6 tools that Claude calls autonomously across multiple turns
3. **Structured JSON generation** (Sonnet) — Stress-test scenario parameters generated as validated JSON
4. **Orchestrated multi-step decision chain** (Sonnet) — AI Autopilot: Claude proposes trades → Python simulates → Claude synthesises recommendation — enriched with macro, sentiment, efficient frontier, stress test, and backtest signals

**2 Custom ML Models (trained via GridSearchCV pipelines):**
- **Macro Risk Model** — Best-of-4 classifier (Logistic Regression, Random Forest, Gradient Boosting, XGBoost) selected by AUC-ROC via 5-fold `TimeSeriesSplit`. Trained on 24 years of macro data (12 features) with balanced class handling.
- **Sentiment Model** — Logistic Regression + TF-IDF with GridSearchCV over C, max_features, and n-gram range. Trained on ~6,000 financial tweets using time-ordered sequential split.

**Cross-Component Signal Pipeline:** ML macro predictions, NLP sentiment scores, efficient frontier optimal weights, stress test vulnerabilities, and backtest performance all flow into a shared signal registry — enriching every LLM prompt with quantitative context. Risk metrics use a dynamic risk-free rate (live US 10Y Treasury yield) and include CVaR (Conditional Value at Risk) alongside Sharpe, Beta, HHI, and Max Drawdown.

See the **ML Transparency** tab for full model evaluation (classification reports, ROC curves, PR curves, calibration, cross-validation, learning curves, feature importances, and model limitations).
""")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ML Transparency doesn't require holdings
    if selected_page == "ML Transparency":
        render_ml_transparency()
        return

    if len(holdings) == 0:
        st.warning("Please add at least one holding in the sidebar to view your portfolio analytics.")
        st.stop()

    # Page Routing
    if selected_page == "Dashboard":
        render_dashboard()
    elif selected_page == "AI Chat":
        render_ai_chat()
    elif selected_page == "Stress Test":
        render_stress_test()
    elif selected_page == "Suggestions":
        render_suggestions()
    elif selected_page == "Macro Radar":
        render_macro_radar()
    elif selected_page == "News & Sentiment":
        render_news()
    elif selected_page == "Efficient Frontier":
        render_efficient_frontier()
    elif selected_page == "Backtest":
        render_backtest()
    elif selected_page == "Standard Portfolios":
        render_popular_portfolios()

if __name__ == "__main__":
    main()
