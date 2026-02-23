import streamlit as st
import pandas as pd
import plotly.express as px
from utils.ai_advisor import compare_portfolio_with_standard

# Example portfolios with mock creators and performance data
EXAMPLE_PORTFOLIOS = [
    {
        "name": "The All-Weather Shield",
        "creator": "Sarah Chen",
        "creator_role": "Risk Analyst",
        "description": "Designed to perform well regardless of the economic environment — inflation, deflation, growth, or recession.",
        "strategy": "Balanced",
        "risk_level": "Conservative",
        "annualized_return": 7.2,
        "volatility": 8.5,
        "sharpe_ratio": 0.85,
        "year_return": 9.1,
        "allocation": {
            "US Stocks (VTI)": 30,
            "Long-term Bonds (TLT)": 40,
            "Mid-term Bonds (IEF)": 15,
            "Gold (GLD)": 7.5,
            "Commodities (DBC)": 7.5,
        },
        "colors": ['#3A6EA5', '#0B1F3A', '#1F8A70', '#F4A261', '#C44536'],
    },
    {
        "name": "The Classic 60/40",
        "creator": "James Rodriguez",
        "creator_role": "Wealth Manager",
        "description": "The time-tested balanced approach. 60% equities for growth, 40% bonds for stability and income.",
        "strategy": "Moderate",
        "risk_level": "Moderate",
        "annualized_return": 8.5,
        "volatility": 10.2,
        "sharpe_ratio": 0.83,
        "year_return": 12.3,
        "allocation": {
            "US Stocks (SPY)": 60,
            "US Aggregate Bonds (BND)": 40,
        },
        "colors": ['#3A6EA5', '#1F8A70'],
    },
    {
        "name": "The Yale Endowment",
        "creator": "Priya Kapoor",
        "creator_role": "Institutional Investor",
        "description": "Inspired by David Swensen's Yale model — heavily diversified with alternative assets to reduce domestic equity reliance.",
        "strategy": "Diversified",
        "risk_level": "Moderate-Aggressive",
        "annualized_return": 9.8,
        "volatility": 12.1,
        "sharpe_ratio": 0.81,
        "year_return": 11.7,
        "allocation": {
            "Domestic Equity (VTI)": 30,
            "Foreign Equity (VXUS)": 15,
            "Emerging Markets (VWO)": 5,
            "Real Estate REITs (VNQ)": 20,
            "US Treasury Bonds (TLT)": 15,
            "TIPS (TIP)": 15,
        },
        "colors": ['#3A6EA5', '#0B1F3A', '#1F8A70', '#F4A261', '#C44536', '#2A9D8F'],
    },
    {
        "name": "The Tech Growth Engine",
        "creator": "Marcus Lee",
        "creator_role": "Software Engineer & Investor",
        "description": "A high-conviction, growth-oriented portfolio focused on technology leaders and innovation disruptors.",
        "strategy": "Growth",
        "risk_level": "Aggressive",
        "annualized_return": 15.3,
        "volatility": 22.4,
        "sharpe_ratio": 0.68,
        "year_return": 28.5,
        "allocation": {
            "NVIDIA (NVDA)": 20,
            "Apple (AAPL)": 20,
            "Microsoft (MSFT)": 20,
            "Amazon (AMZN)": 15,
            "Alphabet (GOOGL)": 15,
            "Tesla (TSLA)": 10,
        },
        "colors": ['#3A6EA5', '#0B1F3A', '#1F8A70', '#F4A261', '#C44536', '#7C3AED'],
    },
    {
        "name": "The Crypto Frontier",
        "creator": "Elena Vasquez",
        "creator_role": "DeFi Researcher",
        "description": "A crypto-native portfolio blending blue-chip digital assets with traditional hedges for the adventurous investor.",
        "strategy": "Speculative",
        "risk_level": "Very Aggressive",
        "annualized_return": 42.1,
        "volatility": 65.0,
        "sharpe_ratio": 0.65,
        "year_return": 85.2,
        "allocation": {
            "Bitcoin (BTC)": 40,
            "Ethereum (ETH)": 30,
            "Solana (SOL)": 10,
            "Gold (GLD)": 10,
            "US Stocks (SPY)": 10,
        },
        "colors": ['#F4A261', '#3A6EA5', '#7C3AED', '#C44536', '#1F8A70'],
    },
    {
        "name": "The Dividend Harvester",
        "creator": "Robert Fischer",
        "creator_role": "Retired Portfolio Manager",
        "description": "Focused on high-quality dividend-paying stocks and bonds to generate consistent passive income.",
        "strategy": "Income",
        "risk_level": "Conservative",
        "annualized_return": 6.8,
        "volatility": 9.0,
        "sharpe_ratio": 0.76,
        "year_return": 7.4,
        "allocation": {
            "Dividend ETF (VYM)": 35,
            "US Bonds (BND)": 25,
            "Intl Dividend (VIGI)": 15,
            "REITs (VNQ)": 15,
            "Utilities (XLU)": 10,
        },
        "colors": ['#1F8A70', '#0B1F3A', '#3A6EA5', '#F4A261', '#C44536'],
    },
]


def _render_portfolio_card(portfolio: dict):
    """Renders a single portfolio card using only native Streamlit components."""

    # Creator info line
    st.caption(f"By **{portfolio['creator']}** · {portfolio['creator_role']}")
    st.markdown(f"#### {portfolio['name']}")
    st.markdown(f"*{portfolio['description']}*")

    col_chart, col_stats = st.columns([1, 1])

    with col_chart:
        # Pie chart
        alloc_df = pd.DataFrame(
            list(portfolio['allocation'].items()),
            columns=['Asset', 'Weight'],
        )
        fig = px.pie(
            alloc_df,
            values='Weight',
            names='Asset',
            hole=0.45,
            color_discrete_sequence=portfolio.get('colors', px.colors.qualitative.Set2),
        )
        fig.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            showlegend=True,
            legend=dict(font=dict(size=10)),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=280,
        )
        fig.update_traces(textposition='inside', textinfo='percent')
        st.plotly_chart(fig, use_container_width=True)

    with col_stats:
        # Key stats using native metrics
        st.markdown("**Performance (Historical Avg.)**")

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Ann. Return", f"{portfolio['annualized_return']}%")
        with m2:
            st.metric("1Y Return", f"{portfolio['year_return']}%",
                       delta=f"{portfolio['year_return']:+.1f}%")

        m3, m4 = st.columns(2)
        with m3:
            st.metric("Volatility", f"{portfolio['volatility']}%")
        with m4:
            st.metric("Sharpe Ratio", f"{portfolio['sharpe_ratio']:.2f}")

        # Risk level and strategy as simple text
        st.markdown(f"**Risk:** {portfolio['risk_level']} · **Strategy:** {portfolio['strategy']}")

    # Allocation table in expander
    with st.expander("View Full Allocation"):
        alloc_table = pd.DataFrame(
            list(portfolio['allocation'].items()),
            columns=['Asset Class', 'Weight (%)'],
        )
        st.dataframe(alloc_table, use_container_width=True, hide_index=True)

    st.markdown("---")


def render_popular_portfolios():
    st.title("Community Portfolios")
    st.write("Explore investment strategies built by our community. Compare their allocation against yours.")

    holdings = st.session_state.holdings

    if not holdings:
        st.warning("No holdings found to compare.")
        return

    # Filter by strategy type
    strategies = ["All"] + sorted(set(p['strategy'] for p in EXAMPLE_PORTFOLIOS))
    selected_strategy = st.selectbox("Filter by Strategy", strategies)

    filtered = EXAMPLE_PORTFOLIOS
    if selected_strategy != "All":
        filtered = [p for p in EXAMPLE_PORTFOLIOS if p['strategy'] == selected_strategy]

    # Render portfolio cards in a 2-column grid
    for i in range(0, len(filtered), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(filtered):
                with col:
                    _render_portfolio_card(filtered[i + j])

    # AI Comparison Section
    st.markdown("---")
    st.markdown("### Compare Your Portfolio")
    st.markdown("Select a community portfolio to see how your allocation stacks up.")

    portfolio_names = [p['name'] for p in EXAMPLE_PORTFOLIOS]
    selected_name = st.selectbox("Choose a portfolio to compare against", portfolio_names)
    selected_portfolio = next(p for p in EXAMPLE_PORTFOLIOS if p['name'] == selected_name)

    col_yours, col_theirs = st.columns(2)

    with col_yours:
        st.markdown("#### Your Portfolio")
        user_alloc = pd.DataFrame(
            [{"Asset": t, "Weight": 1} for t in holdings.keys()]
        )
        fig_user = px.pie(
            user_alloc,
            values='Weight',
            names='Asset',
            hole=0.45,
            color_discrete_sequence=['#3A6EA5', '#0B1F3A', '#1F8A70', '#F4A261', '#C44536', '#2A9D8F'],
        )
        fig_user.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=280,
        )
        fig_user.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_user, use_container_width=True)

    with col_theirs:
        st.markdown(f"#### {selected_portfolio['name']}")
        alloc_df = pd.DataFrame(
            list(selected_portfolio['allocation'].items()),
            columns=['Asset', 'Weight'],
        )
        fig_bench = px.pie(
            alloc_df,
            values='Weight',
            names='Asset',
            hole=0.45,
            color_discrete_sequence=selected_portfolio.get('colors', px.colors.qualitative.Set2),
        )
        fig_bench.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=280,
        )
        fig_bench.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_bench, use_container_width=True)

    if st.button("Run Claude Comparison"):
        user_portfolio_data = {
            'holdings': holdings,
            'weights': {k: 1 / len(holdings) for k in holdings.keys()},
        }
        user_risk_score = 65

        with st.spinner("Claude is analyzing the structural differences..."):
            analysis = compare_portfolio_with_standard(
                user_portfolio_data=user_portfolio_data,
                user_risk_score=user_risk_score,
                standard_portfolio_name=selected_portfolio['name'],
                standard_portfolio_desc=selected_portfolio['description'],
                eli10_mode=st.session_state.get('eli10_mode', False),
            )

            st.info(analysis)
