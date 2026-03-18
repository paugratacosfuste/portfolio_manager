import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
from utils.data_fetcher import fetch_historical_data, fetch_current_prices
from utils.portfolio_metrics import get_dynamic_risk_free_rate


def render_efficient_frontier():
    st.markdown("<h1>Efficient Frontier & Portfolio Optimization</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p>Visualize the risk-return tradeoff across thousands of possible portfolio combinations using Modern Portfolio Theory (MPT), "
        "then optimize your allocation for a desired return target.</p>",
        unsafe_allow_html=True,
    )

    holdings = st.session_state.holdings
    if not holdings:
        st.warning("No holdings found.")
        return

    tickers = list(holdings.keys())

    if len(tickers) < 2:
        st.info("At least 2 assets are needed for portfolio optimization. Add more holdings in the sidebar.")
        return

    with st.spinner("Fetching historical data for optimization..."):
        prices_df = fetch_historical_data(tickers, period="1y")
        current_prices = fetch_current_prices(tickers)

    if prices_df.empty or len(prices_df) < 50:
        st.warning("Insufficient historical data for optimization.")
        return

    # Filter valid tickers
    valid_tickers = [t for t in tickers if t in prices_df.columns]
    if len(valid_tickers) < 2:
        st.warning("Need at least 2 valid tickers with historical data.")
        return

    prices_df = prices_df[valid_tickers].dropna()
    returns = prices_df.pct_change().dropna()
    n_assets = len(valid_tickers)

    # Annualized metrics
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    risk_free_rate = get_dynamic_risk_free_rate()

    # Current portfolio weights
    total_value = sum(qty * current_prices.get(t, 0) for t, qty in holdings.items())
    current_weights = np.array([
        (holdings.get(t, 0) * current_prices.get(t, 0)) / total_value if total_value > 0 else 1.0 / n_assets
        for t in valid_tickers
    ])

    # ── Monte Carlo Simulation ───────────────────────────────────────────
    st.markdown("### Efficient Frontier")
    st.markdown(
        "<p style='color:#666; font-size:0.9rem;'>Each dot represents a randomly generated portfolio with different weight combinations of your assets. "
        "Brighter colors indicate higher Sharpe ratios (better risk-adjusted returns). The upper-left edge of the cloud is the <b>efficient frontier</b> — "
        "no portfolio can achieve higher return for the same level of risk.</p>",
        unsafe_allow_html=True,
    )

    n_portfolios = 5000
    results = np.zeros((n_portfolios, 3))  # return, volatility, sharpe

    np.random.seed(42)
    for i in range(n_portfolios):
        w = np.random.random(n_assets)
        w /= w.sum()

        port_return = np.dot(w, mean_returns)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        port_sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0

        results[i] = [port_return, port_vol, port_sharpe]

    # Current portfolio metrics
    curr_return = np.dot(current_weights, mean_returns)
    curr_vol = np.sqrt(np.dot(current_weights.T, np.dot(cov_matrix, current_weights)))
    curr_sharpe = (curr_return - risk_free_rate) / curr_vol if curr_vol > 0 else 0

    # ── Optimize: Max Sharpe & Min Variance ──────────────────────────────
    def neg_sharpe(w):
        port_return = np.dot(w, mean_returns)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        return -(port_return - risk_free_rate) / port_vol if port_vol > 0 else 0

    def portfolio_vol(w):
        return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))
    init_w = np.array([1.0 / n_assets] * n_assets)

    # Max Sharpe
    opt_sharpe = minimize(neg_sharpe, init_w, method="SLSQP", bounds=bounds, constraints=constraints)
    max_sharpe_w = opt_sharpe.x
    max_sharpe_ret = np.dot(max_sharpe_w, mean_returns)
    max_sharpe_vol = portfolio_vol(max_sharpe_w)
    max_sharpe_sr = (max_sharpe_ret - risk_free_rate) / max_sharpe_vol

    # Min Variance
    opt_minvar = minimize(portfolio_vol, init_w, method="SLSQP", bounds=bounds, constraints=constraints)
    min_var_w = opt_minvar.x
    min_var_ret = np.dot(min_var_w, mean_returns)
    min_var_vol = portfolio_vol(min_var_w)
    min_var_sr = (min_var_ret - risk_free_rate) / min_var_vol

    # Store optimization results for cross-view consumption (suggestions, LLM prompts)
    st.session_state['optimal_weights'] = {
        'max_sharpe': {
            'weights': {t: round(float(w), 4) for t, w in zip(valid_tickers, max_sharpe_w)},
            'sharpe': round(float(max_sharpe_sr), 2),
            'return': round(float(max_sharpe_ret), 4),
            'vol': round(float(max_sharpe_vol), 4),
        },
        'min_variance': {
            'weights': {t: round(float(w), 4) for t, w in zip(valid_tickers, min_var_w)},
            'sharpe': round(float(min_var_sr), 2),
            'return': round(float(min_var_ret), 4),
            'vol': round(float(min_var_vol), 4),
        },
        'current_sharpe': round(float(curr_sharpe), 2),
        'current_return': round(float(curr_return), 4),
        'current_vol': round(float(curr_vol), 4),
    }

    # ── Scatter Plot ─────────────────────────────────────────────────────
    fig = go.Figure()

    # Monte Carlo portfolios
    fig.add_trace(go.Scatter(
        x=results[:, 1] * 100, y=results[:, 0] * 100,
        mode="markers",
        marker=dict(
            size=3, color=results[:, 2],
            colorscale="Viridis", showscale=True,
            colorbar=dict(title="Sharpe"),
        ),
        name="Random Portfolios",
        hovertemplate="Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
    ))

    # Current portfolio
    fig.add_trace(go.Scatter(
        x=[curr_vol * 100], y=[curr_return * 100],
        mode="markers",
        marker=dict(size=16, color="#C44536", symbol="diamond", line=dict(width=2, color="white")),
        name=f"Your Portfolio (Sharpe={curr_sharpe:.2f})",
        hovertemplate=f"Your Portfolio<br>Vol: {curr_vol*100:.1f}%<br>Return: {curr_return*100:.1f}%<br>Sharpe: {curr_sharpe:.2f}<extra></extra>",
    ))

    # Max Sharpe
    fig.add_trace(go.Scatter(
        x=[max_sharpe_vol * 100], y=[max_sharpe_ret * 100],
        mode="markers",
        marker=dict(size=16, color="#1F8A70", symbol="star", line=dict(width=2, color="white")),
        name=f"Max Sharpe (Sharpe={max_sharpe_sr:.2f})",
        hovertemplate=f"Max Sharpe<br>Vol: {max_sharpe_vol*100:.1f}%<br>Return: {max_sharpe_ret*100:.1f}%<br>Sharpe: {max_sharpe_sr:.2f}<extra></extra>",
    ))

    # Min Variance
    fig.add_trace(go.Scatter(
        x=[min_var_vol * 100], y=[min_var_ret * 100],
        mode="markers",
        marker=dict(size=16, color="#3A6EA5", symbol="hexagon", line=dict(width=2, color="white")),
        name=f"Min Variance (Sharpe={min_var_sr:.2f})",
        hovertemplate=f"Min Variance<br>Vol: {min_var_vol*100:.1f}%<br>Return: {min_var_ret*100:.1f}%<br>Sharpe: {min_var_sr:.2f}<extra></extra>",
    ))

    fig.update_layout(
        xaxis_title="Annualized Volatility (%)",
        yaxis_title="Annualized Return (%)",
        margin=dict(t=20, b=50, l=60, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=550,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Optimal Portfolios Comparison ────────────────────────────────────
    st.markdown("### Key Portfolios Comparison")
    st.markdown(
        "<p style='color:#666; font-size:0.9rem;'>How your current allocation compares against the two mathematically optimal portfolios: "
        "one that maximizes risk-adjusted return (Max Sharpe) and one that minimizes volatility (Min Variance).</p>",
        unsafe_allow_html=True,
    )

    comp_data = {
        "Metric": ["Annualized Return", "Annualized Volatility", "Sharpe Ratio"],
        "Your Portfolio": [f"{curr_return*100:.1f}%", f"{curr_vol*100:.1f}%", f"{curr_sharpe:.2f}"],
        "Max Sharpe": [f"{max_sharpe_ret*100:.1f}%", f"{max_sharpe_vol*100:.1f}%", f"{max_sharpe_sr:.2f}"],
        "Min Variance": [f"{min_var_ret*100:.1f}%", f"{min_var_vol*100:.1f}%", f"{min_var_sr:.2f}"],
    }
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    # ── Custom Target Return Optimization ────────────────────────────────
    st.markdown("---")
    st.markdown("### Portfolio Optimizer")
    st.markdown(
        "<p style='color:#666; font-size:0.9rem;'>Use the slider to set your desired annual return. "
        "The optimizer finds the portfolio with the <b>lowest possible risk</b> that achieves that target, "
        "showing you exactly how to rebalance your holdings.</p>",
        unsafe_allow_html=True,
    )

    min_feasible = float(min(mean_returns) * 100)
    max_feasible = float(max(mean_returns) * 100)

    target_return = st.slider(
        "Target Annual Return (%)",
        min_value=round(min_feasible, 1),
        max_value=round(max_feasible, 1),
        value=round((min_feasible + max_feasible) / 2, 1),
        step=0.5,
    )

    target_ret_decimal = target_return / 100.0

    # Optimize for min variance at target return
    constraints_target = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: np.dot(w, mean_returns) - target_ret_decimal},
    ]

    opt_target = minimize(portfolio_vol, init_w, method="SLSQP", bounds=bounds, constraints=constraints_target)

    if opt_target.success:
        opt_w = opt_target.x
        opt_ret = np.dot(opt_w, mean_returns)
        opt_vol = portfolio_vol(opt_w)
        opt_sr = (opt_ret - risk_free_rate) / opt_vol if opt_vol > 0 else 0

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Optimized Weights")
            opt_weight_df = pd.DataFrame({
                "Ticker": valid_tickers,
                "Current Weight": [f"{w*100:.1f}%" for w in current_weights],
                "Optimized Weight": [f"{w*100:.1f}%" for w in opt_w],
            })
            st.dataframe(opt_weight_df, use_container_width=True, hide_index=True)

        with col_b:
            st.markdown("#### Before vs After")
            before_after = pd.DataFrame({
                "Metric": ["Annual Return", "Annual Volatility", "Sharpe Ratio"],
                "Current": [f"{curr_return*100:.1f}%", f"{curr_vol*100:.1f}%", f"{curr_sharpe:.2f}"],
                "Optimized": [f"{opt_ret*100:.1f}%", f"{opt_vol*100:.1f}%", f"{opt_sr:.2f}"],
            })
            st.dataframe(before_after, use_container_width=True, hide_index=True)
    else:
        st.warning(f"Could not find a feasible portfolio for target return {target_return:.1f}%. Try adjusting the target.")

    # ── Methodology Note ─────────────────────────────────────────────────
    with st.expander("Methodology"):
        st.markdown("""
**Monte Carlo Simulation:** 5,000 random portfolios are generated by sampling random weight vectors and computing
each portfolio's annualized return and volatility using the historical covariance matrix.

**Optimization:** `scipy.optimize.minimize` with SLSQP method finds exact optimal portfolios:
- **Max Sharpe Portfolio:** Maximizes risk-adjusted return (Sharpe ratio)
- **Min Variance Portfolio:** Minimizes total portfolio volatility
- **Target Return Portfolio:** Finds minimum variance portfolio achieving a specified return

**Assumptions:**
- Returns are normally distributed (standard MPT assumption)
- Historical covariance is a reasonable estimate of future covariance
- No transaction costs or taxes
- All assets are infinitely divisible
- 1-year lookback period for estimation
""")
