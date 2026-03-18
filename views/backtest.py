import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.data_fetcher import fetch_historical_data, fetch_current_prices
from utils.portfolio_metrics import get_dynamic_risk_free_rate


def render_backtest():
    st.markdown("<h1>Portfolio Backtesting</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p>Historical performance analysis of your current portfolio allocation compared against standard benchmarks.</p>",
        unsafe_allow_html=True,
    )

    holdings = st.session_state.holdings
    if not holdings:
        st.warning("No holdings found.")
        return

    tickers = list(holdings.keys())

    # ── Configuration ────────────────────────────────────────────────────
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        lookback = st.selectbox("Backtest Period", ["1y", "2y", "3y", "5y"], index=2)
    with col_cfg2:
        benchmark_option = st.selectbox("Benchmark", ["SPY (S&P 500)", "60/40 (SPY + BND)", "Both"])

    # ── Fetch Data ───────────────────────────────────────────────────────
    benchmark_tickers = ["SPY"]
    if "60/40" in benchmark_option or benchmark_option == "Both":
        benchmark_tickers.append("BND")

    all_tickers = list(set(tickers + benchmark_tickers))

    with st.spinner("Fetching historical data..."):
        prices_df = fetch_historical_data(all_tickers, period=lookback)
        current_prices = fetch_current_prices(tickers)

    if prices_df.empty or len(prices_df) < 20:
        st.warning("Insufficient historical data for backtesting. Try shorter period or check tickers.")
        return

    # Filter valid portfolio tickers
    valid_tickers = [t for t in tickers if t in prices_df.columns]
    if not valid_tickers:
        st.warning("No valid tickers with historical data found.")
        return

    prices_df = prices_df.dropna()

    # ── Compute Portfolio Value Series ───────────────────────────────────
    total_value = sum(qty * current_prices.get(t, 0) for t, qty in holdings.items())
    weights = {}
    for t in valid_tickers:
        if total_value > 0:
            weights[t] = (holdings.get(t, 0) * current_prices.get(t, 0)) / total_value
        else:
            weights[t] = 1.0 / len(valid_tickers)

    # Normalize weights to valid tickers only
    w_sum = sum(weights.values())
    if w_sum > 0:
        weights = {t: v / w_sum for t, v in weights.items()}

    portfolio_returns = prices_df[valid_tickers].pct_change().dropna()
    w_array = np.array([weights[t] for t in valid_tickers])
    port_daily_returns = portfolio_returns.values @ w_array
    port_cumulative = (1 + pd.Series(port_daily_returns, index=portfolio_returns.index)).cumprod()

    # Benchmarks
    benchmarks = {}
    if "SPY" in prices_df.columns:
        spy_returns = prices_df["SPY"].pct_change().dropna()
        spy_cumulative = (1 + spy_returns).cumprod()
        # Align to portfolio dates
        common_idx = port_cumulative.index.intersection(spy_cumulative.index)
        benchmarks["SPY"] = spy_cumulative.loc[common_idx]
        port_cumulative_aligned = port_cumulative.loc[common_idx]
    else:
        common_idx = port_cumulative.index
        port_cumulative_aligned = port_cumulative

    if "BND" in prices_df.columns and "SPY" in prices_df.columns:
        spy_ret = prices_df["SPY"].pct_change().dropna()
        bnd_ret = prices_df["BND"].pct_change().dropna()
        common_60_40 = spy_ret.index.intersection(bnd_ret.index)
        blend_ret = 0.6 * spy_ret.loc[common_60_40] + 0.4 * bnd_ret.loc[common_60_40]
        blend_cum = (1 + blend_ret).cumprod()
        benchmarks["60/40"] = blend_cum.loc[blend_cum.index.intersection(common_idx)]

    # ── Cumulative Returns Chart ─────────────────────────────────────────
    st.markdown("### Cumulative Returns")

    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=port_cumulative_aligned.index, y=(port_cumulative_aligned - 1) * 100,
        mode="lines", name="Your Portfolio",
        line=dict(color="#3A6EA5", width=2.5),
    ))

    benchmark_colors = {"SPY": "#999", "60/40": "#E08C3A"}
    for bm_name, bm_series in benchmarks.items():
        if ("SPY" in benchmark_option or benchmark_option == "Both") and bm_name == "SPY":
            fig_cum.add_trace(go.Scatter(
                x=bm_series.index, y=(bm_series - 1) * 100,
                mode="lines", name=bm_name,
                line=dict(color=benchmark_colors.get(bm_name, "#666"), width=2, dash="dash"),
            ))
        if ("60/40" in benchmark_option or benchmark_option == "Both") and bm_name == "60/40":
            fig_cum.add_trace(go.Scatter(
                x=bm_series.index, y=(bm_series - 1) * 100,
                mode="lines", name=bm_name,
                line=dict(color=benchmark_colors.get(bm_name, "#666"), width=2, dash="dot"),
            ))

    fig_cum.update_layout(
        yaxis_title="Cumulative Return (%)",
        margin=dict(t=20, b=40, l=60, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=400, legend=dict(x=0.01, y=0.99),
        hovermode="x unified",
    )
    fig_cum.add_hline(y=0, line_dash="solid", line_color="#ccc")
    st.plotly_chart(fig_cum, use_container_width=True)

    # ── Drawdown Chart ───────────────────────────────────────────────────
    st.markdown("### Drawdown Timeline")

    def compute_drawdown(cumulative_series):
        running_max = cumulative_series.cummax()
        drawdown = (cumulative_series - running_max) / running_max
        return drawdown

    port_dd = compute_drawdown(port_cumulative_aligned)

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=port_dd.index, y=port_dd * 100,
        mode="lines", name="Your Portfolio", fill="tozeroy",
        line=dict(color="#C44536", width=1.5),
        fillcolor="rgba(196, 69, 54, 0.2)",
    ))

    for bm_name, bm_series in benchmarks.items():
        show_bm = False
        if ("SPY" in benchmark_option or benchmark_option == "Both") and bm_name == "SPY":
            show_bm = True
        if ("60/40" in benchmark_option or benchmark_option == "Both") and bm_name == "60/40":
            show_bm = True
        if show_bm:
            bm_dd = compute_drawdown(bm_series)
            fig_dd.add_trace(go.Scatter(
                x=bm_dd.index, y=bm_dd * 100,
                mode="lines", name=bm_name,
                line=dict(color=benchmark_colors.get(bm_name, "#666"), width=1.5, dash="dash"),
            ))

    fig_dd.update_layout(
        yaxis_title="Drawdown (%)",
        margin=dict(t=20, b=40, l=60, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=350, legend=dict(x=0.01, y=-0.15, orientation="h"),
        hovermode="x unified",
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # ── Rolling Sharpe Ratio ─────────────────────────────────────────────
    st.markdown("### Rolling 6-Month Sharpe Ratio")

    window = 126  # ~6 months of trading days
    _rfr = get_dynamic_risk_free_rate()
    risk_free_daily = _rfr / 252

    port_daily_series = pd.Series(port_daily_returns, index=portfolio_returns.index)

    if len(port_daily_series) > window:
        rolling_mean = port_daily_series.rolling(window).mean()
        rolling_std = port_daily_series.rolling(window).std()
        rolling_sharpe = ((rolling_mean - risk_free_daily) * 252) / (rolling_std * np.sqrt(252))
        rolling_sharpe = rolling_sharpe.dropna()

        fig_rs = go.Figure()
        fig_rs.add_trace(go.Scatter(
            x=rolling_sharpe.index, y=rolling_sharpe.values,
            mode="lines", name="Your Portfolio",
            line=dict(color="#3A6EA5", width=2),
        ))
        fig_rs.add_hline(y=0, line_dash="solid", line_color="#ccc")
        fig_rs.add_hline(y=1, line_dash="dash", line_color="#1F8A70", annotation_text="Good (1.0)")
        fig_rs.add_hline(y=-1, line_dash="dash", line_color="#C44536")

        fig_rs.update_layout(
            yaxis_title="Sharpe Ratio",
            margin=dict(t=20, b=40, l=60, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=350,
            hovermode="x unified",
        )
        st.plotly_chart(fig_rs, use_container_width=True)

    # ── Summary Statistics Table ─────────────────────────────────────────
    st.markdown("### Performance Summary")

    _rfr_summary = get_dynamic_risk_free_rate()

    def compute_metrics(daily_returns_series, label):
        total_return = (1 + daily_returns_series).prod() - 1
        n_days = len(daily_returns_series)
        ann_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1
        ann_vol = daily_returns_series.std() * np.sqrt(252)
        sharpe = (ann_return - _rfr_summary) / ann_vol if ann_vol > 0 else 0

        cumulative = (1 + daily_returns_series).cumprod()
        running_max = cumulative.cummax()
        max_dd = ((cumulative - running_max) / running_max).min()

        # Sortino ratio (downside deviation)
        downside = daily_returns_series[daily_returns_series < 0]
        downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else ann_vol
        sortino = (ann_return - _rfr_summary) / downside_std if downside_std > 0 else 0

        return {
            "Portfolio": label,
            "Total Return": f"{total_return*100:.1f}%",
            "Annualized Return": f"{ann_return*100:.1f}%",
            "Annualized Volatility": f"{ann_vol*100:.1f}%",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Sortino Ratio": f"{sortino:.2f}",
            "Max Drawdown": f"{max_dd*100:.1f}%",
        }

    summary_rows = [compute_metrics(port_daily_series.loc[common_idx], "Your Portfolio")]

    if "SPY" in benchmarks:
        spy_daily = prices_df["SPY"].pct_change().dropna().loc[common_idx]
        summary_rows.append(compute_metrics(spy_daily, "SPY"))

    if "60/40" in benchmarks:
        blend_daily = (0.6 * prices_df["SPY"].pct_change() + 0.4 * prices_df["BND"].pct_change()).dropna().loc[
            benchmarks["60/40"].index
        ]
        summary_rows.append(compute_metrics(blend_daily, "60/40"))

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Store backtest results for cross-view consumption (LLM prompts, suggestions)
    st.session_state['backtest_metrics'] = {
        'portfolio': summary_rows[0] if summary_rows else {},
        'spy': summary_rows[1] if len(summary_rows) > 1 else {},
        'lookback': lookback,
    }

    # ── Methodology ──────────────────────────────────────────────────────
    with st.expander("Methodology"):
        st.markdown("""
**Backtesting Approach:** Static rebalancing — the backtest assumes your current portfolio weights are held
constant throughout the lookback period. Daily returns are computed and compounded to show cumulative performance.

**Benchmarks:**
- **SPY:** SPDR S&P 500 ETF — a standard US large-cap equity benchmark
- **60/40:** 60% SPY + 40% BND (Vanguard Total Bond Market ETF) — a classic balanced allocation

**Metrics:**
- **Sharpe Ratio:** (Return - Risk-Free Rate) / Volatility. Risk-free rate = 5%
- **Sortino Ratio:** Like Sharpe but uses only downside deviation. Penalizes bad volatility only
- **Max Drawdown:** Largest peak-to-trough decline during the period

**Limitations:**
- Backtesting uses historical data and does not guarantee future performance
- Static weights (no rebalancing) may diverge from actual portfolio behavior
- Survivorship bias: tickers that exist today may not have existed throughout the full period
""")
