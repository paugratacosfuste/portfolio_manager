"""
AI Stress Test view -- Natural language scenario simulation with Plotly visualizations.
"""
import streamlit as st
import plotly.graph_objects as go
from utils.data_fetcher import fetch_current_prices, fetch_asset_metadata
from utils.stress_test import generate_stress_scenario, apply_stress_to_portfolio


PRESET_SCENARIOS = {
    "Tech Crash": "A major technology sector crash similar to the dot-com bust. Tech stocks lose 40-60%, AI hype collapses, semiconductor demand plummets. NASDAQ drops 35%. Flight to safety into bonds and defensive sectors.",
    "Global Recession": "A severe global recession triggered by a banking crisis. GDP contracts 3-4%, unemployment spikes to 8%, consumer spending collapses. All equity sectors decline, with cyclicals hit hardest. Bonds rally as rates are cut.",
    "Rate Shock": "The Federal Reserve unexpectedly raises rates by 200bps due to resurgent inflation. Bond prices crash, growth stocks are hammered by higher discount rates, real estate declines. Energy and commodities surge.",
    "Crypto Winter": "A major crypto exchange collapses (similar to FTX). All cryptocurrencies drop 70-90%. Contagion spreads to fintech stocks. Traditional assets are relatively unaffected. Regulatory crackdown follows.",
    "Inflation Spiral": "Inflation surges to 12% driven by an energy crisis and supply chain disruptions. Central banks struggle to respond. Real returns collapse. Commodities and energy surge. Growth stocks and bonds are hammered.",
}


def render_stress_test():
    st.markdown("<h1>AI Stress Testing</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p>Describe a market scenario in plain English and see how your portfolio would be affected. "
        "Claude analyzes each holding's exposure and generates realistic drawdown estimates.</p>",
        unsafe_allow_html=True,
    )

    holdings = st.session_state.holdings
    if not holdings:
        st.warning("No holdings found.")
        return

    # Pipeline explanation
    with st.expander("How does this work?"):
        st.markdown("""
**This is a 3-step AI + quantitative pipeline:**

1. **You describe a scenario** (or pick a preset) in natural language
2. **Claude analyzes** each holding's sector, asset class, and exposure to generate structured JSON with per-holding drawdown estimates and rationales
3. **Python applies** the AI-generated parameters quantitatively to your actual portfolio values to compute dollar losses and stressed values

This combines LLM reasoning (understanding *why* tech stocks would drop in a tech crash) with deterministic math (applying the exact percentages to your holdings).
        """)

    tickers = list(holdings.keys())

    # Preset scenario buttons
    st.markdown("### Quick Scenarios")
    cols = st.columns(5)
    selected_preset = None
    for i, (name, _) in enumerate(PRESET_SCENARIOS.items()):
        with cols[i]:
            if st.button(name, key=f"preset_{name}", use_container_width=True):
                selected_preset = name

    # Custom scenario input
    st.markdown("### Or Describe Your Own")
    custom_scenario = st.text_area(
        "Scenario description",
        placeholder="e.g., China invades Taiwan, disrupting global semiconductor supply chains...",
        height=80,
        label_visibility="collapsed",
    )

    # Determine which scenario to run
    scenario_text = None
    if selected_preset:
        scenario_text = PRESET_SCENARIOS[selected_preset]
    elif st.button("Run Custom Stress Test") and custom_scenario.strip():
        scenario_text = custom_scenario.strip()

    if scenario_text:
        with st.spinner("Fetching current prices and metadata..."):
            prices = fetch_current_prices(tickers)
            metadata = fetch_asset_metadata(tickers)

        total_value = sum(holdings[t] * prices.get(t, 0) for t in tickers)
        holdings_with_meta = []
        for t in tickers:
            price = prices.get(t, 0)
            value = holdings[t] * price
            weight = (value / total_value * 100) if total_value > 0 else 0
            meta = metadata.get(t, {})
            holdings_with_meta.append({
                "ticker": t,
                "asset_class": meta.get("asset_class", "Unknown"),
                "sector": meta.get("sector", "Unknown"),
                "weight_pct": weight,
            })

        with st.spinner("Claude is analyzing scenario impacts on each holding..."):
            scenario_data = generate_stress_scenario(scenario_text, holdings_with_meta)

        if "error" in scenario_data:
            st.error(f"Error: {scenario_data['error']}")
            return

        with st.spinner("Applying stress to portfolio..."):
            results = apply_stress_to_portfolio(holdings, prices, scenario_data)

        # ── Results Display ───────────────────────────────────────────────
        st.markdown("---")
        st.markdown(f"### Scenario: {results['scenario_name']}")
        st.markdown(f"*{results['scenario_summary']}*")

        # Summary cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"<div style='background:#fff;padding:15px;border-radius:10px;border:1px solid #E0E7EF;text-align:center;'>"
                f"<div style='color:#666;font-size:0.85rem;font-weight:600;'>CURRENT VALUE</div>"
                f"<div style='font-size:1.5rem;font-weight:700;color:#1F8A70;'>${results['current_total']:,.0f}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"<div style='background:#fff;padding:15px;border-radius:10px;border:1px solid #E0E7EF;text-align:center;'>"
                f"<div style='color:#666;font-size:0.85rem;font-weight:600;'>STRESSED VALUE</div>"
                f"<div style='font-size:1.5rem;font-weight:700;color:#C44536;'>${results['stressed_total']:,.0f}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"<div style='background:#fff;padding:15px;border-radius:10px;border:1px solid #E0E7EF;text-align:center;'>"
                f"<div style='color:#666;font-size:0.85rem;font-weight:600;'>TOTAL LOSS</div>"
                f"<div style='font-size:1.5rem;font-weight:700;color:#C44536;'>${results['total_loss']:,.0f}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f"<div style='background:#fff;padding:15px;border-radius:10px;border:1px solid #E0E7EF;text-align:center;'>"
                f"<div style='color:#666;font-size:0.85rem;font-weight:600;'>PORTFOLIO DRAWDOWN</div>"
                f"<div style='font-size:1.5rem;font-weight:700;color:#C44536;'>{results['total_loss_pct']:.1f}%</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Per-holding impact bar chart
        st.markdown("### Per-Holding Impact")
        impact_data = results["holdings_impact"]
        tickers_sorted = sorted(impact_data, key=lambda x: x["drawdown_pct"])

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            y=[d["ticker"] for d in tickers_sorted],
            x=[d["drawdown_pct"] for d in tickers_sorted],
            orientation="h",
            marker_color=[
                "#1F8A70" if d["drawdown_pct"] >= 0 else "#C44536"
                for d in tickers_sorted
            ],
            text=[f"{d['drawdown_pct']:.1f}%" for d in tickers_sorted],
            textposition="outside",
        ))
        fig_bar.update_layout(
            xaxis_title="Drawdown (%)",
            yaxis_title="",
            margin=dict(t=10, b=40, l=80, r=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=max(300, len(tickers_sorted) * 50),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Before / After grouped bar chart
        st.markdown("### Current vs Stressed Value")
        fig_grouped = go.Figure()
        ticker_names = [d["ticker"] for d in impact_data]
        fig_grouped.add_trace(go.Bar(
            name="Current Value",
            x=ticker_names,
            y=[d["current_value"] for d in impact_data],
            marker_color="#3A6EA5",
        ))
        fig_grouped.add_trace(go.Bar(
            name="Stressed Value",
            x=ticker_names,
            y=[d["stressed_value"] for d in impact_data],
            marker_color="#C44536",
        ))
        fig_grouped.update_layout(
            barmode="group",
            yaxis_title="Value ($)",
            margin=dict(t=10, b=40, l=60, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=400,
        )
        st.plotly_chart(fig_grouped, use_container_width=True)

        # Per-holding rationale cards
        st.markdown("### AI Analysis Per Holding")
        for d in impact_data:
            severity_color = "#1F8A70" if d["drawdown_pct"] >= -5 else (
                "#E08C3A" if d["drawdown_pct"] >= -20 else "#C44536"
            )
            st.markdown(
                f"<div style='background:#fff;padding:15px;border-radius:10px;border-left:4px solid {severity_color};"
                f"margin-bottom:10px;'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                f"<strong>{d['ticker']}</strong>"
                f"<span style='color:{severity_color};font-weight:700;'>{d['drawdown_pct']:.1f}% "
                f"(${d['loss']:,.0f})</span>"
                f"</div>"
                f"<div style='color:#666;font-size:0.9rem;margin-top:5px;'>{d['rationale']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
