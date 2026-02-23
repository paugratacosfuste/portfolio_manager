import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.ai_advisor import compare_portfolio_with_standard

# ── Community portfolio data ──────────────────────────────────────────────────
COMMUNITY_PORTFOLIOS = [
    {
        "name": "Tech Maven",
        "user": "Alex Chen",
        "initials": "AC",
        "avatar_color": "#3A6EA5",
        "risk": "Aggressive",
        "risk_color": "#C44536",
        "allocations": [
            ("AAPL", 25), ("MSFT", 25), ("GOOGL", 20), ("NVDA", 20), ("AMZN", 10)
        ],
        "return_1y": "+32.4%",
        "return_3y": "+18.7%",
        "return_pos": True,
    },
    {
        "name": "Value Hunter",
        "user": "Sarah Rodriguez",
        "initials": "SR",
        "avatar_color": "#1F8A70",
        "risk": "Conservative",
        "risk_color": "#1F8A70",
        "allocations": [
            ("BRK-B", 30), ("JNJ", 20), ("WMT", 20), ("VZ", 15), ("KO", 15)
        ],
        "return_1y": "+8.2%",
        "return_3y": "+11.4%",
        "return_pos": True,
    },
    {
        "name": "All-Weather",
        "user": "Marcus Johnson",
        "initials": "MJ",
        "avatar_color": "#7B68EE",
        "risk": "Moderate",
        "risk_color": "#E08C3A",
        "allocations": [
            ("SPY", 30), ("TLT", 40), ("GLD", 15), ("DBC", 7.5), ("IEF", 7.5)
        ],
        "return_1y": "+5.3%",
        "return_3y": "+9.8%",
        "return_pos": True,
    },
    {
        "name": "Crypto Pioneer",
        "user": "Emma Williams",
        "initials": "EW",
        "avatar_color": "#F7931A",
        "risk": "Aggressive",
        "risk_color": "#C44536",
        "allocations": [
            ("BTC-USD", 40), ("ETH-USD", 30), ("SOL-USD", 15), ("COIN", 15)
        ],
        "return_1y": "+87.3%",
        "return_3y": "+42.1%",
        "return_pos": True,
    },
    {
        "name": "ESG Leader",
        "user": "James Kim",
        "initials": "JK",
        "avatar_color": "#2ECC71",
        "risk": "Growth",
        "risk_color": "#3A6EA5",
        "allocations": [
            ("ICLN", 25), ("TSLA", 20), ("MSFT", 20), ("NEE", 15), ("BND", 20)
        ],
        "return_1y": "+14.7%",
        "return_3y": "+12.3%",
        "return_pos": True,
    },
    {
        "name": "Global Diversifier",
        "user": "Priya Patel",
        "initials": "PP",
        "avatar_color": "#9B59B6",
        "risk": "Moderate",
        "risk_color": "#E08C3A",
        "allocations": [
            ("VTI", 35), ("VXUS", 30), ("BND", 20), ("GLD", 10), ("VNQ", 5)
        ],
        "return_1y": "+11.9%",
        "return_3y": "+10.6%",
        "return_pos": True,
    },
]

PIE_COLORS = ["#0B1F3A", "#3A6EA5", "#7BAFD4", "#A8C4DC", "#D4E5F5",
              "#1F8A70", "#4CAF50", "#FFC107", "#FF5722", "#9C27B0"]


def _build_pie(allocations):
    labels = [a[0] for a in allocations]
    values = [a[1] for a in allocations]
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=PIE_COLORS[:len(labels)]),
        textinfo="percent",
        textfont_size=11,
        showlegend=False,
        hoverinfo="label+percent"
    ))
    fig.update_layout(
        margin=dict(t=0, b=0, l=0, r=0),
        height=160,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _render_card(portfolio):
    allocs = []
    for i, a in enumerate(portfolio["allocations"]):
        color = PIE_COLORS[i % len(PIE_COLORS)]
        allocs.append(f"<div style='border-left: 4px solid {color}; background:#F8FAFC; padding:4px 8px; border-radius:4px; font-size:0.8rem; font-weight:600; color:#0B1F3A; display:flex; align-items:center; gap:6px;'>{a[0]} <span style='font-weight:400; color:#666;'>{a[1]}%</span></div>")
    
    allocations_html = f"<div style='display:flex; flex-wrap:wrap; gap:8px; margin-top:20px; margin-bottom:15px; height:70px; align-content: flex-start;'>{''.join(allocs)}</div>"

    ret_color = "#1F8A70" if portfolio["return_pos"] else "#C44536"

    header_html = f"""
<div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
  <div style="width:36px; height:36px; border-radius:50%; background:{portfolio['avatar_color']}; display:flex; align-items:center; justify-content:center; color:#fff; font-weight:700; font-size:0.85rem; flex-shrink:0;">
    {portfolio['initials']}
  </div>
  <div>
    <div style="font-weight:700; font-size:1rem; color:#0B1F3A; line-height:1.2;">{portfolio['name']}</div>
    <div style="font-size:0.78rem; color:#666; line-height:1.2;">by {portfolio['user']}</div>
  </div>
  <span style="margin-left:auto; background:{portfolio['risk_color']}22; color:{portfolio['risk_color']}; font-size:0.72rem; font-weight:600; padding:2px 8px; border-radius:12px; border:1px solid {portfolio['risk_color']}44;">
    {portfolio['risk']}
  </span>
</div>
""".replace('\n', '')

    perf_html = f"""
<div style="display:flex; gap:16px; margin-top:10px; padding-top:15px; border-top:1px solid #E8EDF2;">
  <div>
    <div style="font-size:0.72rem; color:#888;">1Y Return</div>
    <div style="font-weight:700; color:{ret_color}; font-size:0.95rem;">{portfolio['return_1y']}</div>
  </div>
  <div>
    <div style="font-size:0.72rem; color:#888;">3Y Ann. Return</div>
    <div style="font-weight:700; color:{ret_color}; font-size:0.95rem;">{portfolio['return_3y']}</div>
  </div>
</div>
""".replace('\n', '')

    # Enforce strict identical boxed container size across all portfolios
    with st.container(border=True, height=450):
        st.markdown(header_html, unsafe_allow_html=True)
        st.plotly_chart(_build_pie(portfolio["allocations"]), use_container_width=True, key=f"pie_{portfolio['name']}")
        st.markdown(allocations_html + perf_html, unsafe_allow_html=True)


def render_popular_portfolios():
    st.markdown("<h1>Standard Portfolios Gallery</h1>", unsafe_allow_html=True)
    st.markdown("<p>Explore community portfolios or compare your allocation against benchmark strategies.</p>", unsafe_allow_html=True)

    holdings = st.session_state.holdings

    tab_community, tab_benchmark = st.tabs(["Community Portfolios", "Benchmark Comparison"])

    # ── Tab 1: Community Gallery ──────────────────────────────────────────────
    with tab_community:
        st.markdown("### What others are investing in")
        st.markdown("<p style='color:#666; font-size:0.9rem;'>Mock community portfolios for inspiration. Past returns are illustrative only.</p>", unsafe_allow_html=True)

        col_pairs = [COMMUNITY_PORTFOLIOS[i:i+2] for i in range(0, len(COMMUNITY_PORTFOLIOS), 2)]
        for pair in col_pairs:
            cols = st.columns(2)
            for col, portfolio in zip(cols, pair):
                with col:
                    _render_card(portfolio)

    # ── Tab 2: Benchmark Comparison ───────────────────────────────────────────
    with tab_benchmark:
        if not holdings:
            st.warning("No holdings found to compare.")
            return

        standard_portfolios = {
            "Ray Dalio's All-Weather Portfolio": {
                "desc": "Designed to perform well regardless of the economic environment (inflation, deflation, growth, depression).",
                "weights": {
                    "Stocks (VTI)": "30%",
                    "Long-term Treasuries (TLT)": "40%",
                    "Intermediate Treasuries (IEF)": "15%",
                    "Gold (GLD)": "7.5%",
                    "Commodities (DBC)": "7.5%",
                },
            },
            "The Classic 60/40 Portfolio": {
                "desc": "The traditional balanced approach for moderate risk tolerance.",
                "weights": {
                    "Broad US Stocks (SPY)": "60%",
                    "US Aggregate Bonds (BND)": "40%",
                },
            },
            "The Swensen Model (Yale Endowment)": {
                "desc": "Heavily diversified into alternative assets to reduce reliance on domestic equity.",
                "weights": {
                    "Domestic Equity": "30%",
                    "Foreign Equity": "15%",
                    "Emerging Markets": "5%",
                    "Real Estate (REITs)": "20%",
                    "US Treasury Bonds": "15%",
                    "TIPS": "15%",
                },
            },
        }

        st.markdown("### Select a Benchmark")
        selected_name = st.selectbox("Benchmark Portfolio", list(standard_portfolios.keys()))
        selected_portfolio = standard_portfolios[selected_name]

        col1, col2 = st.columns([1, 1.5])

        with col1:
            st.markdown(f"<div style='font-weight:700; font-size:1.1rem; color:#0B1F3A; margin-bottom:5px;'>{selected_name}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:0.85rem; color:#666; margin-bottom:15px; min-height:40px;'>{selected_portfolio['desc']}</div>", unsafe_allow_html=True)
            
            # Format allocations for pills and pie
            alloc_data = []
            allocs_html = []
            
            for i, (asset, weight_str) in enumerate(selected_portfolio["weights"].items()):
                w_val = float(weight_str.replace('%', ''))
                alloc_data.append((asset, w_val))
                color = PIE_COLORS[i % len(PIE_COLORS)]
                allocs_html.append(f"<div style='border-left: 4px solid {color}; background:#F8FAFC; padding:4px 8px; border-radius:4px; font-size:0.8rem; font-weight:600; color:#0B1F3A; display:flex; align-items:center; gap:6px;'>{asset} <span style='font-weight:400; color:#666;'>{w_val}%</span></div>")
                
            pills = f"<div style='display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; margin-bottom:10px;'>{''.join(allocs_html)}</div>"
            
            with st.container(border=True):
                st.plotly_chart(_build_pie(alloc_data), use_container_width=True, key=f"pie_bench_{selected_name}")
                st.markdown(pills, unsafe_allow_html=True)

        with col2:
            st.markdown("#### AI Compare & Contrast")
            st.info("Curious how your custom portfolio stacks up against the " + selected_name + " structurally?")

            if st.button("Run Claude Comparison"):
                user_portfolio_data = {
                    "holdings": holdings,
                    "weights": {k: 1 / len(holdings) for k in holdings.keys()},
                }
                user_risk_score = 65

                with st.spinner("Claude is analyzing the structural differences..."):
                    analysis = compare_portfolio_with_standard(
                        user_portfolio_data=user_portfolio_data,
                        user_risk_score=user_risk_score,
                        standard_portfolio_name=selected_name,
                        standard_portfolio_desc=selected_portfolio["desc"],
                        eli10_mode=st.session_state.get("eli10_mode", False),
                    )

                    st.info(analysis)
