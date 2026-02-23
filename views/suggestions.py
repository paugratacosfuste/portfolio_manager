import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.data_fetcher import fetch_historical_data, fetch_current_prices
from utils.data_fetcher import fetch_asset_metadata
from utils.portfolio_metrics import calculate_portfolio_volatility, calculate_portfolio_beta, calculate_hhi_index, assess_risk_score

def render_suggestions():
    st.markdown("<h1>Risk Analysis & Suggestions</h1>", unsafe_allow_html=True)
    st.markdown("<p>Comparing what you currently have versus what you should have based on your stated risk profile.</p>", unsafe_allow_html=True)
    
    holdings = st.session_state.holdings
    profile = st.session_state.profile
    
    if not holdings:
        st.warning("No holdings found.")
        return
        
    tickers = list(holdings.keys())
    
    with st.spinner("Calculating Risk Metrics & Fetching Metadata..."):
        prices_df = fetch_historical_data(tickers, period='1y')
        current_prices = fetch_current_prices(tickers)
        
        weights = {}
        total_value = sum(qty * current_prices.get(t, 0) for t, qty in holdings.items())
        for t, qty in holdings.items():
            if total_value > 0:
                weights[t] = (qty * current_prices.get(t, 0)) / total_value
            else:
                weights[t] = 0
                
        valid_tickers_for_risk = [t for t in tickers if t in prices_df.columns]
        
        if len(valid_tickers_for_risk) > 0 and len(prices_df) > 50:
            volatility = calculate_portfolio_volatility(prices_df[valid_tickers_for_risk], weights)
            try:
                spy_df = fetch_historical_data(['SPY'], period='1y')
                mkt = spy_df['SPY'] if 'SPY' in spy_df else prices_df.iloc[:, 0]
                beta = calculate_portfolio_beta(prices_df[valid_tickers_for_risk], mkt, weights)
            except Exception:
                beta = 1.0
            hhi = calculate_hhi_index(weights)
            risk_score = assess_risk_score(volatility, hhi, beta, total_value)
        else:
            volatility, beta, hhi, risk_score = 0.0, 1.0, calculate_hhi_index(weights), 50
            st.info("Currently gathering enough historical market data for full probabilistic risk metrics. Relying on baseline mapping.")
            
        # Get metadata for grouping
        metadata = fetch_asset_metadata(tickers)
        
        asset_class_weights = {}
        sector_weights = {}
        region_weights = {}
        
        for t, w in weights.items():
            meta = metadata.get(t, {"asset_class": "Stocks", "sector": "Other", "region": "United States"})
            
            # Aggregate Asset Class
            ac = meta["asset_class"]
            asset_class_weights[ac] = asset_class_weights.get(ac, 0) + w
            
            # Aggregate Sector
            sec = meta["sector"]
            sector_weights[sec] = sector_weights.get(sec, 0) + w
            
            # Aggregate Region
            reg = meta["region"]
            region_weights[reg] = region_weights.get(reg, 0) + w

    # ── IDEAL TARGETS BASED ON PROFILE ──────────────────────────────────────────
    risk_level = profile.get("risk_tolerance", "Moderate")
    
    targets = {
        "Conservative": {
            "vol_max": 0.08, "beta_target": 0.5,
            "asset_class": {"Bonds/Cash": 50, "Stocks": 50, "Crypto": 0},
            "sector": {"Fixed Income": 70, "Healthcare": 10, "Consumer Defensive": 10, "Broad Market ETF": 10},
            "region": {"United States": 90, "International": 10, "Global/Decentralized": 0}
        },
        "Moderate": {
            "vol_max": 0.12, "beta_target": 0.8,
            "asset_class": {"Bonds/Cash": 30, "Stocks": 55, "Crypto": 15},
            "sector": {"Fixed Income": 40, "Technology": 20, "Financial Services": 10, "Broad Market ETF": 30},
            "region": {"United States": 75, "International": 25, "Global/Decentralized": 0}
        },
        "Growth": {
            "vol_max": 0.18, "beta_target": 1.1,
            "asset_class": {"Bonds/Cash": 5, "Stocks": 75, "Crypto": 20},
            "sector": {"Technology": 40, "Consumer Cyclical": 20, "Broad Market ETF": 20, "Digital Assets": 10, "Fixed Income": 10},
            "region": {"United States": 70, "International": 20, "Global/Decentralized": 10}
        },
        "Aggressive": {
            "vol_max": 0.25, "beta_target": 1.5,
            "asset_class": {"Stocks": 70, "Crypto": 30},
            "sector": {"Technology": 50, "Digital Assets": 30, "Consumer Cyclical": 20},
            "region": {"United States": 60, "International": 10, "Global/Decentralized": 30}
        }
    }
    
    target = targets.get(risk_level, targets["Moderate"])

    # ── METRICS COMPARISON BLOCKS ─────────────────────────────────────────────
    st.markdown(f"### Profile: {risk_level} Investor")
    
    col1, col2, col3 = st.columns(3)
    
    def _metric_card(title, current, ideal, status_color, suffix=""):
        return f"""
        <div style="background:#fff; padding:15px; border-radius:10px; border:1px solid #E0E7EF; text-align:center;">
            <div style="color:#666; font-size:0.85rem; font-weight:600; text-transform:uppercase;">{title}</div>
            <div style="font-size:1.8rem; font-weight:700; color:#0B1F3A; margin:5px 0;">{current}{suffix}</div>
            <div style="font-size:0.8rem; color:{status_color}; font-weight:600; background:{status_color}15; padding:3px 8px; border-radius:12px; display:inline-block;">
                Target: {ideal}{suffix}
            </div>
        </div>
        """
        
    vol_color = "#1F8A70" if volatility <= target["vol_max"] else "#C44536"
    beta_color = "#1F8A70" if abs(beta - target["beta_target"]) < 0.2 else "#E08C3A"
    
    with col1:
        st.markdown(_metric_card("Annualized Volatility", f"{volatility*100:.1f}", f"<{target['vol_max']*100:.1f}", vol_color, "%"), unsafe_allow_html=True)
    with col2:
        st.markdown(_metric_card("Portfolio Beta", f"{beta:.2f}", f"~{target['beta_target']:.2f}", beta_color), unsafe_allow_html=True)
    with col3:
        st.markdown(_metric_card("Concentration (HHI)", f"{hhi:.0f}", "<2000", "#1F8A70" if hhi < 2000 else "#C44536"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ALLOCATION COMPARISON ───────────────────────────────────────────────
    st.markdown("### Allocation Gap Analysis")
    st.markdown("<p style='color:#666; font-size:0.9rem;'>Visualizing the mismatch between your current allocation (blue bars) and your risk-adjusted targets (black markers). Actions are suggested if a category is off by more than 5%.</p>", unsafe_allow_html=True)
    
    def render_comparison_bars(current_dict, target_dict, title):
        curr_perc = {k: v * 100 for k, v in current_dict.items()}
        all_keys = list(set(list(curr_perc.keys()) + list(target_dict.keys())))
        all_keys.sort(key=lambda k: target_dict.get(k, 0), reverse=True)
        
        html = f"<div style='margin-bottom:30px;'><h4 style='color:#3A6EA5; margin-bottom:20px;'>{title}</h4>"
        
        for k in all_keys:
            c_val = curr_perc.get(k, 0)
            t_val = target_dict.get(k, 0)
            diff = c_val - t_val
            
            if diff > 5:
                status_color = "#C44536" # Red
                status_text = f"Reduce by {diff:.1f}%"
                base_color = "#C44536"
                base_width = t_val
                action_width = diff
                action_opacity = 0.4
            elif diff < -5:
                status_color = "#1F8A70" # Green
                status_text = f"Increase by {-diff:.1f}%"
                base_color = "#1F8A70"
                base_width = c_val
                action_width = -diff
                action_opacity = 0.4
            else:
                status_color = "#3A6EA5" # Blue
                status_text = "On Target"
                base_color = "#3A6EA5"
                base_width = c_val
                action_width = 0
                action_opacity = 0
                
            html += f"""
<div style="margin-bottom: 24px;">
  <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 8px;">
    <div>
      <div style="font-weight: 700; color: #0B1F3A; font-size: 1rem;">{k}</div>
      <div style="font-size: 0.85rem; color: #666; margin-top: 2px;">
        Current: <span style="font-weight:600; color:{status_color};">{c_val:.1f}%</span> 
        &nbsp;|&nbsp; 
        Target: <span style="font-weight:600; color:#0B1F3A;">{t_val:.1f}%</span>
      </div>
    </div>
    <div style="font-weight: 600; font-size: 0.8rem; color: {status_color}; background: {status_color}15; padding: 4px 12px; border-radius: 12px; border: 1px solid {status_color}33;">
      {status_text}
    </div>
  </div>
  <div style="position: relative; width: 100%; height: 28px; background: #F0F4F8; border-radius: 6px; overflow: hidden; box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);">
    <!-- Base Solid Bar -->
    <div style="position: absolute; top: 0; left: 0; height: 100%; width: {base_width}%; background: {base_color}; border-radius: 6px 0 0 6px;"></div>
    <!-- Gap Fill (Translucent) -->
    <div style="position: absolute; top: 0; left: {base_width}%; height: 100%; width: {action_width}%; background: {base_color}; opacity: {action_opacity}; border-radius: 0 6px 6px 0;"></div>
    <!-- Target Marker line -->
    <div style="position: absolute; top: 0; left: {t_val}%; height: 100%; width: 3px; background: #0B1F3A; z-index: 10;"></div>
  </div>
</div>
"""
            
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

    render_comparison_bars(asset_class_weights, target["asset_class"], "Asset Class Diversification")
    st.markdown("<div style='height:1px; background:#E8EDF2; margin:10px 0 30px 0;'></div>", unsafe_allow_html=True)
    
    render_comparison_bars(sector_weights, target["sector"], "Sector Diversification")
    st.markdown("<div style='height:1px; background:#E8EDF2; margin:10px 0 30px 0;'></div>", unsafe_allow_html=True)
    
    render_comparison_bars(region_weights, target["region"], "Regional Diversification")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    # ── AI ADVICE ──────────────────────────────────────────────────────────
    st.markdown("### AI Rebalancing Strategies")
    
    portfolio_data = {'holdings': holdings, 'weights': weights, 'total_value': total_value}
    metrics_data = {'volatility': volatility, 'beta': beta, 'hhi': hhi, 'risk_score': risk_score}
    from utils.ai_advisor import generate_portfolio_advice
    
    if st.button("Generate Actionable Swap In/Out Recommendations (Claude)"):
        with st.spinner("Claude is analyzing your structural gaps..."):
            advice = generate_portfolio_advice(
                portfolio_data=portfolio_data,
                risk_metrics=metrics_data,
                macro_prediction={'probability': 0.5, 'prediction': 0},
                user_profile=profile,
                eli10_mode=st.session_state.get('eli10_mode', False)
            )
            st.info(f"**Structural Gap Analysis:**\n\n{advice}")
