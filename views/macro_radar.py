import streamlit as st
import pandas as pd
import joblib
import os
import yfinance as yf

def render_macro_radar():
    st.markdown("<h1>Macro Radar: Market Forecasting</h1>", unsafe_allow_html=True)
    st.markdown("<p>Understand how macroeconomic headwinds or tailwinds affect your portfolio.</p>", unsafe_allow_html=True)
    
    # ML TRANSPARENCY SECTION
    with st.expander("About the Machine Learning Model (Transparency)"):
        st.markdown("""
        **Model Overview:**
        This prediction is powered by a **Random Forest Classifier** trained entirely offline.
        
        **Data Source:**
        The model was trained on 24 years of daily historical data (Jan 2000 - Jan 2024) fetched via `yfinance`. 
        Features include:
        - **S&P 500 (^GSPC)**: Broad market returns.
        - **VIX (^VIX)**: Market volatility / Fear Index.
        - **US 10-Year Treasury Yield (^TNX)**: Interest rate proxy.
        - **US Dollar Index (DX-Y.NYB)**: Currency strength.
        
        **Performance Metrics:**
        - **Accuracy:** ~78.9%
        - **Target:** Predicting if the S&P 500 will have a negative return (Correction) 21 trading days into the future.
        """)
        
    model_path = "ml_pipeline/macro_risk_model.joblib"
    
    if not os.path.exists(model_path):
        st.warning("Macro Risk Model not found. Please ensure Phase 2 (train_model.py) was completed.")
        return
        
    with st.spinner("Loading ML Model and fetching latest macro data..."):
        try:
            model = joblib.load(model_path)
                
            # Fetch latest data to run inference — download individually for reliability
            ticker_map = {'^GSPC': 'SP500', '^TNX': 'US10Y', '^VIX': 'VIX'}
            dxy_tickers = ['DX-Y.NYB', 'DX=F']  # fallback for Dollar Index
            frames = {}

            def _download_close(yf_ticker):
                t_data = yf.download(yf_ticker, period="2y", progress=False)
                if t_data.empty:
                    return None
                if isinstance(t_data.columns, pd.MultiIndex):
                    return t_data['Close'].iloc[:, 0]
                elif 'Close' in t_data.columns:
                    return t_data['Close']
                return t_data.iloc[:, 0]

            for yf_ticker, col_name in ticker_map.items():
                try:
                    close = _download_close(yf_ticker)
                    if close is not None and len(close) > 0:
                        frames[col_name] = close
                except Exception:
                    pass

            # Try DXY tickers with fallback
            for dxy_ticker in dxy_tickers:
                try:
                    close = _download_close(dxy_ticker)
                    if close is not None and len(close) > 0:
                        frames['DXY'] = close
                        break
                except Exception:
                    pass

            missing = [k for k in ['SP500', 'US10Y', 'VIX', 'DXY'] if k not in frames]
            if missing:
                st.error(f"Could not fetch data for: {missing}. Try refreshing.")
                return

            df = pd.DataFrame(frames).dropna(how='all').ffill().dropna()
            
            df['SP500_Return'] = df['SP500'].pct_change()
            df['VIX_Change'] = df['VIX'].diff()
            df['US10Y_Change'] = df['US10Y'].diff()
            df['DXY_Return'] = df['DXY'].pct_change()
            
            # Recreate missing ML pipeline rolling features
            df['SP500_20d_vol'] = df['SP500_Return'].rolling(20).std()
            df['SP500_200d_ma_diff'] = df['SP500'] / df['SP500'].rolling(200).mean() - 1
            df['VIX_zscore'] = (df['VIX'] - df['VIX'].rolling(252).mean()) / df['VIX'].rolling(252).std()
            df['US10Y_20d_std'] = df['US10Y_Change'].rolling(20).std()
            
            latest_data = df.dropna().iloc[-1:]
            
            prediction = model.predict(latest_data)[0]
            probability = model.predict_proba(latest_data)[0][1] # Probability of class 1 (Correction)
            
            st.markdown("### 1-Month Market Outlook")
            
            col1, col2 = st.columns(2)
            with col1:
                color = "#C44536" if prediction == 1 else "#1F8A70"
                text = "Market Correction Risk" if prediction == 1 else "Market Stable"
                html1 = f"""
                <div style='background-color:#FFFFFF; padding:1.5rem; border-radius:10px; border: 1px solid #D9D9D9; text-align:center;'>
                    <h5>Predicted Scenario</h5>
                    <h2 style='color:{color};'>{text}</h2>
                </div>
                """
                st.markdown(html1, unsafe_allow_html=True)
                
            with col2:
                html2 = f"""
                <div style='background-color:#FFFFFF; padding:1.5rem; border-radius:10px; border: 1px solid #D9D9D9; text-align:center;'>
                    <h5>Probability of Correction</h5>
                    <h2 style='color:#0B1F3A;'>{probability*100:.1f}%</h2>
                </div>
                """
                st.markdown(html2, unsafe_allow_html=True)
                
            # Explain with Claude
            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button("Get AI Macro Analysis"):
                from utils.ai_advisor import client
                if client:
                    prompt = f"The ML model predicts a {probability*100:.1f}% chance of a market correction next month based on VIX, S&P levels, DXY and Bond yields. Explain what this means for a general stock portfolio."
                    
                    sys_prompt = "You are an expert macro economist. Keep it brief and objective."
                    if st.session_state.get('eli10_mode', False):
                        sys_prompt = "Explain this market weather report to a 10 year old."
                        
                    res = client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=400,
                        system=sys_prompt,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    st.info(res.content[0].text)
                else:
                    st.error("Set ANTHROPIC_API_KEY to generate text.")
                    
        except Exception as e:
            st.error(f"Error running inference: {e}")
