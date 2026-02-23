import streamlit as st
import pandas as pd
import pickle
import os
import yfinance as yf

def render_macro_radar():
    st.markdown("<h1>Macro Radar: Market Forecasting</h1>", unsafe_allow_html=True)
    st.markdown("<p>Understand how macroeconomic headwinds or tailwinds affect your portfolio.</p>", unsafe_allow_html=True)
    
    # ML TRANSPARENCY SECTION
    with st.expander("ℹ️ About the Machine Learning Model (Transparency)"):
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
        
    model_path = "ml_pipeline/macro_risk_model.pkl"
    
    if not os.path.exists(model_path):
        st.warning("Macro Risk Model not found. Please ensure Phase 2 (train_model.py) was completed.")
        return
        
    with st.spinner("Loading ML Model and fetching latest macro data..."):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                
            # Fetch latest data to run inference
            tickers = ["^GSPC", "^VIX", "^TNX", "DX-Y.NYB"]
            data = yf.download(tickers, period="2mo", progress=False)
            
            if 'Adj Close' in data.columns.levels[0]:
                data = data['Adj Close']
            else:
                data = data['Close']
                
            df = data.dropna(how='all').ffill().dropna()
            df.columns = ['SP500', 'US10Y', 'VIX', 'DXY']
            
            df['SP500_Return'] = df['SP500'].pct_change()
            df['VIX_Change'] = df['VIX'].diff()
            df['US10Y_Change'] = df['US10Y'].diff()
            df['DXY_Return'] = df['DXY'].pct_change()
            
            latest_data = df.dropna().iloc[-1:]
            
            prediction = model.predict(latest_data)[0]
            probability = model.predict_proba(latest_data)[0][1] # Probability of class 1 (Correction)
            
            st.markdown("### 1-Month Market Outlook")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<div style='background-color:#FFFFFF; padding:1.5rem; border-radius:10px; border: 1px solid #D9D9D9; text-align:center;'>", unsafe_allow_html=True)
                st.markdown("<h5>Predicted Scenario</h5>", unsafe_allow_html=True)
                if prediction == 1:
                    st.markdown("<h2 style='color:#C44536;'>Market Correction Risk</h2>", unsafe_allow_html=True)
                else:
                    st.markdown("<h2 style='color:#1F8A70;'>Market Stable</h2>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col2:
                st.markdown("<div style='background-color:#FFFFFF; padding:1.5rem; border-radius:10px; border: 1px solid #D9D9D9; text-align:center;'>", unsafe_allow_html=True)
                st.markdown("<h5>Probability of Correction</h5>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='color:#0B1F3A;'>{probability*100:.1f}%</h2>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            # Explain with Claude
            if st.button("Get AI Macro Analysis"):
                from utils.ai_advisor import client
                if client:
                    prompt = f"The ML model predicts a {probability*100:.1f}% chance of a market correction next month based on VIX, S&P levels, DXY and Bond yields. Explain what this means for a general stock portfolio."
                    
                    sys_prompt = "You are an expert macro economist. Keep it brief and objective."
                    if st.session_state.get('eli10_mode', False):
                        sys_prompt = "Explain this market weather report to a 10 year old."
                        
                    res = client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=400,
                        system=sys_prompt,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    st.info(res.content[0].text)
                else:
                    st.error("Set ANTHROPIC_API_KEY to generate text.")
                    
        except Exception as e:
            st.error(f"Error running inference: {e}")
