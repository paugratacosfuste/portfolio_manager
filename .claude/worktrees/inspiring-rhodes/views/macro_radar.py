import streamlit as st
import pandas as pd
import pickle
import os
import yfinance as yf

def render_macro_radar():
    st.title("Macro Radar: Market Forecasting")
    st.write("Understand how macroeconomic headwinds or tailwinds affect your portfolio.")

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

    model_path = "ml_pipeline/macro_risk_model.pkl"

    if not os.path.exists(model_path):
        st.warning("Macro Risk Model not found. Please ensure Phase 2 (train_model.py) was completed.")
        return

    # Load model and fetch data
    prediction = None
    probability = None
    error_msg = None

    with st.spinner("Loading ML Model and fetching latest macro data..."):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            tickers = ["^GSPC", "^VIX", "^TNX", "DX-Y.NYB"]
            data = yf.download(tickers, period="2mo", progress=False)

            if 'Adj Close' in data:
                data = data['Adj Close']
            else:
                data = data['Close']

            df = data.dropna(how='all').ffill().dropna()
            col_mapping = {}
            for col in df.columns:
                col_str = str(col)
                if 'GSPC' in col_str:
                    col_mapping[col] = 'SP500'
                elif 'VIX' in col_str:
                    col_mapping[col] = 'VIX'
                elif 'TNX' in col_str:
                    col_mapping[col] = 'US10Y'
                elif 'DX' in col_str:
                    col_mapping[col] = 'DXY'
            if col_mapping:
                df = df.rename(columns=col_mapping)
            else:
                df.columns = ['DXY', 'SP500', 'US10Y', 'VIX']

            df['SP500_Return'] = df['SP500'].pct_change()
            df['VIX_Change'] = df['VIX'].diff()
            df['US10Y_Change'] = df['US10Y'].diff()
            df['DXY_Return'] = df['DXY'].pct_change()

            feature_cols = ['SP500_Return', 'VIX_Change', 'US10Y_Change', 'DXY_Return']
            latest_data = df[feature_cols].dropna().iloc[-1:]

            prediction = model.predict(latest_data)[0]
            probability = model.predict_proba(latest_data)[0][1]

        except Exception as e:
            error_msg = str(e)

    # Display results OUTSIDE the spinner
    if error_msg:
        st.error(f"Error running inference: {error_msg}")
        return

    if prediction is not None and probability is not None:
        st.subheader("1-Month Market Outlook")

        col1, col2 = st.columns(2)
        with col1:
            scenario_label = "Market Correction Risk" if prediction == 1 else "Market Stable"
            scenario_delta = "High Risk" if prediction == 1 else "Low Risk"
            delta_color = "inverse" if prediction == 1 else "normal"
            st.metric(
                label="Predicted Scenario",
                value=scenario_label,
                delta=scenario_delta,
                delta_color=delta_color,
            )

        with col2:
            st.metric(
                label="Probability of Correction",
                value=f"{probability * 100:.1f}%",
                delta=f"{'Above' if probability > 0.5 else 'Below'} 50% threshold",
                delta_color="inverse" if probability > 0.5 else "normal",
            )

        st.markdown("---")

        if st.button("Get AI Macro Analysis"):
            from utils.ai_advisor import client
            if client:
                prompt = f"The ML model predicts a {probability * 100:.1f}% chance of a market correction next month based on VIX, S&P levels, DXY and Bond yields. Explain what this means for a general stock portfolio."

                sys_prompt = "You are an expert macro economist. Keep it brief and objective."
                if st.session_state.get('eli10_mode', False):
                    sys_prompt = "Explain this market weather report to a 10 year old."

                try:
                    res = client.messages.create(
                        model="claude-haiku-3-5-20241022",
                        max_tokens=400,
                        system=sys_prompt,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    st.info(res.content[0].text)
                except Exception as e:
                    st.error(f"Error calling Claude: {e}")
            else:
                st.error("Set ANTHROPIC_API_KEY to generate text.")
