import streamlit as st
import pandas as pd
import joblib
import os
import yfinance as yf
from ml_pipeline.features import fetch_macro_data, build_macro_features, MACRO_FEATURE_NAMES_V1

def render_macro_radar():
    st.markdown("<h1>Macro Radar: Market Forecasting</h1>", unsafe_allow_html=True)
    st.markdown("<p>Understand how macroeconomic headwinds or tailwinds affect your portfolio.</p>", unsafe_allow_html=True)
    
    model_path = "ml_pipeline/macro_risk_model.joblib"

    if not os.path.exists(model_path):
        st.warning("Macro Risk Model not found. Please ensure Phase 2 (train_model.py) was completed.")
        return

    # Load model FIRST so we can inspect it in the transparency section
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading ML model: {e}")
        return

    # Read training metadata from eval artifacts for accurate descriptions
    _clf_name = type(model.named_steps["clf"]).__name__
    _data_end = "present"
    _eval_path = "ml_pipeline/macro_eval_results.joblib"
    if os.path.exists(_eval_path):
        try:
            _eval_meta = joblib.load(_eval_path)
            _data_end = _eval_meta.get('data_end_date', 'present')
        except Exception:
            pass

    # ML TRANSPARENCY SECTION
    with st.expander("About the Machine Learning Model (Transparency)"):
        st.markdown(f"""
        **Model Overview:**
        This prediction is powered by a **{_clf_name}** selected as the best performer from a 4-model comparison
        (Logistic Regression, Random Forest, Gradient Boosting, XGBoost) via `GridSearchCV` with 5-fold `TimeSeriesSplit`, scored by AUC-ROC.

        **Data Source:**
        Trained on daily macro data (Jan 2000 – {_data_end}) fetched via `yfinance`.
        Features (12 engineered from 4 macro instruments):
        - **S&P 500 (^GSPC)**: Price levels, daily returns, 20-day realized volatility, 200-day MA deviation
        - **VIX (^VIX)**: Level, daily change, 252-day z-score
        - **US 10-Year Treasury Yield (^TNX)**: Level, daily change, 20-day rolling std
        - **US Dollar Index (DX-Y.NYB)**: Level, daily return

        **Target:** Predicting if the S&P 500 will drop > 5% over the next 21 trading days.

        **Note:** See the **ML Transparency** tab for full evaluation metrics (classification report, ROC/PR curves, calibration, cross-validation, learning curves, feature importances).
        """)

    # Fetch macro data using centralized pipeline (eliminates code duplication)
    with st.spinner("Fetching latest macro data..."):
        try:
            df = fetch_macro_data(period="2y")
        except ValueError as e:
            st.error(f"Could not fetch data: {e}. Try refreshing.")
            return

    # Feature engineering using shared module (identical to training)
    try:
        df = build_macro_features(df)
        # Select only v1 features (12) first, THEN dropna — avoids v2 features causing row loss
        latest_data = df[MACRO_FEATURE_NAMES_V1].dropna().iloc[-1:]
    except Exception as e:
        st.error(f"Error computing features: {e}")
        return

    # Run prediction
    try:
        prediction = model.predict(latest_data)[0]
        probability = model.predict_proba(latest_data)[0][1]
    except Exception as e:
        st.error(f"Error running model inference: {e}")
        return

    # Store macro prediction for cross-view consumption
    st.session_state['macro_prediction'] = {
        'probability': float(probability),
        'prediction': int(prediction),
        'vix': round(float(latest_data['VIX'].iloc[0]), 1),
        'sp500_vs_200ma': round(float(latest_data['SP500_200d_ma_diff'].iloc[0] * 100), 1),
    }

    # Track prediction history for consistency monitoring
    if 'macro_prediction_history' not in st.session_state:
        st.session_state['macro_prediction_history'] = []
    import datetime as _dt
    st.session_state['macro_prediction_history'].append({
        'timestamp': _dt.datetime.now().isoformat(),
        'prediction': int(prediction),
        'probability': round(float(probability), 3),
    })
    # Keep last 30 predictions
    st.session_state['macro_prediction_history'] = st.session_state['macro_prediction_history'][-30:]

    # Model staleness warning
    eval_path = "ml_pipeline/macro_eval_results.joblib"
    if os.path.exists(eval_path):
        try:
            _eval = joblib.load(eval_path)
            data_end = _eval.get('data_end_date', '2024-01-01')
            _end_dt = _dt.datetime.strptime(data_end, '%Y-%m-%d')
            _months_old = (_dt.datetime.now() - _end_dt).days / 30
            if _months_old > 12:
                st.warning(f"Model trained on data ending {data_end} ({_months_old:.0f} months ago). Consider retraining for better accuracy.")
        except Exception:
            pass

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

    # Show key macro indicators driving the prediction
    st.markdown("### Key Macro Indicators")
    ind_col1, ind_col2, ind_col3, ind_col4 = st.columns(4)
    with ind_col1:
        vix_val = float(latest_data['VIX'].iloc[0])
        vix_color = "#C44536" if vix_val > 25 else ("#E08C3A" if vix_val > 18 else "#1F8A70")
        st.markdown(f"<div style='background:#fff;padding:12px;border-radius:8px;border:1px solid #E0E7EF;text-align:center;'><div style='color:#666;font-size:0.75rem;'>VIX</div><div style='font-size:1.3rem;font-weight:700;color:{vix_color};'>{vix_val:.1f}</div></div>", unsafe_allow_html=True)
    with ind_col2:
        ma_diff = float(latest_data['SP500_200d_ma_diff'].iloc[0]) * 100
        ma_color = "#1F8A70" if ma_diff > 0 else "#C44536"
        st.markdown(f"<div style='background:#fff;padding:12px;border-radius:8px;border:1px solid #E0E7EF;text-align:center;'><div style='color:#666;font-size:0.75rem;'>S&P vs 200d MA</div><div style='font-size:1.3rem;font-weight:700;color:{ma_color};'>{ma_diff:+.1f}%</div></div>", unsafe_allow_html=True)
    with ind_col3:
        vol_20d = float(latest_data['SP500_20d_vol'].iloc[0]) * 100
        vol_color = "#C44536" if vol_20d > 2 else "#1F8A70"
        st.markdown(f"<div style='background:#fff;padding:12px;border-radius:8px;border:1px solid #E0E7EF;text-align:center;'><div style='color:#666;font-size:0.75rem;'>20d Realized Vol</div><div style='font-size:1.3rem;font-weight:700;color:{vol_color};'>{vol_20d:.2f}%</div></div>", unsafe_allow_html=True)
    with ind_col4:
        vix_z = float(latest_data['VIX_zscore'].iloc[0])
        z_color = "#C44536" if vix_z > 1 else ("#E08C3A" if vix_z > 0.5 else "#1F8A70")
        st.markdown(f"<div style='background:#fff;padding:12px;border-radius:8px;border:1px solid #E0E7EF;text-align:center;'><div style='color:#666;font-size:0.75rem;'>VIX Z-Score</div><div style='font-size:1.3rem;font-weight:700;color:{z_color};'>{vix_z:+.2f}</div></div>", unsafe_allow_html=True)

    # Prediction consistency tracker
    pred_history = st.session_state.get('macro_prediction_history', [])
    if len(pred_history) > 1:
        recent_preds = [p['prediction'] for p in pred_history[-10:]]
        stable_count = sum(1 for p in recent_preds if p == 0)
        correction_count = len(recent_preds) - stable_count
        st.caption(f"Prediction consistency (last {len(recent_preds)} checks): {stable_count} stable, {correction_count} correction signals")

    # Explain with Claude
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("Get AI Macro Analysis"):
        from utils.ai_advisor import client, track_llm_usage
        import time as _time
        if client:
            try:
                prompt = f"The ML model predicts a {probability*100:.1f}% chance of a market correction next month based on VIX, S&P levels, DXY and Bond yields. Explain what this means for a general stock portfolio."

                sys_prompt = "You are an expert macro economist. Keep it brief and objective."
                if st.session_state.get('eli10_mode', False):
                    sys_prompt = "Explain this market weather report to a 10 year old."

                _model = "claude-haiku-4-5-20251001"
                _t0 = _time.time()
                res = client.messages.create(
                    model=_model,
                    max_tokens=400,
                    system=sys_prompt,
                    messages=[{"role": "user", "content": prompt}]
                )
                track_llm_usage(res, _model, _time.time() - _t0)
                st.info(res.content[0].text)
            except Exception as e:
                st.error(f"Error getting AI analysis: {e}")
        else:
            st.error("Set ANTHROPIC_API_KEY to generate text.")
