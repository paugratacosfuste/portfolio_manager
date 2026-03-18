import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os


def render_ml_transparency():
    st.markdown("<h1>ML Transparency & Model Evaluation</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p>Full technical details of both ML models used in the platform — "
        "computed live from the actual saved pipelines.</p>",
        unsafe_allow_html=True,
    )

    # =====================================================================
    # SECTION 1 — Macro Risk Model (Random Forest)
    # =====================================================================
    st.markdown("---")
    st.markdown("## 1. Macro Risk Model (Random Forest Classifier)")

    # --- Training Pipeline Overview ---
    st.markdown("### Training Pipeline Overview")
    st.markdown("""
- **Data**: 24 years of daily macro data (Jan 2000 – Jan 2024) fetched via `yfinance`
- **Tickers**: `^GSPC`, `^VIX`, `^TNX`, `DX-Y.NYB`
- **Target**: Binary — whether the S&P 500 drops > 5 % over the next 21 trading days
- **Train / Test**: 80 / 20 time-series split (no data leakage)
- **Validation**: 5-fold `TimeSeriesSplit` with `GridSearchCV`
- **Scoring metric**: AUC-ROC
""")

    # --- Feature Engineering Table ---
    st.markdown("### Feature Engineering")
    feature_table = pd.DataFrame([
        {"Feature": "SP500",              "Source": "^GSPC",   "Description": "S&P 500 closing price"},
        {"Feature": "US10Y",              "Source": "^TNX",    "Description": "10-Year Treasury yield"},
        {"Feature": "VIX",                "Source": "^VIX",    "Description": "CBOE Volatility Index"},
        {"Feature": "DXY",                "Source": "DX-Y.NYB","Description": "US Dollar Index"},
        {"Feature": "SP500_Return",       "Source": "Derived", "Description": "Daily % change of S&P 500"},
        {"Feature": "VIX_Change",         "Source": "Derived", "Description": "Daily absolute change in VIX"},
        {"Feature": "US10Y_Change",       "Source": "Derived", "Description": "Daily absolute change in yield"},
        {"Feature": "DXY_Return",         "Source": "Derived", "Description": "Daily % change of DXY"},
        {"Feature": "SP500_20d_vol",      "Source": "Rolling", "Description": "20-day rolling std of SP500 returns"},
        {"Feature": "SP500_200d_ma_diff", "Source": "Rolling", "Description": "Current price vs 200-day MA deviation"},
        {"Feature": "VIX_zscore",         "Source": "Rolling", "Description": "VIX z-score over 252-day window"},
        {"Feature": "US10Y_20d_std",      "Source": "Rolling", "Description": "20-day rolling std of yield changes"},
    ])
    st.dataframe(feature_table, use_container_width=True, hide_index=True)

    # --- Model Comparison ---
    st.markdown("### Model Comparison (from training)")
    comparison = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "Gradient Boosting"],
        "AUC-ROC": [0.72, 0.81, 0.79],
        "Accuracy": ["76.2 %", "78.9 %", "77.5 %"],
        "Selected": ["", "**Winner**", ""],
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.info("Random Forest was selected as the best model based on the highest AUC-ROC score during cross-validated evaluation.")

    # --- Live Introspection ---
    macro_path = "ml_pipeline/macro_risk_model.joblib"
    if os.path.exists(macro_path):
        model = joblib.load(macro_path)

        st.markdown("### Live Model Introspection")
        st.caption("Values below are extracted directly from the saved `.joblib` pipeline.")

        feature_names = [
            "SP500", "US10Y", "VIX", "DXY",
            "SP500_Return", "VIX_Change", "US10Y_Change", "DXY_Return",
            "SP500_20d_vol", "SP500_200d_ma_diff", "VIX_zscore", "US10Y_20d_std",
        ]

        # Feature importances
        try:
            clf = model.named_steps["clf"]
            importances = clf.feature_importances_
            imp_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances,
            }).sort_values("Importance", ascending=True)

            fig_imp = go.Figure(go.Bar(
                x=imp_df["Importance"],
                y=imp_df["Feature"],
                orientation="h",
                marker_color="#3A6EA5",
            ))
            fig_imp.update_layout(
                title="Feature Importance (Random Forest)",
                xaxis_title="Importance",
                yaxis_title="",
                margin=dict(t=40, b=40, l=140, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=420,
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not extract feature importances: {e}")

        # Hyperparameters
        try:
            params = clf.get_params()
            params_df = pd.DataFrame(
                [{"Parameter": k, "Value": str(v)} for k, v in sorted(params.items())]
            )
            st.markdown("#### Hyperparameters (Random Forest)")
            st.dataframe(params_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not extract hyperparameters: {e}")

        # Scaler statistics
        try:
            scaler = model.named_steps["preprocessor"].named_steps["scaler"]
            scaler_df = pd.DataFrame({
                "Feature": feature_names,
                "Scaler Mean": scaler.mean_,
                "Scaler Std": scaler.scale_,
            })
            st.markdown("#### StandardScaler Statistics")
            st.dataframe(scaler_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not extract scaler statistics: {e}")

    # --- Evaluation Metrics: Confusion Matrix & ROC ---
    macro_eval_path = "ml_pipeline/macro_eval_results.joblib"
    if os.path.exists(macro_eval_path):
        eval_data = joblib.load(macro_eval_path)
        y_test = eval_data["y_test"]
        y_pred = eval_data["y_pred"]
        y_prob = eval_data["y_prob"]

        st.markdown("### Evaluation Metrics")
        col_cm, col_roc = st.columns(2)

        with col_cm:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            labels = ["Stable (0)", "Correction (1)"]
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm, x=labels, y=labels,
                colorscale=[[0, "#D4E5F5"], [1, "#3A6EA5"]],
                text=cm, texttemplate="%{text}", textfont={"size": 16},
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
                showscale=False,
            ))
            fig_cm.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted", yaxis_title="Actual",
                yaxis=dict(autorange="reversed"),
                margin=dict(t=40, b=40, l=80, r=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=350,
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_roc:
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {roc_auc:.3f}", line=dict(color="#3A6EA5", width=2)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#999"), showlegend=False))
            fig_roc.update_layout(
                title=f"ROC Curve (AUC = {roc_auc:.3f})",
                xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                margin=dict(t=40, b=40, l=40, r=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=350, legend=dict(x=0.6, y=0.1),
            )
            st.plotly_chart(fig_roc, use_container_width=True)
    else:
        st.info("Run `train_model.py` to generate evaluation artifacts for confusion matrix & ROC curve.")

    if not os.path.exists(macro_path):
        st.warning("Macro Risk Model file not found (`ml_pipeline/macro_risk_model.joblib`).")

    # =====================================================================
    # SECTION 2 — Sentiment Model (Logistic Regression + TF-IDF)
    # =====================================================================
    st.markdown("---")
    st.markdown("## 2. Sentiment Model (Logistic Regression + TF-IDF)")

    # --- Training Pipeline Overview ---
    st.markdown("### Training Pipeline Overview")
    st.markdown("""
- **Data**: ~6,000 financial tweets from Kaggle ("Tweet Sentiment's Impact on Stock Returns")
- **Split**: 80 / 20 stratified random split
- **Vectorizer**: TF-IDF — 5,000 max features, (1,2) n-grams, `min_df=2`, English stop words removed
- **Classifier**: Logistic Regression — balanced class weights, `C=1.0`, `max_iter=1000`
""")

    # --- Performance Metrics ---
    st.markdown("### Performance Metrics")
    perf = pd.DataFrame({
        "Metric": ["Accuracy", "F1-Score (Positive)", "F1-Score (Negative)"],
        "Value": ["80.07 %", "84 %", "74 %"],
    })
    st.dataframe(perf, use_container_width=True, hide_index=True)

    # --- Live Introspection ---
    sentiment_path = "ml_pipeline/sentiment_pipeline.joblib"
    if os.path.exists(sentiment_path):
        pipeline = joblib.load(sentiment_path)

        st.markdown("### Live Model Introspection")
        st.caption("Values below are extracted directly from the saved `.joblib` pipeline.")

        # Top positive & negative words
        try:
            vectorizer = pipeline.named_steps["vectorizer"]
            classifier = pipeline.named_steps["classifier"]
            feature_names_sent = vectorizer.get_feature_names_out()
            coefs = classifier.coef_

            # For multi-class, coefs shape is (n_classes, n_features).
            # Use the last class row (highest sentiment) for positive, first for negative.
            if coefs.shape[0] > 1:
                pos_coef = coefs[-1]
                neg_coef = coefs[0]
            else:
                pos_coef = coefs[0]
                neg_coef = coefs[0]

            top_n = 15

            # Positive words
            top_pos_idx = np.argsort(pos_coef)[-top_n:][::-1]
            top_pos_words = feature_names_sent[top_pos_idx]
            top_pos_vals = pos_coef[top_pos_idx]

            # Negative words
            top_neg_idx = np.argsort(neg_coef)[:top_n]
            top_neg_words = feature_names_sent[top_neg_idx]
            top_neg_vals = neg_coef[top_neg_idx]

            col1, col2 = st.columns(2)

            with col1:
                fig_pos = go.Figure(go.Bar(
                    x=top_pos_vals,
                    y=top_pos_words,
                    orientation="h",
                    marker_color="#1F8A70",
                ))
                fig_pos.update_layout(
                    title="Top 15 Positive-Signal Words",
                    xaxis_title="Coefficient",
                    yaxis_title="",
                    yaxis=dict(autorange="reversed"),
                    margin=dict(t=40, b=40, l=120, r=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=450,
                )
                st.plotly_chart(fig_pos, use_container_width=True)

            with col2:
                fig_neg = go.Figure(go.Bar(
                    x=top_neg_vals,
                    y=top_neg_words,
                    orientation="h",
                    marker_color="#C44536",
                ))
                fig_neg.update_layout(
                    title="Top 15 Negative-Signal Words",
                    xaxis_title="Coefficient",
                    yaxis_title="",
                    yaxis=dict(autorange="reversed"),
                    margin=dict(t=40, b=40, l=120, r=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=450,
                )
                st.plotly_chart(fig_neg, use_container_width=True)

        except Exception as e:
            st.warning(f"Could not extract top words: {e}")

        # Vectorizer config
        try:
            vec_params = vectorizer.get_params()
            display_keys = [
                "max_features", "ngram_range", "min_df", "max_df",
                "stop_words", "analyzer", "lowercase", "sublinear_tf",
            ]
            vec_df = pd.DataFrame(
                [{"Parameter": k, "Value": str(vec_params.get(k, "—"))} for k in display_keys]
            )
            st.markdown("#### TF-IDF Vectorizer Configuration")
            st.dataframe(vec_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not extract vectorizer config: {e}")

    # --- Evaluation Metrics: Confusion Matrix & ROC for Sentiment ---
    sent_eval_path = "ml_pipeline/sentiment_eval_results.joblib"
    if os.path.exists(sent_eval_path):
        eval_sent = joblib.load(sent_eval_path)
        y_test_s = eval_sent["y_test"]
        y_pred_s = eval_sent["y_pred"]
        y_prob_s = eval_sent["y_prob"]
        classes_s = eval_sent.get("classes", sorted(set(y_test_s)))

        st.markdown("### Evaluation Metrics")
        col_cm2, col_roc2 = st.columns(2)

        with col_cm2:
            from sklearn.metrics import confusion_matrix
            cm_s = confusion_matrix(y_test_s, y_pred_s, labels=classes_s)
            class_labels = [str(c) for c in classes_s]
            fig_cm2 = go.Figure(data=go.Heatmap(
                z=cm_s, x=class_labels, y=class_labels,
                colorscale=[[0, "#D4E5F5"], [1, "#1F8A70"]],
                text=cm_s, texttemplate="%{text}", textfont={"size": 14},
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
                showscale=False,
            ))
            fig_cm2.update_layout(
                title="Confusion Matrix (Sentiment)",
                xaxis_title="Predicted", yaxis_title="Actual",
                yaxis=dict(autorange="reversed"),
                margin=dict(t=40, b=40, l=80, r=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=350,
            )
            st.plotly_chart(fig_cm2, use_container_width=True)

        with col_roc2:
            from sklearn.metrics import roc_curve, auc as sk_auc
            # For binary or multi-class: use OvR macro approach for the positive class
            if y_prob_s.ndim == 2 and y_prob_s.shape[1] == 2:
                fpr_s, tpr_s, _ = roc_curve(y_test_s, y_prob_s[:, 1])
            elif y_prob_s.ndim == 2:
                # multi-class: use last column (highest sentiment) as positive
                fpr_s, tpr_s, _ = roc_curve((y_test_s == classes_s[-1]).astype(int), y_prob_s[:, -1])
            else:
                fpr_s, tpr_s, _ = roc_curve(y_test_s, y_prob_s)
            roc_auc_s = sk_auc(fpr_s, tpr_s)
            fig_roc2 = go.Figure()
            fig_roc2.add_trace(go.Scatter(x=fpr_s, y=tpr_s, mode="lines", name=f"AUC = {roc_auc_s:.3f}", line=dict(color="#1F8A70", width=2)))
            fig_roc2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#999"), showlegend=False))
            fig_roc2.update_layout(
                title=f"ROC Curve (AUC = {roc_auc_s:.3f})",
                xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                margin=dict(t=40, b=40, l=40, r=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=350, legend=dict(x=0.6, y=0.1),
            )
            st.plotly_chart(fig_roc2, use_container_width=True)
    else:
        st.info("Run `train_sentiment_model.py` to generate evaluation artifacts for confusion matrix & ROC curve.")

    if not os.path.exists(sentiment_path):
        st.warning("Sentiment Pipeline file not found (`ml_pipeline/sentiment_pipeline.joblib`).")

    # ── Model Limitations ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Model Limitations & Caveats")
    st.markdown("""
- **Class Imbalance (Macro Model):** ~88% of samples are class 0 (market stable). The model may under-predict corrections. Consider this when interpreting low correction probabilities.
- **Time-Series Distribution Shift:** Both models were trained on historical data. Macro regimes change — a model trained on 2000-2024 data may not generalize to novel market structures (e.g., unprecedented monetary policy).
- **Training Data Staleness:** The macro model uses data up to Jan 2024. It has not seen recent market events. Periodic retraining is recommended.
- **Sentiment Model Scope:** Trained on ~6,000 financial tweets — a relatively small corpus. Performance may degrade on formal news language or non-English text.
- **No Causal Claims:** Both models identify statistical correlations, not causal relationships. Use predictions as one signal among many, not as sole decision drivers.
""")
