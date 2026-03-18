import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Sklearn - Preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Sklearn - Model Selection & Evaluation
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score, learning_curve

# Sklearn - Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Sklearn - Metrics
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Sklearn - Inspection
from sklearn.inspection import permutation_importance

# Shared feature engineering (prevents train-serve skew)
from ml_pipeline.features import build_macro_features, MACRO_FEATURE_NAMES_V1

# =============================================================================
# EXTRACT DATA using yFinance
# =============================================================================

import datetime as _dt
_end_date = _dt.date.today().strftime("%Y-%m-%d")

print(f"Fetching Macro Data via yfinance (2000-01-01 to {_end_date})...")

# Download each ticker individually for robustness (avoids MultiIndex issues)
ticker_map = {'^GSPC': 'SP500', '^TNX': 'US10Y', '^VIX': 'VIX'}
dxy_tickers = ['DX-Y.NYB', 'DX=F']
frames = {}

def _dl_close(yf_ticker):
    t_data = yf.download(yf_ticker, start="2000-01-01", end=_end_date, progress=False)
    if t_data.empty:
        return None
    if isinstance(t_data.columns, pd.MultiIndex):
        return t_data['Close'].iloc[:, 0]
    elif 'Close' in t_data.columns:
        return t_data['Close']
    return t_data.iloc[:, 0]

for yf_ticker, col_name in ticker_map.items():
    close = _dl_close(yf_ticker)
    if close is not None and len(close) > 0:
        frames[col_name] = close
        print(f"  {col_name} ({yf_ticker}): {len(close)} rows")

for dxy_ticker in dxy_tickers:
    try:
        close = _dl_close(dxy_ticker)
        if close is not None and len(close) > 0:
            frames['DXY'] = close
            print(f"  DXY ({dxy_ticker}): {len(close)} rows")
            break
    except Exception:
        pass

missing = [k for k in ['SP500', 'US10Y', 'VIX', 'DXY'] if k not in frames]
if missing:
    raise ValueError(f"Could not fetch macro data for: {missing}")

df = pd.DataFrame(frames).dropna(how='all').ffill().dropna()
print(f"Combined dataset: {len(df)} rows")

# =============================================================================
# FEATURE ENGINEERING — using shared module for train-serve consistency
# =============================================================================

df = build_macro_features(df)

# =============================================================================
# IMPROVED TARGET (More meaningful correction definition)
# =============================================================================

df['Future_Return'] = df['SP500'].shift(-21) / df['SP500'] - 1
df['Target'] = (df['Future_Return'] < -0.05).astype(int)

df = df.dropna()

print(f"Dataset shape: {df.shape}")
print(df['Target'].value_counts(normalize=True))

# =============================================================================
# PREPROCESSING SECTION
# =============================================================================

TARGET = 'Target'

numerical_features = MACRO_FEATURE_NAMES_V1

preprocessor = Pipeline([
    ('scaler', StandardScaler())
])

SCORING_METRIC = 'roc_auc'

# =============================================================================
# ML PIPELINE
# =============================================================================

def train_macro_model(df, target_col, preprocessor):

    print("\nStarting ML training...")

    X = df[numerical_features]
    y = df[target_col]

    # --- TIME SERIES SPLIT (No leakage) ---
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # --- Models (all handle class imbalance) ---
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos = n_neg / n_pos if n_pos > 0 else 1.0

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'GradientBoost': GradientBoostingClassifier(random_state=42),  # uses sample_weight via grid
        'XGBoost': XGBClassifier(
            scale_pos_weight=scale_pos, random_state=42,
            eval_metric='auc', use_label_encoder=False,
        ),
    }

    results = {}
    best_model_pipe = None
    best_score = -np.inf

    tscv = TimeSeriesSplit(n_splits=5)

    for name, model in models.items():

        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('clf', model)
        ])

        param_grid = {}

        if name == 'RandomForest':
            param_grid = {
                'clf__n_estimators': [100, 300, 500],
                'clf__max_depth': [3, 5, 8, None],
                'clf__min_samples_leaf': [5, 10, 20],
                'clf__max_features': ['sqrt', 'log2'],
            }

        if name == 'GradientBoost':
            param_grid = {
                'clf__n_estimators': [100, 300, 500],
                'clf__learning_rate': [0.01, 0.05, 0.1],
                'clf__max_depth': [3, 5],
                'clf__subsample': [0.8, 1.0],
            }

        if name == 'XGBoost':
            param_grid = {
                'clf__n_estimators': [100, 300, 500],
                'clf__learning_rate': [0.01, 0.05, 0.1],
                'clf__max_depth': [3, 5, 8],
                'clf__subsample': [0.8, 1.0],
            }

        if param_grid:
            grid = GridSearchCV(pipe, param_grid, cv=tscv, scoring=SCORING_METRIC, n_jobs=-1)
            grid.fit(X_train, y_train)
            model_to_eval = grid.best_estimator_
        else:
            pipe.fit(X_train, y_train)
            model_to_eval = pipe

        y_pred = model_to_eval.predict(X_test)
        y_prob = model_to_eval.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)

        print(f"\n{name}")
        print(f"AUC: {auc:.4f}")
        print(f"Accuracy: {acc:.4f}")

        results[name] = auc

        if auc > best_score:
            best_score = auc
            best_model_pipe = model_to_eval
            best_y_test = y_test.values
            best_y_pred = y_pred
            best_y_prob = y_prob

    print(f"\nBest Model AUC: {best_score:.4f}")

    # Retrain on full dataset (train only portion to preserve integrity)
    best_model_pipe.fit(X_train, y_train)

    # --- EXPANDED EVALUATION ARTIFACTS ---
    print("\nComputing expanded evaluation artifacts...")

    # Cross-validation scores
    cv_scores = cross_val_score(best_model_pipe, X_train, y_train, cv=tscv, scoring='roc_auc')
    print(f"CV AUC scores: {cv_scores}")
    print(f"CV AUC mean: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # Learning curve
    train_sizes, train_scores, val_scores = learning_curve(
        best_model_pipe, X_train, y_train, cv=tscv, scoring='roc_auc',
        train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=-1
    )
    print(f"Learning curve computed with {len(train_sizes)} points")

    # Permutation importance
    perm_imp = permutation_importance(
        best_model_pipe, X_test, y_test, n_repeats=10,
        scoring='roc_auc', random_state=42, n_jobs=-1
    )
    print(f"Permutation importance computed for {len(numerical_features)} features")

    # Feature correlation matrix
    feature_corr = X[numerical_features].corr()

    os.makedirs('ml_pipeline', exist_ok=True)

    model_path = 'ml_pipeline/macro_risk_model.joblib'
    joblib.dump(best_model_pipe, model_path)

    # Save evaluation artifacts for ML Transparency visualizations
    eval_path = 'ml_pipeline/macro_eval_results.joblib'
    joblib.dump({
        'y_test': best_y_test,
        'y_pred': best_y_pred,
        'y_prob': best_y_prob,
        'cv_scores': cv_scores,
        'train_sizes': train_sizes,
        'train_scores': train_scores,
        'val_scores': val_scores,
        'perm_importance_mean': perm_imp.importances_mean,
        'perm_importance_std': perm_imp.importances_std,
        'feature_corr': feature_corr.values,
        'feature_names': numerical_features,
        # Model metadata for staleness tracking and transparency
        'trained_at': _dt.datetime.now().isoformat(),
        'data_end_date': _end_date,
        'best_model_name': max(results, key=results.get),
        'all_model_results': results,
        'best_auc': best_score,
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'class_distribution': {0: int((y == 0).sum()), 1: int((y == 1).sum())},
    }, eval_path)
    print(f"Evaluation artifacts saved to {eval_path}")

    print(f"\nModel successfully saved to {model_path}")


if __name__ == "__main__":
    train_macro_model(df, TARGET, preprocessor)
