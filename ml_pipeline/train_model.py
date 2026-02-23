import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Sklearn - Preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Sklearn - Model Selection & Evaluation
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# Sklearn - Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Sklearn - Metrics
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# =============================================================================
# EXTRACT DATA using yFinance
# =============================================================================

print("Fetching Macro Data via yfinance...")

tickers = ["^GSPC", "^VIX", "^TNX", "DX-Y.NYB"]
data = yf.download(tickers, start="2000-01-01", end="2024-01-01", progress=False)

if 'Adj Close' in data.columns.levels[0]:
    data = data['Adj Close']
else:
    data = data['Close']

df = data.dropna(how='all').ffill().dropna()

df.columns = ['SP500', 'US10Y', 'VIX', 'DXY']

# =============================================================================
# FEATURE ENGINEERING (Improved but preserving names)
# =============================================================================

df['SP500_Return'] = df['SP500'].pct_change()
df['VIX_Change'] = df['VIX'].diff()
df['US10Y_Change'] = df['US10Y'].diff()
df['DXY_Return'] = df['DXY'].pct_change()

# Rolling Features (NEW — additive, does not rename existing vars)
df['SP500_20d_vol'] = df['SP500_Return'].rolling(20).std()
df['SP500_200d_ma_diff'] = df['SP500'] / df['SP500'].rolling(200).mean() - 1
df['VIX_zscore'] = (df['VIX'] - df['VIX'].rolling(252).mean()) / df['VIX'].rolling(252).std()
df['US10Y_20d_std'] = df['US10Y_Change'].rolling(20).std()

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

numerical_features = [
    'SP500', 'US10Y', 'VIX', 'DXY',
    'SP500_Return', 'VIX_Change', 'US10Y_Change', 'DXY_Return',
    'SP500_20d_vol', 'SP500_200d_ma_diff', 'VIX_zscore', 'US10Y_20d_std'
]

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

    # --- Models ---
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoost': GradientBoostingClassifier(random_state=42)
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
                'clf__n_estimators': [100, 300],
                'clf__max_depth': [3, 5, None]
            }

        if name == 'GradientBoost':
            param_grid = {
                'clf__n_estimators': [100, 300],
                'clf__learning_rate': [0.05, 0.1]
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

    print(f"\nBest Model AUC: {best_score:.4f}")

    # Retrain on full dataset (train only portion to preserve integrity)
    best_model_pipe.fit(X_train, y_train)

    os.makedirs('ml_pipeline', exist_ok=True)

    with open('ml_pipeline/macro_risk_model.pkl', 'wb') as f:
        pickle.dump(best_model_pipe, f)

    print("\nModel successfully saved to ml_pipeline/macro_risk_model.pkl")


if __name__ == "__main__":
    train_macro_model(df, TARGET, preprocessor)