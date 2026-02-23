import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Sklearn - Preprocessing
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Sklearn - Model Selection & Evaluation
from sklearn.model_selection import train_test_split, GridSearchCV

# Sklearn - Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Sklearn - Metrics
from sklearn.metrics import accuracy_score

# =============================================================================
# EXTRACT DATA using yFinance
# =============================================================================

print("Fetching Macro Data via yfinance...")

# Tickers: S&P 500, VIX, US 10Y Treasury, Dollar Index
tickers = ["^GSPC", "^VIX", "^TNX", "DX-Y.NYB"]
data = yf.download(tickers, start="2000-01-01", end="2024-01-01", progress=False)

if 'Adj Close' in data.columns.levels[0]:
    data = data['Adj Close']
else:
    data = data['Close']

# Drop entirely missing rows and forward fill
df = data.dropna(how='all').ffill().dropna()

# Rename columns for clarity
df.columns = ['SP500', 'US10Y', 'VIX', 'DXY']

# Calculate Returns
df['SP500_Return'] = df['SP500'].pct_change()
df['VIX_Change'] = df['VIX'].diff()
df['US10Y_Change'] = df['US10Y'].diff()
df['DXY_Return'] = df['DXY'].pct_change()

# Target: 1 if S&P 500 return over next 1 month (21 days) is negative (correction), 0 otherwise
df['Target'] = (df['SP500'].shift(-21) < df['SP500']).astype(int)

# Drop rows with NaN (due to pct_change and shift)
df = df.dropna()

print(f"Dataset shape: {df.shape}")
print(df['Target'].value_counts(normalize=True))

# =============================================================================
# PREPROCESSING SECTION
# =============================================================================

TARGET = 'Target'

COLUMNS_TO_DROP = []

numerical_features = ['SP500', 'US10Y', 'VIX', 'DXY', 'SP500_Return', 'VIX_Change', 'US10Y_Change', 'DXY_Return']
categorical_features = []

# Pattern A: All numerical, no missing
preprocessor = Pipeline([('scaler', StandardScaler())])

SCORING_METRIC = 'accuracy'

# =============================================================================
# ML PIPELINE (Condensed from template)
# =============================================================================

def train_macro_model(df, target_col, preprocessor):
    print("\nStarting ML training...")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define models
    models = {
        'LogisticRegression': Pipeline([('preprocessor', preprocessor), ('clf', LogisticRegression(max_iter=1000, random_state=42))]),
        'RandomForest': Pipeline([('preprocessor', preprocessor), ('clf', RandomForestClassifier(n_estimators=100, random_state=42))]),
        'GradientBoost': Pipeline([('preprocessor', preprocessor), ('clf', GradientBoostingClassifier(random_state=42))])
    }
    
    # Train and evaluate
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        score = accuracy_score(y_test, model.predict(X_test))
        results[name] = score
        print(f"{name} Accuracy: {score:.4f}")
        
    best_model_name = max(results, key=results.get)
    print(f"\nBest Model: {best_model_name} ({results[best_model_name]:.4f})")
    
    best_model_pipe = models[best_model_name]
    best_model_pipe.fit(X_train, y_train)
    
    # Ensure directory exists
    os.makedirs('ml_pipeline', exist_ok=True)
    
    # Save the model
    with open('ml_pipeline/macro_risk_model.pkl', 'wb') as f:
        pickle.dump(best_model_pipe, f)
        
    print("\n✅ Model successfully saved to ml_pipeline/macro_risk_model.pkl")

if __name__ == "__main__":
    train_macro_model(df, TARGET, preprocessor)
