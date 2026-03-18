"""
Shared macro feature engineering module.
Used by both training (train_model.py) and inference (chatbot_tools.py, macro_radar.py).
Eliminates code duplication and prevents train-serve skew.
"""
import pandas as pd
import numpy as np
import yfinance as yf


# Canonical feature list — must match training and inference
MACRO_FEATURE_NAMES = [
    'SP500', 'US10Y', 'VIX', 'DXY',
    'SP500_Return', 'VIX_Change', 'US10Y_Change', 'DXY_Return',
    'SP500_20d_vol', 'SP500_200d_ma_diff', 'VIX_zscore', 'US10Y_20d_std',
    # Interaction features
    'VIX_x_SP500_vol', 'Yield_Equity_Divergence',
    # Lag features
    'SP500_Return_lag5', 'VIX_lag5',
]

# The original 12 features (for backward compatibility with existing models)
MACRO_FEATURE_NAMES_V1 = [
    'SP500', 'US10Y', 'VIX', 'DXY',
    'SP500_Return', 'VIX_Change', 'US10Y_Change', 'DXY_Return',
    'SP500_20d_vol', 'SP500_200d_ma_diff', 'VIX_zscore', 'US10Y_20d_std',
]


def build_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies canonical feature engineering to a DataFrame with columns:
    SP500, US10Y, VIX, DXY.

    Returns the same DataFrame with all derived features added.
    Used identically in training and inference to prevent train-serve skew.
    """
    df = df.copy()

    # --- First-order dynamics ---
    df['SP500_Return'] = df['SP500'].pct_change()
    df['VIX_Change'] = df['VIX'].diff()
    df['US10Y_Change'] = df['US10Y'].diff()
    df['DXY_Return'] = df['DXY'].pct_change()

    # --- Rolling / Normalized features ---
    df['SP500_20d_vol'] = df['SP500_Return'].rolling(20).std()
    df['SP500_200d_ma_diff'] = df['SP500'] / df['SP500'].rolling(200).mean() - 1
    df['VIX_zscore'] = (
        (df['VIX'] - df['VIX'].rolling(252).mean()) /
        df['VIX'].rolling(252).std()
    )
    df['US10Y_20d_std'] = df['US10Y_Change'].rolling(20).std()

    # --- Interaction features (new) ---
    df['VIX_x_SP500_vol'] = df['VIX'] * df['SP500_20d_vol']
    df['Yield_Equity_Divergence'] = df['US10Y_Change'] - df['SP500_Return']

    # --- Lag features (new) ---
    df['SP500_Return_lag5'] = df['SP500_Return'].shift(5)
    df['VIX_lag5'] = df['VIX'].shift(5)

    return df


def fetch_macro_data(period: str = "2y") -> pd.DataFrame:
    """
    Downloads macro data from yfinance and returns a clean DataFrame
    with columns: SP500, US10Y, VIX, DXY.

    Handles ticker fallbacks (DXY has multiple possible tickers).
    """
    ticker_map = {'^GSPC': 'SP500', '^TNX': 'US10Y', '^VIX': 'VIX'}
    dxy_tickers = ['DX-Y.NYB', 'DX=F']
    frames = {}

    def _dl_close(yf_ticker):
        t_data = yf.download(yf_ticker, period=period, progress=False)
        if t_data.empty:
            return None
        if isinstance(t_data.columns, pd.MultiIndex):
            return t_data['Close'].iloc[:, 0]
        elif 'Close' in t_data.columns:
            return t_data['Close']
        return t_data.iloc[:, 0]

    for yf_ticker, col_name in ticker_map.items():
        try:
            close = _dl_close(yf_ticker)
            if close is not None and len(close) > 0:
                frames[col_name] = close
        except Exception:
            pass

    for dxy_ticker in dxy_tickers:
        try:
            close = _dl_close(dxy_ticker)
            if close is not None and len(close) > 0:
                frames['DXY'] = close
                break
        except Exception:
            pass

    missing = [k for k in ['SP500', 'US10Y', 'VIX', 'DXY'] if k not in frames]
    if missing:
        raise ValueError(f"Could not fetch macro data for: {missing}")

    df = pd.DataFrame(frames).dropna(how='all').ffill().dropna()
    return df


def get_latest_macro_features(feature_version: int = 1) -> pd.DataFrame:
    """
    Convenience function: fetches macro data, engineers features,
    and returns the latest row ready for model inference.

    Args:
        feature_version: 1 for original 12 features (existing model),
                        2 for expanded 16 features (retrained model)
    """
    df = fetch_macro_data(period="2y")
    df = build_macro_features(df)
    df = df.dropna()

    if df.empty:
        raise ValueError("Not enough historical data for macro features.")

    feature_names = MACRO_FEATURE_NAMES_V1 if feature_version == 1 else MACRO_FEATURE_NAMES
    return df[feature_names].iloc[-1:]
