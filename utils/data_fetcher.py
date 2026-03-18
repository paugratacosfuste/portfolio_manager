import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional
import datetime
import streamlit as st


@st.cache_data(ttl=300, show_spinner=False)
def fetch_current_prices(tickers: List[str]) -> Dict[str, float]:
    prices = {}
    if not tickers:
        return prices
    try:
        data = yf.download(tickers, period="5d", progress=False, auto_adjust=True)
        if data.empty:
            return prices
        # yfinance >=1.2 always returns MultiIndex columns; flatten if needed
        if isinstance(data.columns, pd.MultiIndex):
            close_data = data['Close'].ffill()
        elif 'Close' in data.columns:
            close_data = data['Close'].ffill()
        else:
            return prices
        for ticker in tickers:
            try:
                if isinstance(close_data, pd.Series):
                    price = float(close_data.iloc[-1])
                elif ticker in close_data.columns:
                    price = float(close_data[ticker].iloc[-1])
                elif len(tickers) == 1:
                    price = float(close_data.iloc[-1, 0]) if hasattr(close_data, 'iloc') else float('nan')
                else:
                    price = float('nan')
            except Exception:
                price = float('nan')
            prices[ticker] = 0.0 if pd.isna(price) else price
    except Exception as e:
        print(f"Error fetching current prices: {e}")
    return prices

@st.cache_data(ttl=300, show_spinner=False)
def fetch_historical_data(tickers: List[str], period: str = "1y") -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    try:
        data = yf.download(tickers, period=period, progress=False, auto_adjust=True)
        if data.empty:
            return pd.DataFrame()
        # yfinance >=1.2 always returns MultiIndex columns; extract Close
        if isinstance(data.columns, pd.MultiIndex):
            df = data['Close']
        elif 'Close' in data.columns:
            df = data[['Close']].rename(columns={'Close': tickers[0]}) if len(tickers) == 1 else data['Close']
        else:
            return pd.DataFrame()
        if isinstance(df, pd.Series):
            df = df.to_frame(name=tickers[0])
        # Strip 'Ticker' level name from columns to match expected flat format
        df.columns.name = None
        return df
    except Exception as e:
        print(f"Error fetching historical data: {e}")
    return pd.DataFrame()

def fetch_recent_news(tickers: List[str], limit: int = 10) -> Dict[str, List[Dict]]:
    news_dict = {}
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            news = t.news[:limit] if hasattr(t, 'news') else []
            summarized_news = []
            for item in news:
                content = item.get('content', {})
                title = item.get('title') or content.get('title', 'No Title')
                publisher = item.get('publisher') or content.get('provider', {}).get('displayName', 'Unknown')
                link = item.get('link') or content.get('clickThroughUrl', {}).get('url', '#') or '#'
                pub_time = item.get('providerPublishTime', 0)
                if pub_time:
                    timestamp = datetime.datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d %H:%M')
                else:
                    pub_date = content.get('pubDate', '')
                    timestamp = pub_date[:16] if pub_date else 'Unknown'
                summarized_news.append({
                    "title": title,
                    "link": link,
                    "publisher": publisher,
                    "timestamp": timestamp
                })
            news_dict[ticker] = summarized_news
        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")
            news_dict[ticker] = []
    return news_dict

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_asset_metadata(tickers: List[str]) -> Dict[str, Dict]:
    meta = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            
            quote_type = info.get("quoteType", "")
            category = info.get("category", "") or ""
            
            # Asset Class
            if quote_type == "CRYPTOCURRENCY":
                asset_class = "Crypto"
            elif "Bond" in category or "Fixed Income" in category or quote_type == "INDEX":
                asset_class = "Bonds/Cash"
            else:
                asset_class = "Stocks"
                
            # Sector
            sector = info.get("sector", None)
            if not sector:
                if asset_class == "Crypto":
                    sector = "Digital Assets"
                elif asset_class == "Bonds/Cash":
                    sector = "Fixed Income"
                else:
                    sector = "Broad Market ETF"
            
            # Region
            region = info.get("country", None)
            if not region:
                if asset_class == "Crypto":
                    region = "Global/Decentralized"
                else:
                    region = "United States"
            elif region == "United States":
                region = "United States"
            else:
                region = "International"
                
            meta[ticker] = {
                "asset_class": asset_class,
                "sector": sector,
                "region": region
            }
        except Exception as e:
            print(f"Warning: Could not fetch metadata for {ticker}: {e}")
            meta[ticker] = {"asset_class": "Stocks", "sector": "Other", "region": "United States"}
            
    return meta
