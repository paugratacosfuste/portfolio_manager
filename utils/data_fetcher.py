import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional
import datetime

def fetch_current_prices(tickers: List[str]) -> Dict[str, float]:
    prices = {}
    if not tickers:
        return prices
    try:
        data = yf.download(tickers, period="5d", progress=False)
        if 'Close' in data:
            close_data = data['Close'].ffill()
            for ticker in tickers:
                if len(tickers) == 1:
                    price = float(close_data.iloc[-1])
                else:
                    if ticker in close_data.columns:
                        price = float(close_data[ticker].iloc[-1])
                    else:
                        price = float('nan')
                if pd.isna(price):
                    prices[ticker] = 0.0
                else:
                    prices[ticker] = price
    except Exception as e:
        print(f"Error fetching current prices: {e}")
    return prices

def fetch_historical_data(tickers: List[str], period: str = "1y") -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    try:
        data = yf.download(tickers, period=period, progress=False)
        if 'Adj Close' in data:
            df = data['Adj Close']
            if isinstance(df, pd.Series):
                df = df.to_frame(name=tickers[0])
            return df
        elif 'Close' in data:
            df = data['Close']
            if isinstance(df, pd.Series):
                df = df.to_frame(name=tickers[0])
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
        except:
            meta[ticker] = {"asset_class": "Stocks", "sector": "Other", "region": "United States"}
            
    return meta
