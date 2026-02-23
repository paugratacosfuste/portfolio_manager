import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional
import datetime

def fetch_current_prices(tickers: List[str]) -> Dict[str, float]:
    """
    Fetches the most recent closing prices for a list of tickers.
    Returns a dictionary mapping ticker to price.
    """
    prices = {}
    if not tickers:
        return prices
    
    try:
        data = yf.download(tickers, period="1d", progress=False)
        if 'Close' in data:
            for ticker in tickers:
                if len(tickers) == 1:
                    price = float(data['Close'].iloc[-1])
                else:
                    price = float(data['Close'][ticker].iloc[-1])
                prices[ticker] = price
    except Exception as e:
        print(f"Error fetching current prices: {e}")
    
    return prices

def fetch_historical_data(tickers: List[str], period: str = "1y") -> pd.DataFrame:
    """
    Fetches historical adjusted close prices for a list of tickers over a given period.
    Period can be "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max".
    """
    if not tickers:
        return pd.DataFrame()
    
    try:
        data = yf.download(tickers, period=period, progress=False)
        if 'Adj Close' in data:
            df = data['Adj Close']
            # If only one ticker, yf returns a Series, convert to DataFrame
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

def fetch_recent_news(tickers: List[str], limit: int = 5) -> Dict[str, List[Dict]]:
    """
    Fetches recent news headlines for a list of tickers.
    Returns a dictionary mapping ticker to a list of news articles (dict with title, link, publisher).
    """
    news_dict = {}
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            news = t.news[:limit] if hasattr(t, 'news') else []
            summarized_news = []
            for item in news:
                summarized_news.append({
                    "title": item.get('title', 'No Title'),
                    "link": item.get('link', '#'),
                    "publisher": item.get('publisher', 'Unknown'),
                    "timestamp": datetime.datetime.fromtimestamp(item.get('providerPublishTime', 0)).strftime('%Y-%m-%d %H:%M')
                })
            news_dict[ticker] = summarized_news
        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")
            news_dict[ticker] = []
            
    return news_dict
