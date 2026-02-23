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
        data = yf.download(tickers, period="5d", progress=False)
        if data.empty:
            return prices
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    price = float(data['Close'].dropna().iloc[-1])
                else:
                    price = float(data['Close'][ticker].dropna().iloc[-1])
                prices[ticker] = price
            except (KeyError, IndexError):
                print(f"Warning: Could not fetch price for {ticker}")
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
        if data.empty:
            return pd.DataFrame()
        # Try Adj Close first, then Close
        if 'Adj Close' in data:
            df = data['Adj Close']
        elif 'Close' in data:
            df = data['Close']
        else:
            return pd.DataFrame()
        # If only one ticker, yf returns a Series, convert to DataFrame
        if isinstance(df, pd.Series):
            df = df.to_frame(name=tickers[0])
        return df
    except Exception as e:
        print(f"Error fetching historical data: {e}")

    return pd.DataFrame()

def fetch_recent_news(tickers: List[str], limit: int = 5) -> Dict[str, List[Dict]]:
    """
    Fetches recent news headlines for a list of tickers.
    Returns a dictionary mapping ticker to a list of news articles.
    Handles both old and new yfinance API formats.
    """
    news_dict = {}
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            raw_news = []

            # Try to get news - handle different yfinance API versions
            if hasattr(t, 'news'):
                result = t.news
                if isinstance(result, list):
                    raw_news = result[:limit]
                elif isinstance(result, dict):
                    # Newer yfinance versions may return dict with nested structure
                    if 'news' in result:
                        raw_news = result['news'][:limit]
                    elif 'items' in result:
                        raw_news = result['items'][:limit]

            summarized_news = []
            for item in raw_news:
                if not isinstance(item, dict):
                    continue

                title = None
                link = '#'
                publisher = 'Unknown'
                timestamp_str = ''

                # Standard format (most yfinance versions)
                title = item.get('title') or item.get('headline')
                link = item.get('link') or item.get('url', '#')
                publisher = item.get('publisher') or item.get('source', 'Unknown')

                # Handle nested content structure (newer yfinance)
                if not title and 'content' in item:
                    content = item['content']
                    if isinstance(content, dict):
                        title = content.get('title')
                        canonical = content.get('canonicalUrl')
                        if isinstance(canonical, dict):
                            link = canonical.get('url', '#')
                        elif isinstance(canonical, str):
                            link = canonical
                        provider = content.get('provider', {})
                        if isinstance(provider, dict):
                            publisher = provider.get('displayName', 'Unknown')

                # Handle publisher if it's a dict
                if isinstance(publisher, dict):
                    publisher = publisher.get('displayName') or publisher.get('name', 'Unknown')

                # Timestamp handling
                ts = item.get('providerPublishTime') or item.get('publishedAt') or item.get('publish_time', 0)
                if isinstance(ts, (int, float)) and ts > 0:
                    timestamp_str = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')
                elif isinstance(ts, str):
                    timestamp_str = ts[:16]

                if title:
                    summarized_news.append({
                        "title": title,
                        "link": link,
                        "publisher": publisher if isinstance(publisher, str) else 'Unknown',
                        "timestamp": timestamp_str
                    })

            news_dict[ticker] = summarized_news
        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")
            news_dict[ticker] = []

    return news_dict
