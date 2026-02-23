import yfinance as yf
tickers = ["SPY", "BND", "QQQ"]
for t in tickers:
    tick = yf.Ticker(t)
    info = tick.info
    print(t, "-> quoteType:", info.get("quoteType"), " category:", info.get("category"))
