import pandas as pd
import yfinance as yf
tickers = ['TSLA','BND','SPY']
start_date = '2015-07-01'
end_date = '2025-07-31'

data = {}
for t in tickers:

    df = yf.download(t, start=start_date, end=end_date, progress=False, auto_adjust=False)
    # keep necessary columns and save
    df = df[['Open','High','Low','Close','Adj Close','Volume']].reset_index()

    df.to_csv(f'data/raw/{t}.csv', index=False)
    data[t] = df


# quick check
for t in tickers:
    print(t, data[t].shape)    