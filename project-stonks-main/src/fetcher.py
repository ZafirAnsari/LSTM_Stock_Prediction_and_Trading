
import pandas as pd
import yfinance as yf
from datetime import date
import time
from talib import abstract
from tqdm import tqdm

# Input Start and End Date
# start = date.today() - timedelta(weeks=104*1)
start = ""
print(start)
end = date.today()
print(end)
t0 = time.time()

# create empty dataframe
stock_final = pd.DataFrame()
tickers = ["AA"]

# iterate over each symbol
EMA = abstract.Function('ema')
MACD = abstract.Function('macd')
STDDEV = abstract.Function('stddev')
AD = abstract.Function('ad')
NATR = abstract.Function('natr')

# Close, 'Volume', 'EMA', 'MACD','STDDEV', 'AD', 'ADOSC','NATR'
vix = yf.download("^VIX", period="max", progress=True)
vix = vix.rename(columns={"Close": "Volatility"})
vix = vix[["Volatility"]]

for i in tqdm(tickers):

    # print the symbol which is being downloaded
    print(str(tickers.index(i)) + str(' : ') + i, sep=',', end=',', flush=True)

    # download the stock price
    stock = []
    stock = yf.download(i, interval="1d", progress = True)

    # stock = yf.download(i, start=f"{start}", end=f"{end}", interval="1d", progress = True)

    # append the individual stock prices
    if len(stock) == 0:
        None
    else:
        stock = pd.merge(stock, vix, on="Date")
        calc_ema = EMA(stock["Close"], timeperiod=26)
        calc_macd = MACD(stock["Close"])
        calc_stddev = STDDEV(stock["Close"])
        calc_ad = AD(stock["High"],stock["Low"],stock["Close"], stock["Volume"])
        calc_natr = NATR(stock["High"],stock["Low"],stock["Close"])
        stock["EMA"] = calc_ema
        stock["MACD"] = calc_macd[0]
        stock["STDDEV"] = calc_stddev
        stock["AD"] = calc_ad
        stock["NATR"] = calc_natr
        # stock["Target"] = stock["Close"].shift(periods=-7) # 7 time periods ago

    stock.insert(loc=0, column='Name', value=i)
    stock.insert(loc=1, column='Date', value=stock.index)
    stock.drop(columns=["Volume", "Close"],inplace=True)
    stock.rename(columns={"Adj Close": "close", "Name": "ticker"}, inplace=True)
    stock.rename(columns=dict(zip(stock.columns, [col.lower() for col in stock.columns])), inplace=True)
    stock.reset_index(drop=True, inplace=True)
    stock["close"] = round(stock["close"], 2)
    stock["open"] = round(stock["open"], 2)
    stock["high"] = round(stock["high"], 2)
    stock["low"] = round(stock["low"], 2)
    print(stock.columns)

    stock_final = stock_final.append(stock, sort=False)

t1 = time.time()

total = t1 - t0

print(stock_final.tail(30))
print(str(round(total, 2)) + " seconds")
stock_final.to_csv('../YFinanceStockData/AA.csv', index=False)
