import pandas as pd
import glob
import os
import yfinance

PATH = 'sentiment_data'


lst = glob.glob(PATH+ '/*.csv')

# to change if new memecoins or coins
MEMECOINS = {
    'DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', 'FLOKIUSDT', 'BONKUSDT',
    'BOMEUSDT', 'WIFUSDT', '1MBABYDOGEUSDT', 'MEMEUSDT', 'DOGSUSDT',
    'NEIROUSDT', 'TURBOUSDT', 'PNUTUSDT', 'LUNCUSDT', 'USTCUSDT',
    '1000SATSUSDT', '1000CATUSDT', '1000CHEEMSUSDT', 'BABYUSDT',
    'BANANAUSDT', 'BROCCOLI714USDT', 'BANANAS31USDT', 'GIGGLEUSDT',
    'TURTLEUSDT', 'HMSTRUSDT', 'NOTUSDT', 'TRUMPUSDT', 'PEOPLEUSDT',
    'SPELLUSDT', 'ACTUSDT', 'SUNUSDT', 'WINUSDT', 'HOTUSDT',
    'XVGUSDT', 'FUNUSDT', 'COSUSDT', 'DENTUSDT', 'ACHUSDT',
    'SLPUSDT', 'CVCUSDT',
}

NAME_TO_TICKER = {
    'bitcoin': 'BTCUSDT',
    'ethereum': 'ETHUSDT',
    'ripple': 'XRPUSDT',
    'litecoin': 'LTCUSDT',
    'cardano': 'ADAUSDT',
    'polkadot': 'DOTUSDT',
    'chainlink': 'LINKUSDT',
    'stellar': 'XLMUSDT',
    'dogecoin': 'DOGEUSDT',
    'shiba_inu': 'SHIBUSDT',
    'solana': 'SOLUSDT',
    'avalanche': 'AVAXUSDT',
    'polygon': 'MATICUSDT',
    'cosmos': 'ATOMUSDT',
    'algorand': 'ALGOUSDT',
    'vechain': 'VETUSDT',
    'hedera': 'HBARUSDT',
    'uniswap': 'UNIUSDT',
    'tron': 'TRXUSDT',
    'neo': 'NEOUSDT',
    'iota': 'IOTAUSDT',
    'zcash': 'ZECUSDT',
    'dash': 'DASHUSDT',
    'nano': 'XNOUSDT',
    'bitcoin_cash': 'BCHUSDT',
    'digibyte': 'DGBUSDT',
    'binance_coin': 'BNBUSDT',
    'zilliqa': 'ZILUSDT',
}

data = []
for filepath in glob.glob(PATH + '/*.csv'):
    # extract just the crypto name from the filename
    name = os.path.basename(filepath).removesuffix('.csv').removeprefix('augmento_sentiment_indicator_')

    ticker = NAME_TO_TICKER.get(name)
    if ticker and ticker not in MEMECOINS:
        file = pd.read_csv(filepath)
        file['TICKER'] = ticker
        data.append(file[['datetime','score_combined_pct','TICKER']])

df = pd.concat(data, ignore_index=True, axis=0)

import yfinance as yf

# Map Binance tickers to Yahoo Finance symbols
TICKER_TO_YF = {
    'BTCUSDT': 'BTC-USD',
    'ETHUSDT': 'ETH-USD',
    'XRPUSDT': 'XRP-USD',
    'LTCUSDT': 'LTC-USD',
    'ADAUSDT': 'ADA-USD',
    'DOTUSDT': 'DOT-USD',
    'LINKUSDT': 'LINK-USD',
    'XLMUSDT': 'XLM-USD',
    'SOLUSDT': 'SOL-USD',
    'AVAXUSDT': 'AVAX-USD',
    'MATICUSDT': 'MATIC-USD',
    'ATOMUSDT': 'ATOM-USD',
    'ALGOUSDT': 'ALGO-USD',
    'VETUSDT': 'VET-USD',
    'HBARUSDT': 'HBAR-USD',
    'UNIUSDT': 'UNI-USD',
    'TRXUSDT': 'TRX-USD',
    'NEOUSDT': 'NEO-USD',
    'IOTAUSDT': 'IOTA-USD',
    'ZECUSDT': 'ZEC-USD',
    'DASHUSDT': 'DASH-USD',
    'XNOUSDT': 'XNO-USD',
    'BCHUSDT': 'BCH-USD',
    'DGBUSDT': 'DGB-USD',
    'BNBUSDT': 'BNB-USD',
    'ZILUSDT': 'ZIL-USD',
}

# Fetch market caps
market_caps = {}
for ticker, yf_symbol in TICKER_TO_YF.items():
    try:
        info = yf.Ticker(yf_symbol).info
        mc = info.get('marketCap')
        if mc:
            market_caps[ticker] = mc
    except Exception as e:
        print(f"Failed for {yf_symbol}: {e}")

mc_df = pd.DataFrame(
    list(market_caps.items()), columns=['TICKER', 'market_cap']
)

df = df.merge(mc_df, on='TICKER', how='left')
df = df.dropna(subset=['market_cap'])

df['datetime'] = pd.to_datetime(df['datetime']).dt.date

def weighted_avg(group):
    weights = group['market_cap']
    values = group['score_combined_pct']
    return (weights * values).sum() / weights.sum()

index = (
    df.groupby('datetime')
    .apply(weighted_avg)
    .rename('sentiment_index')
    .reset_index()
)


index.to_csv('data/sentiment_index.csv')