### SCRIPTS TO SELECT THE TICKERS WE WANT ###

from utils.binance_data import BinanceData
from binance.client import Client
from pathlib import Path
from settings.tickers_config import TICKER_CONFIGS

import os

BASE_CURRENCY = 'USDT' # most common in binance
TO_EXCLUDE = ['UP', 'DOWN', 'BULL', 'BEAR'] # leverage tokens offered by Binance

def save_tickers(tickers: list, filepath: str = 'data/tickers.txt'):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write('\n'.join(tickers))
    print(f'Saved {len(tickers)} tickers to {filepath}')


def load_tickers(filepath:str) -> list:
    with open(filepath, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]


if __name__=='__main__':

    binance = BinanceData()

    exchange_info = binance.binance_client.get_exchange_info()

    meme_coins = load_tickers('data/meme_coins.txt')

    clean_tickers = [
        s['symbol'] for s in exchange_info['symbols']
        if s['status'] == 'TRADING'
        and s['symbol'].endswith(BASE_CURRENCY)
        and not any(kw in s['symbol'] for kw in TO_EXCLUDE)
        and s['symbol'] not in meme_coins
    ]

    save_tickers(clean_tickers)

    TICKER_CONFIGS['ticker_list'] = clean_tickers

    binance.run(TICKER_CONFIGS)

