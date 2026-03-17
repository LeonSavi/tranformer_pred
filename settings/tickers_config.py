from binance.client import Client

TICKER_CONFIGS = {
        'start_date': '1 Jan, 2017', # TS starts 17-08-2017
        'end_date':   '16 March, 2026',
        'time_interval': Client.KLINE_INTERVAL_1DAY,
        'tech_indicator_list': None,   # None = auto-select via correlation analysis
        'correlation_threshold': 0.9, # as in the paper
        'vix': False,
    }