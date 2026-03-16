from binance.client import Client

TICKER_CONFIGS = {
        'start_date': '1 Jan, 2011',
        'end_date':   '15 March, 2026',
        'time_interval': Client.KLINE_INTERVAL_4HOUR,
        'tech_indicator_list': None,   # None = auto-select via correlation analysis
        'correlation_threshold': 0.9,
        'vix': False,
    }