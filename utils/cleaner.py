import torch
import pandas as pd
import numpy as np
from utils.scaler import CryptoRollingScaler

MARKET_INDICATORS_PATH = 'data/market_indicators.csv'

MARKET_INDICATOR_COLS = [
    'VIX', 'HY_OAS_diff1', 'HY_OAS_diff5', 'DXY', 'DXY_diff1', 'DXY_diff5',
    'T10Y2Y', 'Fear_Greed', 'RVOL_VIX', 'VIX_term_slope', 'HY_IG_spread',
    'T10Y3M', 'GOLD', 'yield_curve_inverted', 'GPRT', 'FF_surprise',
]

PKL_COLS = ['close', 'volume', 'tic', 'rsi', 'macd', 'cci', 'dx', 'roc', 'ultosc',
            'willr', 'obv', 'ht_dcphase', 'atr', 'natr', 'bb_width', 'ema_cross',
            'candle_body', 'upper_wick', 'lower_wick', 'sentiment_index']

# Post-merge columns (full set after market indicators are joined)
COLUMNS      = PKL_COLS + MARKET_INDICATOR_COLS

PRICE_COLS      = ['close', 'volume', 'obv']
INDICATOR_COLS  = ['rsi', 'macd', 'cci', 'dx', 'roc', 'ultosc', 'willr',
                   'ht_dcphase', 'atr', 'natr', 'bb_width', 'ema_cross',
                   'candle_body', 'upper_wick', 'lower_wick', 'sentiment_index']


PRICE_AND_INDICATOR_COLS = [
    'close', 'volume', 'rsi', 'macd', 'cci', 'dx', 'roc', 'ultosc',
    'willr', 'obv', 'ht_dcphase', 'atr', 'natr', 'bb_width', 'ema_cross',
    'candle_body', 'upper_wick', 'lower_wick', 'sentiment_index'
]

NUMERIC_COLS = PRICE_AND_INDICATOR_COLS


class CleanerTS:

    def __init__(self,
                 dir: str = 'data/train_data_1d.pkl',
                 window: int = 7,           # bug fix: was `windows` → `window`
                 scaler_window: int = 90):

        self.window        = window
        self.scaler_window = scaler_window
        self.directory     = dir
        self.raw_data      = None           # populated after run_cleaner()
        self.scaled_data   = None           # populated after run_scaler()
        self.scaler        = CryptoRollingScaler(window=scaler_window)

    # ── private ───────────────────────────────────────────────────────────────

    def _fix_row(self, df: pd.DataFrame, missing_date: pd.Timestamp, tic: str):
        window_before = df[df.index < missing_date].tail(self.window)
        window_after  = df[df.index > missing_date].head(self.window)
        neighbors     = pd.concat([window_before, window_after]).drop_duplicates()

        if neighbors.empty:
            return None

        prev_row = window_before.iloc[-1] if not window_before.empty else window_after.iloc[0]
        next_row = window_after.iloc[0]   if not window_after.empty  else window_before.iloc[-1]

        # use PRICE_AND_INDICATOR_COLS — macro cols don't exist yet at this stage
        interpolated = (prev_row[PRICE_AND_INDICATOR_COLS] + next_row[PRICE_AND_INDICATOR_COLS]) / 2
        local_std    = neighbors[PRICE_AND_INDICATOR_COLS].std().fillna(0)
        noise        = pd.Series(np.random.normal(0, local_std * 0.1), index=PRICE_AND_INDICATOR_COLS)

        filled_row        = interpolated + noise
        filled_row['tic'] = tic
        filled_row.name   = missing_date

        # return only original cols — macro cols will be joined later via merge
        original_cols = PRICE_AND_INDICATOR_COLS + ['tic']
        return filled_row[original_cols]
    

    def _load_market_indicators(self) -> pd.DataFrame:
        mi = pd.read_csv(MARKET_INDICATORS_PATH)
        # fix the Unnamed: 0 date column
        mi = mi.rename(columns={'Unnamed: 0': 'timestamp'})
        mi['timestamp'] = pd.to_datetime(mi['timestamp'])
        mi = mi.set_index('timestamp')
        # normalize Fear_Greed to [0,1] — already bounded, no fitting needed
        if 'Fear_Greed' in mi.columns:
            mi['Fear_Greed'] = mi['Fear_Greed'] / 100.0
        return mi

    # ── public ────────────────────────────────────────────────────────────────

    def run_cleaner(self) -> pd.DataFrame:
        data = pd.read_pickle(self.directory).dropna()
        data.index = data.index.normalize()         # strips 01:00 / 02:00 DST shifts
        
        data = (
        data.reset_index()
            .drop_duplicates(subset=['timestamp', 'tic'])
            .set_index('timestamp')
         )

        new_rows = []
        n_issues = 0

        for tic in data.tic.unique():
            check   = data[data.tic == tic].copy()
            dt_set  = set(check.index)
            mn_idx  = min(dt_set)
            mx_idx  = max(dt_set)
            curr    = mn_idx

            while curr <= mx_idx:                   
                if curr not in dt_set:
                    # print(f'Missing for {tic}: {curr}')
                    fixed_row = self._fix_row(check, curr, tic)  
                    if fixed_row is not None:
                        new_rows.append(fixed_row) 
                        n_issues += 1
                curr += pd.Timedelta(days=1)

        if new_rows:                                
            data = pd.concat(
                [data, pd.DataFrame(new_rows)]
            ).sort_index()

        print(f'Solved {n_issues} issues')
        data = data.reset_index()
        data = data.rename(columns={'index': 'timestamp'}) if 'timestamp' not in data.columns else data

        # ── merge market indicators ───────────────────────────────────────────
        mi = self._load_market_indicators()
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.merge(mi, on='timestamp', how='left')

        # ffill weekends/holidays where macro data is stale, then fill remaining
        for col in MARKET_INDICATOR_COLS:
            if col in data.columns:
                data[col] = data[col].ffill().bfill().fillna(0)

        self.raw_data = data
        return data
    
    
    def run_scaler(self, data: pd.DataFrame = None, cutoff_date: str = '2025-01-01') -> pd.DataFrame:
        data = self.raw_data if data is None else data
        if data is None:
            raise RuntimeError('No data found — run run_cleaner() first.')

        # 1. rolling price scaler (existing)
        scaled = self.scaler.fit_transform(
            data,
            price_cols=PRICE_COLS,
            indicator_cols=INDICATOR_COLS,
        )

        # 2. macro scaler (new) — fitted on train split only
        scaled = self.scaler.fit_transform_macro(scaled, cutoff_date=cutoff_date)
        self.scaler.save_macro_scalers()

        self.scaled_data = scaled
        return scaled

    # also update run() to pass cutoff_date through
    def run(self, cutoff_date: str = '2025-01-01') -> pd.DataFrame:
        self.run_cleaner()
        return self.run_scaler(cutoff_date=cutoff_date)