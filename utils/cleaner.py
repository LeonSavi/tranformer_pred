import torch
import pandas as pd
import numpy as np
from utils.scaler import CryptoRollingScaler

COLUMNS = ['close', 'volume', 'tic', 'rsi', 'macd', 'cci', 'dx', 'roc', 'ultosc',
           'willr', 'obv', 'ht_dcphase', 'atr', 'natr', 'bb_width', 'ema_cross',
           'candle_body', 'upper_wick', 'lower_wick']

PRICE_COLS      = ['close', 'volume', 'obv']
INDICATOR_COLS  = ['rsi', 'macd', 'cci', 'dx', 'roc', 'ultosc', 'willr',
                   'ht_dcphase', 'atr', 'natr', 'bb_width', 'ema_cross',
                   'candle_body', 'upper_wick', 'lower_wick']
NUMERIC_COLS    = [col for col in COLUMNS if col != 'tic']


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

        interpolated = (prev_row[NUMERIC_COLS] + next_row[NUMERIC_COLS]) / 2
        local_std    = neighbors[NUMERIC_COLS].std().fillna(0)
        noise        = pd.Series(np.random.normal(0, local_std * 0.1), index=NUMERIC_COLS)

        filled_row        = interpolated + noise
        filled_row['tic'] = tic
        filled_row.name   = missing_date
        return filled_row[COLUMNS]

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
        data = data.reset_index()   # timestamp → column, index becomes 0,1,2...
        data = data.rename(columns={'index': 'timestamp'}) if 'timestamp' not in data.columns else data

        self.raw_data = data
        return data

    def run_scaler(self, data: pd.DataFrame = None) -> pd.DataFrame:
        data = self.raw_data if data is None else data

        if data is None:
            raise RuntimeError('No data found — run run_cleaner() first.')

        scaled = self.scaler.fit_transform(
            data,
            price_cols=PRICE_COLS,
            indicator_cols=INDICATOR_COLS,
        )

        self.scaled_data = scaled
        return scaled

    def run(self) -> pd.DataFrame:
        """Convenience method: clean then scale in one call."""
        self.run_cleaner()
        return self.run_scaler()