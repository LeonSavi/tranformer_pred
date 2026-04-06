# utils/scaler.py
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler, StandardScaler
import pickle

# Which macro cols get which scaler
ROBUST_MACRO_COLS   = ['VIX', 'RVOL_VIX', 'GPRT', 'HY_IG_spread',
                        'HY_OAS_diff1', 'HY_OAS_diff5', 'GOLD']
STANDARD_MACRO_COLS = ['DXY', 'DXY_diff1', 'DXY_diff5', 'T10Y2Y',
                        'T10Y3M', 'VIX_term_slope', 'FF_surprise']
PASSTHROUGH_MACRO   = ['Fear_Greed', 'yield_curve_inverted']  # already [0,1] or binary

class CryptoRollingScaler:
    def __init__(self, window: int = 90, eps: float = 1e-8):
        self.window = window
        self.eps = eps

    def fit_transform(
        self,
        df: pd.DataFrame,
        price_cols: list,
        indicator_cols: list,
    ) -> pd.DataFrame:

        df = df.copy()

        # ── ensure timestamp is a column, not just the index ─────────────────
        if 'timestamp' not in df.columns:
            df = df.reset_index()           # moves named index → column

        df = df.sort_values(['tic', 'timestamp']).reset_index(drop=True)

        scaled_chunks = []

        for tic, group in df.groupby('tic', sort=False):
            group = group.copy().sort_values('timestamp').reset_index(drop=True)

            for col in price_cols:
                if col not in group.columns:
                    continue
                rolling = group[col].shift(1).rolling(
                    window=self.window, min_periods=5
                )
                mean = rolling.mean()
                std  = rolling.std().clip(lower=self.eps).fillna(self.eps)

                group[f'scale_mean_{col}'] = mean
                group[f'scale_std_{col}']  = std
                group[col] = (group[col] - mean) / std

            group[indicator_cols] = (
                group[indicator_cols]
                .ffill()
                .fillna(0)
            )

            # drop warm-up rows where rolling stats are unreliable
            group = group.iloc[self.window:]
            scaled_chunks.append(group)

        if not scaled_chunks:
            raise ValueError('No data survived scaling — check window size vs ticker length.')

        result = pd.concat(scaled_chunks, ignore_index=True)
        print(f'Scaler output: {result.shape} | tickers: {result.tic.nunique()}')
        return result

    def inverse_transform(
        self,
        predictions: np.ndarray,
        scale_means: np.ndarray,
        scale_stds:  np.ndarray,
    ) -> np.ndarray:
        return predictions * scale_stds[:, None] + scale_means[:, None]
    
    def fit_transform_macro(
        self,
        df: pd.DataFrame,
        cutoff_date: str,           # fit only on rows before this date
    ) -> pd.DataFrame:
        """
        Fit macro scalers on train split, transform the full DataFrame.
        Must be called AFTER fit_transform (price scaling).
        """
        df = df.copy()
        train_mask = df['timestamp'] < pd.to_datetime(cutoff_date)

        self.macro_scalers_ = {}

        for col in ROBUST_MACRO_COLS:
            if col not in df.columns:
                continue
            scaler = RobustScaler()
            scaler.fit(df.loc[train_mask, [col]].fillna(0))
            df[col] = scaler.transform(df[[col]].fillna(0))
            self.macro_scalers_[col] = scaler

        for col in STANDARD_MACRO_COLS:
            if col not in df.columns:
                continue
            scaler = StandardScaler()
            scaler.fit(df.loc[train_mask, [col]].fillna(0))
            df[col] = scaler.transform(df[[col]].fillna(0))
            self.macro_scalers_[col] = scaler

        # passthrough cols need no scaling — already normalized in cleaner
        print(f'  Macro scalers fitted on {train_mask.sum()} train rows, '
            f'transformed {len(df)} total rows')
        return df

    def save_macro_scalers(self, path: str = 'outputs/macro_scalers.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self.macro_scalers_, f)
        print(f'  Macro scalers saved to {path}')