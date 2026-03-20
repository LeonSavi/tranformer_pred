# utils/scaler.py
import numpy as np
import pandas as pd

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