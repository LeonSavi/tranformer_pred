from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from pytorch_forecasting.data import GroupNormalizer
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
import torch
import pandas as pd

# https://pytorch-forecasting.readthedocs.io/en/v1.0.0/api/pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer.html
# NOT TESTED

def prepare_tft_dataset(flat_df: pd.DataFrame,
                         max_encoder_length: int = 168,   # e.g. 1 week of hourly data
                         max_prediction_length: int = 24, # predict next 24 hours
                         cutoff_date: str = None):
    
    df = flat_df.copy().reset_index()
    
    # TFT needs an integer time index per group
    df = df.sort_values(['tic', 'timestamp'])
    df['time_idx'] = df.groupby('tic').cumcount()

    # Known future features — calendar info you always know ahead of time
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    cutoff = cutoff_date or df['timestamp'].quantile(0.8)
    training_cutoff = df[df['timestamp'] <= cutoff]['time_idx'].max()

    training = TimeSeriesDataSet(
        df[df['time_idx'] <= training_cutoff],
        time_idx='time_idx',
        target='close',               # what you're predicting
        group_ids=['tic'],            # one time series per ticker

        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,

        # Static — constant per ticker (no fundamentals in crypto,
        # but ticker identity itself is useful)
        static_categoricals=['tic'],

        # Time-varying but UNKNOWN in future (you won't know these ahead of time)
        time_varying_unknown_reals=[
            'close', 'volume',
            'rsi', 'macd', 'cci', 'dx',
            'roc', 'ultosc', 'willr', 'obv', 'ht_dcphase'
        ],

        # Time-varying and KNOWN in future (calendar features)
        time_varying_known_reals=['time_idx', 'hour', 'day_of_week'],
        time_varying_known_categoricals=['is_weekend'],

        # Normalize each ticker independently — critical for crypto
        # since BTC and an altcoin differ in magnitude by orders of magnitude
        target_normalizer=GroupNormalizer(groups=['tic'], transformation='softplus'),

        allow_missing_timesteps=True,  # crypto has gaps on some exchanges
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, df,
        min_prediction_idx=training_cutoff + 1,
        stop_randomization=True
    )

    return training, validation


def train_tft(training, validation):

    train_dl = training.to_dataloader(train=True, batch_size=64, num_workers=4)
    val_dl   = validation.to_dataloader(train=False, batch_size=64, num_workers=4)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=64,           # main capacity knob — start small for crypto
        lstm_layers=2,
        dropout=0.1,
        attention_head_size=4,
        hidden_continuous_size=32,
        loss=QuantileLoss(),      # gives you P10/P50/P90 predictions
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    trainer = pl.Trainer(
        max_epochs=50,
        gradient_clip_val=0.1,    # important for crypto — gradients can explode
        callbacks=[EarlyStopping(monitor='val_loss', patience=5)],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    )

    trainer.fit(tft, train_dl, val_dl)
    return tft