# scripts/tft_arch.py
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from pytorch_forecasting.data.encoders import TorchNormalizer
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
import torch
import pandas as pd


DEFAULT_SETTINGS = {
    # dataloader
    'batch_size'                : 64,
    'num_workers'               : 4,
    # model
    'learning_rate'             : 0.03,
    'hidden_size'               : 64,
    'lstm_layers'               : 2,
    'dropout'                   : 0.1,
    'attention_head_size'       : 4,
    'hidden_continuous_size'    : 32,
    'log_interval'              : 10,
    'reduce_on_plateau_patience': 4,
    # trainer
    'max_epochs'                : 50,
    'gradient_clip_val'         : 0.1,
    'early_stopping_patience'   : 5,
}


def prepare_tft_dataset(
    flat_df: pd.DataFrame,
    max_encoder_length: int = 90,
    max_prediction_length: int = 7,
    cutoff_date: str = None,
):
    df = flat_df.copy()

    # timestamp must be a column at this point (scaler already reset_index'd)
    if 'timestamp' not in df.columns:
        df = df.reset_index()

    df = df.sort_values(['tic', 'timestamp']).reset_index(drop=True)
    df['time_idx']    = df.groupby('tic').cumcount()
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    # ── fix: must be string for TimeSeriesDataSet categorical ────────────────
    df['is_weekend'] = df['timestamp'].dt.dayofweek.ge(5).map(
        {True: 'yes', False: 'no'}
    )

    cutoff          = cutoff_date or df['timestamp'].quantile(0.8)
    training_cutoff = df[df['timestamp'] <= cutoff]['time_idx'].max()

    scale_cols = [c for c in df.columns
                  if c.startswith('scale_mean_') or c.startswith('scale_std_')]

    training = TimeSeriesDataSet(
        df[df['time_idx'] <= training_cutoff],
        time_idx='time_idx',
        target='close',
        group_ids=['tic'],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=['tic'],
        time_varying_unknown_reals=[
            'close', 'volume',
            'rsi', 'macd', 'cci', 'dx',
            'roc', 'ultosc', 'willr', 'obv', 'ht_dcphase',
        ] + scale_cols,
        time_varying_known_reals=['time_idx', 'day_of_week'],
        time_varying_known_categoricals=['is_weekend'],   # now a string column
        target_normalizer=TorchNormalizer(method='identity'),
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, df,
        min_prediction_idx=training_cutoff + 1,
        stop_randomization=True,
    )

    return training, validation, df


def train_tft(training, validation, settings: dict = {}):
    # merge: user values override defaults, missing keys fall back to defaults
    cfg = {**DEFAULT_SETTINGS, **settings}

    train_dl = training.to_dataloader(train=True,  batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])
    val_dl   = validation.to_dataloader(train=False, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=cfg['learning_rate'],
        hidden_size=cfg['hidden_size'],
        lstm_layers=cfg['lstm_layers'],
        dropout=cfg['dropout'],
        attention_head_size=cfg['attention_head_size'],
        hidden_continuous_size=cfg['hidden_continuous_size'],
        loss=QuantileLoss(),
        log_interval=cfg['log_interval'],
        reduce_on_plateau_patience=cfg['reduce_on_plateau_patience'],
    )

    trainer = pl.Trainer(
        max_epochs=cfg['max_epochs'],
        gradient_clip_val=cfg['gradient_clip_val'],
        callbacks=[EarlyStopping(monitor='val_loss', patience=cfg['early_stopping_patience'])],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    )

    trainer.fit(tft, train_dl, val_dl)
    return tft