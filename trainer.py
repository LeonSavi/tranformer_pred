# main.py
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.cleaner import CleanerTS
from scripts.tft_arch import prepare_tft_dataset, train_tft
from pytorch_forecasting.data.encoders import TorchNormalizer
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss

torch.set_float32_matmul_precision('high')

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH          = 'data/train_data_1d.pkl'
MODEL_PATH         = 'outputs/tft_crypto.ckpt'
RESULTS_PATH       = 'outputs/backtest_results.csv'
MAX_ENCODER_LENGTH = 90
MAX_PRED_LENGTH    = 7
SCALER_WINDOW      = 90
CUTOFF_DATE        = '2025-06-01'   # train before / validate+backtest after


# ── MODEL SETTINGS ────────────────────────────────────────────────────────────────────

SETTINGS = {
    # dataloader
    'batch_size'                : 512,    # 256 → 1024, fills VRAM properly
    'num_workers'               : 12,      # 8 → 12

    # model
    'learning_rate'             : 0.00075,  # lower LR with bigger batch (linear scaling rule)
    'hidden_size'               : 256,     # 128 → 256
    'lstm_layers'               : 3,       # 2 → 3
    'dropout'                   : 0.2,     # keep
    'attention_head_size'       : 8,       # 4 → 8
    'hidden_continuous_size'    : 128,     # 64 → 128

    'reduce_on_plateau_patience': 3,
    'max_epochs'                : 100,
    'gradient_clip_val'         : 0.1,
    'early_stopping_patience'   : 10,

    'precision'                 : 'bf16-mixed',
    'accumulate_grad_batches'   : 1
}

# ── 1. Clean & Scale ──────────────────────────────────────────────────────────
def get_data() -> pd.DataFrame:
    cleaner = CleanerTS(
        dir=DATA_PATH,
        window=7,
        scaler_window=SCALER_WINDOW,
    )
    scaled_df = cleaner.run()           # clean → scale in one call
    print(f'Data ready: {scaled_df.shape}  |  tickers: {scaled_df.tic.nunique()}')
    return scaled_df, cleaner.scaler    # return scaler for inverse transform


# ── 2. Train ──────────────────────────────────────────────────────────────────
def run_training(scaled_df: pd.DataFrame):
    training, validation, prep_df = prepare_tft_dataset(
        scaled_df,
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PRED_LENGTH,
        cutoff_date=CUTOFF_DATE,
    )
    tft = train_tft(training, validation,SETTINGS)
    os.makedirs('outputs', exist_ok=True)
    tft.trainer.save_checkpoint(MODEL_PATH)
    torch.save(tft.state_dict(), 'outputs/tft_weights.pth')
    return tft, training, validation, prep_df 


# ── 3. Backtest ───────────────────────────────────────────────────────────────
def run_backtest(tft, validation, prep_df: pd.DataFrame, scaler) -> pd.DataFrame:
    """
    prep_df is the output of prepare_tft_dataset() — it has time_idx,
    scale_mean_close, scale_std_close columns.
    """
    val_dl = validation.to_dataloader(train=False, batch_size=64, num_workers=4)

    raw_preds = tft.predict(val_dl, mode='quantiles', return_index=True)
    predictions = raw_preds.output
    index       = raw_preds.index

    records = []
    for i, row in index.iterrows():
        tic      = row['tic']
        time_idx = row['time_idx']

        origin = prep_df[
            (prep_df['tic'] == tic) &
            (prep_df['time_idx'] == time_idx)
        ]
        if origin.empty:
            continue

        scale_mean = origin['scale_mean_close'].values[0]
        scale_std  = origin['scale_std_close'].values[0]
        preds      = predictions[i]

        actuals = prep_df[
            (prep_df['tic'] == tic) &
            (prep_df['time_idx'] > time_idx) &
            (prep_df['time_idx'] <= time_idx + MAX_PRED_LENGTH)
        ]['close'].values

        for h in range(min(MAX_PRED_LENGTH, len(actuals))):
            p10, p50, p90 = preds[h].tolist()
            records.append({
                'tic'           : tic,
                'time_idx'      : time_idx + h + 1,
                'actual_scaled' : actuals[h],
                'p50_scaled'    : p50,
                'actual'        : actuals[h] * scale_std + scale_mean,
                'p10'           : p10        * scale_std + scale_mean,
                'p50'           : p50        * scale_std + scale_mean,
                'p90'           : p90        * scale_std + scale_mean,
            })

    results = pd.DataFrame(records)
    results.to_csv(RESULTS_PATH, index=False)
    print(f'Backtest saved → {RESULTS_PATH}')
    return results


# ── 4. Metrics ────────────────────────────────────────────────────────────────
def compute_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """MAE, RMSE and quantile coverage per ticker."""
    rows = []
    for tic, grp in results.groupby('tic'):
        mae  = (grp['actual'] - grp['p50']).abs().mean()
        rmse = ((grp['actual'] - grp['p50']) ** 2).mean() ** 0.5
        # % of actuals that fall inside the P10-P90 interval
        coverage = ((grp['actual'] >= grp['p10']) &
                    (grp['actual'] <= grp['p90'])).mean()
        rows.append({
            'tic'      : tic,
            'mae'      : round(mae, 4),
            'rmse'     : round(rmse, 4),
            'p10_p90_coverage' : round(coverage, 3),
        })

    metrics = pd.DataFrame(rows).sort_values('mae')
    print('\n── Backtest Metrics (top 10 by MAE) ──')
    print(metrics.head(10).to_string(index=False))
    return metrics


# ── 5. Plot ───────────────────────────────────────────────────────────────────
def plot_backtest(results: pd.DataFrame, tics: list[str] = None, n_tics: int = 4):
    tics = tics or results['tic'].unique()[:n_tics]
    fig, axes = plt.subplots(len(tics), 1, figsize=(14, 4 * len(tics)))
    if len(tics) == 1:
        axes = [axes]

    for ax, tic in zip(axes, tics):
        grp = results[results['tic'] == tic].sort_values('time_idx')
        ax.fill_between(grp['time_idx'], grp['p10'], grp['p90'],
                        alpha=0.2, label='P10-P90', color='steelblue')
        ax.plot(grp['time_idx'], grp['p50'],    label='P50',   color='steelblue')
        ax.plot(grp['time_idx'], grp['actual'], label='Actual', color='black', lw=0.8)
        ax.set_title(tic)
        ax.legend()

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/backtest_plot.png', dpi=150)
    print('Plot saved → outputs/backtest_plot.png')
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    scaled_df, scaler = get_data()

    # scaled_df = scaled_df.sample(n=50000)

    if os.path.exists(MODEL_PATH):
        training, validation, prep_df = prepare_tft_dataset(
            scaled_df,
            max_encoder_length=MAX_ENCODER_LENGTH,
            max_prediction_length=MAX_PRED_LENGTH,
            cutoff_date=CUTOFF_DATE,
        )
        tft = TemporalFusionTransformer.load_from_checkpoint(
            MODEL_PATH,
            map_location='cuda' if torch.cuda.is_available() else 'cpu',
        )
    else:
        tft, _, validation, prep_df = run_training(scaled_df)

    # ── pass prep_df (has time_idx) not scaled_df ────────────────────────────
    results = run_backtest(tft, validation, prep_df, scaler)
    metrics  = compute_metrics(results)
    plot_backtest(results, n_tics=10)