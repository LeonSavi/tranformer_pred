# analysis.py — TFT vs HF Time Series Transformer: full comparison
# ─────────────────────────────────────────────────────────────────────────────
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download

from utils.cleaner import CleanerTS
from scripts.tft_arch import prepare_tft_dataset
from pytorch_forecasting import TemporalFusionTransformer
from transformers import (
    TimeSeriesTransformerConfig,
    TimeSeriesTransformerForPrediction,
)
from settings.train_settings import SETTINGS

torch.set_float32_matmul_precision('high')
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.facecolor': 'white',
})

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH          = 'data/train_data_1d.pkl'
MAX_ENCODER_LENGTH = 90
MAX_PRED_LENGTH    = 7
SCALER_WINDOW      = 90
CUTOFF_DATE        = '2025-01-01'

# HuggingFace repos
TFT_REPO_ID  = 'LeoSavi/TFT_Crypto'
TFT_FILENAME = 'tft_crypto.ckpt'
HF_REPO_ID   = 'LeoSavi/HF_TST_Crypto'
HF_FILENAME  = 'hf_ts_transformer.pt'

# local paths
MODEL_DIR = 'outputs/models'
PLOT_DIR  = 'outputs/plots'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

COVARIATE_COLS = [
    'volume', 'rsi', 'macd', 'cci', 'dx', 'roc',
    'ultosc', 'willr', 'obv', 'ht_dcphase',
    'atr', 'natr', 'bb_width', 'ema_cross',
    'candle_body', 'upper_wick', 'lower_wick',
    'day_of_week','sentiment_index',
]

# ═════════════════════════════════════════════════════════════════════════════
#  1. DATA
# ═════════════════════════════════════════════════════════════════════════════
def get_data():
    cleaner = CleanerTS(
        dir=DATA_PATH,
        window=7,
        scaler_window=SCALER_WINDOW,
    )
    scaled_df = cleaner.run()
    print(f'Data ready: {scaled_df.shape}  |  tickers: {scaled_df.tic.nunique()}')

    training, validation, prep_df = prepare_tft_dataset(
        scaled_df,
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PRED_LENGTH,
        cutoff_date=CUTOFF_DATE,
    )
    cutoff_time_idx = prep_df[
        prep_df['timestamp'] <= CUTOFF_DATE
    ]['time_idx'].max()

    return training, validation, prep_df, cutoff_time_idx


# ═════════════════════════════════════════════════════════════════════════════
#  2. DOWNLOAD MODELS FROM HUGGINGFACE
# ═════════════════════════════════════════════════════════════════════════════
def download_models():
    print('Downloading TFT checkpoint...')
    tft_path = hf_hub_download(
        repo_id=TFT_REPO_ID,
        filename=TFT_FILENAME,
        local_dir=MODEL_DIR,
    )
    print('Downloading HF-TST weights...')
    hf_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        local_dir=MODEL_DIR,
    )
    print(f'  TFT  → {tft_path}')
    print(f'  HF   → {hf_path}')
    return tft_path, hf_path


def load_tft(tft_path):
    tft = TemporalFusionTransformer.load_from_checkpoint(
        tft_path, map_location=DEVICE,
    )
    tft.eval()
    print(f'TFT loaded  |  params: {sum(p.numel() for p in tft.parameters()):,}')
    return tft


def load_hf_model(hf_path, n_time_features):
    config = TimeSeriesTransformerConfig(
        prediction_length=MAX_PRED_LENGTH,
        context_length=MAX_ENCODER_LENGTH,
        input_size=1,
        num_time_features=n_time_features,
        d_model=SETTINGS['hidden_size'],
        encoder_layers=SETTINGS['lstm_layers'],
        decoder_layers=SETTINGS['lstm_layers'],
        encoder_attention_heads=SETTINGS['attention_head_size'],
        decoder_attention_heads=SETTINGS['attention_head_size'],
        encoder_ffn_dim=SETTINGS['hidden_size'] * 2,
        decoder_ffn_dim=SETTINGS['hidden_size'] * 2,
        dropout=SETTINGS['dropout'],
        distribution_output='student_t',
        num_parallel_samples=200,
        lags_sequence=[1, 2, 3, 4, 5, 6, 7],
    )
    model = TimeSeriesTransformerForPrediction(config).to(DEVICE)
    model.load_state_dict(torch.load(hf_path, map_location=DEVICE))
    model.eval()
    print(f'HF-TST loaded  |  params: {sum(p.numel() for p in model.parameters()):,}')
    return model


# ═════════════════════════════════════════════════════════════════════════════
#  3. HF DATASET + COLLATE
# ═════════════════════════════════════════════════════════════════════════════
class CryptoTimeSeriesDataset(Dataset):
    """Sliding-window dataset that mirrors the TFT's train/val split."""

    def __init__(
        self,
        df: pd.DataFrame,
        context_length: int,
        prediction_length: int,
        covariate_cols: list[str],
    ):
        self.context_length    = context_length
        self.prediction_length = prediction_length
        self.samples           = []

        total_len = context_length + prediction_length

        for tic, grp in df.groupby('tic'):
            grp = grp.sort_values('time_idx').reset_index(drop=True)
            n   = len(grp)

            target = grp['close'].values.astype(np.float32)
            covs   = grp[covariate_cols].values.astype(np.float32)
            means  = grp['scale_mean_close'].values.astype(np.float32)   # ← missing
            stds   = grp['scale_std_close'].values.astype(np.float32)    # ← missing
            tidxs  = grp['time_idx'].values
            MAX_LAG = 7

            for start in range(n - total_len - MAX_LAG + 1):
                end_ctx  = start + context_length + MAX_LAG
                end_pred = end_ctx + prediction_length

                self.samples.append({
                    'past_values'          : target[start:end_ctx],
                    'past_time_features'   : covs[start:end_ctx],
                    'past_observed_mask'   : np.ones(context_length + MAX_LAG, dtype=np.float32),
                    'future_values'        : target[end_ctx:end_pred],
                    'future_time_features' : covs[end_ctx:end_pred],
                    'origin_time_idx'      : int(tidxs[end_ctx - 1]),
                    'tic'                  : tic,                          # ← missing
                    'scale_mean'           : float(means[end_ctx - 1]),    # ← missing
                    'scale_std'            : float(stds[end_ctx - 1]),     # ← missing
                })
                
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'past_values'          : torch.tensor(s['past_values']),
            'past_time_features'   : torch.tensor(s['past_time_features']),
            'past_observed_mask'   : torch.tensor(s['past_observed_mask']),
            'future_values'        : torch.tensor(s['future_values']),
            'future_time_features' : torch.tensor(s['future_time_features']),
            'tic'                  : s['tic'],
            'origin_time_idx'      : s['origin_time_idx'],
            'scale_mean'           : s['scale_mean'],
            'scale_std'            : s['scale_std'],
        }
        


def _collate(batch):
    return {
        'past_values'          : torch.stack([b['past_values'] for b in batch]),
        'past_time_features'   : torch.stack([b['past_time_features'] for b in batch]),
        'past_observed_mask'   : torch.stack([b['past_observed_mask'] for b in batch]),
        'future_values'        : torch.stack([b['future_values'] for b in batch]),
        'future_time_features' : torch.stack([b['future_time_features'] for b in batch]),
        'tic'                  : [b['tic'] for b in batch],
        'origin_time_idx'      : [b['origin_time_idx'] for b in batch],
        'scale_mean'           : [b['scale_mean'] for b in batch],
        'scale_std'            : [b['scale_std'] for b in batch],
    }


# ═════════════════════════════════════════════════════════════════════════════
#  4. BACKTESTS
# ═════════════════════════════════════════════════════════════════════════════
def run_tft_backtest(tft, validation, prep_df):
    val_dl = validation.to_dataloader(train=False, batch_size=64, num_workers=4)

    raw_preds   = tft.predict(val_dl, mode='quantiles', return_index=True)
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
                'horizon'       : h + 1,
                'actual_scaled' : actuals[h],
                'p50_scaled'    : p50,
                'actual'        : actuals[h] * scale_std + scale_mean,
                'p10'           : p10        * scale_std + scale_mean,
                'p50'           : p50        * scale_std + scale_mean,
                'p90'           : p90        * scale_std + scale_mean,
            })

    df = pd.DataFrame(records)
    print(f'TFT backtest: {len(df):,} predictions')
    return df


def run_hf_backtest(model, prep_df, cutoff_time_idx):
    covariate_cols = [c for c in COVARIATE_COLS if c in prep_df.columns]

    ds = CryptoTimeSeriesDataset(
        prep_df, MAX_ENCODER_LENGTH, MAX_PRED_LENGTH, covariate_cols,
    )
    ds.samples = [
        s for s in ds.samples if s['origin_time_idx'] >= cutoff_time_idx
    ]

    dl = DataLoader(
        ds, batch_size=128, shuffle=False,
        num_workers=4, collate_fn=_collate,
    )

    records = []
    with torch.no_grad():
        for batch in dl:
            out = model.generate(
                past_values=batch['past_values'].to(DEVICE),
                past_time_features=batch['past_time_features'].to(DEVICE),
                past_observed_mask=batch['past_observed_mask'].to(DEVICE),
                future_time_features=batch['future_time_features'].to(DEVICE),
            )
            samples = out.sequences.cpu().numpy()

            p10 = np.percentile(samples, 10, axis=1)
            p50 = np.percentile(samples, 50, axis=1)
            p90 = np.percentile(samples, 90, axis=1)

            future_vals = batch['future_values'].numpy()
            tickers     = batch['tic']
            origins     = batch['origin_time_idx']
            means       = batch['scale_mean']
            stds        = batch['scale_std']

            for j in range(len(tickers)):
                sm, ss = means[j], stds[j]
                for h in range(MAX_PRED_LENGTH):
                    records.append({
                        'tic'           : tickers[j],
                        'time_idx'      : origins[j] + h + 1,
                        'horizon'       : h + 1,
                        'actual_scaled' : future_vals[j, h],
                        'p50_scaled'    : p50[j, h],
                        'actual'        : future_vals[j, h] * ss + sm,
                        'p10'           : p10[j, h]         * ss + sm,
                        'p50'           : p50[j, h]         * ss + sm,
                        'p90'           : p90[j, h]         * ss + sm,
                    })

    df = pd.DataFrame(records)
    print(f'HF-TST backtest: {len(df):,} predictions')
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  5. METRICS
# ═════════════════════════════════════════════════════════════════════════════
def compute_metrics(results, label=''):
    rows = []
    for tic, grp in results.groupby('tic'):
        err = grp['actual'] - grp['p50']
        mae  = err.abs().mean()
        rmse = (err ** 2).mean() ** 0.5
        coverage = ((grp['actual'] >= grp['p10']) &
                    (grp['actual'] <= grp['p90'])).mean()

        actual_diff = grp['actual_scaled'].diff()
        pred_diff   = grp['p50_scaled'].diff()
        valid       = actual_diff.notna() & pred_diff.notna()
        dir_acc = ((actual_diff[valid] > 0) == (pred_diff[valid] > 0)).mean() \
                  if valid.sum() > 0 else np.nan

        rows.append({
            'tic': tic, 'mae': round(mae, 4), 'rmse': round(rmse, 4),
            'p10_p90_coverage': round(coverage, 3),
            'directional_acc': round(dir_acc, 3),
        })

    metrics = pd.DataFrame(rows).sort_values('mae')
    if label:
        print(f'\n── {label}  (top 10 by MAE) ──')
        print(metrics.head(10).to_string(index=False))
    return metrics


def compute_horizon_metrics(results):
    rows = []
    for h, grp in results.groupby('horizon'):
        err = grp['actual'] - grp['p50']
        rows.append({
            'horizon': int(h),
            'mae': err.abs().mean(),
            'rmse': (err ** 2).mean() ** 0.5,
            'p10_p90_coverage': ((grp['actual'] >= grp['p10']) &
                                  (grp['actual'] <= grp['p90'])).mean(),
        })
    return pd.DataFrame(rows)


def pinball_loss(results):
    losses = {}
    for q, col in [(0.1, 'p10'), (0.5, 'p50'), (0.9, 'p90')]:
        err = results['actual'] - results[col]
        losses[col] = float((q * err.clip(lower=0) + (1 - q) * (-err).clip(lower=0)).mean())
    losses['mean'] = np.mean([losses['p10'], losses['p50'], losses['p90']])
    return losses


def diebold_mariano_test(res_a, res_b, label_a='TFT', label_b='HF-TST'):
    merged = res_a.merge(res_b, on=['tic', 'time_idx', 'horizon'], suffixes=('_a', '_b'))
    e_a = (merged['actual_a'] - merged['p50_a']) ** 2
    e_b = (merged['actual_b'] - merged['p50_b']) ** 2
    d   = e_a - e_b

    d_bar   = d.mean()
    se      = d.std() / np.sqrt(len(d))
    dm_stat = d_bar / se
    p_value = 2 * stats.norm.sf(abs(dm_stat))
    winner  = label_a if d_bar < 0 else label_b

    print(f'\n── Diebold-Mariano Test ──')
    print(f'  DM stat  : {dm_stat:.4f}')
    print(f'  p-value  : {p_value:.6f}')
    print(f'  Winner   : {winner}')
    return {'dm_stat': dm_stat, 'p_value': p_value, 'winner': winner}


# ═════════════════════════════════════════════════════════════════════════════
#  6. PLOTS
# ═════════════════════════════════════════════════════════════════════════════

def plot_t1_comparison(tft_res, hf_res, prep_df, tics=None, n_tics=4):
    """Main paper figure — t+1: actual vs TFT vs HF-TST."""
    tft_t1 = tft_res[tft_res['horizon'] == 1]
    hf_t1  = hf_res[hf_res['horizon'] == 1]
    tics   = tics or tft_t1['tic'].unique()[:n_tics]

    fig, axes = plt.subplots(len(tics), 1, figsize=(14, 3.5 * len(tics)))
    if len(tics) == 1:
        axes = [axes]

    for ax, tic in zip(axes, tics):
        date_map = (
            prep_df[prep_df['tic'] == tic][['time_idx', 'timestamp']]
            .drop_duplicates().set_index('time_idx')['timestamp']
        )
        t = tft_t1[tft_t1['tic'] == tic].sort_values('time_idx')
        h = hf_t1[hf_t1['tic'] == tic].sort_values('time_idx')
        td = t['time_idx'].map(date_map)
        hd = h['time_idx'].map(date_map)

        ax.fill_between(td, t['p10'].values, t['p90'].values,
                        alpha=0.12, color='#1f77b4', label='TFT P10–P90')
        ax.plot(td, t['actual'].values, color='black', lw=1.2, label='Actual')
        ax.plot(td, t['p50'].values, color='#1f77b4', lw=1.0, label='TFT (t+1)')
        ax.plot(hd, h['p50'].values, color='#d62728', lw=1.0, ls='--', label='HF-TST (t+1)')

        ax.set_title(tic, fontweight='bold')
        ax.legend(loc='upper left', framealpha=0.8)
        ax.set_ylabel('Price ($)')
        ax.tick_params(axis='x', rotation=30)

    axes[-1].set_xlabel('Date')
    fig.suptitle('t+1 Forecast — TFT vs Time Series Transformer',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(f'{PLOT_DIR}/t1_comparison.png', bbox_inches='tight')
    print(f'Saved → {PLOT_DIR}/t1_comparison.png')
    plt.show()


def plot_horizon_degradation(tft_h, hf_h):
    """MAE / RMSE / Coverage vs horizon."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col, label in zip(
        axes,
        ['mae', 'rmse', 'p10_p90_coverage'],
        ['MAE', 'RMSE', 'P10–P90 Coverage'],
    ):
        ax.plot(tft_h['horizon'], tft_h[col], 'o-', color='#1f77b4', label='TFT', lw=1.5)
        ax.plot(hf_h['horizon'],  hf_h[col],  's--', color='#d62728', label='HF-TST', lw=1.5)
        ax.set_xlabel('Horizon (days)')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()
        ax.set_xticks(range(1, MAX_PRED_LENGTH + 1))

    fig.suptitle('Accuracy by Forecast Horizon', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(f'{PLOT_DIR}/horizon_degradation.png', bbox_inches='tight')
    print(f'Saved → {PLOT_DIR}/horizon_degradation.png')
    plt.show()


def plot_ticker_scatter(tft_m, hf_m):
    """Each dot = one ticker.  Below diagonal → TFT wins."""
    merged = tft_m.merge(hf_m, on='tic', suffixes=('_tft', '_hf'))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(merged['mae_tft'], merged['mae_hf'],
               alpha=0.5, edgecolors='k', linewidths=0.3, s=40)

    lim = max(merged['mae_tft'].max(), merged['mae_hf'].max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', lw=0.8, label='Equal')
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel('TFT — MAE')
    ax.set_ylabel('HF-TST — MAE')
    ax.set_title('Per-Ticker MAE', fontweight='bold')
    ax.set_aspect('equal')
    ax.legend()

    n_tft = (merged['mae_tft'] < merged['mae_hf']).sum()
    ax.text(0.05, 0.92, f'TFT wins {n_tft}/{len(merged)} tickers',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig.savefig(f'{PLOT_DIR}/ticker_scatter.png', bbox_inches='tight')
    print(f'Saved → {PLOT_DIR}/ticker_scatter.png')
    plt.show()


def plot_volatility_breakdown(tft_res, hf_res, prep_df):
    """MAE split by high-vol / low-vol using NATR."""
    if 'natr' not in prep_df.columns:
        print('  natr not found — skipping volatility plot')
        return

    tmp = prep_df.copy()
    tmp['vol_smooth'] = tmp.groupby('tic')['natr'].transform(
        lambda x: x.rolling(14, min_periods=1).mean()
    )
    median_vol = tmp['vol_smooth'].median()
    tmp['regime'] = np.where(tmp['vol_smooth'] >= median_vol, 'High Vol', 'Low Vol')
    rmap = tmp[['tic', 'time_idx', 'regime']].drop_duplicates()

    rows = []
    for label, res in [('TFT', tft_res), ('HF-TST', hf_res)]:
        m = res.merge(rmap, on=['tic', 'time_idx'], how='left')
        for regime, grp in m.groupby('regime'):
            rows.append({'model': label, 'regime': regime,
                         'mae': (grp['actual'] - grp['p50']).abs().mean()})

    vdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(2)
    w = 0.3
    for i, (mdl, clr) in enumerate([('TFT', '#1f77b4'), ('HF-TST', '#d62728')]):
        vals = vdf[vdf['model'] == mdl].sort_values('regime')['mae'].values
        ax.bar(x + i * w, vals, w, label=mdl, color=clr)

    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(['High Vol', 'Low Vol'])
    ax.set_ylabel('MAE')
    ax.set_title('MAE by Volatility Regime', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    fig.savefig(f'{PLOT_DIR}/volatility_breakdown.png', bbox_inches='tight')
    print(f'Saved → {PLOT_DIR}/volatility_breakdown.png')
    plt.show()


def plot_aggregate_comparison(tft_m, hf_m):
    """Side-by-side bars: MAE, RMSE, Coverage, Dir Acc."""
    agg_tft = tft_m[['mae', 'rmse', 'p10_p90_coverage', 'directional_acc']].mean()
    agg_hf  = hf_m[['mae', 'rmse', 'p10_p90_coverage', 'directional_acc']].mean()
    labels  = ['MAE', 'RMSE', 'P10-P90\nCoverage', 'Directional\nAccuracy']

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, lbl, tv, hv in zip(axes, labels, agg_tft.values, agg_hf.values):
        bars = ax.bar(['TFT', 'HF-TST'], [tv, hv],
                      color=['#1f77b4', '#d62728'], width=0.5)
        ax.set_title(lbl, fontweight='bold')
        ax.bar_label(bars, fmt='%.3f', fontsize=9)

    fig.suptitle('Aggregate Metrics', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(f'{PLOT_DIR}/aggregate_comparison.png', bbox_inches='tight')
    print(f'Saved → {PLOT_DIR}/aggregate_comparison.png')
    plt.show()


def save_summary_table(tft_m, hf_m, tft_h, hf_h, tft_pb, hf_pb, dm):
    """CSV table ready for the paper."""
    at = tft_m[['mae', 'rmse', 'p10_p90_coverage', 'directional_acc']].mean()
    ah = hf_m[['mae', 'rmse', 'p10_p90_coverage', 'directional_acc']].mean()

    summary = pd.DataFrame({
        'Metric': [
            'MAE', 'RMSE', 'P10–P90 Coverage', 'Directional Accuracy',
            'Pinball Loss', 't+1 MAE', 't+7 MAE',
            'DM Statistic', 'DM p-value',
        ],
        'TFT': [
            f"{at['mae']:.4f}", f"{at['rmse']:.4f}",
            f"{at['p10_p90_coverage']:.3f}", f"{at['directional_acc']:.3f}",
            f"{tft_pb['mean']:.4f}",
            f"{tft_h[tft_h['horizon']==1]['mae'].values[0]:.4f}",
            f"{tft_h[tft_h['horizon']==7]['mae'].values[0]:.4f}",
            f"{dm['dm_stat']:.4f}", f"{dm['p_value']:.6f}",
        ],
        'HF-TST': [
            f"{ah['mae']:.4f}", f"{ah['rmse']:.4f}",
            f"{ah['p10_p90_coverage']:.3f}", f"{ah['directional_acc']:.3f}",
            f"{hf_pb['mean']:.4f}",
            f"{hf_h[hf_h['horizon']==1]['mae'].values[0]:.4f}",
            f"{hf_h[hf_h['horizon']==7]['mae'].values[0]:.4f}",
            '—', '—',
        ],
    })

    path = f'{PLOT_DIR}/summary_table.csv'
    summary.to_csv(path, index=False)
    print(f'\n── Summary Table ──')
    print(summary.to_string(index=False))
    print(f'Saved → {path}')
    return summary


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':

    # 1 — data
    training, validation, prep_df, cutoff_time_idx = get_data()

    # 2 — download both models from HuggingFace
    tft_path, hf_path = download_models()

    # 3 — load models
    covariate_cols = [c for c in COVARIATE_COLS if c in prep_df.columns]
    tft      = load_tft(tft_path)
    hf_model = load_hf_model(hf_path, n_time_features=len(covariate_cols))

    # 4 — backtests (same validation window)
    print('\n══ Running Backtests ══')
    tft_results = run_tft_backtest(tft, validation, prep_df)
    hf_results  = run_hf_backtest(hf_model, prep_df, cutoff_time_idx)

    tft_results.to_csv('outputs/tft_backtest.csv', index=False)
    hf_results.to_csv('outputs/hf_backtest.csv', index=False)

    # 5 — metrics
    print('\n══ Metrics ══')
    tft_metrics = compute_metrics(tft_results, label='TFT')
    hf_metrics  = compute_metrics(hf_results,  label='HF-TST')

    tft_horizon = compute_horizon_metrics(tft_results)
    hf_horizon  = compute_horizon_metrics(hf_results)

    tft_pinball = pinball_loss(tft_results)
    hf_pinball  = pinball_loss(hf_results)
    print(f'\nPinball — TFT:    {tft_pinball}')
    print(f'Pinball — HF-TST: {hf_pinball}')

    dm = diebold_mariano_test(tft_results, hf_results)

    # 6 — plots
    print('\n══ Plots ══')
    plot_t1_comparison(tft_results, hf_results, prep_df, n_tics=4)
    plot_horizon_degradation(tft_horizon, hf_horizon)
    plot_ticker_scatter(tft_metrics, hf_metrics)
    plot_volatility_breakdown(tft_results, hf_results, prep_df)
    plot_aggregate_comparison(tft_metrics, hf_metrics)
    save_summary_table(
        tft_metrics, hf_metrics,
        tft_horizon, hf_horizon,
        tft_pinball, hf_pinball, dm,
    )

    print('\n✓ Done. All outputs in outputs/plots/')