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

from huggingface_hub import HfApi, login
from utils.API import HF_TOKEN
from settings import *

torch.set_float32_matmul_precision('high')

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH          = 'data/train_data_1d.pkl'
MODEL_PATH         = 'outputs/tft_crypto.ckpt'
RESULTS_PATH       = 'outputs/backtest_results.csv'
MAX_ENCODER_LENGTH = 90
MAX_PRED_LENGTH    = 7
SCALER_WINDOW      = 90
CUTOFF_DATE        = '2025-12-01'   # train before / validate+backtest after

REPO_ID            = 'LeoSavi/TFT_Crypto'



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

def load_to_hf():

    login(token=HF_TOKEN)
    api = HfApi()

    api.upload_file(
        path_or_fileobj='outputs/tft_weights.pth',
        path_in_repo="TFT_model.pth",
        repo_id=REPO_ID,
        commit_message="Update model"
    )

    api.upload_file(
        path_or_fileobj=MODEL_PATH,
        path_in_repo="TFT_model.ckpt",
        repo_id=REPO_ID,
        commit_message="Update model"
    )

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':

    scaled_df, scaler = get_data()

    tft, _, validation, prep_df = run_training(scaled_df)

    load_to_hf()