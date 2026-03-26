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