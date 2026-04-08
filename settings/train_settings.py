# ── MODEL SETTINGS ────────────────────────────────────────────────────────────────────
SETTINGS = {
    'batch_size'                : 512,    
    'num_workers'               : 12,     
    'learning_rate'             : 0.00075, 
    'hidden_size'               : 128,    
    'lstm_layers'               : 3,      
    'dropout'                   : 0.3,    
    'attention_head_size'       : 16,     
    'hidden_continuous_size'    : 64,    
    'reduce_on_plateau_patience': 5,
    'max_epochs'                : 100,
    'gradient_clip_val'         : 0.1,
    'early_stopping_patience'   : 12,
    'precision'                 : 'bf16-mixed',
    'accumulate_grad_batches'   : 2
}


# ── UPDATED HF-TST SETTINGS ──────────────────────────────────────────────────────────
SETTINGS_TST = {
    'batch_size'                : 1024,
    'num_workers'               : 12,
    'model_type'                : 'huggingface',
    'd_model'                   : 128,
    'encoder_layers'            : 5,
    'decoder_layers'            : 5,
    'attention_heads'           : 8,
    'd_ff'                      : 512,
    'activation_function'       : 'gelu',
    'learning_rate'             : 0.0001,
    'dropout'                   : 0.3,
    'max_epochs'                : 100,
    'early_stopping_patience'   : 15,
    'gradient_clip_val'         : 0.1,
    'reduce_on_plateau_patience': 6,
    'reduce_on_plateau_factor'  : 0.7, 
}