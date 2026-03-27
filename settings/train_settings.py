# ── MODEL SETTINGS ────────────────────────────────────────────────────────────────────
SETTINGS = {
    # dataloader
    'batch_size'                : 1024,    # 256 → 1024, fills VRAM properly
    'num_workers'               : 12,      # 8 → 12

    # model
    'learning_rate_hf'          : 0.0001,
    'learning_rate'             : 0.00075,  # lower LR with bigger batch (linear scaling rule)
    'hidden_size'               : 128,    
    'lstm_layers'               : 3,      
    'dropout'                   : 0.3,    
    'attention_head_size'       : 4,     
    'hidden_continuous_size'    : 64,    

    'reduce_on_plateau_patience': 3,
    'max_epochs'                : 100,
    'gradient_clip_val'         : 0.1,
    'early_stopping_patience'   : 10,

    'precision'                 : 'bf16-mixed',
    'accumulate_grad_batches'   : 1
}


# ── UPDATED HF-TST SETTINGS ──────────────────────────────────────────────────────────
SETTINGS_TST = {
    'batch_size'                : 1024,
    'num_workers'               : 12,

    'model_type'                : 'huggingface', 
    'd_model'                   : 128,      
    'encoder_layers'            : 4,        
    'decoder_layers'            : 4,        
    'attention_heads'           : 8,       
    'd_ff'                      : 512,      
    'activation_function'       : 'gelu',   
    
    'learning_rate'          : 0.0001,  
    'dropout'                   : 0.3,     
    'max_epochs'                : 100,
    'early_stopping_patience'   : 12,       
    'precision'                 : 'bf16-mixed',

    'gradient_clip_val'         :0.1,
    'reduce_on_plateau_patience': 3
}