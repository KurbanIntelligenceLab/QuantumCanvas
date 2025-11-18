"""
Standardized configuration for QM9 experiments
Ensures fair comparison across all models
"""

# Training hyperparameters (identical for all models)
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 50,
    'lr_scratch': 1e-4,          # Learning rate for training from scratch
    'lr_finetune': 1e-5,         # Learning rate for fine-tuning (lower to avoid catastrophic forgetting)
    'weight_decay': 0.0,
    'optimizer': 'adam',
    'scheduler': 'reduce_on_plateau',
    'scheduler_factor': 0.8,
    'scheduler_patience': 10,
    'scheduler_min_lr': 1e-6,
    'early_stopping_patience': 30,
    'loss': 'mae',  # L1Loss
    'num_workers': 4,
}

# QM9 dataset split (standard split)
QM9_SPLIT = {
    'train_size': 110000,
    'val_size': 10000,
    'test_size': 10000,  # Remaining molecules
}

# QM9 target properties
QM9_TARGETS = {
    'homo': 2,   # HOMO energy (eV)
    'lumo': 3,   # LUMO energy (eV)
    'gap': 4,    # HOMO-LUMO gap (eV)
}

# Seeds for reproducibility
SEEDS = [42, 123, 456]

# Model architectures (MUST MATCH two-body checkpoint parameters for transfer learning!)
MODEL_CONFIGS = {
    'schnet': {
        'hidden_channels': 96,      # From two-body checkpoint
        'num_filters': 96,          # From two-body checkpoint
        'num_interactions': 6,      # From two-body checkpoint
        'num_gaussians': 50,        # From two-body checkpoint
        'cutoff': 5.0,
        'readout': 'add'
    },
    'faenet': {
        'cutoff': 5.0,
        'hidden_channels': 128,     # From two-body checkpoint
        'num_filters': 128,         # From two-body checkpoint
        'num_interactions': 4,      # From two-body checkpoint
        'num_gaussians': 8,
        'max_num_neighbors': 20,
        'tag_hidden_channels': 8,
        'pg_hidden_channels': 8,
        'phys_hidden_channels': 0,
        'phys_embeds': False,
        'act': 'silu',
        'preprocess': 'base_preprocess',
        'complex_mp': False,
        'mp_type': 'base',
        'graph_norm': True,
        'second_layer_MLP': False,
        'skip_co': 'add',
        'energy_head': None,
        'regress_forces': None,
        'force_decoder_type': 'mlp',
    },
    'gotennet': {
        'n_atom_basis': 64,         # From two-body checkpoint
        'n_interactions': 3,        # From two-body checkpoint
        'cutoff': 5.0,
        'num_heads': 2,
        'n_rbf': 10,                # From two-body checkpoint
    }
}

# Two-body checkpoint mapping
TWOBODY_TARGET_MAP = {
    'homo': 'e_homo_ev',
    'lumo': 'e_lumo_ev',
    'gap': 'e_g_ev'
}


def get_checkpoint_path(model_type: str, target_name: str, seed: int):
    """Get two-body checkpoint path for transfer learning"""
    from pathlib import Path
    
    if target_name not in TWOBODY_TARGET_MAP:
        return None
    
    twobody_target = TWOBODY_TARGET_MAP[target_name]
    checkpoint_path = Path(f'results_twobody/{twobody_target}/{model_type}/seed_{seed}/best_model.pt')
    
    return checkpoint_path if checkpoint_path.exists() else None

