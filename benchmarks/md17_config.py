"""
Standardized configuration for MD17 experiments
Ensures fair comparison across all models
"""

# Training hyperparameters (identical for all models)
TRAINING_CONFIG = {
    'train_size': 950,
    'val_size': 50,
    # test_size is remaining molecules
    'batch_size': 32,
    'epochs': 50,
    'lr_scratch': 1e-4,          # Learning rate for training from scratch
    'lr_finetune': 1e-5,         # Learning rate for fine-tuning
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

# MD17 molecules to benchmark
MD17_MOLECULES = ['benzene', 'aspirin', 'ethanol']

# Seeds for reproducibility
SEEDS = [42, 123, 456]

# Model architectures (MUST MATCH two-body checkpoint parameters for transfer learning!)
MODEL_CONFIGS = {
    'schnet': {
        'hidden_channels': 96,
        'num_filters': 96,
        'num_interactions': 6,
        'num_gaussians': 50,
        'cutoff': 5.0,
        'readout': 'add'
    },
    'faenet': {
        'cutoff': 5.0,
        'hidden_channels': 128,
        'num_filters': 128,
        'num_interactions': 4,
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
        'n_atom_basis': 64,
        'n_interactions': 3,
        'cutoff': 5.0,
        'num_heads': 2,
        'n_rbf': 10,
    }
}

# Two-body checkpoint mapping (MD17 predicts energy, map to closest two-body target)
# Using total_energy as the most relevant for MD17 energy prediction
TWOBODY_TARGET_MAP = {
    'energy': 'total_energy_ev',  # MD17 energy → two-body total energy
}


def get_checkpoint_path(model_type: str, molecule: str, seed: int):
    """Get two-body checkpoint path for transfer learning
    
    For MD17, we use total_energy checkpoints as the source
    since MD17 is about energy prediction
    """
    from pathlib import Path
    
    twobody_target = TWOBODY_TARGET_MAP['energy']
    checkpoint_path = Path(f'results_twobody/{twobody_target}/{model_type}/seed_{seed}/best_model.pt')
    
    return checkpoint_path if checkpoint_path.exists() else None

