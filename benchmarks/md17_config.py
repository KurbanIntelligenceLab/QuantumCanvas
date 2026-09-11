# Fine-tuning settings read by scripts/run_transfer.py.
TRAINING_CONFIG = {
    'batch_size': 32,
    'lr_scratch': 1e-4,
    'lr_finetune': 1e-5,
    'early_stopping_patience': 30,
}

MD17_MOLECULES = ['benzene', 'aspirin', 'ethanol']

MODEL_CONFIGS = {
    'schnet': {
        'hidden_channels': 96,
        'num_filters': 96,
        'num_interactions': 6,
        'num_gaussians': 50,
        'cutoff': 5.0,
        'readout': 'add'
    },
    'gotennet': {
        'n_atom_basis': 64,
        'n_interactions': 3,
        'cutoff': 5.0,
        'num_heads': 2,
        'n_rbf': 10,
    }
}

# Two-body label to pretrain on (scripts/pretrain_twobody.py --target) for each downstream target.
TWOBODY_TARGET_MAP = {
    'energy': 'total_energy_ev',
}
