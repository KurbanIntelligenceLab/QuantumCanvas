TRAINING_CONFIG = {
    'train_ratio': 0.8,
    'val_ratio': 0.2,
    'batch_size': 32,
    'epochs': 100,
    'lr': 1e-4,
    'weight_decay': 0.0,
    'optimizer': 'adam',
    'scheduler': 'reduce_on_plateau',
    'scheduler_factor': 0.8,
    'scheduler_patience': 10,
    'scheduler_min_lr': 1e-6,
    'early_stopping_patience': 20,
    'loss': 'mse',
    'num_workers': 0,
}

DATASET_CONFIG = {
    'base_dir': './data/CrysMTM',
    'train_temps': [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000],
    'max_rotations': None,
    'modalities': ['xyz', 'element'],
    'as_pyg_data': True,
}

TARGET_PROPERTIES = [
    'HOMO',
    'LUMO',

]

SEEDS = [42, 123, 456]

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

TWOBODY_TARGET_MAP = {
    'HOMO': 'e_homo_ev',
    'LUMO': 'e_lumo_ev',

}

def get_checkpoint_path(model_type: str, target_name: str, seed: int):

    from pathlib import Path

    twobody_target = TWOBODY_TARGET_MAP.get(target_name)

    if twobody_target is None:
        print(f"Warning: No two-body mapping found for CrysMTM target '{target_name}'")
        return None

    checkpoint_path = Path(f'results_twobody/{twobody_target}/{model_type}/seed_{seed}/best_model.pt')

    if not checkpoint_path.exists():
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return None

    return checkpoint_path

