TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 50,
    'lr_scratch': 1e-4,
    'lr_finetune': 1e-5,
    'weight_decay': 0.0,
    'optimizer': 'adam',
    'scheduler': 'reduce_on_plateau',
    'scheduler_factor': 0.8,
    'scheduler_patience': 10,
    'scheduler_min_lr': 1e-6,
    'early_stopping_patience': 30,
    'loss': 'mae',
    'num_workers': 4,
}

QM9_SPLIT = {
    'train_size': 110000,
    'val_size': 10000,
    'test_size': 10000,
}

QM9_TARGETS = {
    'homo': 2,
    'lumo': 3,
    'gap': 4,
}

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

TWOBODY_TARGET_MAP = {
    'homo': 'e_homo_ev',
    'lumo': 'e_lumo_ev',
    'gap': 'e_g_ev'
}

def get_checkpoint_path(model_type: str, target_name: str, seed: int):

    from pathlib import Path

    if target_name not in TWOBODY_TARGET_MAP:
        return None

    twobody_target = TWOBODY_TARGET_MAP[target_name]
    checkpoint_path = Path(f'results_twobody/{twobody_target}/{model_type}/seed_{seed}/best_model.pt')

    return checkpoint_path if checkpoint_path.exists() else None

