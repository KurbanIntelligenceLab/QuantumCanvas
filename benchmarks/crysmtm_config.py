"""
Standardized configuration for CrysMTM (TiO2 Crystal Materials) experiments
Ensures fair comparison across all models
"""

# Training hyperparameters (identical for all models)
TRAINING_CONFIG = {
    'train_ratio': 0.8,  # 80% train, 20% val
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
    'loss': 'mse',  # MSE Loss
    'num_workers': 0,  # Set to 0 for Windows compatibility
}

# Dataset configuration
DATASET_CONFIG = {
    'base_dir': './data/CrysMTM',
    'train_temps': [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000],
    'max_rotations': None,  # Use all available rotations
    'modalities': ['xyz', 'element'],  # For PyG compatibility
    'as_pyg_data': True,
}

# Target properties to predict
# Order matches the dataset labels: [HOMO, LUMO, Eg, Ef, Et, Eta, disp, vol, bond]
# Limited to HOMO, LUMO, and Eg for focused training
TARGET_PROPERTIES = [
    'HOMO',    # 0: Highest Occupied Molecular Orbital (eV)
    'LUMO',    # 1: Lowest Unoccupied Molecular Orbital (eV)
    # 'Eg',      # 2: Band gap (eV)
]

# Full list of available properties (for reference):
# ALL_PROPERTIES = ['HOMO', 'LUMO', 'Eg', 'Ef', 'Et', 'Eta', 'disp', 'vol', 'bond']

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
    'gotennet': {
        'n_atom_basis': 64,
        'n_interactions': 3,
        'cutoff': 5.0,
        'num_heads': 2,
        'n_rbf': 10,
    }
}

# Two-body checkpoint mapping (for transfer learning)
# Map CrysMTM targets to closest two-body targets
TWOBODY_TARGET_MAP = {
    'HOMO': 'e_homo_ev',
    'LUMO': 'e_lumo_ev',
    # 'Eg': 'e_g_ev',
}

# Mappings for additional properties (if needed in future):
# 'Ef': 'mu_ev',  # Chemical potential is similar to Fermi level
# 'Et': 'total_energy_ev',
# 'Eta': 'eta_ev',  # Chemical hardness
# 'disp': 'distance_ang',  # Geometric property
# 'vol': None,  # No direct mapping
# 'bond': 'distance_ang',  # Geometric property


def get_checkpoint_path(model_type: str, target_name: str, seed: int):
    """Get two-body checkpoint path for transfer learning
    
    For CrysMTM, we map targets to corresponding two-body targets:
    - HOMO -> e_homo_ev
    - LUMO -> e_lumo_ev  
    - Eg -> e_g_ev
    
    Args:
        model_type: 'schnet' or 'gotennet'
        target_name: CrysMTM target property (e.g., 'HOMO', 'LUMO', 'Eg')
        seed: Random seed (42, 123, or 456)
    
    Returns:
        Path to checkpoint if exists, None otherwise
    """
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

