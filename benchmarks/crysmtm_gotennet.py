"""
CrysMTM (TiO2) GotenNet Training Script

Automatically trains GotenNet on CrysMTM dataset for:
- 9 target properties: HOMO, LUMO, Eg, Ef, Et, Eta, disp, vol, bond
- 3 seeds: 42, 123, 456
- 2 types: from scratch AND fine-tuned (from two-body checkpoints)
- Predicts: various electronic and structural properties
"""

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from gotennet import GotenNetWrapper
from gotennet.models.components.layers import CosineCutoff
from torch_geometric.data import Data
from torch_scatter import scatter_mean
import numpy as np
import time
import argparse
from pathlib import Path
import json
from benchmarks.crysmtm_config import (
    TRAINING_CONFIG, DATASET_CONFIG, TARGET_PROPERTIES, SEEDS, MODEL_CONFIGS, get_checkpoint_path
)
from benchmarks.crysmtm.regression_dataloader import RegressionLoader
from torch.utils.data import Subset


class GotenNetRegressor(nn.Module):
    """GotenNet wrapper matching the two-body training structure"""
    def __init__(self, n_atom_basis=32, n_interactions=2, cutoff=5.0, num_heads=2, n_rbf=4):
        super().__init__()
        self.gotennet = GotenNetWrapper(
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions,
            cutoff_fn=CosineCutoff(cutoff),
            num_heads=num_heads,
            n_rbf=n_rbf
        )
        # Add a final regression layer for prediction
        self.regressor = nn.Linear(n_atom_basis, 1)
    
    def forward(self, z, pos, batch):
        # Create input dictionary for GotenNet
        data = Data(z=z, pos=pos, batch=batch)
        
        # Get GotenNet embeddings - returns tuple (h, X)
        h, X = self.gotennet(data)
        
        # Use the scalar features h for regression
        # Global mean pooling
        if batch is None:
            x = torch.mean(h, dim=0, keepdim=True)
        else:
            # Use scatter_mean for proper batch handling
            x = scatter_mean(h, batch, dim=0)
        
        # Final regression
        return self.regressor(x).squeeze()


def set_all_seeds(seed):
    """Set all random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, optimizer, device, criterion, target_index, mean, std):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass - GotenNetRegressor expects z, pos, batch
        out = model(batch.z, batch.pos, batch.batch)
        
        # Extract target
        if batch.y.dim() == 1:
            target = batch.y
        else:
            target = batch.y[:, target_index].squeeze()
        
        # Normalize targets
        targets_normalized = (target - mean) / std
        
        # Compute loss on normalized values
        loss = criterion(out.squeeze(), targets_normalized)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
    
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, criterion, target_index, mean, std):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    for batch in loader:
        batch = batch.to(device)
        
        # Forward pass - GotenNetRegressor expects z, pos, batch
        out = model(batch.z, batch.pos, batch.batch)
        
        # Extract target
        if batch.y.dim() == 1:
            target = batch.y
        else:
            target = batch.y[:, target_index].squeeze()
        
        # Normalize targets for loss computation
        targets_normalized = (target - mean) / std
        
        # Compute loss on normalized values
        loss = criterion(out.squeeze(), targets_normalized)
        total_loss += loss.item() * batch.num_graphs
        
        # Denormalize predictions for metric computation
        out_denormalized = out * std + mean
        
        predictions.append(out_denormalized.cpu().numpy())
        targets.append(target.cpu().numpy())
    
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    
    # MAE and RMSE on original scale
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    return total_loss / len(loader.dataset), mae, rmse


def split_dataset(dataset, val_ratio=0.2, seed=42):
    """Split dataset into train and validation"""
    indices = np.arange(len(dataset))
    np.random.seed(seed)
    np.random.shuffle(indices)
    split = int(len(indices) * (1 - val_ratio))
    train_idx, val_idx = indices[:split], indices[split:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def run_single_experiment(target_index: int, target_name: str, seed: int, 
                          checkpoint_path: Path = None, 
                          experiment_type: str = 'scratch',
                          batch_size: int = 32, 
                          epochs: int = 100, 
                          lr: float = 1e-4,
                          weight_decay: float = 0.0):
    """Run a single training experiment
    
    Args:
        target_index: Index of target property in the dataset
        target_name: Name of target property
        experiment_type: 'scratch' or 'fine_tune'
    """
    
    # Configuration - use standardized config
    model_config = MODEL_CONFIGS['gotennet']
    config = {
        **TRAINING_CONFIG,  # Standardized training settings
        **model_config,     # Model architecture from crysmtm_config
        'target_index': target_index,
        'target_name': target_name,
        'seed': seed,
        'checkpoint_path': checkpoint_path,
        'train_from_scratch': checkpoint_path is None,
        'experiment_type': experiment_type,
        'save_dir': f'results_crysmtm/{target_name}/gotennet/seed_{seed}/{experiment_type}',
        # Override with custom values if provided
        'batch_size': batch_size if batch_size != 32 else TRAINING_CONFIG['batch_size'],
        'epochs': epochs if epochs != 100 else TRAINING_CONFIG['epochs'],
        'lr': lr,
        'weight_decay': weight_decay,
    }
    
    print(f"\nTarget: {target_name} (index {target_index})")
    print(f"Experiment Type: {config['experiment_type'].upper()}")
    print(f"Seed: {config['seed']}")
    print(f"Save Directory: {config['save_dir']}")
    
    if config['checkpoint_path']:
        print(f"Fine-tuning from: {config['checkpoint_path']}")
    
    print("\nGotenNet Architecture:")
    print(f"  n_atom_basis={config['n_atom_basis']}, n_interactions={config['n_interactions']}, "
          f"num_heads={config['num_heads']}, n_rbf={config['n_rbf']}, cutoff={config['cutoff']}")
    print(f"\nHyperparameters: batch_size={config['batch_size']}, epochs={config['epochs']}, "
          f"lr={config['lr']}, weight_decay={config['weight_decay']}")
    print()
    
    # Set random seed
    set_all_seeds(config['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load CrysMTM dataset
    print("Loading CrysMTM dataset...")
    
    # Define temperature filter as a proper function (not lambda) for Windows multiprocessing
    def temp_filter(temp):
        return temp in DATASET_CONFIG['train_temps']
    
    full_dataset = RegressionLoader(
        label_dir=DATASET_CONFIG['base_dir'],
        temperature_filter=temp_filter,
        modalities=DATASET_CONFIG['modalities'],
        max_rotations=DATASET_CONFIG['max_rotations'],
        as_pyg_data=DATASET_CONFIG['as_pyg_data'],
        normalize_labels=False,  # We'll handle normalization manually
    )
    
    print(f"Dataset size: {len(full_dataset)} samples")

    # Split train/val
    train_dataset, val_dataset = split_dataset(
        full_dataset, 
        val_ratio=config['val_ratio'],
        seed=config['seed']
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=TRAINING_CONFIG['num_workers']
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=TRAINING_CONFIG['num_workers']
    )
    
    # Compute normalization statistics from training set
    print("\nComputing normalization statistics from training set...")
    train_targets = []
    for idx in train_dataset.indices:
        data = full_dataset[idx]
        if data.y.dim() == 1:
            train_targets.append(data.y[target_index].item())
        else:
            train_targets.append(data.y[0, target_index].item())
    
    train_targets = torch.tensor(train_targets)
    target_mean = train_targets.mean().item()
    target_std = train_targets.std().item()
    print(f"{target_name} mean: {target_mean:.4f}, std: {target_std:.4f}")
    
    # Convert to torch tensors on device
    target_mean = torch.tensor(target_mean, device=device)
    target_std = torch.tensor(target_std, device=device)
    
    # Initialize model
    print("\nInitializing GotenNet model...")
    model = GotenNetRegressor(
        n_atom_basis=config['n_atom_basis'],
        n_interactions=config['n_interactions'],
        cutoff=config['cutoff'],
        num_heads=config['num_heads'],
        n_rbf=config['n_rbf']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load checkpoint if fine-tuning
    start_epoch = 1
    if config['checkpoint_path'] and not config['train_from_scratch']:
        print(f"\n{'='*70}")
        print(f"Loading checkpoint for fine-tuning: {config['checkpoint_path']}")
        print(f"{'='*70}")
        
        checkpoint = torch.load(config['checkpoint_path'], map_location=device, weights_only=False)
        
        # Load state dict (handle different checkpoint formats)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Both two-body and CrysMTM use same GotenNetRegressor wrapper structure
        # Keys should match directly - just load
        try:
            model.load_state_dict(state_dict, strict=True)
            print(">>> Checkpoint loaded successfully (strict=True)")
        except RuntimeError as e:
            print("Warning: Could not load with strict=True, trying strict=False...")
            print(f"Error: {e}")
            model.load_state_dict(state_dict, strict=False)
            print(">>> Checkpoint loaded with strict=False (some parameters may not match)")
        
        print("Fine-tuning from pre-trained two-body GotenNet model\n")
    elif config['train_from_scratch']:
        print("Training from scratch (randomly initialized weights)\n")
    else:
        print("Training from scratch (no checkpoint provided)\n")
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=TRAINING_CONFIG['scheduler_factor'], 
        patience=TRAINING_CONFIG['scheduler_patience'], 
        min_lr=TRAINING_CONFIG['scheduler_min_lr']
    )
    criterion = torch.nn.MSELoss()  # MSE for CrysMTM

    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70 + "\n")
    
    best_val_mae = float('inf')
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_rmse': [], 'epoch_times': [], 'learning_rates': []}
    
    # Create save directory
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(start_epoch, config['epochs'] + 1):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion, target_index, target_mean, target_std)
        
        # Evaluate
        val_loss, val_mae, val_rmse = evaluate(model, val_loader, device, criterion, target_index, target_mean, target_std)
        
        # Learning rate scheduling
        scheduler.step(val_mae)
        
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history (convert to float for JSON serialization)
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['val_mae'].append(float(val_mae))
        history['val_rmse'].append(float(val_rmse))
        history['epoch_times'].append(float(epoch_time))
        history['learning_rates'].append(float(current_lr))
        
        # Print progress
        print(f"Epoch {epoch:03d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val MAE: {val_mae:.4f} | "
              f"Val RMSE: {val_rmse:.4f} | "
              f"Time: {epoch_time:.2f}s | "
              f"LR: {current_lr:.6f}")
        
        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_model_path = save_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'val_loss': val_loss,
                'config': config,
                'target_mean': target_mean.item(),
                'target_std': target_std.item()
            }, best_model_path)
            print(f"  → New best model saved! (MAE: {val_mae:.4f})")
        
        # Early stopping
        if epoch - best_epoch > TRAINING_CONFIG['early_stopping_patience']:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Final evaluation
    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)
    
    # Load best model
    best_model_path = save_dir / 'best_model.pt'
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on validation set
    val_loss, val_mae, val_rmse = evaluate(model, val_loader, device, criterion, target_index, target_mean, target_std)
    
    print(f"\nBest Epoch: {checkpoint['epoch']}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val MAE: {val_mae:.4f}")
    print(f"Val RMSE: {val_rmse:.4f}")
    print(f"Target: {target_name}")
    print("\n" + "=" * 70)
    
    # Save results
    # Convert config to JSON-serializable format
    config_serializable = {k: (str(v) if isinstance(v, Path) else v) for k, v in config.items()}
    
    # CrysMTM pipeline lacks a dedicated test split; report validation metrics
    # under test_* keys for downstream summaries.
    results = {
        'model_type': 'gotennet',
        'dataset': 'crysmtm',
        'target_name': target_name,
        'target_index': target_index,
        'seed': config['seed'],
        'experiment_type': config['experiment_type'],
        'best_epoch': int(checkpoint['epoch']),
        'best_val_mae': float(checkpoint['val_mae']),
        'best_val_rmse': float(checkpoint['val_rmse']),
        'best_val_loss': float(checkpoint['val_loss']),
        'final_val_loss': float(val_loss),
        'final_val_mae': float(val_mae),
        'final_val_rmse': float(val_rmse),
        'test_loss': float(val_loss),
        'test_mae': float(val_mae),
        'test_rmse': float(val_rmse),
        'normalization': {
            'target_mean': float(target_mean.item()),
            'target_std': float(target_std.item())
        },
        'config': config_serializable,
        'history': history,
        'fine_tuned': not config['train_from_scratch'] and config['checkpoint_path'] is not None,
        'checkpoint_source': str(config['checkpoint_path']) if config['checkpoint_path'] else None,
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'total_train_time_sec': sum(history['epoch_times']),
        'total_train_time_min': sum(history['epoch_times']) / 60.0,
        'avg_epoch_time_sec': np.mean(history['epoch_times'])
    }
    
    results_path = save_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")
    
    return model, val_mae


def main():
    """
    Main function that runs all experiments:
    - 9 targets: HOMO, LUMO, Eg, Ef, Et, Eta, disp, vol, bond
    - 3 seeds: 42, 123, 456
    - 2 types: from scratch AND fine-tuned
    - Total: 54 experiments (9 × 3 × 2)
    """
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train GotenNet on CrysMTM dataset for property prediction (both scratch and fine-tuned)'
    )
    parser.add_argument('--scratch-only', action='store_true',
                        help='Only train from scratch (skip fine-tuning)')
    parser.add_argument('--finetune-only', action='store_true',
                        help='Only fine-tune (skip from scratch)')
    parser.add_argument('--targets', nargs='+', default=None,
                        help='Specific targets to train on (default: all)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=TRAINING_CONFIG['lr'],
                        help=f"Learning rate (default: {TRAINING_CONFIG['lr']})")
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    
    args = parser.parse_args()
    
    # Define experiments from config
    if args.targets:
        targets = [(TARGET_PROPERTIES.index(t), t) for t in args.targets if t in TARGET_PROPERTIES]
    else:
        targets = list(enumerate(TARGET_PROPERTIES))
    
    seeds = SEEDS
    experiment_types = []
    
    # Run fine-tune first, then scratch
    if not args.scratch_only:
        experiment_types.append('fine_tune')
    if not args.finetune_only:
        experiment_types.append('scratch')
    
    if not experiment_types:
        print("Error: Cannot skip both scratch and fine-tune!")
        return
    
    total_experiments = len(targets) * len(seeds) * len(experiment_types)
    
    print("\n" + "=" * 80)
    print("CrysMTM BENCHMARK: GotenNet on Property Prediction")
    print("=" * 80)
    print(f"\nExperiment Types: {', '.join(experiment_types)}")
    print(f"Targets: {', '.join([t[1] for t in targets])}")
    print(f"Seeds: {seeds}")
    print(f"Data Split: {TRAINING_CONFIG['train_ratio']:.0%} train, {TRAINING_CONFIG['val_ratio']:.0%} val")
    print("\nTwo-Body Transfer Learning Mapping:")
    from benchmarks.crysmtm_config import TWOBODY_TARGET_MAP
    for target in [t[1] for t in targets]:
        twobody = TWOBODY_TARGET_MAP.get(target, 'N/A')
        print(f"  {target} -> {twobody}")
    print(f"\nTotal experiments: {len(targets)} targets × {len(seeds)} seeds × {len(experiment_types)} types = {total_experiments}")
    print("\n" + "=" * 80 + "\n")
    
    results_summary = []
    experiment_count = 0
    
    for target_index, target_name in targets:
        for seed in seeds:
            for exp_type in experiment_types:
                experiment_count += 1
                
                print("\n" + "=" * 80)
                print(f"EXPERIMENT {experiment_count}/{total_experiments}: {target_name} | Seed {seed} | {exp_type.upper()}")
                print("=" * 80)
                
                # Get checkpoint path if fine-tuning
                checkpoint_path = None
                
                if exp_type == 'fine_tune':
                    # Map CrysMTM property to two-body property
                    from benchmarks.crysmtm_config import TWOBODY_TARGET_MAP
                    twobody_property = TWOBODY_TARGET_MAP.get(target_name)
                    checkpoint_path = get_checkpoint_path('gotennet', target_name, seed)
                    if checkpoint_path:
                        print(f"Using two-body checkpoint: {target_name} -> {twobody_property}")
                        print(f"Checkpoint path: {checkpoint_path}")
                    else:
                        print(f"WARNING: No two-body checkpoint found for {target_name} ({twobody_property}), skipping fine-tune experiment")
                        continue
                else:
                    print("Training from scratch (randomly initialized)")
                
                try:
                    # Run experiment
                    model, val_mae = run_single_experiment(
                        target_index=target_index,
                        target_name=target_name,
                        seed=seed,
                        checkpoint_path=checkpoint_path,
                        experiment_type=exp_type,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        lr=args.lr,
                        weight_decay=args.weight_decay
                    )
                    
                    results_summary.append({
                        'target': target_name,
                        'seed': seed,
                        'type': exp_type,
                        'val_mae': val_mae,
                    })
                    
                    print(f"\n>>> Completed: {target_name} seed {seed} {exp_type} | Val MAE: {val_mae:.4f}\n")
                    
                except Exception as e:
                    print(f"\nERROR in {target_name} seed {seed} {exp_type}: {e}\n")
                    import traceback
                    traceback.print_exc()
    
    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    for target_index, target_name in targets:
        print(f"\n{target_name}:")
        
        for exp_type in experiment_types:
            target_results = [r for r in results_summary if r['target'] == target_name and r['type'] == exp_type]
            if target_results:
                maes = [r['val_mae'] for r in target_results]
                print(f"  {exp_type.upper()}: {np.mean(maes):.4f} ± {np.std(maes):.4f}")
                for r in target_results:
                    print(f"    Seed {r['seed']}: {r['val_mae']:.4f}")
    
    print("\n" + "=" * 80)
    print(f"Completed {len(results_summary)}/{total_experiments} experiments!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    # Check dependencies
    try:
        import torch_geometric
        print(f"PyTorch Geometric version: {torch_geometric.__version__}")
    except ImportError:
        print("ERROR: PyTorch Geometric is not installed!")
        print("\nPlease install with:")
        print("  pip install torch-geometric")
        exit(1)
    
    try:
        from gotennet import GotenNetWrapper
        print("GotenNet is installed")
    except ImportError:
        print("ERROR: GotenNet is not installed!")
        print("\nPlease install GotenNet")
        exit(1)
    
    print("All dependencies satisfied\n")
    main()
