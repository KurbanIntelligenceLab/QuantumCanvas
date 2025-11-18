"""
MD17 GotenNet Training Script

Automatically trains GotenNet on MD17 dataset for:
- 3 molecules: benzene, aspirin, ethanol
- 3 seeds: 42, 123, 456
- 2 types: from scratch AND fine-tuned (from two-body total_energy checkpoint)
- Predicts: molecular energy
"""

import torch
import torch.nn as nn
from torch_geometric.datasets import MD17
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
from benchmarks.md17_config import (
    TRAINING_CONFIG, MD17_MOLECULES, SEEDS, MODEL_CONFIGS, get_checkpoint_path
)


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


def train_epoch(model, loader, optimizer, device, criterion, mean, std):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass - GotenNetRegressor expects z, pos, batch
        out = model(data.z, data.pos, data.batch)
        
        # Normalize targets
        targets_normalized = (data.energy.squeeze() - mean) / std
        
        # Compute loss on normalized values
        loss = criterion(out.squeeze(), targets_normalized)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, criterion, mean, std):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    for data in loader:
        data = data.to(device)
        
        # Forward pass - GotenNetRegressor expects z, pos, batch
        out = model(data.z, data.pos, data.batch)
        
        # Normalize targets for loss computation
        targets_normalized = (data.energy.squeeze() - mean) / std
        
        # Compute loss on normalized values
        loss = criterion(out.squeeze(), targets_normalized)
        total_loss += loss.item() * data.num_graphs
        
        # Denormalize predictions for metric computation
        out_denormalized = out * std + mean
        
        predictions.append(out_denormalized.cpu().numpy())
        targets.append(data.energy.cpu().numpy())
    
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    
    # MAE on original scale
    mae = np.mean(np.abs(predictions - targets))
    
    return total_loss / len(loader.dataset), mae


def run_single_experiment(molecule: str, seed: int, 
                          checkpoint_path: Path = None, 
                          experiment_type: str = 'scratch',
                          batch_size: int = 32, 
                          epochs: int = 50, 
                          lr: float = 1e-4,
                          weight_decay: float = 0.0):
    """Run a single training experiment
    
    Args:
        molecule: MD17 molecule name
        experiment_type: 'scratch' or 'fine_tune'
    """
    
    # Configuration - use standardized config
    model_config = MODEL_CONFIGS['gotennet']
    config = {
        **TRAINING_CONFIG,  # Standardized training settings
        **model_config,     # Model architecture from md17_config
        'molecule': molecule,
        'seed': seed,
        'checkpoint_path': checkpoint_path,
        'train_from_scratch': checkpoint_path is None,
        'experiment_type': experiment_type,
        'save_dir': f'results_md17/{molecule}/gotennet/seed_{seed}/{experiment_type}',
        # Override with custom values if provided
        'batch_size': batch_size if batch_size != 32 else TRAINING_CONFIG['batch_size'],
        'epochs': epochs if epochs != 50 else TRAINING_CONFIG['epochs'],
        'lr': lr,
        'weight_decay': weight_decay,
    }
    
    print(f"\nMolecule: {molecule.upper()}")
    print(f"Experiment Type: {config['experiment_type'].upper()}")
    print(f"Seed: {config['seed']}")
    print(f"Save Directory: {config['save_dir']}")
    
    if config['checkpoint_path']:
        print(f"Fine-tuning from: {config['checkpoint_path']}")
    
    print(f"\nGotenNet Architecture:")
    print(f"  n_atom_basis={config['n_atom_basis']}, n_interactions={config['n_interactions']}, "
          f"num_heads={config['num_heads']}, n_rbf={config['n_rbf']}, cutoff={config['cutoff']}")
    print(f"\nHyperparameters: batch_size={config['batch_size']}, epochs={config['epochs']}, "
          f"lr={config['lr']}, weight_decay={config['weight_decay']}")
    print()
    
    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Device - Force CPU due to torch_cluster CUDA compilation issues
    # TODO: Use CUDA once torch_cluster is properly compiled
    device = torch.device('cpu')
    print(f"Using device: {device} (forced CPU due to torch_cluster compatibility)\n")
    
    # Load MD17 dataset
    print(f"Loading MD17 dataset for {molecule}...")
    path = './data/MD17'
    dataset = MD17(path, name=molecule)
    
    print(f"Dataset size: {len(dataset)} conformations")
    
    # Split dataset
    train_size = config['train_size']
    val_size = config['val_size']
    test_size = 1000  # Limit test set to 1000 samples
    
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    test_dataset = dataset[train_size + val_size:train_size + val_size + test_size]
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)} (limited to {test_size})")
    
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
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=TRAINING_CONFIG['num_workers']
    )
    
    # Compute normalization statistics from training set
    print("\nComputing normalization statistics from training set...")
    train_energies = torch.cat([data.energy for data in train_dataset])
    energy_mean = train_energies.mean().item()
    energy_std = train_energies.std().item()
    print(f"Energy mean: {energy_mean:.4f}, std: {energy_std:.4f}")
    
    # Convert to torch tensors on device
    energy_mean = torch.tensor(energy_mean, device=device)
    energy_std = torch.tensor(energy_std, device=device)
    
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
        
        # Both two-body and MD17 use same GotenNetRegressor wrapper structure
        # Keys should match directly - just load
        try:
            model.load_state_dict(state_dict, strict=True)
            print(">>> Checkpoint loaded successfully (strict=True)")
        except RuntimeError as e:
            print(f"Warning: Could not load with strict=True, trying strict=False...")
            print(f"Error: {e}")
            model.load_state_dict(state_dict, strict=False)
            print(">>> Checkpoint loaded with strict=False (some parameters may not match)")
        
        print(f"Fine-tuning from pre-trained two-body GotenNet model\n")
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
    criterion = torch.nn.L1Loss()
    
    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70 + "\n")
    
    best_val_mae = float('inf')
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'epoch_times': [], 'learning_rates': []}
    
    # Create save directory
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(start_epoch, config['epochs'] + 1):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion, energy_mean, energy_std)
        
        # Evaluate
        val_loss, val_mae = evaluate(model, val_loader, device, criterion, energy_mean, energy_std)
        
        # Learning rate scheduling
        scheduler.step(val_mae)
        
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history (convert to float for JSON serialization)
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['val_mae'].append(float(val_mae))
        history['epoch_times'].append(float(epoch_time))
        history['learning_rates'].append(float(current_lr))
        
        # Print progress
        print(f"Epoch {epoch:03d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val MAE: {val_mae:.4f} | "
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
                'val_loss': val_loss,
                'config': config,
                'energy_mean': energy_mean.item(),
                'energy_std': energy_std.item()
            }, best_model_path)
            print(f"  → New best model saved! (MAE: {val_mae:.4f})")
        
        # Early stopping
        if epoch - best_epoch > TRAINING_CONFIG['early_stopping_patience']:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Test evaluation
    print("\n" + "=" * 70)
    print("Final Evaluation on Test Set")
    print("=" * 70)
    
    # Load best model
    best_model_path = save_dir / 'best_model.pt'
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_mae = evaluate(model, test_loader, device, criterion, energy_mean, energy_std)
    
    # Calculate RMSE on denormalized predictions
    test_rmse = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.z, data.pos, data.batch)
            # Denormalize predictions
            out_denormalized = out * energy_std + energy_mean
            squared_errors = (out_denormalized.squeeze() - data.energy.squeeze()) ** 2
            test_rmse += squared_errors.sum().item()
    test_rmse = np.sqrt(test_rmse / len(test_loader.dataset))
    
    print(f"\nBest Epoch: {checkpoint['epoch']}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Molecule: {molecule}")
    print("\n" + "=" * 70)
    
    # Save results
    # Convert config to JSON-serializable format
    config_serializable = {k: (str(v) if isinstance(v, Path) else v) for k, v in config.items()}
    
    results = {
        'model_type': 'gotennet',
        'dataset': 'md17',
        'molecule': molecule,
        'seed': config['seed'],
        'experiment_type': config['experiment_type'],
        'best_epoch': int(checkpoint['epoch']),
        'best_val_mae': float(checkpoint['val_mae']),
        'best_val_loss': float(checkpoint['val_loss']),
        'test_loss': float(test_loss),
        'test_mae': float(test_mae),
        'test_rmse': float(test_rmse),
        'normalization': {
            'energy_mean': float(energy_mean.item()),
            'energy_std': float(energy_std.item())
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
    
    return model, test_mae


def main():
    """
    Main function that runs all experiments:
    - 3 molecules: benzene, aspirin, ethanol
    - 3 seeds: 42, 123, 456
    - 2 types: from scratch AND fine-tuned
    - Total: 18 experiments (3 × 3 × 2)
    """
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train GotenNet on MD17 dataset for energy prediction (both scratch and fine-tuned)'
    )
    parser.add_argument('--scratch-only', action='store_true',
                        help='Only train from scratch (skip fine-tuning)')
    parser.add_argument('--finetune-only', action='store_true',
                        help='Only fine-tune (skip from scratch)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=TRAINING_CONFIG['lr_scratch'],
                        help=f"Learning rate for from-scratch (default: {TRAINING_CONFIG['lr_scratch']})")
    parser.add_argument('--finetune-lr', type=float, default=TRAINING_CONFIG['lr_finetune'],
                        help=f"Learning rate for fine-tuning (default: {TRAINING_CONFIG['lr_finetune']})")
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    
    args = parser.parse_args()
    
    # Define experiments from config
    molecules = MD17_MOLECULES
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
    
    total_experiments = len(molecules) * len(seeds) * len(experiment_types)
    
    print("\n" + "=" * 80)
    print("MD17 BENCHMARK: GotenNet on Energy Prediction")
    print("=" * 80)
    print(f"\nExperiment Types: {', '.join(experiment_types)}")
    print(f"Molecules: {', '.join([m.upper() for m in molecules])}")
    print(f"Seeds: {seeds}")
    print(f"Data Split: {TRAINING_CONFIG['train_size']} train, {TRAINING_CONFIG['val_size']} val, 1000 test")
    print(f"Total experiments: {len(molecules)} molecules × {len(seeds)} seeds × {len(experiment_types)} types = {total_experiments}")
    print("\n" + "=" * 80 + "\n")
    
    results_summary = []
    experiment_count = 0
    
    for molecule in molecules:
        for seed in seeds:
            for exp_type in experiment_types:
                experiment_count += 1
                
                print("\n" + "=" * 80)
                print(f"EXPERIMENT {experiment_count}/{total_experiments}: {molecule.upper()} | Seed {seed} | {exp_type.upper()}")
                print("=" * 80)
                
                # Get checkpoint path if fine-tuning
                checkpoint_path = None
                current_lr = args.lr
                
                if exp_type == 'fine_tune':
                    checkpoint_path = get_checkpoint_path('gotennet', molecule, seed)
                    current_lr = args.finetune_lr
                    if checkpoint_path:
                        print(f"Using checkpoint: {checkpoint_path}")
                        print(f"Using lower learning rate for fine-tuning: {current_lr}")
                    else:
                        print(f"WARNING: No checkpoint found, skipping fine-tune experiment")
                        continue
                else:
                    print("Training from scratch (randomly initialized)")
                
                try:
                    # Run experiment
                    model, test_mae = run_single_experiment(
                        molecule=molecule,
                        seed=seed,
                        checkpoint_path=checkpoint_path,
                        experiment_type=exp_type,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        lr=current_lr,
                        weight_decay=args.weight_decay
                    )
                    
                    results_summary.append({
                        'molecule': molecule,
                        'seed': seed,
                        'type': exp_type,
                        'test_mae': test_mae,
                    })
                    
                    print(f"\n>>> Completed: {molecule.upper()} seed {seed} {exp_type} | Test MAE: {test_mae:.4f}\n")
                    
                except Exception as e:
                    print(f"\nERROR in {molecule.upper()} seed {seed} {exp_type}: {e}\n")
                    import traceback
                    traceback.print_exc()
    
    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    for molecule in molecules:
        print(f"\n{molecule.upper()}:")
        
        for exp_type in experiment_types:
            mol_results = [r for r in results_summary if r['molecule'] == molecule and r['type'] == exp_type]
            if mol_results:
                maes = [r['test_mae'] for r in mol_results]
                print(f"  {exp_type.upper()}: {np.mean(maes):.4f} ± {np.std(maes):.4f}")
                for r in mol_results:
                    print(f"    Seed {r['seed']}: {r['test_mae']:.4f}")
    
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
        print(f"GotenNet is installed")
    except ImportError:
        print("ERROR: GotenNet is not installed!")
        print("\nPlease install GotenNet")
        exit(1)
    
    main()

