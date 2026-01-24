"""
QM9 SchNet Training Script

Automatically trains SchNet on QM9 dataset for:
- 3 targets: HOMO, LUMO, Gap
- 3 seeds: 42, 123, 456
- Option to fine-tune from two-body checkpoints or train from scratch
"""

import torch
import torch.nn as nn
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet
import numpy as np
import time
import argparse
from pathlib import Path
import json
from benchmarks.qm9_config import (
    TRAINING_CONFIG, QM9_SPLIT, QM9_TARGETS, SEEDS, MODEL_CONFIGS, get_checkpoint_path
)


class SchNetRegressor(nn.Module):
    """SchNet wrapper matching the two-body training structure."""

    def __init__(
        self,
        hidden_channels=16,
        num_filters=16,
        num_interactions=2,
        num_gaussians=8,
        cutoff=5.0,
        readout="add",
    ):
        super().__init__()
        self.schnet = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            readout=readout,
            dipole=False,
            mean=None,
            std=None,
            atomref=None,
        )

    def forward(self, z, pos, batch):
        return self.schnet(z, pos, batch)


def train_epoch(model, loader, optimizer, device, criterion, target_idx=7):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass - SchNet returns graph-level predictions
        out = model(data.z, data.pos, data.batch)
        
        # Compute loss
        loss = criterion(out.squeeze(), data.y[:, target_idx])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, criterion, target_idx=7):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    for data in loader:
        data = data.to(device)
        
        # Forward pass
        out = model(data.z, data.pos, data.batch)
        
        # Compute loss
        loss = criterion(out.squeeze(), data.y[:, target_idx])
        total_loss += loss.item() * data.num_graphs
        
        predictions.append(out.cpu().numpy())
        targets.append(data.y[:, target_idx].cpu().numpy())
    
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    
    mae = np.mean(np.abs(predictions - targets))
    
    return total_loss / len(loader.dataset), mae




def run_single_experiment(target_idx: int, target_name: str, seed: int, 
                          checkpoint_path: Path = None, 
                          experiment_type: str = 'scratch',
                          batch_size: int = 32, 
                          epochs: int = 50, 
                          lr: float = 1e-4,
                          weight_decay: float = 0.0):
    """Run a single training experiment
    
    Args:
        experiment_type: 'scratch' or 'fine_tune'
    """
    
    # Configuration - use standardized config
    model_config = MODEL_CONFIGS['schnet']
    config = {
        **TRAINING_CONFIG,  # Standardized training settings
        **model_config,     # Model architecture from qm9_config
        'target': target_idx,
        'seed': seed,
        'checkpoint_path': checkpoint_path,
        'train_from_scratch': checkpoint_path is None,
        'experiment_type': experiment_type,
        'save_dir': f'results_qm9/{target_name}/schnet/seed_{seed}/{experiment_type}',
        # Override with custom values if provided
        'batch_size': batch_size if batch_size != 32 else TRAINING_CONFIG['batch_size'],
        'epochs': epochs if epochs != 50 else TRAINING_CONFIG['epochs'],
        'lr': lr,
        'weight_decay': weight_decay,
    }
    
    # Target property names
    target_names = [
        'dipole_moment',      # 0: Dipole moment (D)
        'isotropic_polarizability',  # 1: Isotropic polarizability (Bohr^3)
        'homo',               # 2: HOMO energy (eV)
        'lumo',               # 3: LUMO energy (eV)
        'gap',                # 4: HOMO-LUMO gap (eV)
        'electronic_spatial_extent',  # 5: Electronic spatial extent (Bohr^2)
        'zpve',               # 6: Zero point vibrational energy (eV)
        'U0',                 # 7: Internal energy at 0K (eV)
        'U',                  # 8: Internal energy at 298.15K (eV)
        'H',                  # 9: Enthalpy at 298.15K (eV)
        'G',                  # 10: Free energy at 298.15K (eV)
        'Cv',                 # 11: Heat capacity at 298.15K (cal/mol/K)
    ]
    
    print(f"\nTarget Property: {target_names[config['target']]} (index {config['target']})")
    print(f"Experiment Type: {config['experiment_type'].upper()}")
    print(f"Seed: {config['seed']}")
    print(f"Save Directory: {config['save_dir']}")
    
    if config['checkpoint_path']:
        print(f"Fine-tuning from: {config['checkpoint_path']}")
    
    print("\nSchNet Architecture:")
    print(f"  hidden_channels={config['hidden_channels']}, num_filters={config['num_filters']}, "
          f"num_interactions={config['num_interactions']}, num_gaussians={config['num_gaussians']}, cutoff={config['cutoff']}")
    print(f"\nHyperparameters: batch_size={config['batch_size']}, epochs={config['epochs']}, "
          f"lr={config['lr']}, weight_decay={config['weight_decay']}")
    print()
    
    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load QM9 dataset
    print("Loading QM9 dataset...")
    path = './data/QM9'
    dataset = QM9(path)
    
    print(f"Dataset size: {len(dataset)} molecules")
    print(f"Number of features per node: {dataset.num_features}")
    print(f"Number of target properties: {dataset.num_classes}")
    print(f"Example molecule: {dataset[0]}")
    print()
    
    # Split dataset (standard QM9 split from config)
    train_size = QM9_SPLIT['train_size']
    val_size = QM9_SPLIT['val_size']
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    test_dataset = dataset[train_size + val_size:]
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
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

    # Compute normalization statistics from training set (for reporting consistency)
    print("\nComputing target normalization statistics from training set...")
    train_targets = []
    for data in train_dataset:
        train_targets.append(data.y[0, config['target']].item())
    train_targets = torch.tensor(train_targets, dtype=torch.float32)
    target_mean = train_targets.mean().item()
    target_std = train_targets.std().item()
    print(f"{target_names[config['target']]} mean: {target_mean:.6f}, std: {target_std:.6f}")

    # Initialize model (using the wrapper to match checkpoint structure)
    print("\nInitializing SchNet model...")
    model = SchNetRegressor(
        hidden_channels=config['hidden_channels'],
        num_filters=config['num_filters'],
        num_interactions=config['num_interactions'],
        num_gaussians=config['num_gaussians'],
        cutoff=config['cutoff'],
        readout=config['readout']
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

        # Two-body and QM9 now share the same wrapper structure
        try:
            model.load_state_dict(state_dict, strict=True)
            print("✓ Checkpoint loaded successfully (strict=True)")
        except RuntimeError as e:
            print("Warning: Could not load with strict=True, trying strict=False...")
            print(f"Error: {e}")
            model.load_state_dict(state_dict, strict=False)
            print("✓ Checkpoint loaded with strict=False (some parameters may not match)")
        
        print("Fine-tuning from pre-trained two-body model\n")
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
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion, config['target'])
        
        # Evaluate
        val_loss, val_mae = evaluate(model, val_loader, device, criterion, config['target'])
        
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
                'config': config
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
    
    test_loss, test_mae = evaluate(model, test_loader, device, criterion, config['target'])
    
    # Calculate RMSE
    test_rmse = 0
    for data in test_loader:
        data = data.to(device)
        out = model(data.z, data.pos, data.batch)
        squared_errors = (out.squeeze() - data.y[:, config['target']]) ** 2
        test_rmse += squared_errors.sum().item()
    test_rmse = np.sqrt(test_rmse / len(test_loader.dataset))
    
    print(f"\nBest Epoch: {checkpoint['epoch']}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Target: {target_names[config['target']]}")
    print("\n" + "=" * 70)
    
    # Save results
    # Convert config to JSON-serializable format
    config_serializable = {k: (str(v) if isinstance(v, Path) else v) for k, v in config.items()}
    
    results = {
        'model_type': 'schnet',
        'dataset': 'qm9',
        'target': target_names[config['target']],
        'target_idx': config['target'],
        'seed': config['seed'],
        'experiment_type': config['experiment_type'],
        'best_epoch': int(checkpoint['epoch']),
        'best_val_mae': float(checkpoint['val_mae']),
        'best_val_loss': float(checkpoint['val_loss']),
        'test_loss': float(test_loss),
        'test_mae': float(test_mae),
        'test_rmse': float(test_rmse),
        'normalization': {
            'target_mean': float(target_mean),
            'target_std': float(target_std),
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
    - 3 targets: HOMO (2), LUMO (3), Gap (4)
    - 3 seeds: 42, 123, 456
    - 2 types: from scratch AND fine-tuned
    - Total: 18 experiments (3 × 3 × 2)
    """
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train SchNet on QM9 dataset for HOMO/LUMO/Gap prediction (both scratch and fine-tuned)'
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
    experiments = [(QM9_TARGETS[name], name) for name in ['homo', 'lumo', 'gap']]
    seeds = SEEDS
    experiment_types = []
    
    # Run fine-tune first, then scratch (better to start with transfer learning)
    if not args.scratch_only:
        experiment_types.append('fine_tune')
    if not args.finetune_only:
        experiment_types.append('scratch')
    
    if not experiment_types:
        print("Error: Cannot skip both scratch and fine-tune!")
        return
    
    total_experiments = len(experiments) * len(seeds) * len(experiment_types)
    
    print("\n" + "=" * 80)
    print("QM9 BENCHMARK: SchNet on HOMO/LUMO/Gap")
    print("=" * 80)
    print(f"\nExperiment Types: {', '.join(experiment_types)}")
    print(f"Targets: {', '.join([name.upper() for _, name in experiments])}")
    print(f"Seeds: {seeds}")
    print(f"Total experiments: {len(experiments)} targets × {len(seeds)} seeds × {len(experiment_types)} types = {total_experiments}")
    print("\n" + "=" * 80 + "\n")
    
    results_summary = []
    experiment_count = 0
    
    for target_idx, target_name in experiments:
        for seed in seeds:
            for exp_type in experiment_types:
                experiment_count += 1
                
                print("\n" + "=" * 80)
                print(f"EXPERIMENT {experiment_count}/{total_experiments}: {target_name.upper()} | Seed {seed} | {exp_type.upper()}")
                print("=" * 80)
                
                # Get checkpoint path if fine-tuning
                checkpoint_path = None
                current_lr = args.lr
                
                if exp_type == 'fine_tune':
                    checkpoint_path = get_checkpoint_path('schnet', target_name, seed)
                    current_lr = args.finetune_lr
                    if checkpoint_path:
                        print(f"Using checkpoint: {checkpoint_path}")
                        print(f"Using lower learning rate for fine-tuning: {current_lr}")
                    else:
                        print("⚠️  Warning: No checkpoint found, skipping fine-tune experiment")
                        continue
                else:
                    print("Training from scratch (randomly initialized)")
                
                try:
                    # Run experiment
                    model, test_mae = run_single_experiment(
                        target_idx=target_idx,
                        target_name=target_name,
                        seed=seed,
                        checkpoint_path=checkpoint_path,
                        experiment_type=exp_type,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        lr=current_lr,
                        weight_decay=args.weight_decay
                    )
                    
                    results_summary.append({
                        'target': target_name,
                        'seed': seed,
                        'type': exp_type,
                        'test_mae': test_mae,
                    })
                    
                    print(f"\n✅ Completed: {target_name.upper()} seed {seed} {exp_type} | Test MAE: {test_mae:.4f}\n")
                    
                except Exception as e:
                    print(f"\n❌ Error in {target_name.upper()} seed {seed} {exp_type}: {e}\n")
                    import traceback
                    traceback.print_exc()
    
    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    for target_idx, target_name in experiments:
        print(f"\n{target_name.upper()}:")
        
        for exp_type in experiment_types:
            target_results = [r for r in results_summary if r['target'] == target_name and r['type'] == exp_type]
            if target_results:
                maes = [r['test_mae'] for r in target_results]
                print(f"  {exp_type.upper()}: {np.mean(maes):.4f} ± {np.std(maes):.4f}")
                for r in target_results:
                    print(f"    Seed {r['seed']}: {r['test_mae']:.4f}")
    
    print("\n" + "=" * 80)
    print(f"Completed {len(results_summary)}/{total_experiments} experiments!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    # Check if PyTorch Geometric is installed
    try:
        import torch_geometric
        print(f"PyTorch Geometric version: {torch_geometric.__version__}")
    except ImportError:
        print("ERROR: PyTorch Geometric is not installed!")
        print("\nPlease install with:")
        print("  pip install torch-geometric")
        exit(1)
    
    main()

