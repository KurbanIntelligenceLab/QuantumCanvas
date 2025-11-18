"""
QM9 GotenNet Training Script

Automatically trains GotenNet on QM9 dataset for:
- 3 targets: HOMO, LUMO, Gap
- 3 seeds: 42, 123, 456
- 2 types: from scratch AND fine-tuned
"""

import torch
import torch.nn as nn
from torch_geometric.datasets import QM9
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
from benchmarks.qm9_config import (
    TRAINING_CONFIG, QM9_SPLIT, QM9_TARGETS, SEEDS, MODEL_CONFIGS, get_checkpoint_path
)


class GotenNetRegressor(nn.Module):
    """GotenNet wrapper matching the two-body training structure."""

    def __init__(self, n_atom_basis=32, n_interactions=2, cutoff=5.0, num_heads=2, n_rbf=4):
        super().__init__()
        self.gotennet = GotenNetWrapper(
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions,
            cutoff_fn=CosineCutoff(cutoff),
            num_heads=num_heads,
            n_rbf=n_rbf,
        )
        self.regressor = nn.Linear(n_atom_basis, 1)

    def forward(self, z, pos, batch):
        data = Data(z=z, pos=pos, batch=batch)
        h, _ = self.gotennet(data)
        pooled = scatter_mean(h, batch, dim=0)
        return self.regressor(pooled).squeeze()


def train_epoch(model, loader, optimizer, device, criterion, target_idx=7):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
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
    """Run a single training experiment"""
    
    # Configuration - use standardized config
    model_config = MODEL_CONFIGS['gotennet']
    config = {
        **TRAINING_CONFIG,  # Standardized training settings
        **model_config,     # Model architecture from qm9_config
        'target': target_idx,
        'seed': seed,
        'checkpoint_path': checkpoint_path,
        'train_from_scratch': checkpoint_path is None,
        'experiment_type': experiment_type,
        'save_dir': f'results_qm9/{target_name}/gotennet/seed_{seed}/{experiment_type}',
        # Override with custom values if provided
        'batch_size': batch_size if batch_size != 32 else TRAINING_CONFIG['batch_size'],
        'epochs': epochs if epochs != 50 else TRAINING_CONFIG['epochs'],
        'lr': lr,
        'weight_decay': weight_decay,
    }
    
    # Target property names
    target_names = [
        'dipole_moment', 'isotropic_polarizability', 'homo', 'lumo', 'gap',
        'electronic_spatial_extent', 'zpve', 'U0', 'U', 'H', 'G', 'Cv'
    ]
    
    print(f"\nTarget: {target_names[config['target']]} (index {config['target']})")
    print(f"Type: {config['experiment_type'].upper()} | Seed: {config['seed']}")
    if config['checkpoint_path']:
        print(f"Fine-tuning from: {config['checkpoint_path']}")
    
    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load QM9 dataset
    print("Loading QM9 dataset...")
    dataset = QM9('./data/QM9')
    
    # Split dataset (standard QM9 split from config)
    train_size = QM9_SPLIT['train_size']
    val_size = QM9_SPLIT['val_size']
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    test_dataset = dataset[train_size + val_size:]
    
    print(f"Split: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=TRAINING_CONFIG['num_workers']
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=TRAINING_CONFIG['num_workers']
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=TRAINING_CONFIG['num_workers']
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

    # Initialize model
    print("\nInitializing GotenNet model...")
    model = GotenNetRegressor(
        n_atom_basis=config['n_atom_basis'],
        n_interactions=config['n_interactions'],
        cutoff=config['cutoff'],
        num_heads=config['num_heads'],
        n_rbf=config['n_rbf'],
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load checkpoint if fine-tuning
    if config['checkpoint_path'] and not config['train_from_scratch']:
        print(f"\nLoading checkpoint: {config['checkpoint_path']}")
        checkpoint = torch.load(config['checkpoint_path'], map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Both two-body and QM9 use same wrapper structure (GotenNetRegressor)
        try:
            model.load_state_dict(state_dict, strict=True)
            print("✓ Checkpoint loaded (strict=True)")
        except RuntimeError as e:
            print(f"Warning: Trying strict=False... Error: {e}")
            model.load_state_dict(state_dict, strict=False)
            print("✓ Checkpoint loaded (strict=False)")
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=TRAINING_CONFIG['scheduler_factor'], 
        patience=TRAINING_CONFIG['scheduler_patience'], 
        min_lr=TRAINING_CONFIG['scheduler_min_lr']
    )
    criterion = torch.nn.L1Loss()
    
    # Training loop
    print("\nStarting Training...\n")
    
    best_val_mae = float('inf')
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'epoch_times': [], 'learning_rates': []}
    
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, config['epochs'] + 1):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion, config['target'])
        val_loss, val_mae = evaluate(model, val_loader, device, criterion, config['target'])
        scheduler.step(val_mae)
        
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        # Convert to float for JSON serialization
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['val_mae'].append(float(val_mae))
        history['epoch_times'].append(float(epoch_time))
        history['learning_rates'].append(float(current_lr))
        
        print(f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"MAE: {val_mae:.4f} | Time: {epoch_time:.2f}s | LR: {current_lr:.6f}")
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
                'val_loss': val_loss,
                'config': config
            }, save_dir / 'best_model.pt')
            print(f"  → Best model saved (MAE: {val_mae:.4f})")
        
        if epoch - best_epoch > TRAINING_CONFIG['early_stopping_patience']:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Test evaluation
    print("\nFinal Test Evaluation:")
    checkpoint = torch.load(save_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_mae = evaluate(model, test_loader, device, criterion, config['target'])
    
    # Calculate RMSE
    test_rmse = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.z, data.pos, data.batch)
            squared_errors = (out.squeeze() - data.y[:, config['target']]) ** 2
            test_rmse += squared_errors.sum().item()
    test_rmse = np.sqrt(test_rmse / len(test_loader.dataset))
    
    print(f"Best Epoch: {checkpoint['epoch']}")
    print(f"Test MAE: {test_mae:.4f} | Test RMSE: {test_rmse:.4f}")
    
    # Save results
    config_serializable = {k: (str(v) if isinstance(v, Path) else v) for k, v in config.items()}
    
    results = {
        'model_type': 'gotennet',
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
    
    with open(save_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return model, test_mae


def main():
    """Run all experiments: 3 targets × 3 seeds × 2 types = 18 experiments"""
    
    parser = argparse.ArgumentParser(description='Train GotenNet on QM9 (HOMO/LUMO/Gap)')
    parser.add_argument('--scratch-only', action='store_true', help='Only train from scratch')
    parser.add_argument('--finetune-only', action='store_true', help='Only fine-tune')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=TRAINING_CONFIG['lr_scratch'],
                        help=f"Learning rate for scratch (default: {TRAINING_CONFIG['lr_scratch']})")
    parser.add_argument('--finetune-lr', type=float, default=TRAINING_CONFIG['lr_finetune'],
                        help=f"Learning rate for fine-tune (default: {TRAINING_CONFIG['lr_finetune']})")
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay (default: 0.0)')
    
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
        print("Error: Cannot skip both!")
        return
    
    total_experiments = len(experiments) * len(seeds) * len(experiment_types)
    
    print("\n" + "=" * 80)
    print("QM9 BENCHMARK: GotenNet on HOMO/LUMO/Gap")
    print("=" * 80)
    print(f"Types: {', '.join(experiment_types)}")
    print(f"Targets: HOMO, LUMO, Gap | Seeds: {seeds}")
    print(f"Total: {total_experiments} experiments\n" + "=" * 80 + "\n")
    
    results_summary = []
    experiment_count = 0
    
    for target_idx, target_name in experiments:
        for seed in seeds:
            for exp_type in experiment_types:
                experiment_count += 1
                
                print(f"\n{'='*80}")
                print(f"EXPERIMENT {experiment_count}/{total_experiments}: {target_name.upper()} | Seed {seed} | {exp_type.upper()}")
                print("=" * 80)
                
                checkpoint_path = None
                current_lr = args.lr
                
                if exp_type == 'fine_tune':
                    checkpoint_path = get_checkpoint_path('gotennet', target_name, seed)
                    current_lr = args.finetune_lr
                    if not checkpoint_path:
                        print("⚠️  No checkpoint found, skipping")
                        continue
                
                try:
                    model, test_mae = run_single_experiment(
                        target_idx, target_name, seed, checkpoint_path, exp_type,
                        args.batch_size, args.epochs, current_lr, args.weight_decay
                    )
                    
                    results_summary.append({
                        'target': target_name,
                        'seed': seed,
                        'type': exp_type,
                        'test_mae': test_mae,
                    })
                    
                    print(f"\n✅ Completed: {target_name.upper()} seed {seed} {exp_type} | MAE: {test_mae:.4f}\n")
                    
                except Exception as e:
                    print(f"\n❌ Error: {e}\n")
                    import traceback
                    traceback.print_exc()
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    for target_idx, target_name in experiments:
        print(f"\n{target_name.upper()}:")
        for exp_type in experiment_types:
            results = [r for r in results_summary if r['target'] == target_name and r['type'] == exp_type]
            if results:
                maes = [r['test_mae'] for r in results]
                print(f"  {exp_type.upper()}: {np.mean(maes):.4f} ± {np.std(maes):.4f}")
                for r in results:
                    print(f"    Seed {r['seed']}: {r['test_mae']:.4f}")
    
    print(f"\n{'='*80}")
    print(f"Completed {len(results_summary)}/{total_experiments} experiments!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    try:
        import torch_geometric
        from gotennet import GotenNetWrapper
        print(f"PyTorch Geometric: {torch_geometric.__version__}")
        print("GotenNet: OK")
    except ImportError as e:
        print(f"ERROR: Missing dependency - {e}")
        exit(1)
    
    main()
