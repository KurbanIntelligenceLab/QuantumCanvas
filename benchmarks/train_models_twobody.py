import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
import time
from pathlib import Path
import json
from typing import Dict
from benchmarks.twobody_dataloader import TwoBodyDataset
from benchmarks.models import get_model
from benchmarks.benchmark_config import cfg


def train_epoch(model, loader, optimizer, device, criterion, model_type='schnet'):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass based on model type
        if model_type == 'faenet':
            # FAENet needs batch object
            outputs = model(data)
            # FAENet may return dict or tensor
            if isinstance(outputs, dict):
                out = outputs.get("energy", outputs.get("output", list(outputs.values())[0]))
            else:
                out = outputs
        elif model_type == 'egnn':
            # EGNN - updated to use z, pos, batch directly
            out = model(data.z, data.pos, data.batch)
        elif model_type == 'multimodal':
            # MultiModal needs z, pos, batch, and images
            # Convert 10-channel images to 3-channel for ResNet (use first 3 channels)
            images = torch.stack([d.image[:3] for d in data.to_data_list()])  # [B, 3, 32, 32]
            images = images.to(device).float()  # Ensure float32
            out = model(data.z, data.pos, data.batch, images)
        elif model_type == 'quantumshellnet':
            # QuantumShellNet needs images, z, pos, batch
            # Convert 10-channel images to 3-channel (use first 3 channels)
            # images = torch.stack([d.image[:3] for d in data.to_data_list()])  # [B, 3, 32, 32]
            images = images.to(device).float()  # Ensure float32
            out = model(images, data.z, data.pos, data.batch)
        elif model_type == 'vit':
            # Image-only model
            # Convert 10-channel images to 3-channel (use first 3 channels)
            images = torch.stack([d.image[:3] for d in data.to_data_list()])  # [B, 3, 32, 32]
            images = images.to(device).float()  # Ensure float32
            out = model(images)
        else:  # schnet, gotennet, dimenet
            out = model(data.z, data.pos, data.batch)
        
        # Ensure output is properly shaped
        if out.dim() > 1:
            out = out.squeeze()
        
        # Compute loss
        loss = criterion(out, data.y.squeeze())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, criterion, model_type='schnet', denormalize_fn=None):
    """Evaluate the model
    
    Args:
        denormalize_fn: Function to denormalize predictions back to original scale for metrics
    """
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    for data in loader:
        data = data.to(device)
        
        # Forward pass based on model type
        if model_type == 'faenet':
            # FAENet needs batch object
            outputs = model(data)
            # FAENet may return dict or tensor
            if isinstance(outputs, dict):
                out = outputs.get("energy", outputs.get("output", list(outputs.values())[0]))
            else:
                out = outputs
        elif model_type == 'egnn':
            # EGNN - updated to use z, pos, batch directly
            out = model(data.z, data.pos, data.batch)
        elif model_type == 'multimodal':
            # MultiModal needs z, pos, batch, and images
            # Convert 10-channel images to 3-channel for ResNet (use first 3 channels)
            images = torch.stack([d.image[:3] for d in data.to_data_list()])  # [B, 3, 32, 32]
            images = images.to(device).float()  # Ensure float32
            out = model(data.z, data.pos, data.batch, images)
        elif model_type == 'quantumshellnet':
            # QuantumShellNet needs images, z, pos, batch
            # Convert 10-channel images to 3-channel (use first 3 channels)
            images = torch.stack([d.image[:3] for d in data.to_data_list()])  # [B, 3, 32, 32]
            images = images.to(device).float()  # Ensure float32
            out = model(images, data.z, data.pos, data.batch)
        elif model_type == 'vit':
            # Image-only model
            # Convert 10-channel images to 3-channel (use first 3 channels)
            images = torch.stack([d.image[:3] for d in data.to_data_list()])  # [B, 3, 32, 32]
            images = images.to(device).float()  # Ensure float32
            out = model(images)
        else:  # schnet, gotennet, dimenet
            out = model(data.z, data.pos, data.batch)
        
        # Ensure output is properly shaped
        if out.dim() > 1:
            out = out.squeeze()
        
        # Compute loss on normalized values
        loss = criterion(out, data.y.squeeze())
        total_loss += loss.item() * data.num_graphs
        
        # Handle both scalar and batch predictions
        pred_np = out.cpu().numpy()
        target_np = data.y.cpu().numpy()
        
        # Ensure 1D arrays for concatenation
        if pred_np.ndim == 0:
            pred_np = pred_np.reshape(1)
        if target_np.ndim == 0:
            target_np = target_np.reshape(1)
            
        predictions.append(pred_np)
        targets.append(target_np)
    
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    
    # Denormalize predictions and targets for proper metric calculation in original scale
    if denormalize_fn is not None:
        predictions = np.array([denormalize_fn(p) for p in predictions])
        targets = np.array([denormalize_fn(t) for t in targets])
    
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    return total_loss / len(loader.dataset), mae, rmse


def train_model(model_type: str, dataset_path: str, target: str, seed: int,
                device: torch.device, base_save_dir: Path):
    """Train a single model"""
    
    # Create organized directory structure: results_twobody/{target}/{model}/{seed}
    save_dir = base_save_dir / target / model_type / f"seed_{seed}"
    save_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "=" * 70)
    print(f"Training {model_type.upper()} on {target}")
    print("=" * 70)
    print(f"Save directory: {save_dir}")
    
    # Split indices first (before loading dataset) for reproducibility
    # Load full dataset temporarily just to get the number of samples
    temp_dataset = TwoBodyDataset(dataset_path, target_label=target, verbose=False, normalize_labels=False)
    n_samples = len(temp_dataset)
    n_train = int(cfg.train_split * n_samples)
    n_val = int(cfg.val_split * n_samples)
    n_test = n_samples - n_train - n_val
    
    indices = torch.randperm(n_samples, generator=torch.Generator().manual_seed(seed))
    train_indices = indices[:n_train].tolist()
    val_indices = indices[n_train:n_train+n_val].tolist()
    test_indices = indices[n_train+n_val:].tolist()
    
    # Load training dataset WITH normalization (computes stats from training data only)
    print("\nLoading training dataset...")
    train_dataset_full = TwoBodyDataset(dataset_path, target_label=target, verbose=True, normalize_labels=True)
    train_dataset = train_dataset_full[train_indices]
    
    # Get normalization stats from training dataset
    norm_stats = train_dataset_full.get_normalization_stats()
    
    # Load val/test datasets with SAME normalization stats
    print("\nLoading validation dataset...")
    val_dataset_full = TwoBodyDataset(dataset_path, target_label=target, verbose=False, 
                                       normalize_labels=True, normalization_stats=norm_stats)
    val_dataset = val_dataset_full[val_indices]
    
    print("Loading test dataset...")
    test_dataset_full = TwoBodyDataset(dataset_path, target_label=target, verbose=False,
                                        normalize_labels=True, normalization_stats=norm_stats)
    test_dataset = test_dataset_full[test_indices]
    
    print(f"Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, 
                             shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, 
                           shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, 
                            shuffle=False, num_workers=cfg.num_workers)
    
    # Initialize model
    print(f"\nInitializing {model_type} model...")
    model_config = cfg.model_configs.get(model_type, {})
    
    try:
        model = get_model(model_type, **model_config).to(device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"Error initializing model: {e}")
        return None
    
    # Loss function
    criterion = nn.L1Loss() if cfg.loss_function == 'mae' else nn.MSELoss()
    
    # Optimizer
    optimizer_name = cfg.optimizer_name
    optimizer_params = cfg.optimizer_params.get(optimizer_name, {})
    weight_decay = cfg.training_params['weight_decay']
    
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, 
                                    weight_decay=weight_decay, **optimizer_params)
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                     weight_decay=weight_decay, **optimizer_params)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr,
                                   weight_decay=weight_decay, **optimizer_params)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                    weight_decay=weight_decay)
    
    # Scheduler
    scheduler_name = cfg.scheduler_name
    scheduler_params = cfg.scheduler_params.get(scheduler_name, {})
    
    if scheduler_name == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
    elif scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
    elif scheduler_name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
    elif scheduler_name == 'none':
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                               factor=0.8, patience=10, min_lr=1e-6)
    
    # Create denormalization function
    denormalize_fn = train_dataset_full.denormalize_label
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')  # Track best validation MSE/MAE loss
    best_val_mae = float('inf')
    best_epoch = 0
    history = {
        'train_loss': [], 
        'val_loss': [], 
        'val_mae': [], 
        'val_rmse': [],
        'epoch_times': [],
        'learning_rates': []
    }
    
    total_train_start = time.time()
    
    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()
        
        # Train (on normalized values)
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion, model_type)
        
        # Evaluate (denormalized for metrics in original scale)
        val_loss, val_mae, val_rmse = evaluate(model, val_loader, device, criterion, model_type, denormalize_fn)
        
        # Scheduler step
        if scheduler is not None:
            if cfg.scheduler_name == 'reduce_on_plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Record history
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['val_mae'].append(float(val_mae))
        history['val_rmse'].append(float(val_rmse))
        history['epoch_times'].append(float(epoch_time))
        history['learning_rates'].append(float(optimizer.param_groups[0]['lr']))
        
        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val MAE: {val_mae:.4f} | "
                  f"Val RMSE: {val_rmse:.4f} | "
                  f"Time: {epoch_time:.2f}s")
        
        # Save best model based on validation loss (MSE or MAE)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_mae = val_mae
            best_epoch = epoch
            
            save_path = save_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'config': cfg.to_dict(),
                'model_config': model_config,
                'model_type': model_type,
                'target': target,
                'normalization_stats': norm_stats  # Save normalization stats for inference
            }, save_path)
        
        # Early stopping
        if epoch - best_epoch > cfg.training_params['patience']:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    total_train_time = time.time() - total_train_start
    
    # Test evaluation
    print("\n" + "=" * 70)
    print("Final Evaluation on Test Set")
    print("=" * 70)
    
    # Load best model
    best_model_path = save_dir / 'best_model.pt'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_mae, test_rmse = evaluate(model, test_loader, device, criterion, model_type, denormalize_fn)
    
    print(f"\nBest Epoch: {best_epoch}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Total Training Time: {total_train_time:.2f}s ({total_train_time/60:.2f} min)")
    print(f"Avg Time per Epoch: {np.mean(history['epoch_times']):.2f}s")
    
    # Save comprehensive results
    results = {
        'model_type': model_type,
        'target': target,
        'seed': seed,
        
        # Best metrics (denormalized, in original scale)
        'best_epoch': int(best_epoch),
        'best_val_loss': float(best_val_loss),
        'best_val_mae': float(best_val_mae),
        
        # Test metrics (denormalized, in original scale)
        'test_loss': float(test_loss),
        'test_mae': float(test_mae),
        'test_rmse': float(test_rmse),
        
        # Training times
        'total_train_time_sec': float(total_train_time),
        'total_train_time_min': float(total_train_time / 60),
        'avg_epoch_time_sec': float(np.mean(history['epoch_times'])),
        'total_epochs_trained': len(history['train_loss']),
        
        # Full history
        'history': history,
        
        # Configurations
        'config': cfg.to_dict(),
        'model_config': model_config,
        'optimizer': cfg.optimizer_name,
        'scheduler': cfg.scheduler_name,
        'loss_function': cfg.loss_function,
        
        # Model info
        'num_parameters': sum(p.numel() for p in model.parameters()),
        
        # Normalization info
        'normalization_stats': norm_stats,
    }
    
    # Save results JSON
    results_path = save_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save training history CSV for easy plotting
    history_df = pd.DataFrame(history)
    history_df.to_csv(save_dir / 'training_history.csv', index=False)
    
    print(f"\nResults saved to {save_dir}")
    print(f"  - best_model.pt (checkpoint)")
    print(f"  - results.json (metrics)")
    print(f"  - training_history.csv (epoch-by-epoch)")
    
    return results


def run_single_experiment(seed: int, target: str, device: torch.device, 
                          dataset_path: str, base_save_dir: Path, models_to_train: list) -> Dict:
    """Run a single experiment for a given seed and target"""
    print("\n" + "=" * 70)
    print(f"EXPERIMENT: Target={target}, Seed={seed}")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Dataset: {dataset_path}")
    print(f"  Target: {target}")
    print(f"  Seed: {seed}")

    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Train models
    all_results = {}
    for model_type in models_to_train:
        try:
            results = train_model(
                model_type=model_type,
                dataset_path=dataset_path,
                target=target,
                seed=seed,
                device=device,
                base_save_dir=base_save_dir
            )
            if results:
                all_results[model_type] = results
        except Exception as e:
            print(f"\n❌ Error training {model_type}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary for this experiment
    if len(all_results) > 0:
        print("\n" + "=" * 70)
        print(f"RESULTS: Target={target}, Seed={seed}")
        print("=" * 70)
        print(f"\n{'Model':<15} {'Test Loss':<12} {'Test MAE':<12} {'Test RMSE':<12} {'Time (min)':<12} {'Params':<12}")
        print("-" * 90)
        
        for model_type, results in sorted(all_results.items(), key=lambda x: x[1]['test_mae']):
            print(f"{model_type:<15} {results['test_loss']:<12.4f} "
                  f"{results['test_mae']:<12.4f} {results['test_rmse']:<12.4f} "
                  f"{results['total_train_time_min']:<12.2f} {results['num_parameters']:<12,}")
        
        # Save summary for this seed+target combination
        summary_path = base_save_dir / target / f'benchmark_summary_seed_{seed}.json'
        with open(summary_path, 'w') as f:
            json.dump({
                'target': target,
                'seed': seed,
                'models': all_results
            }, f, indent=2)
        
        print(f"\n✓ Experiment complete!")
        print(f"  Summary: {summary_path}")
    else:
        print(f"\n❌ No models successfully trained for target={target}, seed={seed}")
    
    return all_results


def main():
    # Print configuration summary
    cfg.print_summary()
    
    # Setup from config
    device = torch.device(cfg.device)
    print(f"\nUsing device: {device}")
    
    # Configuration from benchmark_config.py
    dataset_path = cfg.dataset_path
    base_save_dir = Path('results_twobody')
    base_save_dir.mkdir(exist_ok=True, parents=True)
    
    models_to_train = cfg.available_models
    
    print("\n" + "=" * 70)
    print("STARTING COMPREHENSIVE BENCHMARK")
    print("=" * 70)
    print(f"  Total experiments: {len(cfg.targets)} targets × {len(cfg.seeds)} seeds = {cfg.total_experiments} runs")
    print(f"  Total model trainings: {cfg.total_experiments * len(models_to_train)}")
    print()
    
    # Track all results across all experiments
    global_results = {}
    experiment_count = 0
    total_experiments = cfg.total_experiments
    
    # Loop through all targets and seeds
    for target in cfg.targets:
        if target not in global_results:
            global_results[target] = {}
        
        for seed in cfg.seeds:
            experiment_count += 1
            print(f"\n{'#' * 70}")
            print(f"EXPERIMENT {experiment_count}/{total_experiments}")
            print(f"{'#' * 70}")
            
            # Run single experiment
            results = run_single_experiment(
                seed=seed,
                target=target,
                device=device,
                dataset_path=dataset_path,
                base_save_dir=base_save_dir,
                models_to_train=models_to_train
            )
            
            # Store results
            global_results[target][f'seed_{seed}'] = results
    
    # Final comprehensive summary
    print("\n\n" + "=" * 70)
    print("COMPREHENSIVE BENCHMARK SUMMARY")
    print("=" * 70)
    
    # Save comprehensive results
    comprehensive_summary_path = base_save_dir / 'comprehensive_benchmark_summary.json'
    with open(comprehensive_summary_path, 'w') as f:
        json.dump({
            'targets': cfg.targets,
            'seeds': cfg.seeds,
            'models': models_to_train,
            'results': global_results
        }, f, indent=2)
    
    print(f"\n✓ All benchmarks complete!")
    print(f"  Total experiments: {experiment_count}")
    print(f"  Comprehensive summary: {comprehensive_summary_path}")
    print(f"\nResults directory structure:")
    print(f"  results_twobody/")
    print(f"    ├── comprehensive_benchmark_summary.json")
    for target in cfg.targets[:2]:
        print(f"    ├── {target}/")
        for seed in cfg.seeds[:2]:
            print(f"    │   ├── benchmark_summary_seed_{seed}.json")
        for model in models_to_train[:2]:
            print(f"    │   ├── {model}/")
            for seed in cfg.seeds[:2]:
                print(f"    │   │   ├── seed_{seed}/")
                print(f"    │   │   │   ├── best_model.pt")
                print(f"    │   │   │   ├── results.json")
                print(f"    │   │   │   └── training_history.csv")
            print(f"    │   │   └── ...")
        print(f"    │   └── ...")
    print(f"    └── ...")


if __name__ == '__main__':
    # Check dataset exists
    if not Path(cfg.dataset_path).exists():
        print(f"ERROR: Dataset not found at {cfg.dataset_path}")
        print("\nPlease build the dataset first:")
        print("  python build_dataset.py")
        exit(1)
    
    main()

