#!/usr/bin/env python3
"""
Train and evaluate models for modality ablation study.

Compares:
1. Tabular baseline (pooled features + MLP)
2. Tabular Transformer (pooled features as tokens)
3. Vision-only (ViT on 10-channel images, no Z input)
4. Geometry-only (Z + positions, no images)
5. QuantumShellNet (images + Z)
6. Multimodal fusion (SchNet + ResNet)

Usage:
    python rebuttal_experiments/modality/train_modality_comparison.py
    
    # Specific models and targets
    python rebuttal_experiments/modality/train_modality_comparison.py \
        --models tabular_mlp vision_only quantumshellnet \
        --targets e_g_ev total_energy_ev
"""

import sys
from pathlib import Path

# Ensure project root is on path
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import argparse
import json
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from benchmarks.twobody_dataloader import TwoBodyDataset
from benchmarks.models import get_model as get_benchmark_model
from benchmarks.benchmark_config import cfg
from rebuttal_experiments.modality.models import get_modality_model
from rebuttal_experiments.modality.improved_models import get_improved_model


# Default configuration
SEEDS = [42, 123, 456]
TARGETS = ['e_g_ev', 'total_energy_ev', 'e_homo_ev', 'e_lumo_ev', 'dipole_mag_d']
# Baselines + improved fusion models (no old qsn/vit/multimodal)
MODELS = ['tabular_mlp', 'tabular_transformer', 'vision_only', 'geometry_only', 
          'qsn_v2', 'multimodal_v2', 'film_cnn']
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
PATIENCE = 15
DEVICE = None
OUTPUT_DIR = "rebuttal_results/modality_ablation"


def get_model(model_type: str, device: torch.device):
    """Get model by type, handling modality, improved, and benchmark models."""
    modality_models = ['tabular_mlp', 'tabular_transformer', 'vision_only', 'geometry_only']
    improved_models = ['qsn_v2', 'multimodal_v2', 'film_cnn']
    
    if model_type in modality_models:
        model = get_modality_model(model_type)
    elif model_type in improved_models:
        model = get_improved_model(model_type)
    else:
        # Use benchmark models (quantumshellnet, vit, multimodal, etc.)
        model_config = cfg.model_configs.get(model_type, {})
        model = get_benchmark_model(model_type, **model_config)
    
    return model.to(device)


def forward_model(model, data, model_type: str, device: torch.device):
    """Handle forward pass for different model types."""
    
    if model_type in ['tabular_mlp', 'tabular_transformer', 'vision_only']:
        # These need images
        images = torch.stack([d.image for d in data.to_data_list()]).to(device).float()
        return model(images, data.z, data.pos, data.batch)
    
    elif model_type == 'geometry_only':
        # No images needed
        images = torch.stack([d.image for d in data.to_data_list()]).to(device).float()
        return model(images, data.z, data.pos, data.batch)
    
    elif model_type == 'quantumshellnet':
        images = torch.stack([d.image for d in data.to_data_list()]).to(device).float()
        return model(images, data.z, data.pos, data.batch)
    
    elif model_type == 'vit':
        # ViT uses first 3 channels
        images = torch.stack([d.image[:3] for d in data.to_data_list()]).to(device).float()
        return model(images)
    
    elif model_type == 'multimodal':
        images = torch.stack([d.image[:3] for d in data.to_data_list()]).to(device).float()
        return model(data.z, data.pos, data.batch, images)
    
    elif model_type in ['qsn_v2', 'multimodal_v2', 'film_cnn']:
        # Improved models use all 10 channels
        images = torch.stack([d.image for d in data.to_data_list()]).to(device).float()
        return model(images, data.z, data.pos, data.batch)
    
    else:
        # GNN models (schnet, egnn, etc.)
        return model(data.z, data.pos, data.batch)


def train_epoch(model, loader, optimizer, criterion, model_type, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_samples = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = forward_model(model, data, model_type, device)
        loss = criterion(out, data.y.squeeze())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        n_samples += data.num_graphs
    
    return total_loss / n_samples


@torch.no_grad()
def evaluate(model, loader, model_type, device, denormalize_fn=None):
    """Evaluate model and return metrics."""
    model.eval()
    predictions = []
    targets = []
    
    for data in loader:
        data = data.to(device)
        out = forward_model(model, data, model_type, device)
        
        pred_np = out.cpu().numpy()
        target_np = data.y.squeeze().cpu().numpy()
        
        if pred_np.ndim == 0:
            pred_np = pred_np.reshape(1)
        if target_np.ndim == 0:
            target_np = target_np.reshape(1)
        
        predictions.append(pred_np)
        targets.append(target_np)
    
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    
    # Denormalize if needed
    if denormalize_fn is not None:
        predictions = np.array([denormalize_fn(p) for p in predictions])
        targets = np.array([denormalize_fn(t) for t in targets])
    
    mae = float(np.mean(np.abs(predictions - targets)))
    rmse = float(np.sqrt(np.mean((predictions - targets) ** 2)))
    
    return {'mae': mae, 'rmse': rmse, 'predictions': predictions, 'targets': targets}


def train_single(model_type: str, target: str, seed: int, dataset_path: str,
                 batch_size: int, epochs: int, lr: float, patience: int,
                 device: torch.device, output_dir: Path, verbose: bool = True):
    """Train a single model on a single target with a single seed."""
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Load dataset
    full_dataset = TwoBodyDataset(
        dataset_path,
        target_label=target,
        verbose=False,
        normalize_labels=True
    )
    
    # Split dataset
    n_total = len(full_dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create model
    model = get_model(model_type, device)
    n_params = sum(p.numel() for p in model.parameters())
    
    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience // 2)
    criterion = nn.L1Loss()
    
    # Training loop
    best_val_mae = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    train_losses = []
    val_maes = []
    
    start_time = time.time()
    
    pbar = tqdm(range(epochs), desc=f"{model_type}/{target}/seed_{seed}", disable=not verbose)
    for epoch in pbar:
        train_loss = train_epoch(model, train_loader, optimizer, criterion, model_type, device)
        val_metrics = evaluate(model, val_loader, model_type, device)
        
        train_losses.append(train_loss)
        val_maes.append(val_metrics['mae'])
        
        scheduler.step(val_metrics['mae'])
        
        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            best_epoch = epoch
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        pbar.set_postfix({
            'train_loss': f"{train_loss:.4f}",
            'val_mae': f"{val_metrics['mae']:.4f}",
            'best': f"{best_val_mae:.4f}"
        })
        
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break
    
    train_time = time.time() - start_time
    
    # Load best model and evaluate on test set
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, model_type, device, 
                           denormalize_fn=full_dataset.denormalize_label)
    val_metrics_final = evaluate(model, val_loader, model_type, device,
                                 denormalize_fn=full_dataset.denormalize_label)
    
    # Save results
    result = {
        'model_type': model_type,
        'target': target,
        'seed': seed,
        'n_params': n_params,
        'best_epoch': best_epoch,
        'train_time_sec': train_time,
        'val_mae': val_metrics_final['mae'],
        'val_rmse': val_metrics_final['rmse'],
        'test_mae': test_metrics['mae'],
        'test_rmse': test_metrics['rmse'],
        'train_losses': train_losses,
        'val_maes': val_maes,
        'normalization_stats': full_dataset.get_normalization_stats(),
    }
    
    # Save checkpoint
    save_dir = output_dir / target / model_type / f"seed_{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': best_state,
        'model_type': model_type,
        'target': target,
        'seed': seed,
        'normalization_stats': full_dataset.get_normalization_stats(),
    }, save_dir / "best_model.pt")
    
    with open(save_dir / "results.json", 'w', encoding='utf-8') as f:
        # Remove numpy arrays for JSON serialization
        result_json = {k: v for k, v in result.items() 
                      if not isinstance(v, np.ndarray)}
        json.dump(result_json, f, indent=2)
    
    return result


def run_all_experiments(models: list, targets: list, seeds: list,
                        dataset_path: str, batch_size: int, epochs: int,
                        lr: float, patience: int, device: torch.device,
                        output_dir: Path, verbose: bool = True):
    """Run all experiments and aggregate results."""
    
    all_results = []
    
    total_runs = len(models) * len(targets) * len(seeds)
    print(f"\nRunning {total_runs} experiments: {len(models)} models × {len(targets)} targets × {len(seeds)} seeds")
    print(f"Models: {models}")
    print(f"Targets: {targets}")
    print(f"Seeds: {seeds}")
    print(f"Output: {output_dir}\n")
    
    for target in targets:
        for model_type in models:
            for seed in seeds:
                try:
                    result = train_single(
                        model_type=model_type,
                        target=target,
                        seed=seed,
                        dataset_path=dataset_path,
                        batch_size=batch_size,
                        epochs=epochs,
                        lr=lr,
                        patience=patience,
                        device=device,
                        output_dir=output_dir,
                        verbose=verbose
                    )
                    all_results.append(result)
                    print(f"✓ {model_type}/{target}/seed_{seed}: "
                          f"test_mae={result['test_mae']:.4f}, "
                          f"test_rmse={result['test_rmse']:.4f}")
                except Exception as e:
                    print(f"✗ {model_type}/{target}/seed_{seed}: {e}")
                    import traceback
                    traceback.print_exc()
    
    # Aggregate results
    aggregated = aggregate_results(all_results)
    
    # Save aggregated results
    with open(output_dir / "aggregated_results.json", 'w', encoding='utf-8') as f:
        json.dump(aggregated, f, indent=2)
    
    # Generate summary table
    generate_summary_table(aggregated, output_dir)
    
    return all_results, aggregated


def aggregate_results(results: list) -> dict:
    """Aggregate results across seeds."""
    
    # Group by (model_type, target)
    grouped = {}
    for r in results:
        key = (r['model_type'], r['target'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)
    
    aggregated = {}
    for (model_type, target), runs in grouped.items():
        test_maes = [r['test_mae'] for r in runs]
        test_rmses = [r['test_rmse'] for r in runs]
        val_maes = [r['val_mae'] for r in runs]
        
        aggregated[f"{model_type}/{target}"] = {
            'model_type': model_type,
            'target': target,
            'n_seeds': len(runs),
            'test_mae_mean': float(np.mean(test_maes)),
            'test_mae_std': float(np.std(test_maes)),
            'test_rmse_mean': float(np.mean(test_rmses)),
            'test_rmse_std': float(np.std(test_rmses)),
            'val_mae_mean': float(np.mean(val_maes)),
            'val_mae_std': float(np.std(val_maes)),
            'n_params': runs[0]['n_params'],
        }
    
    return aggregated


def generate_summary_table(aggregated: dict, output_dir: Path):
    """Generate a summary table in text and LaTeX format."""
    
    # Get unique models and targets
    models = sorted(set(v['model_type'] for v in aggregated.values()))
    targets = sorted(set(v['target'] for v in aggregated.values()))
    
    # Text table
    lines = []
    lines.append("=" * 100)
    lines.append("MODALITY ABLATION RESULTS (Test MAE ± std)")
    lines.append("=" * 100)
    
    # Header
    header = f"{'Model':<25}" + "".join(f"{t:<15}" for t in targets)
    lines.append(header)
    lines.append("-" * 100)
    
    # Data rows
    for model in models:
        row = f"{model:<25}"
        for target in targets:
            key = f"{model}/{target}"
            if key in aggregated:
                v = aggregated[key]
                row += f"{v['test_mae_mean']:.4f}±{v['test_mae_std']:.4f}  "
            else:
                row += f"{'N/A':<15}"
        lines.append(row)
    
    lines.append("=" * 100)
    
    # Add model comparison section
    lines.append("\nMODEL COMPARISON (averaged across targets)")
    lines.append("-" * 60)
    
    for model in models:
        model_results = [v for k, v in aggregated.items() if v['model_type'] == model]
        if model_results:
            avg_mae = np.mean([r['test_mae_mean'] for r in model_results])
            avg_rmse = np.mean([r['test_rmse_mean'] for r in model_results])
            n_params = model_results[0]['n_params']
            lines.append(f"{model:<25} avg_MAE: {avg_mae:.4f}  avg_RMSE: {avg_rmse:.4f}  params: {n_params:,}")
    
    # Save text report
    report_path = output_dir / "summary_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    print("\n" + "\n".join(lines))
    print(f"\nSaved summary to: {report_path}")
    
    # Generate LaTeX table
    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Modality Ablation Results (Test MAE)}",
        r"\label{tab:modality_ablation}",
        r"\begin{tabular}{l" + "c" * len(targets) + "}",
        r"\toprule",
        "Model & " + " & ".join(targets) + r" \\",
        r"\midrule",
    ]
    
    for model in models:
        row = model.replace("_", r"\_") + " & "
        cells = []
        for target in targets:
            key = f"{model}/{target}"
            if key in aggregated:
                v = aggregated[key]
                cells.append(f"{v['test_mae_mean']:.3f}$\\pm${v['test_mae_std']:.3f}")
            else:
                cells.append("--")
        row += " & ".join(cells) + r" \\"
        latex_lines.append(row)
    
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    latex_path = output_dir / "summary_table.tex"
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(latex_lines))
    
    print(f"Saved LaTeX table to: {latex_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Modality ablation experiments")
    parser.add_argument("--models", nargs="+", default=MODELS,
                        help=f"Models to train. Default: {MODELS}")
    parser.add_argument("--targets", nargs="+", default=TARGETS,
                        help=f"Targets to predict. Default: {TARGETS}")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                        help=f"Random seeds. Default: {SEEDS}")
    parser.add_argument("--dataset_path", type=str, default="dataset_combined.npz")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment config
    config = {
        'models': args.models,
        'targets': args.targets,
        'seeds': args.seeds,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'patience': args.patience,
        'device': str(device),
        'timestamp': datetime.now().isoformat(),
    }
    with open(output_dir / "experiment_config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    # Run experiments
    run_all_experiments(
        models=args.models,
        targets=args.targets,
        seeds=args.seeds,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        device=device,
        output_dir=output_dir,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
