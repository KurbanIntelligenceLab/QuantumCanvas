#!/usr/bin/env python3
"""
Element ID Shuffle/Mask Ablation Experiment.

Tests whether models rely on element identity priors or learn from 
spatial/orbital structure in the images.

Conditions:
1. Normal: Standard evaluation (baseline)
2. Shuffle Z: Shuffle atomic numbers across samples in batch
3. Mask Z: Set all atomic numbers to 0 (unknown element)
4. Random Z: Replace atomic numbers with random values

If shuffling/masking Z causes large MAE increase:
    → Model relies heavily on element identity (potential shortcut)
    
If shuffling/masking Z causes small increase:
    → Model learns from spatial/orbital structure (good!)

Usage:
    python rebuttal_experiments/modality/element_shuffle_ablation.py
    
    # Specific checkpoint
    python rebuttal_experiments/modality/element_shuffle_ablation.py \
        --checkpoint results_twobody/e_g_ev/quantumshellnet/seed_42/best_model.pt
"""

import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import argparse
import json
from typing import Optional, List, Dict, Any

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from benchmarks.twobody_dataloader import TwoBodyDataset
from benchmarks.models import get_model as get_benchmark_model
from benchmarks.benchmark_config import cfg
from rebuttal_experiments.modality.models import get_modality_model
from rebuttal_experiments.modality.improved_models import get_improved_model


# Configuration
RESULTS_DIR = "rebuttal_results"
MODALITY_RESULTS_DIR = "rebuttal_results/modality_ablation"
DATASET_PATH = "dataset_combined.npz"
MODELS = ["qsn_v2", "multimodal_v2", "film_cnn", "tabular_mlp", "vision_only", "geometry_only"]
SEEDS = ["42", "123", "456"]
BATCH_SIZE = 64


def get_model(model_type: str, model_config: dict, device: torch.device):
    """Load model by type."""
    modality_models = ['tabular_mlp', 'tabular_transformer', 'vision_only', 'geometry_only']
    improved_models = ['qsn_v2', 'multimodal_v2', 'film_cnn']
    
    if model_type in modality_models:
        model = get_modality_model(model_type, **model_config)
    elif model_type in improved_models:
        model = get_improved_model(model_type, **model_config)
    else:
        model = get_benchmark_model(model_type, **model_config)
    
    return model.to(device)


def forward_model(model, data, model_type: str, device: torch.device, 
                  z_override: Optional[torch.Tensor] = None):
    """
    Forward pass with optional Z override for ablation.
    
    Args:
        model: The model to evaluate
        data: PyG batch data
        model_type: Model type string
        device: Torch device
        z_override: If provided, use this instead of data.z
    """
    z = z_override if z_override is not None else data.z
    
    if model_type in ['tabular_mlp', 'tabular_transformer', 'vision_only']:
        images = torch.stack([d.image for d in data.to_data_list()]).to(device).float()
        return model(images, z, data.pos, data.batch)
    
    elif model_type == 'geometry_only':
        images = torch.stack([d.image for d in data.to_data_list()]).to(device).float()
        return model(images, z, data.pos, data.batch)
    
    elif model_type == 'quantumshellnet':
        images = torch.stack([d.image for d in data.to_data_list()]).to(device).float()
        return model(images, z, data.pos, data.batch)
    
    elif model_type == 'vit':
        # ViT doesn't use Z at all, so ablation won't affect it
        images = torch.stack([d.image[:3] for d in data.to_data_list()]).to(device).float()
        return model(images)
    
    elif model_type == 'multimodal':
        images = torch.stack([d.image[:3] for d in data.to_data_list()]).to(device).float()
        return model(z, data.pos, data.batch, images)
    
    elif model_type in ['qsn_v2', 'multimodal_v2', 'film_cnn']:
        # Improved models use all 10 channels
        images = torch.stack([d.image for d in data.to_data_list()]).to(device).float()
        return model(images, z, data.pos, data.batch)
    
    else:
        # GNN models
        return model(z, data.pos, data.batch)


def create_z_ablation(z: torch.Tensor, batch: torch.Tensor, 
                      ablation_type: str, rng: torch.Generator) -> torch.Tensor:
    """
    Create ablated atomic numbers based on ablation type.
    
    Args:
        z: Original atomic numbers [N]
        batch: Batch assignment [N]
        ablation_type: One of 'normal', 'shuffle', 'mask', 'random'
        rng: Random generator for reproducibility
        
    Returns:
        Modified atomic numbers tensor
    """
    if ablation_type == 'normal':
        return z
    
    elif ablation_type == 'shuffle':
        # Shuffle Z across all atoms in the batch
        perm = torch.randperm(z.size(0), generator=rng, device=z.device)
        return z[perm]
    
    elif ablation_type == 'shuffle_within_sample':
        # Shuffle Z only within each sample (swap atom1 and atom2)
        z_new = z.clone()
        n_samples = batch.max().item() + 1
        for i in range(n_samples):
            mask = batch == i
            indices = torch.where(mask)[0]
            if len(indices) == 2:
                # Swap the two atoms
                z_new[indices[0]], z_new[indices[1]] = z[indices[1]].clone(), z[indices[0]].clone()
        return z_new
    
    elif ablation_type == 'mask':
        # Set all Z to 0 (unknown element)
        return torch.zeros_like(z)
    
    elif ablation_type == 'random':
        # Replace with random Z values (1-103)
        return torch.randint(1, 104, z.shape, generator=rng, device=z.device, dtype=z.dtype)
    
    elif ablation_type == 'constant':
        # Set all Z to the same value (carbon = 6)
        return torch.full_like(z, 6)
    
    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")


@torch.no_grad()
def evaluate_with_ablation(model, loader, model_type: str, device: torch.device,
                           ablation_type: str, denormalize_fn=None, seed: int = 42):
    """Evaluate model with Z ablation."""
    model.eval()
    predictions = []
    targets = []
    
    rng = torch.Generator(device=device).manual_seed(seed)
    
    for data in loader:
        data = data.to(device)
        
        # Create ablated Z
        z_ablated = create_z_ablation(data.z, data.batch, ablation_type, rng)
        
        # Forward pass with ablated Z
        out = forward_model(model, data, model_type, device, z_override=z_ablated)
        
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
    
    if denormalize_fn is not None:
        predictions = np.array([denormalize_fn(p) for p in predictions])
        targets = np.array([denormalize_fn(t) for t in targets])
    
    mae = float(np.mean(np.abs(predictions - targets)))
    rmse = float(np.sqrt(np.mean((predictions - targets) ** 2)))
    
    return {'mae': mae, 'rmse': rmse}


def run_single_checkpoint(checkpoint_path: str, dataset_path: str,
                          batch_size: int, device: torch.device,
                          ablation_types: List[str]) -> Dict[str, Any]:
    """Run all ablation types on a single checkpoint."""
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model_type = ckpt.get('model_type')
    target = ckpt.get('target')
    seed = ckpt.get('seed', 42)
    norm_stats = ckpt.get('normalization_stats')
    model_config = ckpt.get('model_config', {})
    
    if model_type is None or target is None:
        raise ValueError(f"Checkpoint missing model_type or target: {checkpoint_path}")
    
    # Load model
    model = get_model(model_type, model_config, device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Load dataset
    dataset = TwoBodyDataset(
        dataset_path,
        target_label=target,
        verbose=False,
        normalize_labels=True,
        normalization_stats=norm_stats
    )
    
    # Use full dataset for evaluation (or split for consistency)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Run all ablation types
    results = {
        'checkpoint': checkpoint_path,
        'model_type': model_type,
        'target': target,
        'seed': seed,
        'ablations': {}
    }
    
    for abl_type in ablation_types:
        metrics = evaluate_with_ablation(
            model, loader, model_type, device, abl_type,
            denormalize_fn=dataset.denormalize_label,
            seed=seed
        )
        results['ablations'][abl_type] = metrics
    
    # Compute relative changes from baseline
    baseline_mae = results['ablations']['normal']['mae']
    for abl_type, metrics in results['ablations'].items():
        if abl_type != 'normal':
            delta_mae = metrics['mae'] - baseline_mae
            rel_change = delta_mae / baseline_mae * 100 if baseline_mae > 0 else 0
            results['ablations'][abl_type]['delta_mae'] = delta_mae
            results['ablations'][abl_type]['relative_change_pct'] = rel_change
    
    return results


def discover_checkpoints(results_dirs: List[Path], models: List[str], 
                         seeds: List[str]) -> List[tuple]:
    """Find all available checkpoints."""
    checkpoints = []
    
    for results_dir in results_dirs:
        if not results_dir.is_dir():
            continue
        
        for target_dir in sorted(results_dir.iterdir()):
            if not target_dir.is_dir() or target_dir.name.startswith('.'):
                continue
            
            target = target_dir.name
            
            for model_type in models:
                model_dir = target_dir / model_type
                if not model_dir.is_dir():
                    continue
                
                for s in seeds:
                    ckpt = model_dir / f"seed_{s}" / "best_model.pt"
                    if ckpt.is_file():
                        checkpoints.append((target, model_type, s, str(ckpt)))
    
    return checkpoints


def aggregate_results(all_results: List[Dict]) -> Dict:
    """Aggregate results across seeds and models."""
    
    # Group by (target, model_type)
    grouped = {}
    for r in all_results:
        key = (r['target'], r['model_type'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)
    
    aggregated = {}
    for (target, model_type), runs in grouped.items():
        agg_key = f"{target}/{model_type}"
        aggregated[agg_key] = {
            'target': target,
            'model_type': model_type,
            'n_seeds': len(runs),
            'ablations': {}
        }
        
        # Get all ablation types
        abl_types = list(runs[0]['ablations'].keys())
        
        for abl_type in abl_types:
            maes = [r['ablations'][abl_type]['mae'] for r in runs]
            rmses = [r['ablations'][abl_type]['rmse'] for r in runs]
            
            agg = {
                'mae_mean': float(np.mean(maes)),
                'mae_std': float(np.std(maes)),
                'rmse_mean': float(np.mean(rmses)),
                'rmse_std': float(np.std(rmses)),
            }
            
            if abl_type != 'normal':
                deltas = [r['ablations'][abl_type].get('delta_mae', 0) for r in runs]
                rel_changes = [r['ablations'][abl_type].get('relative_change_pct', 0) for r in runs]
                agg['delta_mae_mean'] = float(np.mean(deltas))
                agg['delta_mae_std'] = float(np.std(deltas))
                agg['relative_change_pct_mean'] = float(np.mean(rel_changes))
                agg['relative_change_pct_std'] = float(np.std(rel_changes))
            
            aggregated[agg_key]['ablations'][abl_type] = agg
    
    return aggregated


def generate_report(aggregated: Dict, output_dir: Path):
    """Generate human-readable report."""
    
    lines = []
    lines.append("=" * 100)
    lines.append("ELEMENT ID SHUFFLE/MASK ABLATION RESULTS")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  - Large positive Delta%: Model relies heavily on element identity (potential shortcut)")
    lines.append("  - Small Delta%: Model learns from spatial/orbital structure (robust!)")
    lines.append("")
    
    # Group by target
    targets = sorted(set(v['target'] for v in aggregated.values()))
    
    for target in targets:
        lines.append(f"\n{'='*80}")
        lines.append(f"TARGET: {target}")
        lines.append(f"{'='*80}")
        
        target_results = {k: v for k, v in aggregated.items() if v['target'] == target}
        
        for key, data in sorted(target_results.items()):
            model_type = data['model_type']
            lines.append(f"\n  {model_type}:")
            
            # Baseline
            normal = data['ablations'].get('normal', {})
            lines.append(f"    Normal (baseline):  MAE = {normal.get('mae_mean', 0):.4f} ± {normal.get('mae_std', 0):.4f}")
            
            # Other ablations
            for abl_type in ['shuffle', 'shuffle_within_sample', 'mask', 'random', 'constant']:
                if abl_type in data['ablations']:
                    abl = data['ablations'][abl_type]
                    delta_pct = abl.get('relative_change_pct_mean', 0)
                    delta_sign = "+" if delta_pct >= 0 else ""
                    lines.append(
                        f"    {abl_type:<20} MAE = {abl['mae_mean']:.4f} ± {abl['mae_std']:.4f}  "
                        f"Delta% = {delta_sign}{delta_pct:.1f}%"
                    )
    
    # Summary table
    lines.append("\n" + "=" * 100)
    lines.append("SUMMARY: Element Identity Dependence (% MAE increase when Z shuffled)")
    lines.append("=" * 100)
    
    models = sorted(set(v['model_type'] for v in aggregated.values()))
    header = f"{'Model':<25}" + "".join(f"{t:<20}" for t in targets)
    lines.append(header)
    lines.append("-" * 100)
    
    for model in models:
        row = f"{model:<25}"
        for target in targets:
            key = f"{target}/{model}"
            if key in aggregated and 'shuffle' in aggregated[key]['ablations']:
                delta = aggregated[key]['ablations']['shuffle'].get('relative_change_pct_mean', 0)
                row += f"{delta:>+.1f}%{'':<14}"
            else:
                row += f"{'N/A':<20}"
        lines.append(row)
    
    lines.append("")
    lines.append("Note: ViT doesn't use element IDs, so ablations have no effect on it.")
    lines.append("      Large positive values indicate high reliance on element identity.")
    lines.append("      Small/negative values indicate the model learns from other features.")
    
    # Save report
    report_path = output_dir / "element_shuffle_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    print("\n".join(lines))
    print(f"\nSaved report to: {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Single checkpoint to evaluate")
    parser.add_argument("--results_dirs", nargs="+", 
                        default=[RESULTS_DIR, MODALITY_RESULTS_DIR])
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--seeds", nargs="+", default=SEEDS)
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="rebuttal_results/element_shuffle_ablation")
    args = parser.parse_args()
    
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ablation types to test
    ablation_types = ['normal', 'shuffle', 'shuffle_within_sample', 'mask', 'random', 'constant']
    
    if args.checkpoint:
        # Single checkpoint mode
        results = run_single_checkpoint(
            args.checkpoint, args.dataset_path, args.batch_size, device, ablation_types
        )
        print(json.dumps(results, indent=2))
        
        with open(output_dir / "single_result.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
    else:
        # Discover and run all checkpoints
        results_dirs = [Path(d) for d in args.results_dirs]
        checkpoints = discover_checkpoints(results_dirs, args.models, args.seeds)
        
        if not checkpoints:
            print(f"No checkpoints found in {args.results_dirs}")
            return
        
        print(f"Found {len(checkpoints)} checkpoints")
        
        all_results = []
        for target, model_type, seed, ckpt_path in tqdm(checkpoints, desc="Evaluating"):
            try:
                result = run_single_checkpoint(
                    ckpt_path, args.dataset_path, args.batch_size, device, ablation_types
                )
                all_results.append(result)
                
                # Save individual result
                save_path = output_dir / target / model_type / f"seed_{seed}.json"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2)
                    
            except Exception as e:
                print(f"Error on {target}/{model_type}/seed_{seed}: {e}")
                import traceback
                traceback.print_exc()
        
        # Aggregate and report
        aggregated = aggregate_results(all_results)
        
        with open(output_dir / "aggregated_results.json", 'w', encoding='utf-8') as f:
            json.dump(aggregated, f, indent=2)
        
        generate_report(aggregated, output_dir)


if __name__ == "__main__":
    main()
