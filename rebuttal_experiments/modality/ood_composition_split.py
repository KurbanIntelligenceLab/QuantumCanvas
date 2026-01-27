#!/usr/bin/env python3
"""
Out-of-Distribution (OOD) Composition Split Experiment.

Tests whether models generalize to unseen element pairs, demonstrating
that pretraining/visual features capture transferable physics rather
than just memorizing element-pair correlations.

Split strategies:
1. held_out_pairs: Random 20% of pair types held out for testing
2. by_period: Train on lighter elements (periods 1-4), test on heavier (periods 5-7)
3. by_type: Train on metal-metal pairs, test on metal-nonmetal pairs
4. by_electronegativity: Train on similar-EN pairs, test on high-EN-difference pairs

Usage:
    python rebuttal_experiments/modality/ood_composition_split.py
    
    # Specific split strategy
    python rebuttal_experiments/modality/ood_composition_split.py \
        --split_strategy held_out_pairs \
        --models quantumshellnet vit tabular_mlp
"""

import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import argparse
import json
import time
from typing import List, Dict, Tuple, Set
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm

from benchmarks.twobody_dataloader import TwoBodyDataset, ELEMENT_TO_Z
from benchmarks.models import get_model as get_benchmark_model
from benchmarks.benchmark_config import cfg
from rebuttal_experiments.modality.models import get_modality_model
from rebuttal_experiments.modality.improved_models import get_improved_model


# Periodic table metadata
ELEMENT_PERIOD = {
    # Period 1
    'H': 1, 'He': 1,
    # Period 2
    'Li': 2, 'Be': 2, 'B': 2, 'C': 2, 'N': 2, 'O': 2, 'F': 2, 'Ne': 2,
    # Period 3
    'Na': 3, 'Mg': 3, 'Al': 3, 'Si': 3, 'P': 3, 'S': 3, 'Cl': 3, 'Ar': 3,
    # Period 4
    'K': 4, 'Ca': 4, 'Sc': 4, 'Ti': 4, 'V': 4, 'Cr': 4, 'Mn': 4, 'Fe': 4,
    'Co': 4, 'Ni': 4, 'Cu': 4, 'Zn': 4, 'Ga': 4, 'Ge': 4, 'As': 4, 'Se': 4,
    'Br': 4, 'Kr': 4,
    # Period 5
    'Rb': 5, 'Sr': 5, 'Y': 5, 'Zr': 5, 'Nb': 5, 'Mo': 5, 'Tc': 5, 'Ru': 5,
    'Rh': 5, 'Pd': 5, 'Ag': 5, 'Cd': 5, 'In': 5, 'Sn': 5, 'Sb': 5, 'Te': 5,
    'I': 5, 'Xe': 5,
    # Period 6
    'Cs': 6, 'Ba': 6, 'La': 6, 'Ce': 6, 'Pr': 6, 'Nd': 6, 'Pm': 6, 'Sm': 6,
    'Eu': 6, 'Gd': 6, 'Tb': 6, 'Dy': 6, 'Ho': 6, 'Er': 6, 'Tm': 6, 'Yb': 6,
    'Lu': 6, 'Hf': 6, 'Ta': 6, 'W': 6, 'Re': 6, 'Os': 6, 'Ir': 6, 'Pt': 6,
    'Au': 6, 'Hg': 6, 'Tl': 6, 'Pb': 6, 'Bi': 6, 'Po': 6, 'At': 6, 'Rn': 6,
    # Period 7
    'Fr': 7, 'Ra': 7, 'Ac': 7, 'Th': 7, 'Pa': 7, 'U': 7, 'Np': 7, 'Pu': 7,
}

# Metals vs Non-metals
METALS = {
    'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn',
    'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',
    'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Cs', 'Ba', 'La', 'Ce',
    'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb',
    'Bi', 'Po', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu'
}

# Pauling electronegativity (approximate)
ELECTRONEGATIVITY = {
    'H': 2.20, 'He': 0, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55, 
    'N': 3.04, 'O': 3.44, 'F': 3.98, 'Ne': 0, 'Na': 0.93, 'Mg': 1.31,
    'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Ar': 0,
    'K': 0.82, 'Ca': 1.00, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66,
    'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65,
    'Ga': 1.81, 'Ge': 2.01, 'As': 2.18, 'Se': 2.55, 'Br': 2.96, 'Kr': 3.00,
    'Rb': 0.82, 'Sr': 0.95, 'Y': 1.22, 'Zr': 1.33, 'Nb': 1.60, 'Mo': 2.16,
    'Tc': 1.90, 'Ru': 2.20, 'Rh': 2.28, 'Pd': 2.20, 'Ag': 1.93, 'Cd': 1.69,
    'In': 1.78, 'Sn': 1.96, 'Sb': 2.05, 'Te': 2.10, 'I': 2.66, 'Xe': 2.60,
    'Cs': 0.79, 'Ba': 0.89, 'La': 1.10, 'Ce': 1.12, 'Pr': 1.13, 'Nd': 1.14,
    'Pm': 1.13, 'Sm': 1.17, 'Eu': 1.20, 'Gd': 1.20, 'Tb': 1.10, 'Dy': 1.22,
    'Ho': 1.23, 'Er': 1.24, 'Tm': 1.25, 'Yb': 1.10, 'Lu': 1.27, 'Hf': 1.30,
    'Ta': 1.50, 'W': 2.36, 'Re': 1.90, 'Os': 2.20, 'Ir': 2.20, 'Pt': 2.28,
    'Au': 2.54, 'Hg': 2.00, 'Tl': 1.62, 'Pb': 2.33, 'Bi': 2.02, 'Po': 2.00,
    'At': 2.20, 'Rn': 0, 'Fr': 0.70, 'Ra': 0.90, 'Ac': 1.10, 'Th': 1.30,
    'Pa': 1.50, 'U': 1.38, 'Np': 1.36, 'Pu': 1.28,
}


class OODSplitDataset(torch.utils.data.Dataset):
    """Wrapper dataset that filters by indices."""
    
    def __init__(self, base_dataset: TwoBodyDataset, indices: List[int]):
        self.base_dataset = base_dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]


def get_pair_name(elements: List[str]) -> str:
    """Get canonical pair name (sorted alphabetically)."""
    return "_".join(sorted(elements))


def create_split_indices(dataset: TwoBodyDataset, split_strategy: str,
                         seed: int = 42, holdout_ratio: float = 0.2
                         ) -> Tuple[List[int], List[int], List[int], Dict]:
    """
    Create train/val/test indices based on split strategy.
    
    Returns:
        train_indices, val_indices, test_indices, split_info
    """
    rng = np.random.RandomState(seed)
    
    # Get all pair types and their sample indices
    pair_to_indices = defaultdict(list)
    for i in range(len(dataset)):
        data = dataset[i]
        pair_name = data.pair_name
        pair_to_indices[pair_name].append(i)
    
    all_pairs = list(pair_to_indices.keys())
    n_pairs = len(all_pairs)
    
    if split_strategy == 'held_out_pairs':
        # Random holdout of pair types
        rng.shuffle(all_pairs)
        n_holdout = int(n_pairs * holdout_ratio)
        test_pairs = set(all_pairs[:n_holdout])
        train_val_pairs = set(all_pairs[n_holdout:])
        
        split_info = {
            'strategy': 'held_out_pairs',
            'n_test_pairs': len(test_pairs),
            'n_train_val_pairs': len(train_val_pairs),
            'test_pairs': list(test_pairs)[:10],  # Sample for logging
        }
    
    elif split_strategy == 'by_period':
        # Train on lighter elements (periods 1-4), test on heavier (periods 5-7)
        train_val_pairs = set()
        test_pairs = set()
        
        for pair_name in all_pairs:
            elems = pair_name.split('_')
            periods = [ELEMENT_PERIOD.get(e, 4) for e in elems]
            max_period = max(periods)
            
            if max_period <= 4:
                train_val_pairs.add(pair_name)
            else:
                test_pairs.add(pair_name)
        
        split_info = {
            'strategy': 'by_period',
            'train_periods': '1-4',
            'test_periods': '5-7',
            'n_test_pairs': len(test_pairs),
            'n_train_val_pairs': len(train_val_pairs),
        }
    
    elif split_strategy == 'by_type':
        # Train on metal-metal, test on metal-nonmetal
        train_val_pairs = set()
        test_pairs = set()
        
        for pair_name in all_pairs:
            elems = pair_name.split('_')
            is_metal = [e in METALS for e in elems]
            
            if all(is_metal):  # Metal-Metal
                train_val_pairs.add(pair_name)
            else:  # At least one non-metal
                test_pairs.add(pair_name)
        
        # If test set is too small or large, adjust
        if len(test_pairs) < 0.1 * n_pairs:
            # Flip: train on metal-nonmetal, test on metal-metal
            train_val_pairs, test_pairs = test_pairs, train_val_pairs
        
        split_info = {
            'strategy': 'by_type',
            'train_type': 'metal-metal' if len(test_pairs) > 0.1 * n_pairs else 'mixed',
            'test_type': 'metal-nonmetal' if len(test_pairs) > 0.1 * n_pairs else 'metal-metal',
            'n_test_pairs': len(test_pairs),
            'n_train_val_pairs': len(train_val_pairs),
        }
    
    elif split_strategy == 'by_electronegativity':
        # Train on similar-EN pairs, test on high-EN-difference pairs
        en_diffs = []
        for pair_name in all_pairs:
            elems = pair_name.split('_')
            ens = [ELECTRONEGATIVITY.get(e, 2.0) for e in elems]
            en_diff = abs(ens[0] - ens[1]) if len(ens) == 2 else 0
            en_diffs.append((pair_name, en_diff))
        
        # Sort by EN difference
        en_diffs.sort(key=lambda x: x[1])
        
        # Bottom 80% (similar EN) for training, top 20% (high EN diff) for testing
        n_train = int(len(en_diffs) * (1 - holdout_ratio))
        train_val_pairs = set(p for p, _ in en_diffs[:n_train])
        test_pairs = set(p for p, _ in en_diffs[n_train:])
        
        split_info = {
            'strategy': 'by_electronegativity',
            'train_description': f'Low EN difference (bottom {100*(1-holdout_ratio):.0f}%)',
            'test_description': f'High EN difference (top {100*holdout_ratio:.0f}%)',
            'n_test_pairs': len(test_pairs),
            'n_train_val_pairs': len(train_val_pairs),
            'en_threshold': en_diffs[n_train][1] if n_train < len(en_diffs) else 0,
        }
    
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}")
    
    # Convert pairs to sample indices
    test_indices = []
    for pair in test_pairs:
        test_indices.extend(pair_to_indices[pair])
    
    train_val_indices = []
    for pair in train_val_pairs:
        train_val_indices.extend(pair_to_indices[pair])
    
    # Split train_val into train and val (90/10)
    rng.shuffle(train_val_indices)
    n_train = int(len(train_val_indices) * 0.9)
    train_indices = train_val_indices[:n_train]
    val_indices = train_val_indices[n_train:]
    
    split_info['n_train_samples'] = len(train_indices)
    split_info['n_val_samples'] = len(val_indices)
    split_info['n_test_samples'] = len(test_indices)
    
    return train_indices, val_indices, test_indices, split_info


def get_model(model_type: str, device: torch.device):
    """Load model by type."""
    modality_models = ['tabular_mlp', 'tabular_transformer', 'vision_only', 'geometry_only']
    improved_models = ['qsn_v2', 'multimodal_v2', 'film_cnn']
    
    if model_type in modality_models:
        model = get_modality_model(model_type)
    elif model_type in improved_models:
        model = get_improved_model(model_type)
    else:
        model_config = cfg.model_configs.get(model_type, {})
        model = get_benchmark_model(model_type, **model_config)
    
    return model.to(device)


def forward_model(model, data, model_type: str, device: torch.device):
    """Handle forward pass for different model types."""
    if model_type in ['tabular_mlp', 'tabular_transformer', 'vision_only', 'geometry_only']:
        images = torch.stack([d.image for d in data.to_data_list()]).to(device).float()
        return model(images, data.z, data.pos, data.batch)
    
    elif model_type == 'quantumshellnet':
        images = torch.stack([d.image for d in data.to_data_list()]).to(device).float()
        return model(images, data.z, data.pos, data.batch)
    
    elif model_type == 'vit':
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
        return model(data.z, data.pos, data.batch)


def train_epoch(model, loader, optimizer, criterion, model_type, device):
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
    
    if denormalize_fn is not None:
        predictions = np.array([denormalize_fn(p) for p in predictions])
        targets = np.array([denormalize_fn(t) for t in targets])
    
    mae = float(np.mean(np.abs(predictions - targets)))
    rmse = float(np.sqrt(np.mean((predictions - targets) ** 2)))
    
    return {'mae': mae, 'rmse': rmse}


def train_and_evaluate_ood(model_type: str, target: str, split_strategy: str,
                           seed: int, dataset_path: str, batch_size: int,
                           epochs: int, lr: float, patience: int,
                           device: torch.device, output_dir: Path,
                           verbose: bool = True) -> Dict:
    """Train on ID data and evaluate on both ID and OOD data."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load dataset
    full_dataset = TwoBodyDataset(
        dataset_path,
        target_label=target,
        verbose=False,
        normalize_labels=True
    )
    
    # Create OOD split
    train_idx, val_idx, test_idx, split_info = create_split_indices(
        full_dataset, split_strategy, seed=seed
    )
    
    if len(test_idx) == 0:
        print(f"Warning: No OOD test samples for {split_strategy}. Skipping.")
        return None
    
    # Create split datasets
    train_dataset = OODSplitDataset(full_dataset, train_idx)
    val_dataset = OODSplitDataset(full_dataset, val_idx)
    test_dataset = OODSplitDataset(full_dataset, test_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create model
    model = get_model(model_type, device)
    n_params = sum(p.numel() for p in model.parameters())
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience // 2)
    criterion = nn.L1Loss()
    
    # Training
    best_val_mae = float('inf')
    best_state = None
    patience_counter = 0
    
    pbar = tqdm(range(epochs), desc=f"{model_type}/{target}/{split_strategy}", disable=not verbose)
    for epoch in pbar:
        train_loss = train_epoch(model, train_loader, optimizer, criterion, model_type, device)
        val_metrics = evaluate(model, val_loader, model_type, device)
        
        scheduler.step(val_metrics['mae'])
        
        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        pbar.set_postfix({'train': f"{train_loss:.4f}", 'val': f"{val_metrics['mae']:.4f}"})
        
        if patience_counter >= patience:
            break
    
    # Load best model
    model.load_state_dict(best_state)
    
    # Evaluate on ID (val) and OOD (test)
    id_metrics = evaluate(model, val_loader, model_type, device,
                         denormalize_fn=full_dataset.denormalize_label)
    ood_metrics = evaluate(model, test_loader, model_type, device,
                          denormalize_fn=full_dataset.denormalize_label)
    
    # Compute generalization gap
    gen_gap_mae = ood_metrics['mae'] - id_metrics['mae']
    gen_gap_pct = gen_gap_mae / id_metrics['mae'] * 100 if id_metrics['mae'] > 0 else 0
    
    result = {
        'model_type': model_type,
        'target': target,
        'split_strategy': split_strategy,
        'seed': seed,
        'n_params': n_params,
        'split_info': split_info,
        'id_mae': id_metrics['mae'],
        'id_rmse': id_metrics['rmse'],
        'ood_mae': ood_metrics['mae'],
        'ood_rmse': ood_metrics['rmse'],
        'generalization_gap_mae': gen_gap_mae,
        'generalization_gap_pct': gen_gap_pct,
    }
    
    # Save checkpoint
    save_dir = output_dir / split_strategy / target / model_type / f"seed_{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': best_state,
        'model_type': model_type,
        'target': target,
        'seed': seed,
        'split_strategy': split_strategy,
        'normalization_stats': full_dataset.get_normalization_stats(),
    }, save_dir / "best_model.pt")
    
    with open(save_dir / "results.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    return result


def run_all_ood_experiments(models: List[str], targets: List[str],
                            split_strategies: List[str], seeds: List[int],
                            dataset_path: str, batch_size: int, epochs: int,
                            lr: float, patience: int, device: torch.device,
                            output_dir: Path, verbose: bool = True) -> Dict:
    """Run all OOD experiments."""
    
    all_results = []
    
    total = len(models) * len(targets) * len(split_strategies) * len(seeds)
    print(f"\nRunning {total} OOD experiments")
    print(f"Models: {models}")
    print(f"Targets: {targets}")
    print(f"Splits: {split_strategies}")
    
    for split_strategy in split_strategies:
        for target in targets:
            for model_type in models:
                for seed in seeds:
                    try:
                        result = train_and_evaluate_ood(
                            model_type=model_type,
                            target=target,
                            split_strategy=split_strategy,
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
                        if result:
                            all_results.append(result)
                            print(f"✓ {split_strategy}/{target}/{model_type}/seed_{seed}: "
                                  f"ID={result['id_mae']:.4f}, OOD={result['ood_mae']:.4f}, "
                                  f"Gap={result['generalization_gap_pct']:+.1f}%")
                    except Exception as e:
                        print(f"✗ {split_strategy}/{target}/{model_type}/seed_{seed}: {e}")
                        import traceback
                        traceback.print_exc()
    
    # Aggregate results
    aggregated = aggregate_ood_results(all_results)
    
    with open(output_dir / "aggregated_results.json", 'w', encoding='utf-8') as f:
        json.dump(aggregated, f, indent=2)
    
    generate_ood_report(aggregated, output_dir)
    
    return aggregated


def aggregate_ood_results(results: List[Dict]) -> Dict:
    """Aggregate OOD results across seeds."""
    
    grouped = {}
    for r in results:
        key = (r['split_strategy'], r['target'], r['model_type'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)
    
    aggregated = {}
    for (split, target, model), runs in grouped.items():
        key = f"{split}/{target}/{model}"
        
        aggregated[key] = {
            'split_strategy': split,
            'target': target,
            'model_type': model,
            'n_seeds': len(runs),
            'id_mae_mean': float(np.mean([r['id_mae'] for r in runs])),
            'id_mae_std': float(np.std([r['id_mae'] for r in runs])),
            'ood_mae_mean': float(np.mean([r['ood_mae'] for r in runs])),
            'ood_mae_std': float(np.std([r['ood_mae'] for r in runs])),
            'gap_mae_mean': float(np.mean([r['generalization_gap_mae'] for r in runs])),
            'gap_mae_std': float(np.std([r['generalization_gap_mae'] for r in runs])),
            'gap_pct_mean': float(np.mean([r['generalization_gap_pct'] for r in runs])),
            'gap_pct_std': float(np.std([r['generalization_gap_pct'] for r in runs])),
            'split_info': runs[0]['split_info'],
        }
    
    return aggregated


def generate_ood_report(aggregated: Dict, output_dir: Path):
    """Generate OOD experiment report."""
    
    lines = []
    lines.append("=" * 100)
    lines.append("OOD COMPOSITION SPLIT RESULTS")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  - Small generalization gap: Model learns transferable features")
    lines.append("  - Large generalization gap: Model memorizes training compositions")
    lines.append("")
    
    # Group by split strategy
    splits = sorted(set(v['split_strategy'] for v in aggregated.values()))
    
    for split in splits:
        lines.append(f"\n{'='*80}")
        lines.append(f"SPLIT STRATEGY: {split}")
        lines.append(f"{'='*80}")
        
        split_results = {k: v for k, v in aggregated.items() if v['split_strategy'] == split}
        
        if split_results:
            # Get split info from first result
            first_result = list(split_results.values())[0]
            split_info = first_result.get('split_info', {})
            lines.append(f"  Train samples: {split_info.get('n_train_samples', 'N/A')}")
            lines.append(f"  Val samples: {split_info.get('n_val_samples', 'N/A')}")
            lines.append(f"  Test (OOD) samples: {split_info.get('n_test_samples', 'N/A')}")
        
        targets = sorted(set(v['target'] for v in split_results.values()))
        
        for target in targets:
            lines.append(f"\n  --- {target} ---")
            
            target_results = {k: v for k, v in split_results.items() if v['target'] == target}
            
            for key, data in sorted(target_results.items()):
                model = data['model_type']
                lines.append(
                    f"    {model:<20} "
                    f"ID: {data['id_mae_mean']:.4f}±{data['id_mae_std']:.4f}  "
                    f"OOD: {data['ood_mae_mean']:.4f}±{data['ood_mae_std']:.4f}  "
                    f"Gap: {data['gap_pct_mean']:+.1f}%±{data['gap_pct_std']:.1f}%"
                )
    
    # Summary table: average gap per model across all splits/targets
    lines.append("\n" + "=" * 100)
    lines.append("SUMMARY: Average Generalization Gap by Model")
    lines.append("=" * 100)
    
    models = sorted(set(v['model_type'] for v in aggregated.values()))
    
    for model in models:
        model_results = [v for v in aggregated.values() if v['model_type'] == model]
        if model_results:
            avg_gap = np.mean([r['gap_pct_mean'] for r in model_results])
            avg_id = np.mean([r['id_mae_mean'] for r in model_results])
            avg_ood = np.mean([r['ood_mae_mean'] for r in model_results])
            lines.append(f"  {model:<25} avg_ID: {avg_id:.4f}  avg_OOD: {avg_ood:.4f}  avg_Gap: {avg_gap:+.1f}%")
    
    # Save
    report_path = output_dir / "ood_composition_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    print("\n".join(lines))
    print(f"\nSaved report to: {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", 
                        default=['tabular_mlp', 'vision_only', 'geometry_only', 'qsn_v2', 'multimodal_v2', 'film_cnn'])
    parser.add_argument("--targets", nargs="+", 
                        default=['e_g_ev', 'total_energy_ev', 'dipole_mag_d'])
    parser.add_argument("--split_strategies", nargs="+",
                        default=['held_out_pairs', 'by_period', 'by_electronegativity'])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--dataset_path", type=str, default="dataset_combined.npz")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="rebuttal_results/ood_composition")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = vars(args).copy()
    config['device'] = str(device)
    with open(output_dir / "config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    run_all_ood_experiments(
        models=args.models,
        targets=args.targets,
        split_strategies=args.split_strategies,
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
