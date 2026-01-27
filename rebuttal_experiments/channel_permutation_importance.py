#!/usr/bin/env python3
"""
Channel permutation importance for image-based models.

Evaluates performance drop when permuting image channels across samples.
Useful for ablation/feature-importance analysis in rebuttal.

Runs by default for all available targets under results_twobody for vit,
quantumshellnet (qsn), and multimodal. Reports per target per model and
average per target.

Usage:
  python rebuttal_experiments/channel_permutation_importance.py
"""

import sys
from pathlib import Path

# Ensure project root is on path when run as script
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import json
from itertools import combinations

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from benchmarks.twobody_dataloader import TwoBodyDataset
from benchmarks.models import get_model

# Defaults (no CLI args)
RESULTS_DIR = "results_twobody"
DATASET_PATH = "dataset_combined.npz"
# Default: only QSN. Set to ["vit", "quantumshellnet", "multimodal"] to run all.
MODELS = ["quantumshellnet"]
SEEDS = ["42", "123", "456"]
BATCH_SIZE = 64
DEVICE = None
NUM_BATCHES = 0
SEED = 42
GROUPED = False


def _maybe_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _extract_images(data, model_type: str, device: torch.device, channel_indices=None):
    """
    Extract images for the model.
    For ViT/Multimodal: if channel_indices is provided, use those channels from the 10-channel image.
    Otherwise, use first 3 channels (default).
    For QSN: always use all 10 channels.
    """
    if model_type in {"vit", "multimodal"}:
        if channel_indices is not None:
            # Use specified channels from the 10-channel image
            images = torch.stack([d.image[channel_indices] for d in data.to_data_list()]).to(device).float()
        else:
            # Default: use first 3 channels
            images = torch.stack([d.image[:3] for d in data.to_data_list()]).to(device).float()
    else:
        images = torch.stack([d.image for d in data.to_data_list()]).to(device).float()
    return images


@torch.no_grad()
def evaluate(model, loader, device, model_type, denormalize_fn=None, permute_channels=None, rng=None, channel_indices=None):
    predictions = []
    targets = []

    for data in loader:
        data = data.to(device)
        images = _extract_images(data, model_type, device, channel_indices=channel_indices)

        if permute_channels:
            for ch in permute_channels:
                perm = torch.randperm(images.size(0), device=device, generator=rng)
                images[:, ch] = images[perm, ch]

        if model_type == "multimodal":
            out = model(data.z, data.pos, data.batch, images)
        elif model_type == "vit":
            out = model(images)
        else:
            out = model(images, data.z, data.pos, data.batch)

        if out.dim() > 1:
            out = out.squeeze()

        pred_np = out.detach().cpu().numpy()
        target_np = data.y.detach().cpu().numpy()

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
    return mae, rmse


def discover_checkpoints(results_dir: Path, models: list, seeds: list):
    """Yield (target, model_type, seed, checkpoint_path) for each available best_model.pt."""
    if not results_dir.is_dir():
        return
    for target_dir in sorted(results_dir.iterdir()):
        if not target_dir.is_dir() or target_dir.suffix or target_dir.name.startswith("."):
            continue
        target = target_dir.name
        for model_type in models:
            model_dir = target_dir / model_type
            if not model_dir.is_dir():
                continue
            for s in seeds:
                ckpt = model_dir / f"seed_{s}" / "best_model.pt"
                if ckpt.is_file():
                    yield (target, model_type, s, str(ckpt))


def run_single(checkpoint_path: str, target: str, model_type: str, dataset_path: str,
               batch_size: int = BATCH_SIZE, device=None, num_batches: int = NUM_BATCHES,
               seed: int = SEED, num_images: int = 10):
    """Run channel permutation importance for one checkpoint. Returns results dict."""
    ckpt = torch.load(checkpoint_path, weights_only=False)
    mt = ckpt.get("model_type") or model_type
    tg = ckpt.get("target") or target
    if mt is None or tg is None:
        raise ValueError(f"model_type and target must be in checkpoint or provided: {checkpoint_path}")

    model_config = ckpt.get("model_config", {})
    model = get_model(mt, **model_config)

    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    norm_stats = ckpt.get("normalization_stats")
    full_dataset = TwoBodyDataset(
        dataset_path,
        target_label=tg,
        verbose=False,
        normalize_labels=True,
        normalization_stats=norm_stats,
    )
    
    # Store denormalize function before potentially creating subset
    denormalize_fn = full_dataset.denormalize_label
    
    # For QSN, limit to num_images (10) samples
    if mt == "quantumshellnet":
        # Create a subset dataset with only first num_images samples
        subset_indices = list(range(min(num_images, len(full_dataset))))
        from torch.utils.data import Subset
        dataset = Subset(full_dataset, subset_indices)
    else:
        dataset = full_dataset
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    if num_batches > 0:
        loader = list(loader)[:num_batches]

    _maybe_sync(dev)
    baseline_mae, baseline_rmse = evaluate(model, loader, dev, mt, denormalize_fn)
    _maybe_sync(dev)

    rng = torch.Generator(device=dev).manual_seed(seed)
    importance = {}
    
    if mt == "quantumshellnet":
        # QSN: Individual channel importance (permute each channel separately)
        for ch in tqdm(range(10), desc="Channels", leave=False):
            mae, rmse = evaluate(model, loader, dev, mt, denormalize_fn, permute_channels=[ch], rng=rng)
            importance[f"ch_{ch}"] = {
                "mae": mae,
                "rmse": rmse,
                "delta_mae": mae - baseline_mae,
                "delta_rmse": rmse - baseline_rmse,
            }
    else:
        # ViT and Multimodal: Permute combinations of 3 channels from all 10
        # For each combination of 3 channels, use those channels and permute them
        channel_combinations = list(combinations(range(10), 3))
        combination_results = {}
        
        for combo in tqdm(channel_combinations, desc="Combinations", leave=False):
            combo_list = list(combo)
            combo_key = f"combo_{combo_list[0]}_{combo_list[1]}_{combo_list[2]}"
            # Use these 3 channels and permute all 3 of them (indices 0, 1, 2 in the extracted 3-channel image)
            mae, rmse = evaluate(model, loader, dev, mt, denormalize_fn, 
                                permute_channels=[0, 1, 2], rng=rng, channel_indices=combo_list)
            combination_results[combo_key] = {
                "channels": combo_list,
                "mae": mae,
                "rmse": rmse,
                "delta_mae": mae - baseline_mae,
                "delta_rmse": rmse - baseline_rmse,
            }
        
        # Compute per-channel importance by averaging over all combinations that include that channel
        for ch in range(10):
            # Find all combinations that include this channel
            relevant_combos = []
            for combo_key, combo_data in combination_results.items():
                if ch in combo_data["channels"]:
                    relevant_combos.append(combo_data)
            
            if relevant_combos:
                avg_delta_mae = float(np.mean([c["delta_mae"] for c in relevant_combos]))
                avg_delta_rmse = float(np.mean([c["delta_rmse"] for c in relevant_combos]))
                avg_mae = float(np.mean([c["mae"] for c in relevant_combos]))
                avg_rmse = float(np.mean([c["rmse"] for c in relevant_combos]))
                
                importance[f"ch_{ch}"] = {
                    "mae": avg_mae,
                    "rmse": avg_rmse,
                    "delta_mae": avg_delta_mae,
                    "delta_rmse": avg_delta_rmse,
                    "n_combinations": len(relevant_combos),
                }

    mean_abs_delta_mae = float(np.mean([abs(imp["delta_mae"]) for imp in importance.values()]))
    mean_abs_delta_rmse = float(np.mean([abs(imp["delta_rmse"]) for imp in importance.values()]))

    return {
        "checkpoint": checkpoint_path,
        "model_type": mt,
        "target": tg,
        "seed": seed,
        "baseline": {"mae": baseline_mae, "rmse": baseline_rmse},
        "importance": importance,
        "mean_abs_delta_mae": mean_abs_delta_mae,
        "mean_abs_delta_rmse": mean_abs_delta_rmse,
    }


def main():
    results_dir = Path(__file__).resolve().parent.parent / RESULTS_DIR
    if not results_dir.is_dir():
        results_dir = Path.cwd() / RESULTS_DIR
    dataset_path = Path(__file__).resolve().parent.parent / DATASET_PATH
    if not dataset_path.is_file():
        dataset_path = Path.cwd() / DATASET_PATH
    dataset_path = str(dataset_path)

    if not results_dir.is_dir():
        print(f"Results directory not found: {results_dir}")
        return

    checkpoints = list(discover_checkpoints(results_dir, MODELS, SEEDS))
    if not checkpoints:
        print(f"No checkpoints found under {results_dir} for {MODELS}. Expected: <target>/<model>/seed_*/best_model.pt")
        return

    print(f"Found {len(checkpoints)} checkpoints. Running channel permutation importance...\n")
    all_results = []
    for target, model_type, seed, ckpt in tqdm(checkpoints, desc="Checkpoints"):
        try:
            r = run_single(ckpt, target, model_type, dataset_path, batch_size=BATCH_SIZE,
                          device=DEVICE, num_batches=NUM_BATCHES, seed=int(seed))
            all_results.append(r)
            # Save per-run json (per seed)
            out_path = results_dir / f"perm_importance_{r['target']}_{r['model_type']}_seed_{seed}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(r, f, indent=2)
        except Exception as e:
            print(f"Error {target}/{model_type}/seed_{seed}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # --- Group results by target and model, then average over seeds ---
    by_target_model = {}
    for r in all_results:
        t, m = r["target"], r["model_type"]
        key = (t, m)
        if key not in by_target_model:
            by_target_model[key] = {
                "baselines": [],
                "importances": [],  # List of importance dicts per seed
                "mean_abs_deltas_mae": [],
                "mean_abs_deltas_rmse": [],
                "seeds": [],
            }
        by_target_model[key]["baselines"].append(r["baseline"])
        by_target_model[key]["importances"].append(r["importance"])
        by_target_model[key]["mean_abs_deltas_mae"].append(r["mean_abs_delta_mae"])
        by_target_model[key]["mean_abs_deltas_rmse"].append(r["mean_abs_delta_rmse"])
        by_target_model[key]["seeds"].append(r["seed"])

    # --- Average over seeds per target per model ---
    by_target = {}
    for (target, model_type), data in by_target_model.items():
        # Average baselines
        avg_baseline_mae = float(np.mean([b["mae"] for b in data["baselines"]]))
        avg_baseline_rmse = float(np.mean([b["rmse"] for b in data["baselines"]]))
        
        # Average per-channel importance over seeds
        # Get all channel keys (should be ch_0 to ch_9)
        channel_keys = set()
        for imp_dict in data["importances"]:
            channel_keys.update(imp_dict.keys())
        channel_keys = sorted(channel_keys)
        
        avg_importance = {}
        for ch_key in channel_keys:
            ch_deltas_mae = []
            ch_deltas_rmse = []
            ch_maes = []
            ch_rmses = []
            for imp_dict in data["importances"]:
                if ch_key in imp_dict:
                    ch_deltas_mae.append(imp_dict[ch_key]["delta_mae"])
                    ch_deltas_rmse.append(imp_dict[ch_key]["delta_rmse"])
                    ch_maes.append(imp_dict[ch_key]["mae"])
                    ch_rmses.append(imp_dict[ch_key]["rmse"])
            
            if ch_deltas_mae:
                avg_importance[ch_key] = {
                    "delta_mae": float(np.mean(ch_deltas_mae)),
                    "delta_rmse": float(np.mean(ch_deltas_rmse)),
                    "mae": float(np.mean(ch_maes)),
                    "rmse": float(np.mean(ch_rmses)),
                }
        
        avg_mean_abs_delta_mae = float(np.mean(data["mean_abs_deltas_mae"]))
        avg_mean_abs_delta_rmse = float(np.mean(data["mean_abs_deltas_rmse"]))
        
        by_target.setdefault(target, {})[model_type] = {
            "baseline_mae": avg_baseline_mae,
            "baseline_rmse": avg_baseline_rmse,
            "mean_abs_delta_mae": avg_mean_abs_delta_mae,
            "mean_abs_delta_rmse": avg_mean_abs_delta_rmse,
            "importance": avg_importance,
            "n_seeds": len(data["seeds"]),
            "seeds": data["seeds"],
        }

    # --- Average per target (over models) ---
    average_per_target = {}
    for target in sorted(by_target.keys()):
        row = by_target[target]
        vals_mae = [row[m]["baseline_mae"] for m in MODELS if m in row]
        vals_rmse = [row[m]["baseline_rmse"] for m in MODELS if m in row]
        delta_mae = [row[m]["mean_abs_delta_mae"] for m in MODELS if m in row]
        delta_rmse = [row[m]["mean_abs_delta_rmse"] for m in MODELS if m in row]
        n = len(vals_mae)
        average_per_target[target] = {
            "n_models": n,
            "avg_baseline_mae": float(np.mean(vals_mae)) if vals_mae else None,
            "avg_baseline_rmse": float(np.mean(vals_rmse)) if vals_rmse else None,
            "avg_mean_abs_delta_mae": float(np.mean(delta_mae)) if delta_mae else None,
            "avg_mean_abs_delta_rmse": float(np.mean(delta_rmse)) if delta_rmse else None,
        }

    # --- Per-channel importance per target (averaged over models and seeds) ---
    per_channel_per_target = {}
    for target in sorted(by_target.keys()):
        row = by_target[target]
        # Get all channel keys
        channel_keys = set()
        for m in MODELS:
            if m in row:
                channel_keys.update(row[m]["importance"].keys())
        channel_keys = sorted(channel_keys)
        
        per_channel = {}
        for ch_key in channel_keys:
            ch_deltas_mae = []
            ch_deltas_rmse = []
            for m in MODELS:
                if m in row and ch_key in row[m]["importance"]:
                    ch_deltas_mae.append(row[m]["importance"][ch_key]["delta_mae"])
                    ch_deltas_rmse.append(row[m]["importance"][ch_key]["delta_rmse"])
            
            if ch_deltas_mae:
                per_channel[ch_key] = {
                    "delta_mae": float(np.mean(ch_deltas_mae)),
                    "delta_rmse": float(np.mean(ch_deltas_rmse)),
                }
        
        per_channel_per_target[target] = per_channel

    # Build report to save
    report = {
        "per_target_per_model": by_target,
        "average_per_target": average_per_target,
        "per_channel_per_target": per_channel_per_target,
        "checkpoints_run": [{"target": t, "model": m, "seed": s, "checkpoint": c} for t, m, s, c in checkpoints],
        "n_checkpoints": len(checkpoints),
    }

    def _json_enc(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if np.isfinite(obj) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    report_path = results_dir / "channel_permutation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=_json_enc)
    print(f"Saved full report (per target per model + average per target + per-channel per target) to: {report_path}")

    # Human-readable .txt report
    lines = []
    lines.append("=" * 80)
    lines.append("PER TARGET PER MODEL (averaged over 3 seeds)")
    lines.append("=" * 80)
    for target in sorted(by_target.keys()):
        row = by_target[target]
        lines.append(f"\n--- {target} ---")
        for model_type in MODELS:
            if model_type in row:
                d = row[model_type]
                lines.append(f"  {model_type:16}  baseline MAE: {d['baseline_mae']:.6f}  baseline RMSE: {d['baseline_rmse']:.6f}  "
                            f"mean|ΔMAE|: {d['mean_abs_delta_mae']:.6f}  mean|ΔRMSE|: {d['mean_abs_delta_rmse']:.6f}  "
                            f"(n_seeds={d['n_seeds']})")

    lines.append("\n" + "=" * 80)
    lines.append("PER-CHANNEL IMPORTANCE PER TARGET (averaged over models and seeds)")
    lines.append("=" * 80)
    for target in sorted(per_channel_per_target.keys()):
        lines.append(f"\n--- {target} ---")
        ch_data = per_channel_per_target[target]
        for ch_key in sorted(ch_data.keys()):
            ch = ch_data[ch_key]
            lines.append(f"  {ch_key:8}  ΔMAE: {ch['delta_mae']:10.6f}  ΔRMSE: {ch['delta_rmse']:10.6f}")

    lines.append("\n" + "=" * 80)
    lines.append(f"AVERAGE PER TARGET (over {', '.join(MODELS)})")
    lines.append("=" * 80)
    for target in sorted(by_target.keys()):
        a = average_per_target[target]
        n, avg_mae, avg_rmse = a["n_models"], a["avg_baseline_mae"], a["avg_baseline_rmse"]
        avg_dm, avg_dr = a["avg_mean_abs_delta_mae"], a["avg_mean_abs_delta_rmse"]
        avg_mae_s = f"{avg_mae:.6f}" if avg_mae is not None else "N/A"
        avg_rmse_s = f"{avg_rmse:.6f}" if avg_rmse is not None else "N/A"
        avg_dm_s = f"{avg_dm:.6f}" if avg_dm is not None else "N/A"
        avg_dr_s = f"{avg_dr:.6f}" if avg_dr is not None else "N/A"
        lines.append(f"  {target:28}  models: {n}  avg baseline MAE: {avg_mae_s}  avg baseline RMSE: {avg_rmse_s}  "
                    f"avg mean|ΔMAE|: {avg_dm_s}  avg mean|ΔRMSE|: {avg_dr_s}")

    report_txt_path = results_dir / "channel_permutation_report.txt"
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved readable report to: {report_txt_path}")

    # Print to stdout as well
    print("\n" + "\n".join(lines))
    print("\nDone.")

if __name__ == "__main__":
    main()
