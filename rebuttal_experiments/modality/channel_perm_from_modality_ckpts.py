#!/usr/bin/env python3
"""
Channel permutation analysis using modality ablation checkpoints (QSN).

Computes per-channel delta MAE and percent change relative to baseline MAE,
then averages across specified targets (and seeds if available).

Example:
  python rebuttal_experiments/channel_perm_from_modality_ckpts.py \
    --results_dir rebuttal_results/modality_ablation \
    --targets dipole_mag_d e_g_ev e_homo_ev e_lumo_ev total_energy_ev \
    --model qsn_v2 \
    --output_dir rebuttal_results/channel_perm
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmarks.twobody_dataloader import TwoBodyDataset
from benchmarks.models import get_model as get_benchmark_model
from benchmarks.benchmark_config import cfg
from rebuttal_experiments.modality.models import get_modality_model
from rebuttal_experiments.modality.improved_models import get_improved_model


def _maybe_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _extract_images(data, device: torch.device):
    images = torch.stack([d.image for d in data.to_data_list()]).to(device).float()
    return images


@torch.no_grad()
def _forward_model(model, data, images, model_type: str, device: torch.device):
    if model_type in ['tabular_mlp', 'tabular_transformer', 'vision_only', 'geometry_only']:
        return model(images, data.z, data.pos, data.batch)
    if model_type in ['qsn_v2', 'multimodal_v2', 'film_cnn', 'quantumshellnet']:
        return model(images, data.z, data.pos, data.batch)
    if model_type == 'vit':
        return model(images[:, :3])
    if model_type == 'multimodal':
        return model(data.z, data.pos, data.batch, images[:, :3])
    return model(data.z, data.pos, data.batch)


@torch.no_grad()
def evaluate(model, loader, device, model_type, denormalize_fn=None, permute_channels=None, rng=None):
    predictions = []
    targets = []
    for data in loader:
        data = data.to(device)
        images = _extract_images(data, device)
        if permute_channels:
            for ch in permute_channels:
                perm = torch.randperm(images.size(0), device=device, generator=rng)
                images[:, ch] = images[perm, ch]
        out = _forward_model(model, data, images, model_type, device)
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
    return mae


def find_checkpoints(results_dir: Path, targets, model, seeds=None):
    ckpts = []
    for target in targets:
        target_dir = results_dir / target / model
        if not target_dir.is_dir():
            continue
        seed_dirs = sorted([d for d in target_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")])
        for seed_dir in seed_dirs:
            if seeds is not None and seed_dir.name.split("_", 1)[-1] not in seeds:
                continue
            ckpt = seed_dir / "best_model.pt"
            if ckpt.is_file():
                ckpts.append((target, seed_dir.name.split("_", 1)[-1], ckpt))
    return ckpts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="rebuttal_results/modality_ablation")
    parser.add_argument("--dataset_path", type=str, default="dataset_combined.npz")
    parser.add_argument("--targets", nargs="+", default=[
        "dipole_mag_d", "e_g_ev", "e_homo_ev", "e_lumo_ev", "total_energy_ev"
    ])
    parser.add_argument("--model", type=str, default="qsn_v2")
    parser.add_argument("--seeds", nargs="+", default=None,
                        help="Optional list of seed ids to include (e.g., 42 123 456)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Limit dataset to first N samples (0 = full)")
    parser.add_argument("--output_dir", type=str, default="rebuttal_results/channel_perm")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for permutations")
    return parser.parse_args()


def build_model(model_type: str, model_config: dict, device: torch.device):
    modality_models = ['tabular_mlp', 'tabular_transformer', 'vision_only', 'geometry_only']
    improved_models = ['qsn_v2', 'multimodal_v2', 'film_cnn']
    if model_type in modality_models:
        model = get_modality_model(model_type)
    elif model_type in improved_models:
        model = get_improved_model(model_type, **model_config)
    else:
        cfg_kwargs = cfg.model_configs.get(model_type, {})
        cfg_kwargs.update(model_config)
        model = get_benchmark_model(model_type, **cfg_kwargs)
    return model.to(device)


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    ckpts = find_checkpoints(results_dir, args.targets, args.model, args.seeds)
    if not ckpts:
        raise SystemExit(f"No checkpoints found under {results_dir} for {args.model} on {args.targets}")

    # Organize by target
    by_target = {}
    for target, seed, ckpt in ckpts:
        by_target.setdefault(target, []).append((seed, ckpt))

    per_target = {}
    for target, seed_ckpts in sorted(by_target.items()):
        seed_results = []
        for seed, ckpt_path in seed_ckpts:
            ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
            model_config = ckpt.get("model_config", {})
            model_type = ckpt.get("model_type", args.model)
            model = build_model(model_type, model_config, device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            norm_stats = ckpt.get("normalization_stats")
            dataset = TwoBodyDataset(
                str(dataset_path),
                target_label=target,
                verbose=False,
                normalize_labels=True,
                normalization_stats=norm_stats,
            )
            denormalize_fn = dataset.denormalize_label

            if args.max_samples > 0:
                from torch.utils.data import Subset
                dataset = Subset(dataset, list(range(min(args.max_samples, len(dataset)))))

            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            _maybe_sync(device)
            baseline_mae = evaluate(model, loader, device, model_type, denormalize_fn)
            _maybe_sync(device)

            rng = torch.Generator(device=device).manual_seed(args.seed)
            deltas = {}
            for ch in range(10):
                mae = evaluate(
                    model,
                    loader,
                    device,
                    model_type,
                    denormalize_fn,
                    permute_channels=[ch],
                    rng=rng,
                )
                deltas[f"ch_{ch}"] = mae - baseline_mae
            seed_results.append({
                "seed": seed,
                "baseline_mae": baseline_mae,
                "delta_mae": deltas,
            })

        # Average across seeds
        baseline_mean = float(np.mean([r["baseline_mae"] for r in seed_results]))
        channel_keys = sorted(seed_results[0]["delta_mae"].keys())
        delta_mean = {}
        for ch in channel_keys:
            delta_mean[ch] = float(np.mean([r["delta_mae"][ch] for r in seed_results]))
        per_target[target] = {
            "baseline_mae": baseline_mean,
            "delta_mae": delta_mean,
            "pct_delta": {ch: (delta_mean[ch] / baseline_mean) * 100.0 for ch in channel_keys},
            "n_seeds": len(seed_results),
            "seeds": [r["seed"] for r in seed_results],
        }

    # Average across targets
    channel_keys = sorted(next(iter(per_target.values()))["pct_delta"].keys())
    avg_pct = {}
    for ch in channel_keys:
        avg_pct[ch] = float(np.mean([per_target[t]["pct_delta"][ch] for t in per_target]))

    summary = {
        "model": args.model,
        "targets": args.targets,
        "n_targets": len(per_target),
        "per_target": per_target,
        "avg_pct_delta": avg_pct,
    }

    json_path = output_dir / f"{args.model}_channel_perm_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Write a compact markdown summary for rebuttal
    md_lines = []
    md_lines.append("## QuantumshellNet Channel Permutation Summary")
    md_lines.append("")
    md_lines.append(f"- Model: `{args.model}`")
    md_lines.append(f"- Targets: {', '.join(args.targets)}")
    md_lines.append(f"- Seeds: {', '.join(sorted({str(s) for t in per_target.values() for s in t['seeds']}))}")
    md_lines.append("")
    md_lines.append("### Average % ΔMAE per Channel (across targets)")
    md_lines.append("")
    md_lines.append("| Channel | Avg % ΔMAE |")
    md_lines.append("| --- | --- |")
    for ch in sorted(avg_pct.keys(), key=lambda c: abs(avg_pct[c]), reverse=True):
        md_lines.append(f"| {ch} | {avg_pct[ch]:.3f}% |")

    md_lines.append("")
    md_lines.append("### Per-Target % ΔMAE (baseline-normalized)")
    for target in per_target:
        md_lines.append(f"**{target}** (baseline MAE: {per_target[target]['baseline_mae']:.6f})")
        md_lines.append("")
        md_lines.append("| Channel | % ΔMAE |")
        md_lines.append("| --- | --- |")
        for ch in channel_keys:
            md_lines.append(f"| {ch} | {per_target[target]['pct_delta'][ch]:.3f}% |")
        md_lines.append("")

    md_path = output_dir / f"{args.model}_channel_perm_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
