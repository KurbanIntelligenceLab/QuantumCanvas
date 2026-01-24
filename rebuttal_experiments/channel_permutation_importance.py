#!/usr/bin/env python3
"""
Channel permutation importance for image-based models.

Evaluates performance drop when permuting image channels across samples.
Useful for ablation/feature-importance analysis in rebuttal.

Usage:
  python benchmarks/channel_permutation_importance.py \
    --checkpoint_path results_twobody/e_g_ev/quantumshellnet/seed_42/best_model.pt \
    --dataset_path dataset_combined.npz \
    --target e_g_ev
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from benchmarks.twobody_dataloader import TwoBodyDataset
from benchmarks.models import get_model


def _maybe_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _extract_images(data, model_type: str, device: torch.device):
    if model_type in {"vit", "multimodal"}:
        images = torch.stack([d.image[:3] for d in data.to_data_list()]).to(device).float()
    else:
        images = torch.stack([d.image for d in data.to_data_list()]).to(device).float()
    return images


def _forward_model(model, data, model_type: str, device: torch.device):
    if model_type == "multimodal":
        images = _extract_images(data, model_type, device)
        return model(data.z, data.pos, data.batch, images)
    if model_type == "vit":
        images = _extract_images(data, model_type, device)
        return model(images)
    if model_type == "quantumshellnet":
        images = _extract_images(data, model_type, device)
        return model(images, data.z, data.pos, data.batch)
    raise ValueError(f"Unsupported model_type for channel importance: {model_type}")


@torch.no_grad()
def evaluate(model, loader, device, model_type, denormalize_fn=None, permute_channels=None, rng=None):
    predictions = []
    targets = []

    for data in loader:
        data = data.to(device)
        images = _extract_images(data, model_type, device)

        if permute_channels:
            # Shuffle specified channels across batch dimension
            for ch in permute_channels:
                perm = torch.randperm(images.size(0), device=device, generator=rng)
                images[:, ch] = images[perm, ch]

        # Run forward pass
        if model_type == "multimodal":
            out = model(data.z, data.pos, data.batch, images)
        elif model_type == "vit":
            out = model(images)
        else:  # quantumshellnet
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="dataset_combined.npz")
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grouped", action="store_true")
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint_path, weights_only=False)

    model_type = args.model_type or checkpoint.get("model_type")
    target = args.target or checkpoint.get("target")
    if model_type is None or target is None:
        raise ValueError("model_type and target must be provided via args or checkpoint")

    model_config = checkpoint.get("model_config", {})
    model = get_model(model_type, **model_config)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    norm_stats = checkpoint.get("normalization_stats")
    dataset = TwoBodyDataset(
        args.dataset_path,
        target_label=target,
        verbose=False,
        normalize_labels=True,
        normalization_stats=norm_stats,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.num_batches > 0:
        loader = list(loader)[: args.num_batches]

    denormalize_fn = dataset.denormalize_label

    # Baseline performance
    _maybe_sync(device)
    baseline_mae, baseline_rmse = evaluate(model, loader, device, model_type, denormalize_fn)
    _maybe_sync(device)

    # Channel groups (default: individual)
    if args.grouped:
        channel_groups = {
            "omap": [0, 1],
            "rip_gaf": [2, 3],
            "rip_mtf": [4, 5],
            "com": [6, 7],
            "q_image": [8, 9],
        }
    else:
        channel_groups = {f"ch_{i}": [i] for i in range(10)}

    rng = torch.Generator(device=device).manual_seed(args.seed)

    importance = {}
    for name, channels in channel_groups.items():
        mae, rmse = evaluate(
            model,
            loader,
            device,
            model_type,
            denormalize_fn,
            permute_channels=channels,
            rng=rng,
        )
        importance[name] = {
            "mae": mae,
            "rmse": rmse,
            "delta_mae": mae - baseline_mae,
            "delta_rmse": rmse - baseline_rmse,
        }

    results = {
        "checkpoint": args.checkpoint_path,
        "model_type": model_type,
        "target": target,
        "baseline": {"mae": baseline_mae, "rmse": baseline_rmse},
        "importance": importance,
    }

    output_json = args.output_json
    if output_json is None:
        output_json = f"results_twobody/perm_importance_{target}_{model_type}.json"

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nSaved permutation importance to: {output_path}")


if __name__ == "__main__":
    main()
