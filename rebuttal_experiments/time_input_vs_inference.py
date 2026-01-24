#!/usr/bin/env python3
"""
Time input-generation (voxel/image encoding) vs model inference.

This script measures:
1) Raw-data input generation time:
   - Image encoding (10-channel image creation)
   - Geometry-only parsing (coords + element->Z)
2) Model inference time on the combined dataset (.npz)
   - Supports GNNs and image-based models

Usage:
  python benchmarks/time_input_vs_inference.py \
    --dataset_path dataset_combined.npz \
    --raw_data_dir raw_data \
    --target e_g_ev \
    --models schnet vit quantumshellnet multimodal \
    --batch_size 64
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from benchmarks.twobody_dataloader import TwoBodyDataset, ELEMENT_TO_Z
from benchmarks.models import get_model
from benchmarks.benchmark_config import cfg
from build_dataset import DetailedOutParser, GeometryParser, ImageEncoder


def _maybe_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_image_generation(raw_data_dir: Path, max_samples: int):
    detailed_files = sorted(raw_data_dir.glob("*/detailed.out"))
    if max_samples > 0:
        detailed_files = detailed_files[:max_samples]

    encoder = ImageEncoder()
    processed = 0
    start = time.time()
    for detailed_file in detailed_files:
        mol_data = DetailedOutParser.parse_file(detailed_file)
        if mol_data is None:
            continue
        _ = encoder.encode(mol_data)
        processed += 1

    total = time.time() - start
    return {
        "samples": processed,
        "total_sec": float(total),
        "per_sample_ms": float(1000.0 * total / max(processed, 1)),
    }


def time_geometry_generation(raw_data_dir: Path, max_samples: int):
    xyz_files = sorted(raw_data_dir.glob("*/geo_end.xyz"))
    if max_samples > 0:
        xyz_files = xyz_files[:max_samples]

    processed = 0
    start = time.time()
    for xyz_file in xyz_files:
        geometry, _ = GeometryParser.parse_xyz(xyz_file)
        if geometry is None:
            continue
        pos = np.array([[a.x, a.y, a.z] for a in geometry], dtype=np.float32)
        elements = [a.element for a in geometry]
        z = np.array([ELEMENT_TO_Z.get(elem, 0) for elem in elements], dtype=np.int64)
        _ = pos, z
        processed += 1

    total = time.time() - start
    return {
        "samples": processed,
        "total_sec": float(total),
        "per_sample_ms": float(1000.0 * total / max(processed, 1)),
    }


def _forward_model(model, data, model_type: str, device: torch.device):
    if model_type == "faenet":
        outputs = model(data)
        if isinstance(outputs, dict):
            return outputs.get("energy", outputs.get("output", list(outputs.values())[0]))
        return outputs
    if model_type == "egnn":
        return model(data.z, data.pos, data.batch)
    if model_type == "multimodal":
        images = torch.stack([d.image[:3] for d in data.to_data_list()]).to(device).float()
        return model(data.z, data.pos, data.batch, images)
    if model_type == "quantumshellnet":
        images = torch.stack([d.image for d in data.to_data_list()]).to(device).float()
        return model(images, data.z, data.pos, data.batch)
    if model_type == "vit":
        images = torch.stack([d.image[:3] for d in data.to_data_list()]).to(device).float()
        return model(images)
    return model(data.z, data.pos, data.batch)


def time_inference(
    model_type: str,
    dataset_path: str,
    target: str,
    batch_size: int,
    device: torch.device,
    num_batches: int,
    warmup_batches: int,
):
    dataset = TwoBodyDataset(
        dataset_path,
        target_label=target,
        verbose=False,
        normalize_labels=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model_config = cfg.model_configs.get(model_type, {})
    model = get_model(model_type, **model_config).to(device)
    model.eval()
    num_parameters = sum(p.numel() for p in model.parameters())

    total_samples = 0
    total_time = 0.0

    with torch.no_grad():
        # Warmup
        for idx, data in enumerate(loader):
            if idx >= warmup_batches:
                break
            data = data.to(device)
            _ = _forward_model(model, data, model_type, device)
            _maybe_sync(device)

        # Timed inference
        for idx, data in enumerate(loader):
            if num_batches > 0 and idx >= num_batches:
                break
            data = data.to(device)
            _maybe_sync(device)
            start = time.time()
            _ = _forward_model(model, data, model_type, device)
            _maybe_sync(device)
            end = time.time()

            batch_size_actual = data.num_graphs
            total_samples += batch_size_actual
            total_time += (end - start)

    per_sample_ms = 1000.0 * total_time / max(total_samples, 1)
    throughput = total_samples / max(total_time, 1e-9)
    return {
        "samples": int(total_samples),
        "total_sec": float(total_time),
        "per_sample_ms": float(per_sample_ms),
        "samples_per_sec": float(throughput),
        "num_parameters": int(num_parameters),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="dataset_combined.npz")
    parser.add_argument("--raw_data_dir", type=str, default=None)
    parser.add_argument("--target", type=str, default="e_g_ev")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["schnet", "egnn", "faenet", "vit", "quantumshellnet", "multimodal"],
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--warmup_batches", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device(cfg.device)

    results = {
        "device": str(device),
        "dataset_path": args.dataset_path,
        "target": args.target,
    }

    if args.raw_data_dir:
        raw_data_dir = Path(args.raw_data_dir)
        if raw_data_dir.exists():
            results["input_generation"] = {
                "image_encoding": time_image_generation(raw_data_dir, args.max_samples),
                "geometry_only": time_geometry_generation(raw_data_dir, args.max_samples),
            }
        else:
            results["input_generation"] = {"error": f"raw_data_dir not found: {raw_data_dir}"}

    results["inference"] = {}
    for model_type in args.models:
        results["inference"][model_type] = time_inference(
            model_type=model_type,
            dataset_path=args.dataset_path,
            target=args.target,
            batch_size=args.batch_size,
            device=device,
            num_batches=args.num_batches,
            warmup_batches=args.warmup_batches,
        )

    output_json = args.output_json
    if output_json is None:
        output_json = f"results_twobody/timing_{args.target}.json"

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nSaved timing results to: {output_path}")


if __name__ == "__main__":
    main()
