#!/usr/bin/env python3
"""Quick utility to check parameter counts for all modality models."""

import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from models import get_modality_model
from improved_models import get_improved_model


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    print("=" * 60)
    print("MODEL PARAMETER COUNTS")
    print("=" * 60)
    
    # Baseline modality models (no vision inductive bias)
    baseline_models = ['tabular_mlp', 'tabular_transformer', 'vision_only', 'geometry_only']
    
    print("\nBaseline Models:")
    print("-" * 40)
    for name in baseline_models:
        try:
            model = get_modality_model(name)
            params = count_params(model)
            print(f"  {name:<25} {params:>10,} params")
        except Exception as e:
            print(f"  {name:<25} ERROR: {e}")
    
    # Improved fusion models
    improved_models = ['qsn_v2', 'multimodal_v2', 'film_cnn']
    
    print("\nImproved Fusion Models:")
    print("-" * 40)
    for name in improved_models:
        try:
            model = get_improved_model(name)
            params = count_params(model)
            print(f"  {name:<25} {params:>10,} params")
        except Exception as e:
            print(f"  {name:<25} ERROR: {e}")
    
    # Try to load benchmark models if available
    print("\nBenchmark Models (for reference):")
    print("-" * 40)
    try:
        from benchmarks.models import QuantumShellNet, ViTRegressor
        
        qsn = QuantumShellNet()
        print(f"  {'quantumshellnet':<25} {count_params(qsn):>10,} params")
        
        vit = ViTRegressor()
        print(f"  {'vit':<25} {count_params(vit):>10,} params")
    except ImportError as e:
        print(f"  (Could not import benchmark models: {e})")
    
    print("=" * 60)
    print("Target: ~300-350K params for fair comparison")
    print("=" * 60)


if __name__ == "__main__":
    main()
