
from benchmarks.modality.models import get_modality_model
from benchmarks.modality.fusion_models import get_fusion_model

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def main():
    print("=" * 60)
    print("MODEL PARAMETER COUNTS")
    print("=" * 60)

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

    improved_models = ['qsn_v2', 'multimodal_v2', 'film_cnn']

    print("\nImproved Fusion Models:")
    print("-" * 40)
    for name in improved_models:
        try:
            model = get_fusion_model(name)
            params = count_params(model)
            print(f"  {name:<25} {params:>10,} params")
        except Exception as e:
            print(f"  {name:<25} ERROR: {e}")

    print("\nBenchmark Models (as configured in benchmark_config.py):")
    print("-" * 40)
    from benchmarks.benchmark_config import cfg
    from benchmarks.models import get_model
    for name in ['quantumshellnet', 'vit', 'multimodal']:
        model = get_model(name, **cfg.model_configs.get(name, {}))
        print(f"  {name:<25} {count_params(model):>10,} params")

    print("=" * 60)

if __name__ == "__main__":
    main()
