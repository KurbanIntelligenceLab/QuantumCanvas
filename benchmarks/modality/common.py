import torch

from benchmarks.benchmark_config import cfg
from benchmarks.models import get_model as get_benchmark_model
from benchmarks.modality.fusion_models import get_fusion_model
from benchmarks.modality.models import get_modality_model

MODALITY_MODELS = ['tabular_mlp', 'tabular_transformer', 'vision_only', 'geometry_only']
FUSION_MODELS = ['qsn_v2', 'multimodal_v2', 'film_cnn']


def test_split(dataset, seed: int, train: float = 0.8, val: float = 0.1):
    """The held-out test subset of the seeded train/val/test split used by the training scripts.

    train_models_twobody (randperm) and train_modality_comparison (random_split) draw the same
    permutation for a given seed, so this recovers either script's test set.
    """
    n = len(dataset)
    n_train, n_val = int(train * n), int(val * n)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return torch.utils.data.Subset(dataset, perm[n_train + n_val:])


def maybe_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def get_model(model_type: str, device: torch.device):
    """Build a modality/fusion model, or fall back to a benchmark model from benchmark_config."""
    if model_type in MODALITY_MODELS:
        model = get_modality_model(model_type)
    elif model_type in FUSION_MODELS:
        model = get_fusion_model(model_type)
    else:
        model_config = cfg.model_configs.get(model_type, {})
        model = get_benchmark_model(model_type, **model_config)

    return model.to(device)
