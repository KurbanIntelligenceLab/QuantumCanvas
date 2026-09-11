"""One training step of every benchmark and modality model on synthetic data."""
import pytest
import torch
from torch_geometric.loader import DataLoader

from benchmarks.benchmark_config import cfg
from benchmarks.models import get_model
from benchmarks.modality import train_modality_comparison as modality
from benchmarks.modality.common import FUSION_MODELS, MODALITY_MODELS
from benchmarks.train_models_twobody import train_epoch
from benchmarks.twobody_dataloader import TwoBodyDataset


@pytest.fixture(scope="module")
def loader(synthetic_npz):
    return DataLoader(TwoBodyDataset(synthetic_npz, target_label='e_g_ev', verbose=False), batch_size=8)


@pytest.mark.parametrize("model_type", cfg.available_models)
def test_benchmark_model_trains(model_type, loader):
    torch.manual_seed(0)
    model = get_model(model_type, **cfg.model_configs.get(model_type, {}))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = train_epoch(model, loader, opt, torch.device('cpu'), torch.nn.L1Loss(), model_type)
    assert torch.isfinite(torch.tensor(loss))


@pytest.mark.parametrize("model_type", MODALITY_MODELS + FUSION_MODELS)
def test_modality_model_trains(model_type, loader):
    torch.manual_seed(0)
    device = torch.device('cpu')
    model = modality.get_model(model_type, device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = modality.train_epoch(model, loader, opt, torch.nn.L1Loss(), model_type, device)
    assert torch.isfinite(torch.tensor(loss))


def test_ablation_test_split_matches_training_split(synthetic_npz):
    from benchmarks.modality.common import test_split as held_out
    ds = TwoBodyDataset(synthetic_npz, target_label='e_g_ev', verbose=False)
    n = len(ds)
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    _, _, trained_test = torch.utils.data.random_split(
        ds, [n_train, n_val, n - n_train - n_val], generator=torch.Generator().manual_seed(7))
    assert held_out(ds, 7).indices == list(trained_test.indices)


def test_crysmtm_group_split_keeps_structures_together():
    from benchmarks.crysmtm import group_split
    groups = [(phase, t) for phase in ('anatase', 'brookite', 'rutile') for t in range(0, 1001, 50) for _ in range(5)]
    parts = group_split(groups, seed=42)
    seen = [{groups[i] for i in part} for part in parts]
    assert sum(len(p) for p in parts) == len(groups)
    assert not (seen[0] & seen[1] or seen[0] & seen[2] or seen[1] & seen[2])
    assert len(seen[0]) == int(0.7 * 63)
