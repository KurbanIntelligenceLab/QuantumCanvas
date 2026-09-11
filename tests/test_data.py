import numpy as np
import torch
from torch_geometric.loader import DataLoader

import quantumcanvas as qc


def test_image_dataset(synthetic_npz):
    ds = qc.TwoBodyDataset(synthetic_npz, target_label='e_g_ev')
    image, target = ds[0]
    assert len(ds) == 24
    assert image.shape == (10, 32, 32) and image.dtype == torch.float32
    assert target.shape == ()


def test_image_dataset_skips_missing_labels(synthetic_npz):
    assert len(qc.TwoBodyDataset(synthetic_npz, target_label='dipole_mag_d')) == 23


def test_graph_dataset_batches(synthetic_npz):
    ds = qc.TwoBodyGraphDataset(synthetic_npz, target_label='e_g_ev', verbose=False)
    batch = next(iter(DataLoader(ds, batch_size=4)))
    assert batch.z.shape == (8,) and batch.pos.shape == (8, 3)
    assert batch.y.shape == (4,)
    assert batch.image.shape == (40, 32, 32)  # images concatenate along dim 0: view(-1, 10, 32, 32)


def test_graph_label_normalization_round_trips(synthetic_npz):
    ds = qc.TwoBodyGraphDataset(synthetic_npz, target_label='e_g_ev', verbose=False)
    raw = float(np.load(synthetic_npz, allow_pickle=True)['labels'][0]['e_g_ev'])
    assert -1.0 <= float(ds[0].y) <= 1.0
    assert abs(ds.denormalize_label(float(ds[0].y)) - raw) < 1e-5
    raw_ds = qc.TwoBodyGraphDataset(synthetic_npz, target_label='e_g_ev', verbose=False, normalize_labels=False)
    assert abs(float(raw_ds[0].y) - raw) < 1e-5


def test_load_npz_keys(synthetic_npz):
    assert set(qc.load_npz(synthetic_npz)) == {'images', 'geometries', 'elements', 'labels', 'metadata', 'pair_names'}


def test_constants():
    assert len(qc.CHANNELS) == 10
    assert len(qc.BENCHMARK_TARGETS) == 19  # 17 distinct quantities; the 3 charge stats coincide


def test_batch_images_and_pair_view_match_per_graph(synthetic_npz):
    from benchmarks.pairs import pair_view
    batch = next(iter(DataLoader(qc.TwoBodyGraphDataset(synthetic_npz, verbose=False), batch_size=5)))
    graphs = batch.to_data_list()
    assert torch.equal(qc.batch_images(batch), torch.stack([g.image for g in graphs]))
    z, pos, dist = pair_view(batch.z, batch.pos, batch.batch)
    assert torch.equal(z, torch.stack([g.z for g in graphs]))
    assert torch.allclose(dist, torch.stack([(g.pos[0] - g.pos[1]).norm() for g in graphs]))
