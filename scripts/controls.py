"""Control experiments on what the image modality contributes.

  spatial     energy gap / dipole magnitude from (a) the full 10-channel image, (b) the same image with its
              pixels permuted identically in every channel (spatial layout destroyed), and (c) an MLP fed the
              six scalars the images are rendered from, all parameter-matched.
  readout     dipole and charge targets with the channels that render them masked out, or kept alone.
  bondlength  bond length from optimized coordinates (SchNet) vs. atomic numbers only, plus a mean predictor.

Writes one row per run to <output_dir>/<experiment>.csv (the layout of the CSVs in results/).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from quantumcanvas import ELEMENT_TO_Z, load_npz

# Channels that directly render each readout target (see quantumcanvas.CHANNELS).
READOUT_CHANNELS = {
    'dipole_mag_d': [4],
    'q_absmean': [5, 6, 8, 9],
    'q_maxabs': [5, 6, 8, 9],
    'q_std': [5, 6, 8, 9],
}
SPATIAL_TARGETS = ['e_g_ev', 'dipole_mag_d']


class ShellNet(nn.Module):
    """QuantumShellNet-style CNN (343,542 parameters) with nuclear-count features for the two atoms."""

    def __init__(self, in_channels=10):
        super().__init__()
        chans = [in_channels, 80, 160, 96, 48, 24]
        self.convs = nn.ModuleList(nn.Conv2d(a, b, kernel_size=3, padding=1, stride=2) for a, b in zip(chans, chans[1:]))
        self.fc1 = nn.Linear(24, 200)
        self.fc2 = nn.Linear(200 + 3, 100)
        self.fc3 = nn.Linear(100 + 3, 50)
        self.fc4 = nn.Linear(50 + 3, 1)
        self.dropout = nn.Dropout(0.3)
        self.act = nn.ReLU()

    def forward(self, images, z):
        # z: [B, 2] atomic numbers. Mass number is approximated as 2Z, so neutrons = Z.
        atom_num = z.float().sum(1, keepdim=True)
        mass_num = 2 * atom_num
        nuclear = torch.cat((mass_num, atom_num, mass_num - atom_num), dim=1)
        x = images
        for conv in self.convs:
            x = self.dropout(self.act(conv(x)))
        x = self.dropout(self.act(self.fc1(x.flatten(1))))
        x = self.dropout(self.act(self.fc2(torch.cat((x, nuclear), 1))))
        x = self.dropout(self.act(self.fc3(torch.cat((x, nuclear), 1))))
        return self.fc4(torch.cat((x, nuclear), 1)).squeeze(-1)


def mlp(in_dim, hidden):
    layers, prev = [], in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.3)]
        prev = h
    return nn.Sequential(*layers, nn.Linear(prev, 1))


class SchNetPairs(nn.Module):
    def __init__(self):
        super().__init__()
        from torch_geometric.nn import SchNet
        self.schnet = SchNet(hidden_channels=96, num_filters=96, num_interactions=6,
                             num_gaussians=50, cutoff=5.0, readout='add')

    def forward(self, z, pos):
        batch = torch.arange(z.shape[0], device=z.device).repeat_interleave(2)
        return self.schnet(z.reshape(-1), pos.reshape(-1, 3), batch).squeeze(-1)


def raw_scalar_features(data):
    """The six scalars the images are rendered from, recovered from the dataset file.

    Columns: electron population of atom A and B, |q|, |mu|, total s+p and total d+f shell population.
    The shell sums are the channel-2/3 images divided by their fixed rendering templates (quantumcanvas.build).
    """
    labels = data['labels']
    images = data['images'].astype(np.float64)
    n, _, h, w = images.shape
    yg, xg = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing='ij')
    r = np.sqrt(xg ** 2 + yg ** 2)
    sp_template = np.exp(-r ** 2 / 0.3)
    df_template = np.exp(-r ** 2 / 0.5) * (1 + np.cos(4 * np.arctan2(yg, xg)))
    return np.stack([
        data['geometries'][:, 0, 3],
        data['geometries'][:, 1, 3],
        [lab['q_maxabs'] for lab in labels],
        [lab['dipole_mag_d'] for lab in labels],
        images[:, 2].reshape(n, -1).sum(1) / sp_template.sum(),
        images[:, 3].reshape(n, -1).sum(1) / df_template.sum(),
    ], axis=1).astype(np.float32)


def split_indices(n, seed, train=0.8, val=0.1):
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).numpy()
    n_tr, n_va = int(train * n), int(val * n)
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


def fit(model, inputs, y, seed, epochs, patience=10, batch_size=64, lr=1e-3, device='cpu'):
    """Train on a seeded 80/10/10 split with targets min-max scaled on train; return test metrics in raw units.

    `inputs` is a tuple of arrays passed positionally to the model; rows align with `y`.
    """
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    tr, va, te = split_indices(len(y), seed)
    lo, hi = float(y[tr].min()), float(y[tr].max())
    hi = hi if hi - lo > 1e-10 else lo + 1.0
    scale = lambda v: 2.0 * (v - lo) / (hi - lo) - 1.0  # noqa: E731
    unscale = lambda v: (v + 1.0) / 2.0 * (hi - lo) + lo  # noqa: E731

    tensors = [torch.as_tensor(x) for x in inputs]
    y_scaled = torch.as_tensor(scale(y), dtype=torch.float32)

    def batches(idx, shuffle):
        idx = rng.permutation(idx) if shuffle else idx
        for i in range(0, len(idx), batch_size):
            b = torch.as_tensor(idx[i:i + batch_size])
            yield [t[b].to(device) for t in tensors], y_scaled[b].to(device)

    def evaluate(idx):
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in batches(idx, False):
                preds.append(model(*xb).cpu().numpy())
        pred = unscale(np.concatenate(preds))
        return pred, y[idx]

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, min_lr=1e-6)
    criterion = nn.L1Loss()
    best_val, best_epoch, best_state = float('inf'), 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in batches(tr, True):
            optimizer.zero_grad()
            criterion(model(*xb), yb).backward()
            optimizer.step()
        pred, target = evaluate(va)
        val_loss = float(np.mean(np.abs(scale(pred) - scale(target))))
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch > patience:
            break

    if best_state is not None:  # otherwise validation never improved (e.g. NaN): keep the final weights
        model.load_state_dict(best_state)
    pred, target = evaluate(te)
    return {
        'n_params': sum(p.numel() for p in model.parameters()), 'best_epoch': best_epoch, 'best_val_loss': best_val,
        'test_mae': float(np.mean(np.abs(pred - target))), 'test_rmse': float(np.sqrt(np.mean((pred - target) ** 2))),
        'n_train': len(tr), 'n_val': len(va), 'n_test': len(te),
    }


def target_rows(data, target):
    """Indices of pairs with a finite `target` label, and those label values."""
    values = np.array([lab.get(target, np.nan) for lab in data['labels']], dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(values))
    return valid, values[valid]


def atomic_numbers(data):
    return np.array([[ELEMENT_TO_Z.get(e, 0) for e in pair] for pair in data['elements']], dtype=np.int64)


def run_spatial(data, seeds, epochs, device):
    z, raw = atomic_numbers(data), raw_scalar_features(data)
    images = data['images'].astype(np.float32)
    for target in SPATIAL_TARGETS:
        valid, y = target_rows(data, target)
        keep = np.isfinite(raw[valid]).all(1)  # same rows for all three arms
        valid, y = valid[keep], y[keep]
        for seed in seeds:
            imgs = images[valid]
            yield {'target': target, 'mode': 'full_image', 'seed': seed,
                   **fit(ShellNet(), (imgs, z[valid]), y, seed, epochs, device=device)}
            # One pixel permutation shared by all channels and samples: keeps each channel's values, destroys layout.
            perm = np.random.RandomState(1000 + seed).permutation(imgs.shape[2] * imgs.shape[3])
            shuffled = imgs.reshape(len(imgs), imgs.shape[1], -1)[:, :, perm].reshape(imgs.shape)
            yield {'target': target, 'mode': 'channel_preserving_shuffle', 'seed': seed,
                   **fit(ShellNet(), (shuffled, z[valid]), y, seed, epochs, device=device)}
            feats = raw[valid]
            tr, _, _ = split_indices(len(y), seed)
            feats = (feats - feats[tr].mean(0)) / (feats[tr].std(0) + 1e-8)
            yield {'target': target, 'mode': 'raw_scalar_mlp', 'seed': seed, 'feat_name': 'raw_scalar',
                   **fit(mlp(feats.shape[1], (400, 450, 350)), (feats,), y, seed, epochs, device=device)}


def run_readout(data, seeds, epochs, device):
    z = atomic_numbers(data)
    images = data['images'].astype(np.float32)
    for target, channels in READOUT_CHANNELS.items():
        valid, y = target_rows(data, target)
        others = [c for c in range(images.shape[1]) if c not in channels]
        for seed in seeds:
            for mode, zeroed in [('all_channels', []), ('masked_encoding_channels', channels),
                                 ('only_encoding_channels', others)]:
                imgs = images[valid].copy()
                imgs[:, zeroed] = 0.0
                yield {'target': target, 'mode': mode, 'seed': seed,
                       **fit(ShellNet(), (imgs, z[valid]), y, seed, epochs, device=device)}


def run_bondlength(data, seeds, epochs, device):
    z = atomic_numbers(data)
    pos = data['geometries'][:, :, :3].astype(np.float32)
    valid, y = target_rows(data, 'distance_ang')
    for seed in seeds:
        yield {'target': 'distance_ang', 'mode': 'leaked_schnet_pos', 'seed': seed,
               **fit(SchNetPairs(), (z[valid], pos[valid]), y, seed, epochs, device=device)}
        tr, va, te = split_indices(len(y), seed)
        zf = z[valid].astype(np.float32)
        zf = (zf - zf[tr].mean(0)) / (zf[tr].std(0) + 1e-8)
        yield {'target': 'distance_ang', 'mode': 'leakage_free_z_only', 'seed': seed,
               **fit(mlp(2, (200, 100, 50)), (zf,), y, seed, epochs, device=device)}
        err = y[te] - y[tr].mean()
        yield {'target': 'distance_ang', 'mode': 'trivial_mean', 'seed': seed,
               'test_mae': float(np.mean(np.abs(err))), 'test_rmse': float(np.sqrt(np.mean(err ** 2))),
               'n_train': len(tr), 'n_val': len(va), 'n_test': len(te)}


EXPERIMENTS = {'spatial': run_spatial, 'readout': run_readout, 'bondlength': run_bondlength}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--experiment', required=True, choices=sorted(EXPERIMENTS))
    p.add_argument('--dataset_path', default='dataset_combined.npz')
    p.add_argument('--output_dir', default='outputs/controls')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_npz(args.dataset_path)
    rows = []
    for row in EXPERIMENTS[args.experiment](data, args.seeds, args.epochs, args.device):
        rows.append(row)
        print(json.dumps(row), flush=True)
    path = out_dir / f'{args.experiment}.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f'wrote {path} ({len(rows)} runs)')
    return rows


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f'done in {time.time() - t0:.0f}s')
