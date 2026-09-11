"""CrysMTM (TiO2 polymorphs) as PyG graphs for the transfer experiments.

Expected layout: <root>/labels.csv and <root>/<phase>/<T>K/xyz/rot_<k>.xyz for the phases anatase,
brookite and rutile at T = 0, 50, ..., 1000 K. Every rotation of one (phase, temperature) structure
shares its labels, so splits must be grouped by `groups` to avoid leakage.
"""
import os
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from quantumcanvas.constants import ELEMENT_TO_Z

PHASES = ["anatase", "brookite", "rutile"]
TEMPERATURES = range(0, 1001, 50)
PROPERTIES = ["HOMO", "LUMO", "Eg", "Ef", "Et", "Eta", "disp", "vol", "bond"]


def load_labels(root):
    """{(phase, temperature): [value for each of PROPERTIES]} from labels.csv."""
    df = pd.read_csv(os.path.join(root, "labels.csv"))
    table = {}
    for row in df.itertuples():
        key = (row.Polymorph.lower(), int(float(str(row.Temperature).replace("K", ""))))
        table.setdefault(key, {})[row.Parameter] = float(row.Value)
    return {k: [v[p] for p in PROPERTIES] for k, v in table.items() if all(p in v for p in PROPERTIES)}


def read_xyz(path):
    symbols, coords = [], []
    with open(path, encoding="utf-8") as f:
        for line in f.readlines()[2:]:
            parts = line.split()
            if len(parts) >= 4:
                symbols.append(parts[0])
                coords.append([float(v) for v in parts[1:4]])
    return symbols, coords


def group_split(groups, seed, fractions=(0.7, 0.1)):
    """Seeded train/val/test index split that keeps every group (one structure's rotations) in a single part."""
    keys = sorted(set(groups))
    order = np.random.RandomState(seed).permutation(len(keys))
    n_train, n_val = int(fractions[0] * len(keys)), int(sum(fractions) * len(keys))
    part = {keys[k]: 0 if rank < n_train else 1 if rank < n_val else 2 for rank, k in enumerate(order)}
    which = np.array([part[g] for g in groups])
    idx = np.arange(len(groups))
    return idx[which == 0], idx[which == 1], idx[which == 2]


class RegressionLoader(Dataset):
    """Each item is Data(z, pos, y[1, 9]) with z the true atomic numbers (Ti=22, O=8)."""

    def __init__(self, label_dir: str, temperature_filter: Optional[Callable[[int], bool]] = None,
                 max_rotations: Optional[int] = None):
        labels = load_labels(label_dir)
        self.samples, self.groups = [], []
        for phase in PHASES:
            for temp in TEMPERATURES:
                if temperature_filter and not temperature_filter(temp):
                    continue
                xyz_dir = os.path.join(label_dir, phase, f"{temp}K", "xyz")
                if (phase, temp) not in labels or not os.path.isdir(xyz_dir):
                    continue
                rotations = sorted(int(f[4:-4]) for f in os.listdir(xyz_dir)
                                   if f.startswith("rot_") and f.endswith(".xyz") and f[4:-4].isdigit())
                for rot in rotations[:max_rotations]:
                    self.samples.append((os.path.join(xyz_dir, f"rot_{rot}.xyz"), labels[(phase, temp)]))
                    self.groups.append((phase, temp))
        self._cache = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx not in self._cache:
            path, y = self.samples[idx]
            symbols, coords = read_xyz(path)
            self._cache[idx] = Data(
                z=torch.tensor([ELEMENT_TO_Z[s] for s in symbols], dtype=torch.long),
                pos=torch.tensor(coords, dtype=torch.float),
                y=torch.tensor([y], dtype=torch.float),
            )
        return self._cache[idx]
