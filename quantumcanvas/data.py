import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset as PyGDataset

from quantumcanvas.constants import (
    DATASET_MD5,
    DEFAULT_DATASET_PATH,
    ELEMENT_TO_Z,
    ZENODO_URL,
)


def _md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def download(path=DEFAULT_DATASET_PATH, force=False):
    """Download dataset_combined.npz from Zenodo and verify its md5. Returns the path."""
    path = Path(path)
    if path.exists() and not force:
        if _md5(path) == DATASET_MD5:
            return path
        raise RuntimeError(f"{path} exists but its md5 does not match; pass force=True to re-download")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.part')
    urllib.request.urlretrieve(ZENODO_URL, tmp)
    if _md5(tmp) != DATASET_MD5:
        tmp.unlink()
        raise RuntimeError("downloaded file failed the md5 check")
    tmp.rename(path)
    return path


def load_npz(path=DEFAULT_DATASET_PATH):
    """Load the raw arrays: images, geometries, elements, labels, metadata, pair_names."""
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def batch_images(batch):
    """Images of a PyG batch from TwoBodyGraphDataset as one [B, C, H, W] tensor (they are stored concatenated)."""
    return batch.image.view(batch.num_graphs, -1, *batch.image.shape[-2:])


def _valid_indices(labels, target_label):
    valid = []
    for i, label_dict in enumerate(labels):
        if isinstance(label_dict, dict) and target_label in label_dict:
            value = label_dict[target_label]
            if value is not None and np.isfinite(value):
                valid.append(i)
    return valid


class TwoBodyDataset(torch.utils.data.Dataset):
    """Image dataset: yields (image [10, 32, 32], target) for one label.

    Samples whose target is missing or non-finite are skipped.
    """

    def __init__(self, npz_path=DEFAULT_DATASET_PATH, target_label='e_g_ev'):
        data = np.load(npz_path, allow_pickle=True)
        self.images = data['images']
        self.labels = data['labels']
        self.pair_names = data['pair_names']
        self.target_label = target_label
        self.valid_indices = _valid_indices(self.labels, target_label)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        image = torch.tensor(self.images[i], dtype=torch.float32)
        target = torch.tensor(float(self.labels[i][self.target_label]), dtype=torch.float32)
        return image, target


class TwoBodyGraphDataset(PyGDataset):
    """PyTorch Geometric dataset: one 2-node graph per element pair.

    Each item is a ``Data(z, pos, x, y, image, pair_name)``. By default ``y`` is
    min-max scaled to [-1, 1]; use ``denormalize_label`` to map predictions back,
    or pass ``normalize_labels=False`` for raw units.
    """

    def __init__(self, npz_path=DEFAULT_DATASET_PATH, target_label='e_g_ev',
                 transform=None, pre_transform=None, verbose=True,
                 normalize_labels=True, normalization_stats=None):

        self.npz_path = Path(npz_path)
        self.target_label = target_label
        self.verbose = verbose
        self.normalize_labels = normalize_labels

        if verbose:
            print(f"Loading dataset from {self.npz_path}...")
        data = np.load(self.npz_path, allow_pickle=True)

        self.geometries = data['geometries']
        self.elements = data['elements']
        self.labels = data['labels']
        self.pair_names = data['pair_names']
        self.images = data['images'] if 'images' in data else None

        if verbose:
            print(f"  Loaded {len(self.geometries)} samples")
            print(f"  Target property: {target_label}")
            if self.images is not None:
                print(f"  Images shape: {self.images.shape}")

        self.valid_indices = _valid_indices(self.labels, target_label)

        if verbose:
            print(f"  Valid samples: {len(self.valid_indices)}/{len(self.labels)}")

        if normalize_labels:
            if normalization_stats is None:
                self._compute_normalization_stats()
            else:
                self.label_min = normalization_stats['min']
                self.label_max = normalization_stats['max']

            if verbose:
                print(f"  Label range: [{self.label_min:.4f}, {self.label_max:.4f}]")
                print("  Normalized to: [-1, 1]")
        else:
            self.label_min = None
            self.label_max = None

        super().__init__(None, transform, pre_transform)

    def fit_normalization(self, indices):
        """Recompute the [-1, 1] scaling from a subset (e.g. the training split), given as dataset positions."""
        self._compute_normalization_stats([self.valid_indices[i] for i in indices])
        return self.get_normalization_stats()

    def _compute_normalization_stats(self, raw_indices=None):
        raw_indices = self.valid_indices if raw_indices is None else raw_indices
        all_labels = np.array([float(self.labels[idx][self.target_label]) for idx in raw_indices])
        self.label_min = float(np.min(all_labels))
        self.label_max = float(np.max(all_labels))

        if abs(self.label_max - self.label_min) < 1e-10:
            self.label_max = self.label_min + 1.0

    def get_normalization_stats(self):
        return {'min': self.label_min, 'max': self.label_max}

    def normalize_label(self, value):
        if not self.normalize_labels or self.label_min is None:
            return value
        return 2.0 * (value - self.label_min) / (self.label_max - self.label_min) - 1.0

    def denormalize_label(self, normalized_value):
        if not self.normalize_labels or self.label_min is None:
            return normalized_value
        return (normalized_value + 1.0) / 2.0 * (self.label_max - self.label_min) + self.label_min

    def len(self):
        return len(self.valid_indices)

    def get(self, idx):
        actual_idx = self.valid_indices[idx]

        geom = self.geometries[actual_idx]
        pos = torch.tensor(geom[:, :3], dtype=torch.float32)

        elements = self.elements[actual_idx]
        z = torch.LongTensor([ELEMENT_TO_Z.get(elem, 0) for elem in elements])

        y_raw = float(self.labels[actual_idx][self.target_label])
        y = torch.FloatTensor([self.normalize_label(y_raw)])

        x = torch.cat([pos, z.unsqueeze(1).float()], dim=1)

        image = None
        if self.images is not None:
            image = torch.tensor(self.images[actual_idx], dtype=torch.float32)

        return Data(
            z=z,
            pos=pos,
            x=x,
            y=y,
            image=image,
            pair_name=self.pair_names[actual_idx],
        )
