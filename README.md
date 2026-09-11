# QuantumCanvas: A Multimodal Benchmark for Learning Two-Body Quantum Interactions

Can Polat (1), Mustafa Kurban (2, 3), Erchin Serpedin (1), Hasan Kurban (4)

1. Department of Electrical and Computer Engineering, Texas A&M University, College Station, Texas, USA
2. Department of Electrical and Computer Engineering, Texas A&M University at Qatar, Doha, Qatar
3. Department of Prosthetics and Orthotics, Ankara University, Ankara, Turkey
4. College of Science and Engineering, Hamad Bin Khalifa University, Doha, Qatar

Published in Machine Learning: Science and Technology (2026), [doi:10.1088/2632-2153/aea5d6](https://doi.org/10.1088/2632-2153/aea5d6).
Dataset: [doi:10.5281/zenodo.20631934](https://doi.org/10.5281/zenodo.20631934).

## Abstract

Most molecular and materials machine-learning models fit correlations across whole molecules or crystals rather than learning the quantum interactions between atomic pairs. Yet bonding, charge redistribution, orbital hybridization, and electronic coupling all emerge from these two-body interactions. We introduce *QuantumCanvas*, a multimodal benchmark that treats the two-body quantum system as the minimal, exhaustively enumerable unit of interatomic interaction. It covers 2,850 element-element pairs, each at a single optimized geometry, and evaluates 17 benchmark target quantities spanning electronic, thermodynamic, dipole, and charge-derived properties; algebraically derived quantities and redundant diagnostic charge rows are identified explicitly, and label availability is reported per target. Each pair is also represented by ten-channel images of orbital populations and charge- and dipole-derived fields that encode angular and electrostatic structure without explicit atomic coordinates. Benchmarking graph, vision, and fusion architectures on element-pair-disjoint splits reveals modality-specific inductive biases: graph encoders achieve the lowest MAE on most reported targets, while late fusion gives the lowest MAE for the reported Mermin free-energy label. Controls on the energy gap and dipole magnitude show that destroying the spatial layout of the images does not degrade accuracy and that a model fed the generating scalars directly outperforms both image variants: the rendering is an alternative encoding of the same scalars, not an independent signal. Pretraining on *QuantumCanvas* lowers mean test error in 11 of 16 encoder-target comparisons across *QM9*, *MD17*, and *CrysMTM*. *QuantumCanvas* thus provides a controlled, physically grounded testbed for studying which signals each modality captures, how they combine, and how they transfer across molecular, dynamical, and crystalline regimes.

**Keywords:** machine learning for interatomic interactions, two-body quantum systems, multimodal benchmark, orbital-image representations, molecular property prediction

## Installation

Requires Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/). Linux, macOS and Windows are supported.

```bash
git clone https://github.com/KurbanIntelligenceLab/QuantumCanvas.git
cd QuantumCanvas
uv sync                      # data loaders only
uv sync --extra benchmarks   # + everything the benchmark scripts need
```

`uv sync` installs the exact versions in `uv.lock`, including `torch==2.8.0` and `torch-geometric==2.5.3`. The `benchmarks` extra adds the PyG extensions (`torch-scatter`, `torch-sparse`, `torch-cluster`, `torch-spline-conv`), FAENet and GotenNet. The extensions come as prebuilt wheels on Linux (CUDA 12.8) and Windows (CPU); on macOS they are compiled from source during the first sync (a few minutes). `numpy` is pinned below 2.0 because DimeNet++ in `torch-geometric` 2.5.3 still uses `np.math`.

## Dataset

The dataset is archived on Zenodo (33.5 MB, CC-BY-4.0). Download it with the md5 check:

```bash
uv run python -c "import quantumcanvas; quantumcanvas.download()"   # writes ./dataset_combined.npz
```

or manually:

```bash
curl -L -o dataset_combined.npz "https://zenodo.org/records/20631934/files/dataset_combined.npz?download=1"
md5sum dataset_combined.npz   # a35d349814ca9e12a8413289c015de49  (macOS: md5)
```

Every script looks for `dataset_combined.npz` in the working directory by default; pass `--dataset_path` to point elsewhere.

Images (PyTorch):

```python
from torch.utils.data import DataLoader
from quantumcanvas import TwoBodyDataset

ds = TwoBodyDataset("dataset_combined.npz", target_label="e_g_ev")
for images, targets in DataLoader(ds, batch_size=32, shuffle=True):
    ...  # images: [32, 10, 32, 32], targets: [32]
```

Graphs (PyTorch Geometric):

```python
from torch_geometric.loader import DataLoader
from quantumcanvas import TwoBodyGraphDataset, batch_images

ds = TwoBodyGraphDataset("dataset_combined.npz", target_label="e_g_ev")
for batch in DataLoader(ds, batch_size=32, shuffle=True):
    out = model(batch.z, batch.pos, batch.batch)   # 2 atoms per graph
    images = batch_images(batch)                   # [32, 10, 32, 32]
```

`TwoBodyGraphDataset` scales `y` to [-1, 1] (`ds.denormalize_label(...)` maps predictions back; `ds.fit_normalization(train_indices)` refits the scaling on a training split; pass `normalize_labels=False` for raw units). Both datasets skip pairs whose target is missing. For the raw arrays, use `quantumcanvas.load_npz(path)`.

## Data format

`dataset_combined.npz` holds 2,850 element pairs:

| Key | Shape | Contents |
|---|---|---|
| `images` | `[2850, 10, 32, 32]` float32 | ten image channels (below) |
| `geometries` | `[2850, 2, 4]` float32 | per atom: x, y, z (Angstrom) and electron population |
| `elements` | `[2850, 2]` | element symbols, e.g. `['Be', 'Rn']` |
| `pair_names` | `[2850]` | e.g. `'Be_Rn'` |
| `labels` | `[2850]` dict | 37 DFTB+ labels per pair (keys below) |
| `metadata` | `[2850]` dict | bond length, Fermi level, total energy, dipole vector |

Image channels (`quantumcanvas.CHANNELS`):

| Ch | Channel | Rendering |
|---|---|---|
| 0 | Orbital population | orbital-weighted population stamp per atom |
| 1 | Angular moment | net magnetic-moment magnitude per atom |
| 2 | s/p shell field | isotropic radial field times total s+p population |
| 3 | d/f shell field | four-fold radial field times total d+f population |
| 4 | Dipole field | radial ring times dipole magnitude |
| 5 | Charge-asymmetry field | quadrupole field times the charge difference of the two atoms |
| 6 | Charge magnitude | stamp per atom times absolute charge |
| 7 | Electron population | stamp per atom times total electron population |
| 8 | Positive charge | stamp at the positively charged atom |
| 9 | Negative charge | stamp at the negatively charged atom |

Benchmark targets (`quantumcanvas.BENCHMARK_TARGETS`), with the number of pairs that have the label:

| Group | Label keys | Unit | Pairs |
|---|---|---|---|
| Electronic | `e_g_ev`, `e_homo_ev`, `e_lumo_ev` | eV | 2,840 |
| | `band_energy_ev` | eV | 2,850 |
| Energy | `total_energy_ev`, `repulsive_energy_ev`, `mermin_free_energy_ev` | eV | 2,850 |
| Conceptual DFT | `i_ev`, `a_ev`, `chi_ev`, `mu_ev`, `eta_ev` | eV | 2,840 |
| | `softness_evinv` | 1/eV | 987 |
| | `electrophilicity_ev` | eV | 987 |
| Dipole | `dipole_mag_d`, `dipole_z_d` | D | 2,850 |
| Charge | `q_maxabs`, `q_absmean`, `q_std` | e | 2,850 |

Every pair is neutral (q_B = -q_A), so the three charge statistics all equal the absolute atomic charge and the 19 keys amount to 17 distinct quantities. Electronegativity, chemical potential, hardness, softness and electrophilicity are algebraic functions of the ionization potential I and electron affinity A: chi = (I + A) / 2, eta = (I - A) / 2, S = 1 / eta, mu = -chi, omega = mu^2 / (2 eta). The other label keys (`distance_ang`, `fermi_level_ev`, `metal_like`, `dipole_x_d`, `dipole_y_d`, `total_charge`, SCC and geometry-convergence diagnostics) are included for analysis.

## Running the benchmarks

Run from the repository root after `uv sync --extra benchmarks`. Every script takes `--help`. Outputs go to `outputs/<experiment>/`.

```bash
# Graph, vision and fusion models on every benchmark target (3 seeds each)
uv run python -m benchmarks.train_models_twobody
uv run python -m benchmarks.train_models_twobody --models schnet gcn --targets e_g_ev --seeds 42

# Parameter-matched modality comparison (tabular vs. vision vs. geometry vs. fusion)
uv run python -m benchmarks.modality.train_modality_comparison

# Ablations on the trained modality checkpoints, and out-of-distribution splits
uv run python -m benchmarks.modality.element_shuffle_ablation          # element-identity shuffling
uv run python -m benchmarks.modality.channel_perm_from_modality_ckpts  # channel-permutation importance
uv run python -m benchmarks.modality.ood_composition_split             # composition / periodic-table splits
uv run python -m benchmarks.modality.run_all_experiments               # comparison, shuffling and OOD in sequence

# Transfer learning: pretrain on a two-body target, then fine-tune from scratch vs. pretrained
uv run python scripts/pretrain_twobody.py --model schnet --target e_g_ev --seed 42
uv run python scripts/run_transfer.py --case qm9:gap --model schnet --seed 42 --arm scratch --lr 1e-4
uv run python scripts/run_transfer.py --case qm9:gap --model schnet --seed 42 --arm pretrained --lr 1e-4 \
    --ckpt outputs/pretrain/e_g_ev/schnet/seed_42/best_model.pt

# Controls: spatial shuffle vs. raw scalars, channel readout, bond-length leakage, charge redundancy
uv run python scripts/controls.py --experiment spatial      # also: readout, bondlength
uv run python scripts/bondlength_dimenet.py --model dimenet --mode leakage_free --seed 42
uv run python scripts/charge_verification.py
```

Transfer cases are `qm9:{gap,homo,lumo}`, `md17:{aspirin,benzene,ethanol}` and `crysmtm:{HOMO,LUMO}`. `TWOBODY_TARGET_MAP` in each `benchmarks/*_config.py` names the two-body target to pretrain on for each case (for example `e_g_ev` for `qm9:gap`, `total_energy_ev` for MD17). The paper used `--lr 1e-4` on both arms. QM9 and MD17 download into `data/` through PyTorch Geometric; CrysMTM must be placed in `data/CrysMTM` manually. Hyperparameters live in `benchmarks/benchmark_config.py` (two-body benchmark) and `benchmarks/{qm9,md17,crysmtm}_config.py` (transfer).

## Differences from the paper

The code was cleaned up after publication. The following fixes change numbers, so rerunning will not reproduce the published tables exactly:

- Modality comparison and OOD splits: the "best" checkpoint was a shallow copy of the model weights, so the last epoch was evaluated. The best validation epoch is now restored.
- Label scaling ([-1, 1] min-max) was fitted on the whole dataset; it is now fitted on the training split (two-body benchmark, modality comparison, OOD splits, pretraining).
- Element-shuffle and channel-permutation ablations evaluated on all pairs, most of them training data; they now use the held-out test split of each checkpoint.
- CrysMTM transfer: rotations of the same (phase, temperature) structure were split across train and test, and atoms were encoded as 0/1 instead of their atomic numbers. The split is now grouped by structure and atoms use Z (Ti = 22, O = 8).
- OOD bond-distance split: read the bond length of the wrong pair when the target had missing labels.

The GCN model was called `egnn` in earlier versions; `egnn` still works as an alias.

## Results

[`results/`](results/) holds the per-run CSVs behind the transfer, control and verification experiments reported in the paper. [`results/README.md`](results/README.md) documents each file and its columns.

## Repository layout

```
quantumcanvas/   installable package: data loaders, constants, Zenodo download, dataset builder
benchmarks/      benchmark models and training (train_models_twobody.py, benchmark_config.py)
  modality/      modality/fusion models, comparison and ablations
  crysmtm/       CrysMTM loader for the transfer experiments
scripts/         pretraining, transfer runs, control experiments
results/         per-run CSVs cited in the paper
tests/           smoke tests on synthetic data (uv run pytest)
build_dataset.py rebuild dataset_combined.npz from raw DFTB+ outputs
```

Rebuilding the dataset requires the raw per-pair DFTB+ outputs (`<raw_data_dir>/<pair>/{detailed.out,geo_end.xyz}` plus `dftb_ptbp_combined.csv` and `bond_distances_all.csv`), which are not part of the Zenodo archive: `uv run python build_dataset.py <raw_data_dir> dataset_combined.npz`.

## Citation

If you use QuantumCanvas, please cite the paper:

```bibtex
@article{polat2026quantumcanvas,
  author  = {Polat, Can and Kurban, Mustafa and Serpedin, Erchin and Kurban, Hasan},
  title   = {QuantumCanvas: a multimodal benchmark for learning two-body quantum interactions},
  journal = {Machine Learning: Science and Technology},
  year    = {2026},
  doi     = {10.1088/2632-2153/aea5d6},
  url     = {http://iopscience.iop.org/article/10.1088/2632-2153/aea5d6}
}
```

and, for the dataset itself:

```bibtex
@misc{polat2026quantumcanvas_dataset,
  author    = {Polat, Can and Kurban, Mustafa and Serpedin, Erchin and Kurban, Hasan},
  title     = {QuantumCanvas: A Multimodal Benchmark for Learning Two-Body Quantum Interactions},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.20631934},
  url       = {https://doi.org/10.5281/zenodo.20631934}
}
```

## License

Code: [MIT](LICENSE). Dataset: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
