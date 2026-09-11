# AGENTS.md

Orientation for coding agents working in this repository. The human-facing guide is [README.md](README.md).

## What this is

Code and results for the QuantumCanvas paper (Machine Learning: Science and Technology, 2026, doi:10.1088/2632-2153/aea5d6). The dataset has 2,850 element pairs, each with a `[10, 32, 32]` image, a 2-atom geometry and 37 DFTB+ labels. It lives on Zenodo and is not in git.

## Setup

```bash
uv sync --extra benchmarks --extra dev              # everything; `uv sync` alone = loaders only
uv run python -c "import quantumcanvas; quantumcanvas.download()"   # ./dataset_combined.npz (md5-checked)
uv run pytest -q && uv run ruff check .             # ~20 s, synthetic data, no download needed
```

Use `uv run` (or `.venv/bin/python`). No `PYTHONPATH` is needed: `quantumcanvas` and `benchmarks` are installed packages. Linux, macOS and Windows are supported; CI runs on Linux and Windows.

## Public API (`quantumcanvas/`)

- `TwoBodyDataset(npz_path, target_label)`: a torch `Dataset` of `(image [10, 32, 32], target)`.
- `TwoBodyGraphDataset(npz_path, target_label, normalize_labels=True)`: a PyG dataset of `Data(z, pos, x, y, image, pair_name)`. `y` is min-max scaled to [-1, 1] unless `normalize_labels=False`; call `fit_normalization(train_indices)` after splitting so the scaling comes from training data only.
- `batch_images(batch)`: the `[B, 10, 32, 32]` images of a graph batch (they are stored concatenated).
- `load_npz`, `download`, and constants `CHANNELS`, `BENCHMARK_TARGETS` (label key to unit), `ELEMENT_TO_Z`.
- `quantumcanvas/build.py`: raw DFTB+ outputs to `dataset_combined.npz` (the raw data is not public).

`benchmarks.twobody_dataloader.TwoBodyDataset` is the graph dataset; the benchmark code imports it under that name. `benchmarks.pairs.pair_view` gives per-pair `[B, 2]` views of a batch (every graph has two atoms).

## Entry points

All of these take `--help`, read `dataset_combined.npz` from the working directory (`--dataset_path` to override), and write to `outputs/<experiment>/` (gitignored).

| Command | Purpose |
|---|---|
| `python -m benchmarks.train_models_twobody` | 9 models x benchmark targets x 3 seeds. Config in `benchmarks/benchmark_config.py`. |
| `python -m benchmarks.modality.train_modality_comparison` | parameter-matched modality/fusion comparison |
| `python -m benchmarks.modality.{element_shuffle_ablation,ood_composition_split,channel_perm_from_modality_ckpts}` | ablations (evaluated on each checkpoint's held-out test split) |
| `python scripts/pretrain_twobody.py`, then `python scripts/run_transfer.py --ckpt outputs/pretrain/...` | transfer to QM9/MD17/CrysMTM (`data/` holds those datasets) |
| `python scripts/controls.py --experiment {spatial,readout,bondlength}`, `scripts/bondlength_dimenet.py`, `scripts/charge_verification.py` | control experiments behind the `results/` control CSVs |

Model registries: `benchmarks/models.py::get_model` holds the benchmark models (`gcn` also accepts its old name `egnn`), and `benchmarks/modality/common.py::get_model` adds the modality/fusion models.

## Conventions and constraints

- `results/*.csv` are cited by the published paper. Do not edit, regenerate or reformat them. `tests/test_results.py` guards their columns.
- Keep the defaults in `benchmark_config.py` and `*_config.py`; they are the benchmark protocol. Add CLI flags rather than changing defaults.
- Splits are seeded; normalization is fitted on the training split; evaluation uses the held-out test split. Keep it that way (see "Differences from the paper" in the README).
- `numpy<2` and `torch-geometric==2.5.3` are deliberate (DimeNet++). Change dependencies only in `pyproject.toml`, then run `uv lock`.
- Keep console output and docs ASCII. Style follows the surrounding code. Lint is `ruff` with rules E9 and F.
- `main` is protected: changes go through a pull request with an admin review.
