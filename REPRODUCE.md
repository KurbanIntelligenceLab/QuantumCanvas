# Reproducing the manuscript floats

## Deposited results

`results/` holds the numeric output behind every table and control
experiment in the manuscript and supplementary information. Any reported
number can be checked against these files without rerunning training. The 96
individual transfer runs are deposited as `results/per_seed.csv`.

Table 4 of the manuscript aggregates directly from that file: group by encoder, case and arm, then take the mean and sample standard
deviation (`ddof=1`) of `test_mae` over the three seeds, with
`delta_pct = 100 * (pretrained_mean - scratch_mean) / scratch_mean`. Applying
that rule to the 96 rows reproduces every printed cell.

`results/README.md` documents each file, its columns and units. The sections
below give the commands that generated the deposited data.

All commands assume the repo root as working directory and `PYTHONPATH=.`
set (the `benchmarks` package is not installed, only importable from the
repo root). Every command below was verified this session to at least
import cleanly and parse its arguments; none were run to completion (no
GPU available in this environment, and QM9/MD17/CrysMTM are not staged
locally). See `development/audit/gpu_staging.md` for the one command that
was run end to end (a two-step smoke test of the SchNet + two-body
pipeline).

## Dataset

```
python build_dataset.py
```
Requires a `raw_data/<pair_name>/{detailed.out,geo_end.xyz}` tree plus
`dftb_ptbp_combined.csv` and `bond_distances_all.csv`, none of which exist
on this machine (see `development/audit/code_inventory.md` section 3).
Output: `dataset_combined.npz`, which is already present at the repo root
(31.9 MB, 2850 samples) and is what every benchmark script below consumes.

## Table 1 (`tab:qc_full`) — comprehensive benchmark, 20 targets x 8 models

```
PYTHONPATH=. python -m benchmarks.train_models_twobody
```
Config: `benchmarks/benchmark_config.py` (no CLI flags; trains every model
in `ModelConfigs.available_models` across every target in `cfg.targets`
and every seed in `cfg.seeds`).
Output: `results_twobody/<target>/<model>/seed_<seed>/{best_model.pt,results.json,training_history.csv}`
and a top-level `comprehensive_benchmark_summary.json`.
No script in the repo turns that summary JSON into the Table 1 LaTeX; the
table must be assembled from `comprehensive_benchmark_summary.json` by
hand or with a maintainer-written formatting step — say so rather than
inventing one.
The DimeNet++ row cannot be produced under the pinned `numpy>=2.0` +
`torch_geometric==2.5.3` combination in this environment: see
`benchmarks/models.py::DimeNetRegressor` and the numpy/torch-sparse
constraints below.

## Table 2 (`tab:modality`) — parameter-matched modality comparison

```
PYTHONPATH=. python -m benchmarks.modality.train_modality_comparison
```
Config: hardcoded `SEEDS`, `TARGETS`, `MODELS`, `EPOCHS=100`, `LR=1e-3` at
the top of `benchmarks/modality/train_modality_comparison.py` (no
benchmark_config.py dependency for hyperparameters).
Output directory: `--output_dir` (default in the script; run `--help` to
confirm the current default) holding per-model per-target per-seed JSON
results.
Import and `--help` verified this session.

## Table 3 (`tab:shuffle`) — element-identity shuffling

```
PYTHONPATH=. python -m benchmarks.modality.element_shuffle_ablation --checkpoint <path/to/best_model.pt>
```
Requires checkpoints already produced by Table 2's training run (the
script evaluates shuffled inputs against trained models; it does not
train from scratch). `--device`, `--output_dir` accepted; run `--help` for
the full flag set. Import and `--help` verified this session.

## Table 4 (`tab:transfer`) — transfer learning to QM9 / MD17 / CrysMTM

Two-stage pipeline:

```
PYTHONPATH=. python -m benchmarks.train_models_twobody
```
produces the two-body pretraining checkpoints
(`results_twobody/<target>/<model>/seed_<seed>/best_model.pt`) that Table 4
fine-tunes from. This stage has no surviving output on this machine (see
`development/audit/code_inventory.md` section 2) and must be run first, in
full, before any transfer case below can load a pretrained encoder.

```
python scripts/run_transfer.py --case qm9:<target> --seed <seed> --arm scratch --model schnet
python scripts/run_transfer.py --case qm9:<target> --seed <seed> --arm pretrained --model schnet --ckpt <path/to/best_model.pt>
python scripts/run_transfer.py --case md17:<molecule> --seed <seed> --arm scratch --model schnet
python scripts/run_transfer.py --case md17:<molecule> --seed <seed> --arm pretrained --model schnet --ckpt <path/to/best_model.pt>
python scripts/run_transfer.py --case crysmtm:<target> --seed <seed> --arm scratch --model schnet
python scripts/run_transfer.py --case crysmtm:<target> --seed <seed> --arm pretrained --model schnet --ckpt <path/to/best_model.pt>
```
`<target>`/`<molecule>` come from `benchmarks/qm9_config.py::QM9_TARGETS`,
the MD17 molecule names accepted by `torch_geometric.datasets.MD17`, and
the nine CrysMTM property keys hardcoded in `scripts/run_transfer.py::get_case`
(`HOMO, LUMO, Eg, Ef, Et, Eta, disp, vol, bond`).
Output: `runs/<case>__<model>__seed<seed>__<arm>/{result.json,best.pt,state.pt}`.
QM9/MD17 auto-download via PyTorch Geometric on first run; CrysMTM requires
the pre-existing `./data/CrysMTM/<phase>/<temp>K/xyz/rot_<n>.xyz` + `labels.csv`
tree, present locally under `data/CrysMTM/` (228 MB, not tracked by git).
`--help` and argument parsing verified this session; not run end-to-end
(no local GPU, and the pretraining stage above has not been run).

Equivalent `benchmarks/qm9_schnet.py`, `benchmarks/md17_schnet.py`,
`benchmarks/crysmtm_schnet.py` (and `_gotennet`/`_faenet` variants) exist
with their own argparse surface and are an alternative, dataset-specific
entry point to the same fine-tuning comparison; `--help` verified for the
schnet and faenet variants (`PYTHONPATH=.` required). The `_gotennet.py`
scripts import `gotennet`, which is installable (see Environment section)
but whose `--help` output was not checked this session; `gotennet` was
instead verified via `scripts/run_transfer.py --model gotennet`, which
produced 48 of the 96 Table 4 transfer runs this session (the GotenNet
rows of Table 4).

## Figures 1–3

No script in the repo reproduces Figures 1 (`fig:introFig`), 2
(`fig:statistics`), or 3 (`fig:transferLearning`, panel schematic + PCA
projection). Say so rather than inventing a command: these appear to be
assembled outside the code in this repository (vector-graphics
composition, per the "Adobe Illustrator exports" comment in
`benchmarks/modality/make_si_figures.py`).

## Figure 4 (`fig:ood`), Figures S1–S3

```
PYTHONPATH=. python benchmarks/modality/make_si_figures.py
```
Writes `figure4.pdf` (main text, condensed out-of-distribution summary),
`figureS1.pdf` (element-identity shuffle heatmap), `figureS2.pdf` (the
four OOD splits not shown in Figure 4), `figureS3.pdf` (channel-permutation
importance) into the working directory.
The values plotted are hardcoded arrays transcribed from the
supplementary LaTeX tables (`tab:supp-ood-*`, `tab:supp-shuffle-full`,
`tab:supp-channels-avg`) at the top of the script, not read live from a
results file — see the script's own module docstring. Import verified
this session; not executed (would write PDFs into the repo root).

## Control experiments (new this round: label-readout ablation,
## spatial-shuffle / raw-scalar controls, bond-length leakage, charge
## conservation)

These four controls were run this review round and their outputs
(`label_readout.{csv,md,png}`, `spatial_controls.{csv,md,png}`,
`bondlength_leakage.{csv,md,png}`, `charge_analysis.md`,
`charge_residuals.csv`) live under `development/2nd_review/` — ablation
material, correctly excluded from the public tree. The code that produced
them is split between two places:

- Label-readout ablation and spatial-shuffle / raw-scalar controls: driven
  by `development/2nd_review/gpu_controls/main.py` (`--experiment readout`
  or `--experiment spatial`), which is GPU-run scratch code, not part of
  the public package. It is not copied into the public tree because it
  was written as one-off ablation infrastructure (hardcoded `ELEMENT_TO_Z`
  duplicated from `benchmarks/twobody_dataloader.py`, no config file, no
  docstring of its protocol beyond the script itself). A maintainer who
  wants this control in the public repo should port it into
  `benchmarks/modality/` following the same pattern as
  `element_shuffle_ablation.py`, rather than copying the scratch script
  verbatim.
- Bond-length leakage check: `scripts/bondlength_dimenet.py`, already in
  the public tree.
  ```
  PYTHONPATH=. python scripts/bondlength_dimenet.py --model schnet --mode leaked --seed 42
  PYTHONPATH=. python scripts/bondlength_dimenet.py --model schnet --mode leakage_free --seed 42
  ```
  `--model dimenet` additionally requires the DimeNet++ fix below.
  Output: `bl/<model>_<mode>_seed<seed>/result.json`. `--help` verified
  this session.
- Charge-conservation residuals: no script in the repo. The manuscript's
  `development/2nd_review/charge_analysis.md` computes
  `|total_charge|` per pair directly from `dataset_combined.npz` labels —
  a short one-off analysis, not committed as a script anywhere in this
  repo. State this gap rather than inventing a command.

## Environment

Two dependency problems were found and confirmed by direct construction
attempts this session, not assumed:

1. **DimeNet++ / numpy 2.0.** `torch_geometric==2.5.3`'s
   `dimenet_utils.py` calls `np.math.factorial`, removed in numpy 2.0
   (`AttributeError: module 'numpy' has no attribute 'math'`, reproduced
   directly by constructing `benchmarks.models.DimeNetRegressor` under
   numpy 2.4.6). This affects the paper's headline bond-length result
   (Table 1, r = 0.008 A, a DimeNet++ number). Pin `numpy<2.0` to
   construct DimeNet++ under this `torch_geometric` version, or upgrade
   `torch_geometric` past the version that fixed this (verify against the
   installed release before relying on it — not checked this session).
2. **`torch-sparse` is required but absent from every requirements file.**
   `DimeNetPlusPlus.forward` calls `torch_geometric.typing.SparseTensor`,
   which raises `ImportError: 'SparseTensor' requires 'torch-sparse'` if
   the package is not installed (reproduced directly this session, after
   fixing problem 1, by running a forward pass). `torch-sparse` failed to
   build in this sandboxed environment for an environment-specific reason
   (a git-write restriction when the build touches `.gitmodules`, not a
   real dependency conflict) — install normally on an unrestricted host.
3. **`gotennet` is installable and was used this session.** The
   `_gotennet.py` fine-tuning scripts and 48 of the 96 Table 4 transfer
   runs (the GotenNet rows) require the `gotennet` package plus its own
   dependency set (`omegaconf`, `hydra-core`, `pytorch-lightning`,
   `torchmetrics`), none of which ship with the base `torch_geometric`
   stack. Confirmed installable and importable this session; see the
   install commands below.

Dependencies are declared in `pyproject.toml` and managed with
[uv](https://docs.astral.sh/uv/). The core set carries the `numpy<2.0`
constraint required by problem 1; the PyTorch Geometric C-extension
packages and the two optional encoder backends are separate extras
because they need a wheel index rather than plain PyPI resolution.

Base environment:

```
uv sync
```

This creates `.venv` and installs the core dependencies from `uv.lock`,
giving a byte-identical environment on every host. Run anything in it with
`uv run`, for example `uv run python scripts/run_transfer.py --help`.

The `pyg-ext` extra (`torch-scatter`, `torch-sparse`, `torch-cluster`,
`torch-spline-conv`) is not a plain PyPI wheel on every platform. These are
published on a version-specific index keyed to the installed `torch` and
CUDA build, so they need an explicit index URL:

```
uv pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    --find-links https://data.pyg.org/whl/torch-2.8.0+cu121.html
```

for a CUDA 12.1 build, or on a CPU-only host:

```
uv pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    --find-links https://data.pyg.org/whl/torch-2.8.0+cpu.html
```

Substitute the index suffix (`+cu121`, `+cu118`, `+cpu`) for the actual
installed `torch` build. Without the index, `torch-sparse` falls back to a
source build that fails in sandboxed environments with a git-write
restriction (see problem 2 above); the wheel index avoids the build
entirely.

The encoder backends install from PyPI directly:

```
uv sync --extra gotennet --extra faenet
```

The PyPI package is named `gotennet` (not `gotennet-pytorch` or any other
variant seen in some forks); the import name matches the package name. Its
own dependencies (`omegaconf`, `hydra-core`, `pytorch-lightning`,
`torchmetrics`) resolve automatically under `uv sync`, unlike a bare
`pip install --no-deps`.

Generate or refresh the lockfile with `uv lock`. Commit `uv.lock` so every
reproduction run resolves to the same versions.
