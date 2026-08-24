# Results deposited for the manuscript

Every numeric claim in the manuscript and supplementary information is
computed from one of the files below, so the reported numbers can be checked
without rerunning any training.

Table 4 of the manuscript is a strict aggregation of `per_seed.csv`: group by
encoder, case and arm, then take the mean and sample standard deviation
(`ddof=1`) of `test_mae` over the three seeds. The percentage change is
`100 * (pretrained_mean - scratch_mean) / scratch_mean`. Applying that rule to
the 96 rows reproduces every printed cell, so the table can be checked without
rerunning training.

## Transfer experiments

| File | Contents |
|---|---|
| `per_seed.csv` | All 96 individual runs: encoder, case, arm, seed, test MAE, best epoch, learning rate, parameter count. The primary transfer deposit; Table 4 aggregates directly from it. |
| `transfer_encoder_headroom.csv` | Per-target comparison of GotenNet from-scratch against SchNet pretrained, with each encoder's own transfer delta. |
| `transfer_protocol_ablation.csv` | 15 runs isolating the effect of the fine-tuning learning rate: scratch, full-checkpoint and encoder-only loading, each at matched and asymmetric learning rates. |

Grid coverage: 2 encoders (SchNet, GotenNet) x 8 downstream targets
(QM9 gap/HOMO/LUMO, MD17 aspirin/benzene/ethanol, CrysMTM HOMO/LUMO) x 2
arms (scratch, pretrained) x 3 seeds (42, 123, 456) = 96 runs, no cells
excluded. All runs use learning rate 1e-4 on both arms; parameter counts are
matched within encoder (266,593 SchNet; 288,257 GotenNet).

MAE units are those of each downstream dataset and are comparable across
encoders within a dataset, not across datasets. Only test MAE was recorded per
run: `scripts/run_transfer.py` accumulates absolute error only, so RMSE is not
part of this deposit.

## Control experiments

| File | Contents |
|---|---|
| `bondlength_leakage_control.csv` | Bond-length prediction under four conditions: optimized coordinates as input, a constant-geometry variant, a coordinate-free control taking only the two atomic numbers, and a constant-predictor baseline. The `valid_control` column marks which row is the leakage-free comparison. |
| `spatial_shuffle_rawscalar_controls.csv` | Parameter-matched three-way control on energy gap and dipole magnitude: intact orbital image, channel-preserving spatial shuffle, and a raw-scalar MLP fed the ungrouped generating scalars. |
| `label_readout_channel_ablation.csv` | Channel-masking ablation on the dipole and charge targets, separating targets rendered directly into an input channel from those that must be inferred. |
| `charge_target_verification.csv` | Charge conservation residuals and the pairwise redundancy checks establishing that the three charge-magnitude targets are one quantity. |

## Reproducing individual runs

`REPRODUCE.md` at the repository root maps each manuscript float to the
script, configuration and command that generates it. The transfer runs come
from `scripts/run_transfer.py`; the control experiments from the scripts
named in that mapping.
