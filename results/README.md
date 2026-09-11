# Results

Per-run CSVs behind the transfer, control and verification experiments in the
paper ([doi:10.1088/2632-2153/aea5d6](https://doi.org/10.1088/2632-2153/aea5d6)).

The transfer-learning table in the paper aggregates `per_seed.csv`: group by
encoder, case and arm, then take the mean and sample standard deviation
(`ddof=1`) of `test_mae` over the three seeds. The percentage change is
`100 * (pretrained_mean - scratch_mean) / scratch_mean`.

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

## Where each file comes from

| File | Source |
|---|---|
| `per_seed.csv` | `scripts/run_transfer.py`, one run per row (`--arm scratch` / `--arm pretrained --ckpt ...` after `scripts/pretrain_twobody.py`) |
| `transfer_encoder_headroom.csv` | aggregated from `per_seed.csv` (per-case means) |
| `transfer_protocol_ablation.csv` | `scripts/run_transfer.py` with `--load {full,encoder}` and `--lr` varied |
| `bondlength_leakage_control.csv` | `scripts/bondlength_dimenet.py` (`--mode leaked` / `leakage_free`) and `scripts/controls.py --experiment bondlength` (coordinate-free and mean-predictor rows), aggregated over seeds |
| `spatial_shuffle_rawscalar_controls.csv` | `scripts/controls.py --experiment spatial` |
| `label_readout_channel_ablation.csv` | `scripts/controls.py --experiment readout` |
| `charge_target_verification.csv` | `scripts/charge_verification.py` |

See the main [README](../README.md#running-the-benchmarks) for how to run the scripts.
