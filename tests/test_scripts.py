"""The control scripts run end to end on synthetic data."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import charge_verification  # noqa: E402
import controls  # noqa: E402

from quantumcanvas import load_npz  # noqa: E402


def test_raw_scalar_features(synthetic_npz):
    assert controls.raw_scalar_features(load_npz(synthetic_npz)).shape == (24, 6)


@pytest.mark.parametrize("experiment,n_rows", [("spatial", 6), ("readout", 12), ("bondlength", 3)])
def test_controls(experiment, n_rows, synthetic_npz, tmp_path):
    rows = controls.main(["--experiment", experiment, "--dataset_path", synthetic_npz, "--output_dir", str(tmp_path),
                          "--seeds", "0", "--epochs", "1", "--device", "cpu"])
    assert len(rows) == n_rows
    assert (tmp_path / f"{experiment}.csv").exists()


def test_charge_verification(synthetic_npz):
    rows = charge_verification.charge_checks(load_npz(synthetic_npz)["labels"])
    assert rows["n_pairs_residual_gt_1e-3"] == 0
    assert rows["independent_targets_of_20"] == 17


def test_modality_train_then_element_shuffle(synthetic_npz, tmp_path):
    """Checkpoints from the modality comparison feed the element-shuffle ablation (held-out test split)."""
    import torch
    from benchmarks.modality import element_shuffle_ablation as shuffle
    from benchmarks.modality import train_modality_comparison as modality
    result = modality.train_single('tabular_mlp', 'e_g_ev', 0, synthetic_npz, batch_size=8, epochs=1, lr=1e-3,
                                   patience=5, device=torch.device('cpu'), output_dir=tmp_path, verbose=False)
    ckpt = next(tmp_path.rglob('best_model.pt'))
    out = shuffle.run_single_checkpoint(str(ckpt), synthetic_npz, 8, torch.device('cpu'), ['normal', 'mask'])
    assert abs(out['ablations']['normal']['mae'] - result['test_mae']) < 1e-5
