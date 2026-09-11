"""The deposited CSVs are cited by the paper; guard their shape and the Table 4 grid."""
from pathlib import Path

import pandas as pd
import pytest

RESULTS = Path(__file__).resolve().parent.parent / "results"

COLUMNS = {
    "per_seed.csv": ["encoder", "case", "arm", "seed", "test_mae", "best_epoch", "lr", "n_params"],
    "transfer_encoder_headroom.csv": ["case", "goten_scratch", "schnet_pretrained", "goten_scratch_better",
                                      "goten_delta_pct", "schnet_delta_pct"],
    "transfer_protocol_ablation.csv": ["config", "seed", "test_mae", "best_val_mae", "best_epoch", "epochs_run",
                                       "lr", "load_mode", "head_dropped", "ep1_val_mae", "n_params"],
    "bondlength_leakage_control.csv": ["architecture", "coordinate_input", "mae_mean", "mae_std", "n_seeds",
                                       "valid_control"],
    "spatial_shuffle_rawscalar_controls.csv": ["target", "mode", "seed", "n_params", "best_epoch", "best_val_loss",
                                               "test_mae", "test_rmse", "n_train", "n_val", "n_test", "feat_name"],
    "label_readout_channel_ablation.csv": ["target", "mode", "seed", "n_params", "best_epoch", "best_val_loss",
                                           "test_mae", "test_rmse", "n_train", "n_val", "n_test"],
    "charge_target_verification.csv": ["check", "value"],
}


@pytest.mark.parametrize("name,columns", COLUMNS.items())
def test_columns(name, columns):
    assert list(pd.read_csv(RESULTS / name).columns) == columns


def test_transfer_grid_is_complete():
    df = pd.read_csv(RESULTS / "per_seed.csv")
    assert len(df) == 96
    assert df.groupby(["encoder", "case", "arm"]).size().eq(3).all()
    assert df.groupby(["encoder", "case"]).ngroups == 16
