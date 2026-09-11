"""Charge conservation and redundancy of the charge targets (results/charge_target_verification.csv).

Every pair is neutral (q_B = -q_A), so q_absmean, q_maxabs and q_std all equal |q_A|: the 19 benchmark
label keys are 17 independent quantities.
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from quantumcanvas import BENCHMARK_TARGETS, load_npz
from quantumcanvas.constants import REDUNDANT_CHARGE_TARGETS


def charge_checks(labels):
    total = np.array([lab['total_charge'] for lab in labels], dtype=np.float64)
    q = {k: np.array([lab[k] for lab in labels], dtype=np.float64) for k in REDUNDANT_CHARGE_TARGETS}
    residual = np.abs(total)
    rows = {
        'charge_conservation_max_abs_residual_e': f"{residual.max():.3e}",
        'charge_conservation_mean_abs_residual_e': f"{residual.mean():.3e}",
        'n_pairs_residual_gt_1e-3': int((residual > 1e-3).sum()),
        'n_pairs_total': len(labels),
        'total_charge_unique_values': len(np.unique(total)),
        'total_charge_std': f"{total.std(ddof=1):.3e}",
    }
    for a, b in [('q_absmean', 'q_maxabs'), ('q_absmean', 'q_std'), ('q_maxabs', 'q_std')]:
        rows[f'R2_{a}_vs_{b}'] = f"{np.corrcoef(q[a], q[b])[0, 1] ** 2:.6f}"
    rows['q_absmean_identical_to_q_std_atol_1e-12'] = bool(np.allclose(q['q_absmean'], q['q_std'], atol=1e-12))
    identical = all(np.allclose(q[a], q[b], atol=1e-12) for a, b in combinations(REDUNDANT_CHARGE_TARGETS, 2))
    # Key name kept from the deposited CSV (the 20-target configuration including bond length).
    rows['independent_targets_of_20'] = len(BENCHMARK_TARGETS) - (len(REDUNDANT_CHARGE_TARGETS) - 1 if identical else 0)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset_path', default='dataset_combined.npz')
    p.add_argument('--output', default='outputs/controls/charge_target_verification.csv')
    args = p.parse_args(argv)
    rows = charge_checks(load_npz(args.dataset_path)['labels'])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'check': list(rows), 'value': list(rows.values())}).to_csv(out, index=False)
    for k, v in rows.items():
        print(f"{k:45s} {v}")
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
