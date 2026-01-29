"""
Compute channel importance as % (averaged over targets and over MAE/RMSE).
Reads channel_permutation_report.txt, outputs table for rebuttal docs.
"""
import re
from pathlib import Path


def parse_report(path: Path):
    """Parse PER-CHANNEL IMPORTANCE section; return dict target -> list of (abs_dmae, abs_drmse) for ch_0..ch_9."""
    text = path.read_text(encoding="utf-8")
    in_block = False
    result = {}
    current_target = None
    ch_data = []
    for line in text.splitlines():
        if "PER-CHANNEL IMPORTANCE PER TARGET" in line:
            in_block = True
            continue
        if in_block and "AVERAGE PER TARGET" in line:
            break
        if not in_block:
            continue
        target_m = re.match(r"^---\s+(\S+)\s+---\s*$", line)
        if target_m:
            if current_target is not None and len(ch_data) == 10:
                result[current_target] = ch_data
            current_target = target_m.group(1)
            ch_data = []
            continue
        ch_m = re.match(r"\s+ch_(\d+)\s+.*?ΔMAE:\s*([-\d.]+)\s+ΔRMSE:\s*([-\d.]+)", line)
        if ch_m:
            ch_data.append((abs(float(ch_m.group(2))), abs(float(ch_m.group(3)))))
    if current_target is not None and len(ch_data) == 10:
        result[current_target] = ch_data
    return result


def compute_pct_importance(parsed):
    """For each target: pct_MAE_ch = 100 * |ΔMAE_ch| / sum(|ΔMAE|), same for RMSE. Then average over targets."""
    n_ch = 10
    sum_mae = [0.0] * n_ch
    sum_rmse = [0.0] * n_ch
    n_targets = 0
    for target, ch_data in parsed.items():
        mae_vals = [ch_data[i][0] for i in range(n_ch)]
        rmse_vals = [ch_data[i][1] for i in range(n_ch)]
        total_mae = sum(mae_vals)
        total_rmse = sum(rmse_vals)
        if total_mae <= 0 and total_rmse <= 0:
            continue
        if total_mae > 0:
            for i in range(n_ch):
                sum_mae[i] += 100.0 * mae_vals[i] / total_mae
        if total_rmse > 0:
            for i in range(n_ch):
                sum_rmse[i] += 100.0 * rmse_vals[i] / total_rmse
        n_targets += 1
    if n_targets == 0:
        return None, None, None
    avg_pct_mae = [sum_mae[i] / n_targets for i in range(n_ch)]
    avg_pct_rmse = [sum_rmse[i] / n_targets for i in range(n_ch)]
    avg_pct_both = [(avg_pct_mae[i] + avg_pct_rmse[i]) / 2.0 for i in range(n_ch)]
    return avg_pct_mae, avg_pct_rmse, avg_pct_both


def main():
    root = Path(__file__).resolve().parent.parent
    for name in ["rebuttal_results_100_epochs", "rebuttal_results"]:
        report_path = root / name / "channel_permutation_report.txt"
        if not report_path.exists():
            print(f"Skip {report_path} (not found)")
            continue
        parsed = parse_report(report_path)
        avg_mae, avg_rmse, avg_both = compute_pct_importance(parsed)
        if avg_mae is None:
            print(f"{name}: no targets parsed, skip")
            continue
        print(f"\n=== {name} ===\nChannel importance (% of total sensitivity, avg over {len(parsed)} targets)")
        print("Averaged over losses (MAE + RMSE):")
        print("| Channel | MAE % | RMSE % | Avg % |")
        print("|---------|-------|--------|-------|")
        for i in range(10):
            print(f"| ch_{i} | {avg_mae[i]:.1f} | {avg_rmse[i]:.1f} | {avg_both[i]:.1f} |")
        print()


if __name__ == "__main__":
    main()
