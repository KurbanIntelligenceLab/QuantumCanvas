import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
import subprocess

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

def run_modality_comparison(args):

    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Modality Comparison (Tabular vs Vision vs Fusion)")
    print("=" * 80)

    cmd = [
        sys.executable,
        str(_script_dir / "train_modality_comparison.py"),
        "--output_dir", "results_modality/modality_ablation",
    ]

    if args.quick:
        cmd.extend([
            "--models", "tabular_mlp", "vision_only", "qsn_v2",
            "--targets", "e_g_ev",
            "--seeds", "42",
            "--epochs", "10",
        ])
    elif args.models:
        cmd.extend(["--models"] + args.models)

    if args.targets:
        cmd.extend(["--targets"] + args.targets)

    if args.device:
        cmd.extend(["--device", args.device])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(_project_root))
    return result.returncode == 0

def run_element_shuffle(args):

    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Element ID Shuffle/Mask Ablation")
    print("=" * 80)

    cmd = [
        sys.executable,
        str(_script_dir / "element_shuffle_ablation.py"),
        "--output_dir", "results_modality/element_shuffle_ablation",
    ]

    if args.quick:
        cmd.extend([
            "--models", "qsn_v2", "vision_only",
            "--seeds", "42",
        ])
    elif args.models:
        cmd.extend(["--models"] + args.models)

    if args.device:
        cmd.extend(["--device", args.device])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(_project_root))
    return result.returncode == 0

def run_ood_composition(args):

    print("\n" + "=" * 80)
    print("EXPERIMENT 3: OOD Composition Split")
    print("=" * 80)

    cmd = [
        sys.executable,
        str(_script_dir / "ood_composition_split.py"),
        "--output_dir", "results_modality/ood_composition",
    ]

    if args.quick:
        cmd.extend([
            "--models", "tabular_mlp", "vision_only", "qsn_v2",
            "--targets", "e_g_ev",
            "--split_strategies", "held_out_pairs",
            "--seeds", "42",
            "--epochs", "10",
        ])
    elif args.models:
        cmd.extend(["--models"] + args.models)

    if args.targets:
        cmd.extend(["--targets"] + args.targets)

    if args.device:
        cmd.extend(["--device", args.device])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(_project_root))
    return result.returncode == 0

def generate_final_report(output_dir: Path):

    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("QUANTUMCANVAS ablation: MODALITY ABLATION EXPERIMENTS")
    report_lines.append(f"Generated: {datetime.now().isoformat()}")
    report_lines.append("=" * 100)

    modality_report = output_dir / "modality_ablation" / "summary_report.txt"
    if modality_report.exists():
        report_lines.append("\n\n" + "=" * 80)
        report_lines.append("PART 1: MODALITY COMPARISON RESULTS")
        report_lines.append("=" * 80)
        report_lines.append(modality_report.read_text())

    shuffle_report = output_dir / "element_shuffle_ablation" / "element_shuffle_report.txt"
    if shuffle_report.exists():
        report_lines.append("\n\n" + "=" * 80)
        report_lines.append("PART 2: ELEMENT ID SHUFFLE ABLATION RESULTS")
        report_lines.append("=" * 80)
        report_lines.append(shuffle_report.read_text())

    ood_report = output_dir / "ood_composition" / "ood_composition_report.txt"
    if ood_report.exists():
        report_lines.append("\n\n" + "=" * 80)
        report_lines.append("PART 3: OOD COMPOSITION SPLIT RESULTS")
        report_lines.append("=" * 80)
        report_lines.append(ood_report.read_text())


    final_report_path = output_dir / "ABLATION_SUMMARY.txt"
    with open(final_report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))

    print(f"\n{'=' * 80}")
    print(f"FINAL REPORT SAVED: {final_report_path}")
    print(f"{'=' * 80}")

def main():
    parser = argparse.ArgumentParser(description="Run all modality ablation experiments")
    parser.add_argument("--experiments", nargs="+",
                        choices=["modality_comparison", "element_shuffle", "ood_composition"],
                        default=["modality_comparison", "element_shuffle", "ood_composition"],
                        help="Which experiments to run")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test run with minimal settings")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Override default models")
    parser.add_argument("--targets", nargs="+", default=None,
                        help="Override default targets")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--skip_report", action="store_true",
                        help="Skip generating final report")
    args = parser.parse_args()

    print("=" * 80)
    print("QUANTUMCANVAS ablation: MODALITY ABLATION EXPERIMENTS")
    print("=" * 80)
    print(f"Experiments to run: {args.experiments}")
    print(f"Quick mode: {args.quick}")
    print(f"Device: {args.device or 'auto'}")

    output_dir = _project_root / "results_modality"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        'experiments': args.experiments,
        'quick': args.quick,
        'models': args.models,
        'targets': args.targets,
        'device': args.device,
        'timestamp': datetime.now().isoformat(),
    }
    with open(output_dir / "run_config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    success = True

    if "modality_comparison" in args.experiments:
        if not run_modality_comparison(args):
            print("WARNING: Modality comparison experiment failed")
            success = False

    if "element_shuffle" in args.experiments:
        if not run_element_shuffle(args):
            print("WARNING: Element shuffle experiment failed")
            success = False

    if "ood_composition" in args.experiments:
        if not run_ood_composition(args):
            print("WARNING: OOD composition experiment failed")
            success = False

    if not args.skip_report:
        generate_final_report(output_dir)

    print("\n" + "=" * 80)
    if success:
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    else:
        print("SOME EXPERIMENTS FAILED - CHECK LOGS")
    print("=" * 80)

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
