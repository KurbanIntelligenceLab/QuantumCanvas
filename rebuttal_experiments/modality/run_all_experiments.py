#!/usr/bin/env python3
"""
Unified runner for all modality ablation experiments.

This script runs all rebuttal experiments in sequence:
1. Modality comparison (tabular vs vision vs fusion)
2. Element ID shuffle/mask ablation  
3. OOD composition split

Results are saved to rebuttal_results/ and can be used to generate
rebuttal figures and tables.

Usage:
    # Run everything
    python rebuttal_experiments/modality/run_all_experiments.py
    
    # Run specific experiments
    python rebuttal_experiments/modality/run_all_experiments.py \
        --experiments modality_comparison element_shuffle
    
    # Quick test run
    python rebuttal_experiments/modality/run_all_experiments.py --quick
"""

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
    """Run modality comparison experiment."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Modality Comparison (Tabular vs Vision vs Fusion)")
    print("=" * 80)
    
    cmd = [
        sys.executable,
        str(_script_dir / "train_modality_comparison.py"),
        "--output_dir", "rebuttal_results/modality_ablation",
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
    """Run element shuffle ablation experiment."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Element ID Shuffle/Mask Ablation")
    print("=" * 80)
    
    cmd = [
        sys.executable,
        str(_script_dir / "element_shuffle_ablation.py"),
        "--output_dir", "rebuttal_results/element_shuffle_ablation",
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
    """Run OOD composition split experiment."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: OOD Composition Split")
    print("=" * 80)
    
    cmd = [
        sys.executable,
        str(_script_dir / "ood_composition_split.py"),
        "--output_dir", "rebuttal_results/ood_composition",
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
    """Generate a unified final report combining all experiment results."""
    
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("QUANTUMCANVAS REBUTTAL: MODALITY ABLATION EXPERIMENTS")
    report_lines.append(f"Generated: {datetime.now().isoformat()}")
    report_lines.append("=" * 100)
    
    # 1. Modality comparison summary
    modality_report = output_dir / "modality_ablation" / "summary_report.txt"
    if modality_report.exists():
        report_lines.append("\n\n" + "=" * 80)
        report_lines.append("PART 1: MODALITY COMPARISON RESULTS")
        report_lines.append("=" * 80)
        report_lines.append(modality_report.read_text())
    
    # 2. Element shuffle summary
    shuffle_report = output_dir / "element_shuffle_ablation" / "element_shuffle_report.txt"
    if shuffle_report.exists():
        report_lines.append("\n\n" + "=" * 80)
        report_lines.append("PART 2: ELEMENT ID SHUFFLE ABLATION RESULTS")
        report_lines.append("=" * 80)
        report_lines.append(shuffle_report.read_text())
    
    # 3. OOD composition summary
    ood_report = output_dir / "ood_composition" / "ood_composition_report.txt"
    if ood_report.exists():
        report_lines.append("\n\n" + "=" * 80)
        report_lines.append("PART 3: OOD COMPOSITION SPLIT RESULTS")
        report_lines.append("=" * 80)
        report_lines.append(ood_report.read_text())
    
    # Key findings for rebuttal
    report_lines.append("\n\n" + "=" * 100)
    report_lines.append("KEY FINDINGS FOR REBUTTAL")
    report_lines.append("=" * 100)
    report_lines.append("""
Based on the experiments above, the key points for the rebuttal are:

1. VISION ADDS VALUE OVER TABULAR:
   - Compare test MAE of vision models (ViT, QuantumShellNet) vs tabular baseline
   - If vision > tabular: "Convolutional processing of spatial orbital structure 
     captures information that cannot be recovered from pooled statistics alone."

2. MODELS DON'T JUST MEMORIZE ELEMENT IDENTITIES:
   - Check the % MAE increase when element IDs are shuffled
   - Small increase: "Models learn from orbital/spatial features, not just element lookup"
   - Large increase for some models: "This motivates using vision-based representations"

3. MODELS GENERALIZE TO UNSEEN COMPOSITIONS:
   - Check the OOD generalization gap
   - Small gap: "Pretraining on QuantumCanvas captures transferable quantum interactions
     that generalize to unseen element pairs."

SUGGESTED REBUTTAL SENTENCES:
- "Our ablations show that vision-based processing of orbital density images outperforms
   equivalent tabular baselines by X%, demonstrating that spatial structure is informative."
- "Element ID shuffling increases MAE by only Y% for QuantumShellNet, showing the model
   learns from spatial orbital features rather than memorizing element correlations."
- "On held-out compositions, vision models show Z% smaller generalization gap compared
   to geometry-only baselines, confirming transferable feature learning."
""")
    
    # Save final report
    final_report_path = output_dir / "REBUTTAL_SUMMARY.txt"
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
    print("QUANTUMCANVAS REBUTTAL: MODALITY ABLATION EXPERIMENTS")
    print("=" * 80)
    print(f"Experiments to run: {args.experiments}")
    print(f"Quick mode: {args.quick}")
    print(f"Device: {args.device or 'auto'}")
    
    output_dir = _project_root / "rebuttal_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save run config
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
    
    # Run experiments
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
    
    # Generate final report
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
