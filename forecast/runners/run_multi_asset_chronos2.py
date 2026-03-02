from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compatibility wrapper for Chronos-2 multi-asset benchmark")
    parser.add_argument("--data-glob", type=str, default="data/*_timeseries.csv")
    parser.add_argument("--time-col", type=str, default="time")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 5, 20])
    parser.add_argument("--context-length", type=int, default=120)
    parser.add_argument("--min-train-size", type=int, default=365)
    parser.add_argument("--test-size", type=int, default=30)
    parser.add_argument("--step-size", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--model-id", type=str, default="amazon/chronos-2")
    parser.add_argument("--auto-adjust-splits", action="store_true")
    parser.add_argument("--skip-failed-assets", action="store_true")
    parser.add_argument("--limit-assets", type=int, default=0)
    parser.add_argument("--include-assets", type=str, default="")
    parser.add_argument("--exclude-assets", type=str, default="")
    parser.add_argument(
        "--paper-strict",
        action="store_true",
        help="Fail if any foundation backend is non-native or if any asset run degrades.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Persist and aggregate per-fold prediction/quantile traces across assets",
    )
    parser.add_argument("--output", type=str, default="results/multi_asset_chronos2_summary.csv")
    args = parser.parse_args()

    output_path = Path(args.output)
    cmd = [
        "python",
        "forecast/runners/run_multi_asset_foundation.py",
        "--data-glob",
        args.data_glob,
        "--time-col",
        args.time_col,
        "--horizon",
        str(args.horizon),
        "--context-length",
        str(args.context_length),
        "--min-train-size",
        str(args.min_train_size),
        "--test-size",
        str(args.test_size),
        "--step-size",
        str(args.step_size),
        "--batch-size",
        str(args.batch_size),
        "--models",
        "chronos2",
        "--model-overrides",
        f"chronos2={args.model_id}",
        "--output",
        str(output_path),
    ]
    if args.auto_adjust_splits:
        cmd.append("--auto-adjust-splits")
    if args.skip_failed_assets:
        cmd.append("--skip-failed-assets")
    if args.limit_assets > 0:
        cmd.extend(["--limit-assets", str(args.limit_assets)])
    if args.include_assets:
        cmd.extend(["--include-assets", args.include_assets])
    if args.exclude_assets:
        cmd.extend(["--exclude-assets", args.exclude_assets])
    if args.save_predictions:
        cmd.append("--save-predictions")
    if args.paper_strict:
        cmd.append("--paper-strict")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
