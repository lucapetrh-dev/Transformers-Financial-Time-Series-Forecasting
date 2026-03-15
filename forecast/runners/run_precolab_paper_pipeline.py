from __future__ import annotations

import argparse
import shutil
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)


def _copy_file_with_retries(src: Path, dst: Path, retries: int = 7) -> None:
    last_exc: OSError | None = None
    for attempt in range(1, retries + 1):
        tmp = dst.with_suffix(f"{dst.suffix}.tmp_sync")
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, tmp)
            tmp.replace(dst)
            return
        except OSError as exc:
            last_exc = exc
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            time.sleep(min(0.25 * attempt, 2.0))
    raise OSError(f"Failed to copy {src} -> {dst} after {retries} attempts") from last_exc


def _sync_tree(src_root: Path, dst_root: Path) -> None:
    files = sorted(p for p in src_root.rglob("*") if p.is_file())
    print(f"\n>>> Materializing {len(files)} artifacts into {dst_root}")
    for i, src in enumerate(files, start=1):
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        _copy_file_with_retries(src, dst)
        if i % 100 == 0 or i == len(files):
            print(f"Materialized {i}/{len(files)} artifacts")


def _join(root: Path, rel: str) -> str:
    return str(root / rel)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic pre-Colab strict paper pipeline")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 5, 20])
    parser.add_argument("--assets", type=str, default="btc,eth,ada,doge,xmr,xrp")
    parser.add_argument("--results-root", type=str, default="results/paper")
    parser.add_argument(
        "--staging-root",
        type=str,
        default="",
        help="Optional temporary results root. If omitted and results-root is under results/paper, /tmp staging is used automatically.",
    )
    parser.add_argument("--keep-staging", action="store_true", help="Keep staging directory after successful materialization")
    parser.add_argument("--skip-chronos", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()

    h = str(args.horizon)
    py = sys.executable
    results_root = Path(args.results_root)
    results_root_str = results_root.as_posix()
    auto_stage = "results/paper" in results_root_str
    if args.staging_root:
        execution_root = Path(args.staging_root)
        cleanup_staging = False
    elif auto_stage:
        execution_root = Path(tempfile.mkdtemp(prefix="paper_pipeline_", dir="/tmp"))
        cleanup_staging = not args.keep_staging
        print(f"\n>>> Using staging results root to avoid transient filesystem errors: {execution_root}")
    else:
        execution_root = results_root
        cleanup_staging = False

    execution_root.mkdir(parents=True, exist_ok=True)

    _run(
        [
            py,
            "forecast/runners/run_data_manifest.py",
            "--data-glob",
            ".data/hourly/*_timeseries_hourly.csv",
            "--include-assets",
            args.assets,
            "--provenance-config",
            "forecast/config/data_provenance.json",
            "--output",
            _join(execution_root, "data_manifest.csv"),
        ]
    )
    _run(
        [
            py,
            "forecast/runners/run_contract_audit.py",
            "--contract-path",
            "forecast/config/paper_experiment.json",
            "--manifest-path",
            _join(execution_root, "data_manifest.csv"),
            "--require-provenance",
            "--output",
            _join(execution_root, "contract_asset_manifest.csv"),
        ]
    )

    _run(
        [
            py,
            "forecast/runners/run_multi_asset_baselines.py",
            "--data-glob",
            ".data/hourly/*_timeseries_hourly.csv",
            "--include-assets",
            args.assets,
            "--horizon",
            h,
            "--run-paired-sentiment",
            "--no-arima",
            "--no-xgboost",
            "--min-train-size",
            "300",
            "--test-size",
            "60",
            "--step-size",
            "240",
            "--auto-adjust-splits",
            "--save-predictions",
            "--output",
            _join(execution_root, f"multi_asset_baselines_h{h}_paired_summary.csv"),
        ]
    )

    _run(
        [
            py,
            "forecast/runners/run_multi_asset_transformers.py",
            "--data-glob",
            ".data/hourly/*_timeseries_hourly.csv",
            "--include-assets",
            args.assets,
            "--horizon",
            h,
            "--models",
            "patchtst_like,itransformer_like",
            "--objective-track",
            "both",
            "--point-loss",
            "mse",
            "--run-paired-sentiment",
            "--min-train-size",
            "300",
            "--test-size",
            "60",
            "--step-size",
            "240",
            "--auto-adjust-splits",
            "--save-predictions",
            "--save-history",
            "--output",
            _join(execution_root, f"multi_asset_transformers_h{h}_paired_summary.csv"),
        ]
    )

    if not args.skip_chronos:
        _run(
            [
                py,
                "forecast/runners/run_multi_asset_chronos2.py",
                "--data-glob",
                ".data/hourly/*_timeseries_hourly.csv",
                "--include-assets",
                args.assets,
                "--horizon",
                h,
                "--min-train-size",
                "300",
                "--test-size",
                "60",
                "--step-size",
                "240",
                "--auto-adjust-splits",
                "--save-predictions",
                "--paper-strict",
                "--output",
                _join(execution_root, f"multi_asset_chronos2_h{h}_summary.csv"),
            ]
        )

    if not args.skip_ablations:
        _run(
            [
                py,
                "forecast/runners/run_feature_ablations.py",
                "--data-glob",
                ".data/hourly/*_timeseries_hourly.csv",
                "--include-assets",
                args.assets,
                "--horizon",
                h,
                "--n-splits",
                "4",
                "--embargo",
                "5",
                "--label-horizon",
                "1",
                "--output",
                _join(execution_root, f"feature_ablation_h{h}_summary.csv"),
            ]
        )

    if not args.skip_inference:
        _run(
            [
                py,
                "forecast/runners/run_statistical_inference.py",
                "--results-root",
                str(execution_root),
                "--horizon",
                h,
                "--output-prefix",
                f"PAPER_H{h}",
                "--bootstrap-samples",
                "1000",
                "--mcs-bootstrap-samples",
                "500",
            ]
        )

    _run(
        [
            py,
            "forecast/runners/run_paper_summary.py",
            "--results-root",
            str(execution_root),
            "--horizon",
            h,
            "--output-prefix",
            f"PAPER_H{h}",
            "--strict",
        ]
    )
    _run(
        [
            py,
            "forecast/runners/run_reproducibility_audit.py",
            "--results-root",
            str(execution_root),
            "--output-dir",
            _join(execution_root, "reproducibility"),
            "--strict",
        ]
    )

    if execution_root != results_root:
        _sync_tree(execution_root, results_root)
        print(f"Materialized artifacts into: {results_root}")
        if cleanup_staging:
            shutil.rmtree(execution_root, ignore_errors=True)


if __name__ == "__main__":
    main()
