from __future__ import annotations

import argparse
import shlex
import subprocess
import sys


def _run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic pre-Colab strict paper pipeline")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 5, 20])
    parser.add_argument("--assets", type=str, default="btc,eth,ada,doge,xmr,xrp")
    parser.add_argument("--results-root", type=str, default="results/paper")
    parser.add_argument("--skip-chronos", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()

    h = str(args.horizon)
    py = sys.executable
    results_root = args.results_root

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
            f"{results_root}/data_manifest.csv",
        ]
    )
    _run(
        [
            py,
            "forecast/runners/run_contract_audit.py",
            "--contract-path",
            "forecast/config/paper_experiment.json",
            "--manifest-path",
            f"{results_root}/data_manifest.csv",
            "--require-provenance",
            "--output",
            f"{results_root}/contract_asset_manifest.csv",
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
            f"{results_root}/multi_asset_baselines_h{h}_paired_summary.csv",
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
            f"{results_root}/multi_asset_transformers_h{h}_paired_summary.csv",
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
                f"{results_root}/multi_asset_chronos2_h{h}_summary.csv",
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
                f"{results_root}/feature_ablation_h{h}_summary.csv",
            ]
        )

    if not args.skip_inference:
        _run(
            [
                py,
                "forecast/runners/run_statistical_inference.py",
                "--results-root",
                results_root,
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
            results_root,
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
            results_root,
            "--output-dir",
            f"{results_root}/reproducibility",
            "--strict",
        ]
    )


if __name__ == "__main__":
    main()
