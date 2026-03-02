from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pandas as pd


def test_paper_summary_generator_real_data_outputs(tmp_path: Path) -> None:
    out_prefix = "TEST_H1"
    infer_cmd = [
        sys.executable,
        "forecast/runners/run_statistical_inference.py",
        "--results-root",
        "results/paper",
        "--horizon",
        "1",
        "--output-prefix",
        out_prefix,
        "--bootstrap-samples",
        "20",
        "--bootstrap-block-size",
        "10",
        "--mcs-bootstrap-samples",
        "20",
        "--mcs-block-size",
        "10",
    ]
    subprocess.run(infer_cmd, check=True)

    cmd = [
        sys.executable,
        "forecast/runners/run_paper_summary.py",
        "--results-root",
        "results/paper",
        "--horizon",
        "1",
        "--output-prefix",
        out_prefix,
    ]
    subprocess.run(cmd, check=True)

    root = Path("results/paper")
    required = [
        root / f"{out_prefix}_best_by_asset_mode.csv",
        root / f"{out_prefix}_best_by_asset.csv",
        root / f"{out_prefix}_sentiment_delta.csv",
        root / f"{out_prefix}_family_performance.csv",
        root / f"{out_prefix}_ablation_mode_wins.csv",
        root / f"{out_prefix}_metric_ci95.csv",
        root / f"{out_prefix}_directional_binomial.csv",
        root / f"{out_prefix}_mcs.csv",
        root / f"{out_prefix}_RESULTS_SUMMARY.md",
    ]
    for path in required:
        assert path.exists(), f"Missing output: {path}"

    best_df = pd.read_csv(root / f"{out_prefix}_best_by_asset.csv")
    family_df = pd.read_csv(root / f"{out_prefix}_family_performance.csv")
    ci_df = pd.read_csv(root / f"{out_prefix}_metric_ci95.csv")
    assert not best_df.empty
    assert not family_df.empty
    assert not ci_df.empty
    assert best_df["asset"].nunique() >= 2
    assert {"mae_ci95_low", "mae_ci95_high"}.issubset(best_df.columns)
