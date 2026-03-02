from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd


def test_ensemble_benchmark_runner_real_data(tmp_path: Path) -> None:
    out_path = tmp_path / "ensemble_h1_summary.csv"
    cmd = [
        "python",
        "forecast/runners/run_ensemble_benchmark.py",
        "--results-root",
        "results/paper",
        "--horizon",
        "1",
        "--output",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)

    assert out_path.exists()
    folds_path = out_path.with_name(out_path.stem + "_folds.csv")
    pred_path = out_path.with_name(out_path.stem + "_predictions.csv")
    audit_path = out_path.with_name(out_path.stem + "_comparability_audit.csv")
    assert folds_path.exists()
    assert pred_path.exists()
    assert audit_path.exists()

    summary = pd.read_csv(out_path)
    assert not summary.empty
    assert "ensemble_ridge_foundation" in set(summary["model"].astype(str).tolist())
    assert "sharpe_50bps_mean" in summary.columns
