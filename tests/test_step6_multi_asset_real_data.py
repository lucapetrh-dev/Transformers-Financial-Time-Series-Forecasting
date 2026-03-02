from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd


def test_multi_asset_baseline_runner_real_data(tmp_path: Path) -> None:
    out_path = tmp_path / "multi_asset_summary.csv"
    cmd = [
        "python",
        "forecast/runners/run_multi_asset_baselines.py",
        "--data-glob",
        "data/*_timeseries.csv",
        "--horizon",
        "1",
        "--min-train-size",
        "350",
        "--test-size",
        "40",
        "--step-size",
        "400",
        "--auto-adjust-splits",
        "--run-paired-sentiment",
        "--no-arima",
        "--no-xgboost",
        "--output",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)

    assert out_path.exists()
    rank_path = out_path.with_name(out_path.stem + "_ranks.csv")
    audit_path = out_path.with_name(out_path.stem + "_comparability_audit.csv")
    assert rank_path.exists()
    assert audit_path.exists()

    summary = pd.read_csv(out_path)
    assert not summary.empty
    assert {"asset", "mode", "model", "mae_mean"}.issubset(summary.columns)
    assert summary["asset"].nunique() >= 2
    assert "no_sentiment" in set(summary["mode"].tolist())
    assert "with_sentiment" in set(summary["mode"].tolist())

    audit = pd.read_csv(audit_path)
    assert not audit.empty
    assert audit["consistent_target_space_all"].all()
    assert set(audit["expected_target_space"]) == {"log_return"}
