from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd


def test_multi_asset_transformer_runner_real_data(tmp_path: Path) -> None:
    out_path = tmp_path / "multi_asset_transformers_summary.csv"
    cmd = [
        "python",
        "forecast/runners/run_multi_asset_transformers.py",
        "--data-glob",
        ".data/hourly/*_timeseries_hourly.csv",
        "--limit-assets",
        "2",
        "--horizon",
        "1",
        "--lookback",
        "48",
        "--min-train-size",
        "300",
        "--test-size",
        "60",
        "--step-size",
        "180",
        "--epochs",
        "2",
        "--batch-size",
        "64",
        "--models",
        "dlinear_like,tcn_like",
        "--probabilistic-mode",
        "quantile",
        "--auto-adjust-splits",
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
    assert {"asset", "mode", "model", "mae_mean", "coverage_80_mean", "wis_80_mean"}.issubset(summary.columns)
    assert summary["asset"].nunique() == 2
    assert {"dlinear_like", "tcn_like", "random_walk_sequence"}.issubset(set(summary["model"].tolist()))

    audit = pd.read_csv(audit_path)
    assert not audit.empty
    assert audit["consistent_target_space_all"].all()
    assert set(audit["expected_target_space"]) == {"log_return"}
