from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd


def test_multi_asset_chronos2_runner_real_data(tmp_path: Path) -> None:
    out_path = tmp_path / "multi_asset_chronos2_summary.csv"
    cmd = [
        "python",
        "forecast/runners/run_multi_asset_chronos2.py",
        "--data-glob",
        "data/*_timeseries.csv",
        "--limit-assets",
        "2",
        "--horizon",
        "1",
        "--context-length",
        "120",
        "--min-train-size",
        "500",
        "--test-size",
        "20",
        "--step-size",
        "400",
        "--batch-size",
        "64",
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
    assert {"asset", "model", "mae_mean"}.issubset(summary.columns)
    assert "chronos2_zero_shot" in set(summary["model"].tolist())
    assert summary["asset"].nunique() >= 2

    audit = pd.read_csv(audit_path)
    assert not audit.empty
    assert audit["consistent_target_space_all"].all()
    assert set(audit["expected_target_space"]) == {"log_return"}

