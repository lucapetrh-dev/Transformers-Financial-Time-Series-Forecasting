from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd


def test_multi_asset_foundation_runner_real_data(tmp_path: Path) -> None:
    out_path = tmp_path / "multi_asset_foundation_summary.csv"
    cmd = [
        "python",
        "forecast/runners/run_multi_asset_foundation.py",
        "--data-glob",
        ".data/hourly/*_timeseries_hourly.csv",
        "--include-assets",
        "btc",
        "--horizon",
        "1",
        "--context-length",
        "120",
        "--min-train-size",
        "300",
        "--test-size",
        "60",
        "--step-size",
        "240",
        "--models",
        "moirai",
        "--auto-adjust-splits",
        "--save-predictions",
        "--output",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)

    assert out_path.exists()
    rank_path = out_path.with_name(out_path.stem + "_ranks.csv")
    audit_path = out_path.with_name(out_path.stem + "_comparability_audit.csv")
    pred_path = out_path.with_name(out_path.stem + "_predictions.csv")
    assert rank_path.exists()
    assert audit_path.exists()
    assert pred_path.exists()

    summary = pd.read_csv(out_path)
    assert not summary.empty
    assert {"asset", "model", "backend_status", "mae_mean"}.issubset(summary.columns)
    assert "moirai_zero_shot" in set(summary["model"].astype(str).tolist())
    assert "native" in set(summary["backend_status"].astype(str).tolist())

    audit = pd.read_csv(audit_path)
    assert not audit.empty
    assert audit["consistent_target_space_all"].all()
    assert set(audit["expected_target_space"]) == {"log_return"}
