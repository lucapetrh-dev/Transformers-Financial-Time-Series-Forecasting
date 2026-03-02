from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd


def test_cpcv_feature_stability_report_real_data(tmp_path: Path) -> None:
    out_path = tmp_path / "cpcv_stability_summary.csv"
    cmd = [
        "python",
        "forecast/runners/run_cpcv_report.py",
        "--data-path",
        "data/btc_timeseries.csv",
        "--horizon",
        "1",
        "--n-splits",
        "4",
        "--embargo",
        "5",
        "--label-horizon",
        "1",
        "--no-arima",
        "--no-xgboost",
        "--feature-stability-report",
        "--stability-repeats",
        "2",
        "--output",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)

    assert out_path.exists()
    folds_path = out_path.with_name(out_path.stem + "_feature_importance_folds.csv")
    summary_path = out_path.with_name(out_path.stem + "_feature_importance_summary.csv")
    metadata_path = out_path.with_name(out_path.stem + "_metadata.json")
    assert folds_path.exists()
    assert summary_path.exists()
    assert metadata_path.exists()

    folds = pd.read_csv(folds_path)
    summary = pd.read_csv(summary_path)
    assert not folds.empty
    assert not summary.empty
    for col in ["feature", "importance_mean", "importance_std", "fold"]:
        assert col in folds.columns
    for col in ["feature", "importance_mean", "importance_std", "rank_by_mean_importance"]:
        assert col in summary.columns
    assert summary["rank_by_mean_importance"].min() == 1

    with metadata_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["feature_stability_report_enabled"] is True
    assert meta["stability_repeats"] == 2
