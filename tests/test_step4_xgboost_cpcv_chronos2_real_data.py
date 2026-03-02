from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

from forecast.pipeline.baselines import has_xgboost


def test_cpcv_report_with_xgboost_real_data(tmp_path: Path) -> None:
    out_path = tmp_path / "cpcv_summary.csv"
    cmd = [
        "python",
        "forecast/runners/run_cpcv_report.py",
        "--data-path",
        "data/btc_timeseries.csv",
        "--horizon",
        "1",
        "--n-splits",
        "5",
        "--embargo",
        "5",
        "--label-horizon",
        "1",
        "--tune-linear-alpha",
        "--alpha-grid",
        "0.01,0.1,1.0",
        "--output",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)

    assert out_path.exists()
    folds_path = out_path.with_name(out_path.stem + "_folds.csv")
    fold_manifest_path = out_path.with_name(out_path.stem + "_fold_manifest.csv")
    metadata_path = out_path.with_name(out_path.stem + "_metadata.json")
    tuning_path = out_path.with_name(out_path.stem + "_tuning.csv")
    quality_path = out_path.with_name(out_path.stem + "_data_quality_summary.csv")
    missingness_path = out_path.with_name(out_path.stem + "_missingness_stages.csv")
    assert folds_path.exists()
    assert fold_manifest_path.exists()
    assert metadata_path.exists()
    assert tuning_path.exists()
    assert quality_path.exists()
    assert missingness_path.exists()

    summary = pd.read_csv(out_path)
    models = set(summary["model"].tolist())
    assert "random_walk" in models
    assert "linear_ridge" in models
    for col in ["coverage_80_mean", "interval_width_80_mean", "wis_80_mean"]:
        assert col in summary.columns
    if has_xgboost():
        assert "xgboost" in models


def test_chronos2_zero_shot_runner_real_data(tmp_path: Path) -> None:
    out_path = tmp_path / "chronos2_summary.csv"
    cmd = [
        "python",
        "forecast/runners/run_chronos2_zero_shot.py",
        "--data-path",
        "data/btc_timeseries.csv",
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
        "--output",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)

    assert out_path.exists()
    folds_path = out_path.with_name(out_path.stem + "_folds.csv")
    fold_manifest_path = out_path.with_name(out_path.stem + "_fold_manifest.csv")
    metadata_path = out_path.with_name(out_path.stem + "_metadata.json")
    quality_path = out_path.with_name(out_path.stem + "_data_quality_summary.csv")
    missingness_path = out_path.with_name(out_path.stem + "_missingness_stages.csv")
    assert folds_path.exists()
    assert fold_manifest_path.exists()
    assert metadata_path.exists()
    assert quality_path.exists()
    assert missingness_path.exists()

    summary = pd.read_csv(out_path)
    models = set(summary["model"].tolist())
    assert "chronos2_zero_shot" in models
    assert "random_walk_scaled" in models
    for col in ["coverage_80_mean", "interval_width_80_mean", "wis_80_mean"]:
        assert col in summary.columns
    row = summary.loc[summary["model"] == "chronos2_zero_shot"]
    assert not row.empty
    assert row["mae_mean"].notna().all()
