from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_sentiment_diagnostics_runner_real_data(tmp_path: Path) -> None:
    out_dir = tmp_path / "sentiment_diagnostics"
    cmd = [
        sys.executable,
        "forecast/runners/run_sentiment_diagnostics.py",
        "--data-glob",
        ".data/hourly/*_timeseries_hourly.csv",
        "--include-assets",
        "btc",
        "--horizon",
        "1",
        "--output-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)

    corr_path = out_dir / "sentiment_correlation_by_lag.csv"
    dim_path = out_dir / "dimensionality_profile.csv"
    pca_path = out_dir / "pca_sentiment_ablation.csv"
    group_path = out_dir / "feature_group_summary.csv"
    metadata_path = out_dir / "metadata.json"

    for path in [corr_path, dim_path, pca_path, group_path, metadata_path]:
        assert path.exists(), f"Missing output: {path}"

    corr_df = pd.read_csv(corr_path)
    dim_df = pd.read_csv(dim_path)
    pca_df = pd.read_csv(pca_path)
    group_df = pd.read_csv(group_path)

    assert not corr_df.empty
    assert not dim_df.empty
    assert not pca_df.empty
    assert not group_df.empty

    assert {"asset", "feature", "lag", "pearson_r", "pearson_p", "spearman_r", "spearman_p", "n_obs"}.issubset(corr_df.columns)
    assert {"asset", "mode", "n_features", "n_train_samples", "p_over_n"}.issubset(dim_df.columns)
    assert {"asset", "mode", "n_pca_components", "n_total_features", "mae", "rmse", "dir_acc"}.issubset(pca_df.columns)
    assert {"asset", "group", "n_features", "mean_abs_corr", "max_abs_corr", "std_corr"}.issubset(group_df.columns)

    assert set(corr_df["asset"].astype(str).str.lower()) == {"btc"}
