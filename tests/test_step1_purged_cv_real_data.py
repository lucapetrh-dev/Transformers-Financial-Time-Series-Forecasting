from __future__ import annotations

import subprocess
from pathlib import Path
import json

import pandas as pd

from forecast.pipeline.data_pipeline import (
    FeatureConfig,
    add_targets,
    build_feature_frame,
    drop_na_for_modeling,
    load_market_data,
    make_lag_features,
)
from forecast.pipeline.tuning import PurgedCVConfig, tune_ridge_alpha_purged_cv


def test_purged_cv_tunes_alpha_on_real_btc_data() -> None:
    raw = load_market_data("data/btc_timeseries.csv", time_col="time")
    features = build_feature_frame(raw, config=FeatureConfig(lookbacks=(20, 60), ewma_span=20, roc_period=5))
    features = add_targets(features, horizons=(1,))
    features = make_lag_features(features, source_col="return_1d", max_lag=20)
    features = drop_na_for_modeling(features)

    feature_cols = [f"return_1d_lag_{k}" for k in range(1, 21)] + ["ewma_ret", "ewma_roc", "ret_mean_20", "ret_std_20"]
    X = features[feature_cols].to_numpy(dtype=float)
    y = features["target_ret_1d"].to_numpy(dtype=float)

    best_alpha, cv_df = tune_ridge_alpha_purged_cv(
        X,
        y,
        alpha_grid=[0.01, 0.1, 1.0],
        cv_config=PurgedCVConfig(n_splits=4, embargo=5, label_horizon=1),
        objective="mae",
    )

    assert best_alpha in {0.01, 0.1, 1.0}
    assert not cv_df.empty
    summary_rows = cv_df[cv_df["fold"] == -1]
    assert len(summary_rows) == 3


def test_baseline_runner_with_tuning_real_data(tmp_path: Path) -> None:
    out_path = tmp_path / "baselines_tuned.csv"
    cmd = [
        "python",
        "forecast/runners/run_baselines.py",
        "--data-path",
        "data/btc_timeseries.csv",
        "--horizon",
        "1",
        "--min-train-size",
        "500",
        "--test-size",
        "60",
        "--step-size",
        "120",
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
    cv_path = out_path.with_name(out_path.stem + "_cv.csv")
    quality_path = out_path.with_name(out_path.stem + "_data_quality_summary.csv")
    missingness_path = out_path.with_name(out_path.stem + "_missingness_stages.csv")
    assert folds_path.exists()
    assert fold_manifest_path.exists()
    assert metadata_path.exists()
    assert cv_path.exists()
    assert quality_path.exists()
    assert missingness_path.exists()

    summary = pd.read_csv(out_path)
    assert "linear_alpha_mean" in summary.columns
    for col in ["coverage_80_mean", "interval_width_80_mean", "wis_80_mean"]:
        assert col in summary.columns
    linear_row = summary.loc[summary["model"] == "linear_ridge"]
    assert not linear_row.empty
    assert linear_row["linear_alpha_mean"].notna().all()

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    assert metadata["target_space"] == "log_return"
    assert metadata["split_method"] == "walk_forward_expanding"
