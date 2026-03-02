from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

from forecast.pipeline.data_pipeline import FeatureConfig, build_feature_frame, load_market_data


def test_build_feature_frame_adds_regime_columns_real_btc_data() -> None:
    raw = load_market_data("data/btc_timeseries.csv", time_col="time")
    features = build_feature_frame(
        raw,
        config=FeatureConfig(
            lookbacks=(20, 60),
            ewma_span=20,
            roc_period=5,
            use_regime_features=True,
            regime_lookbacks=(20, 60),
        ),
        time_col="time",
    )

    for lb in (20, 60):
        for col in [f"realized_vol_{lb}", f"vol_z_{lb}", f"vol_regime_{lb}"]:
            assert col in features.columns
            assert features[col].notna().sum() > 0


def test_baseline_runner_with_regime_features_real_data(tmp_path: Path) -> None:
    out_path = tmp_path / "baseline_regime_summary.csv"
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
        "--use-regime-features",
        "--regime-lookbacks",
        "20,60",
        "--no-arima",
        "--no-xgboost",
        "--output",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)

    assert out_path.exists()
    metadata_path = out_path.with_name(out_path.stem + "_metadata.json")
    quality_path = out_path.with_name(out_path.stem + "_data_quality_summary.csv")
    assert metadata_path.exists()
    assert quality_path.exists()

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    assert metadata["use_regime_features"] is True
    assert metadata["regime_lookbacks"] == [20, 60]

    quality_df = pd.read_csv(quality_path)
    assert int(quality_df.loc[0, "feature_count_total"]) >= 34


def test_cpcv_runner_with_regime_features_real_data(tmp_path: Path) -> None:
    out_path = tmp_path / "cpcv_regime_summary.csv"
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
        "--use-regime-features",
        "--regime-lookbacks",
        "20,60",
        "--no-arima",
        "--no-xgboost",
        "--output",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)

    assert out_path.exists()
    summary = pd.read_csv(out_path)
    assert not summary.empty
    assert {"linear_ridge", "random_walk"}.issubset(set(summary["model"].tolist()))
