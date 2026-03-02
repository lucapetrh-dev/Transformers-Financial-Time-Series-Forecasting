from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from forecast.pipeline.data_pipeline import (
    FeatureConfig,
    add_targets,
    build_feature_frame,
    drop_na_for_modeling,
    load_market_data,
    make_lag_features,
)
from forecast.pipeline.metrics import cost_sensitivity_metrics, deflated_sharpe_ratio, strategy_returns


def test_dsr_and_cost_sensitivity_on_real_btc_data() -> None:
    raw = load_market_data("data/btc_timeseries.csv", time_col="time")
    features = build_feature_frame(raw, config=FeatureConfig(lookbacks=(20, 60), ewma_span=20, roc_period=5))
    features = add_targets(features, horizons=(1,))
    features = make_lag_features(features, source_col="return_1d", max_lag=20)
    features = drop_na_for_modeling(features)

    y_true = features["target_ret_1d"].to_numpy(dtype=float)
    y_pred = features["return_1d_lag_1"].to_numpy(dtype=float)

    metrics = cost_sensitivity_metrics(y_true, y_pred, cost_bps_list=(0.0, 5.0, 10.0, 20.0), n_trials=4)
    assert "dsr" in metrics
    assert 0.0 <= metrics["dsr"] <= 1.0
    assert "sharpe_0bps" in metrics and "sharpe_20bps" in metrics

    rets, _ = strategy_returns(y_true, y_pred, cost_bps=5.0)
    dsr = deflated_sharpe_ratio(rets, n_trials=4)
    assert np.isfinite(dsr)


def test_baseline_runner_outputs_economic_columns_real_data(tmp_path: Path) -> None:
    out_path = tmp_path / "baseline_econ_summary.csv"
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
        "--output",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)

    summary = pd.read_csv(out_path)
    for col in ["dsr_mean", "sharpe_0bps_mean", "sharpe_20bps_mean", "max_drawdown_20bps_mean"]:
        assert col in summary.columns


def test_transformer_runner_outputs_economic_columns_real_data(tmp_path: Path) -> None:
    out_path = tmp_path / "transformer_econ_summary.csv"
    cmd = [
        "python",
        "forecast/runners/run_transformers.py",
        "--data-path",
        "data/btc_timeseries.csv",
        "--horizon",
        "1",
        "--lookback",
        "30",
        "--min-train-size",
        "300",
        "--test-size",
        "60",
        "--step-size",
        "300",
        "--epochs",
        "2",
        "--batch-size",
        "64",
        "--models",
        "patchtst_like,itransformer_like",
        "--output",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)

    summary = pd.read_csv(out_path)
    for col in ["dsr_mean", "sharpe_0bps_mean", "sharpe_20bps_mean", "max_drawdown_20bps_mean"]:
        assert col in summary.columns
