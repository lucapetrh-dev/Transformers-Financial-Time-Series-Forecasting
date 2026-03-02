from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from forecast.pipeline.data_pipeline import FeatureConfig, add_targets, build_feature_frame, load_market_data
from forecast.pipeline.transformers import (
    DLinearLikeRegressor,
    SequenceStandardizer,
    TCNLikeRegressor,
    TrainConfig,
    build_sliding_windows,
    predict_model,
    train_model,
)


def _real_btc_sequences(lookback: int = 48):
    raw = load_market_data(".data/hourly/btc_lunarcrush_timeseries_hourly.csv", time_col="time")
    features = build_feature_frame(raw, config=FeatureConfig(lookbacks=(20, 60), ewma_span=20, roc_period=5))
    features = add_targets(features, horizons=(1,))
    feature_cols = [
        "return_1d",
        "ewma_ret",
        "ewma_roc",
        "ret_mean_20",
        "ret_std_20",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
    ]
    frame = features[["time", *feature_cols, "target_ret_1d"]].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return build_sliding_windows(frame, feature_cols=feature_cols, target_col="target_ret_1d", lookback=lookback)


def test_dlinear_and_tcn_train_on_real_btc_data() -> None:
    X, y = _real_btc_sequences(lookback=48)
    assert len(X) > 300

    X_train, y_train = X[:300], y[:300]
    X_val, y_val = X[300:380], y[300:380]
    X_test = X[380:430]

    scaler = SequenceStandardizer().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    cfg = TrainConfig(epochs=2, batch_size=64, lr=1e-3, patience=2, seed=42)

    dlinear = DLinearLikeRegressor(lookback=48, n_features=X_train.shape[-1], moving_avg=5)
    dlinear = train_model(dlinear, X_train, y_train, X_val, y_val, cfg)
    dlinear_pred = predict_model(dlinear, X_test)
    assert dlinear_pred.shape[0] == X_test.shape[0]
    assert np.isfinite(dlinear_pred).all()

    tcn = TCNLikeRegressor(lookback=48, n_features=X_train.shape[-1], channels=(32, 32), kernel_size=3, dropout=0.1)
    tcn = train_model(tcn, X_train, y_train, X_val, y_val, cfg)
    tcn_pred = predict_model(tcn, X_test)
    assert tcn_pred.shape[0] == X_test.shape[0]
    assert np.isfinite(tcn_pred).all()


def test_transformer_runner_with_replacement_models_real_data(tmp_path: Path) -> None:
    out_path = tmp_path / "transformers_replacement_summary.csv"
    cmd = [
        "python",
        "forecast/runners/run_transformers.py",
        "--data-path",
        ".data/hourly/btc_lunarcrush_timeseries_hourly.csv",
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
        "--output",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)

    summary = pd.read_csv(out_path)
    models = set(summary["model"].tolist())
    assert "dlinear_like" in models
    assert "tcn_like" in models
    assert "random_walk_sequence" in models
