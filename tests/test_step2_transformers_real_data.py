from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from forecast.pipeline.data_pipeline import FeatureConfig, add_targets, build_feature_frame, load_market_data
from forecast.pipeline.transformers import (
    ITransformerLikeRegressor,
    PatchTSTLikeRegressor,
    SequenceStandardizer,
    TrainConfig,
    build_sliding_windows,
    predict_model,
    train_model,
)


def _real_btc_sequences(lookback: int = 30):
    raw = load_market_data("data/btc_timeseries.csv", time_col="time")
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
    return build_sliding_windows(frame, feature_cols, "target_ret_1d", lookback)


def test_patch_and_itransformer_train_on_real_btc_data() -> None:
    X, y = _real_btc_sequences(lookback=30)
    assert len(X) > 500

    X_train, y_train = X[:400], y[:400]
    X_val, y_val = X[400:500], y[400:500]
    X_test = X[500:560]

    scaler = SequenceStandardizer().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    cfg = TrainConfig(epochs=2, batch_size=64, lr=1e-3, patience=2, seed=42)

    patch = PatchTSTLikeRegressor(lookback=30, n_features=X_train.shape[-1])
    patch = train_model(patch, X_train, y_train, X_val, y_val, cfg)
    patch_pred = predict_model(patch, X_test)
    assert patch_pred.shape[0] == X_test.shape[0]
    assert np.isfinite(patch_pred).all()

    itr = ITransformerLikeRegressor(lookback=30, n_features=X_train.shape[-1])
    itr = train_model(itr, X_train, y_train, X_val, y_val, cfg)
    itr_pred = predict_model(itr, X_test)
    assert itr_pred.shape[0] == X_test.shape[0]
    assert np.isfinite(itr_pred).all()


def test_transformer_runner_real_data(tmp_path: Path) -> None:
    out_path = tmp_path / "transformers_summary.csv"
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
        "180",
        "--epochs",
        "3",
        "--batch-size",
        "64",
        "--models",
        "patchtst_like,itransformer_like",
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
    assert "patchtst_like" in models
    assert "itransformer_like" in models
    assert "random_walk_sequence" in models
    for col in ["coverage_80_mean", "interval_width_80_mean", "wis_80_mean"]:
        assert col in summary.columns
