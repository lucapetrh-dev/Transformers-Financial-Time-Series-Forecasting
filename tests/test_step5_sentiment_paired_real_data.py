from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

from forecast.pipeline.data_pipeline import FeatureConfig, add_targets, build_feature_frame, load_market_data
from forecast.pipeline.sentiment import detect_sentiment_columns


def _paired_paths(base_path: Path) -> tuple[Path, Path, Path]:
    no_sent = base_path.with_name(base_path.stem + "_no_sent" + base_path.suffix)
    with_sent = base_path.with_name(base_path.stem + "_with_sent" + base_path.suffix)
    cmp_path = base_path.with_name(base_path.stem + "_sentiment_comparison.csv")
    return no_sent, with_sent, cmp_path


def _assert_paired_outputs(base_path: Path) -> None:
    no_sent, with_sent, cmp_path = _paired_paths(base_path)
    assert no_sent.exists(), f"Missing {no_sent}"
    assert with_sent.exists(), f"Missing {with_sent}"
    assert cmp_path.exists(), f"Missing {cmp_path}"

    for summary_path in (no_sent, with_sent):
        folds_path = summary_path.with_name(summary_path.stem + "_folds.csv")
        manifest_path = summary_path.with_name(summary_path.stem + "_fold_manifest.csv")
        metadata_path = summary_path.with_name(summary_path.stem + "_metadata.json")
        quality_path = summary_path.with_name(summary_path.stem + "_data_quality_summary.csv")
        missingness_path = summary_path.with_name(summary_path.stem + "_missingness_stages.csv")
        assert folds_path.exists(), f"Missing {folds_path}"
        assert manifest_path.exists(), f"Missing {manifest_path}"
        assert metadata_path.exists(), f"Missing {metadata_path}"
        assert quality_path.exists(), f"Missing {quality_path}"
        assert missingness_path.exists(), f"Missing {missingness_path}"
        quality_df = pd.read_csv(quality_path)
        assert not quality_df.empty
        assert "imputation_policy" in quality_df.columns
        assert quality_df.loc[0, "imputation_policy"] == "dropna_only_no_value_imputation"
        if summary_path.stem.endswith("_with_sent"):
            assert quality_df.loc[0, "sentiment_lag_effective"] >= 1

    cmp_df = pd.read_csv(cmp_path)
    assert not cmp_df.empty
    assert "model" in cmp_df.columns
    delta_cols = [c for c in cmp_df.columns if c.endswith("_delta_with_minus_no")]
    assert len(delta_cols) > 0


def test_detect_sentiment_columns_real_btc_data() -> None:
    raw = load_market_data("data/btc_timeseries.csv", time_col="time")
    features = build_feature_frame(raw, config=FeatureConfig(lookbacks=(20, 60), ewma_span=20, roc_period=5))
    features = add_targets(features, horizons=(1, 5, 20))

    sentiment_cols = detect_sentiment_columns(features, exclude_cols={"log_price", "return_1d", "target_ret_1d"})
    assert len(sentiment_cols) > 0


def test_baseline_runner_paired_sentiment_real_data(tmp_path: Path) -> None:
    out_path = tmp_path / "baseline_paired_summary.csv"
    cmd = [
        "python",
        "forecast/runners/run_baselines.py",
        "--data-path",
        "data/btc_timeseries.csv",
        "--horizon",
        "1",
        "--min-train-size",
        "350",
        "--test-size",
        "40",
        "--step-size",
        "400",
        "--run-paired-sentiment",
        "--no-arima",
        "--no-xgboost",
        "--output",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    _assert_paired_outputs(out_path)


def test_transformer_runner_paired_sentiment_real_data(tmp_path: Path) -> None:
    out_path = tmp_path / "transformer_paired_summary.csv"
    cmd = [
        "python",
        "forecast/runners/run_transformers.py",
        "--data-path",
        "data/btc_timeseries.csv",
        "--horizon",
        "1",
        "--lookback",
        "20",
        "--min-train-size",
        "240",
        "--test-size",
        "40",
        "--step-size",
        "400",
        "--epochs",
        "1",
        "--batch-size",
        "64",
        "--models",
        "patchtst_like",
        "--run-paired-sentiment",
        "--output",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    _assert_paired_outputs(out_path)


def test_cpcv_runner_paired_sentiment_real_data(tmp_path: Path) -> None:
    out_path = tmp_path / "cpcv_paired_summary.csv"
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
        "--run-paired-sentiment",
        "--no-arima",
        "--no-xgboost",
        "--output",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    _assert_paired_outputs(out_path)
