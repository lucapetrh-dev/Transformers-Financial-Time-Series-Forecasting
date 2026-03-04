from __future__ import annotations

import argparse
import glob
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forecast.analysis.sentiment_diagnostics import (
    compute_dimensionality_profile,
    compute_feature_group_summary,
    compute_sentiment_target_correlations,
    run_pca_sentiment_ablation,
)
from forecast.pipeline.assets import asset_name_from_path, filter_asset_paths, parse_asset_list
from forecast.pipeline.data_pipeline import FeatureConfig, add_targets, build_feature_frame, load_market_data, make_lag_features
from forecast.pipeline.experiment_contract import save_metadata_json
from forecast.pipeline.sentiment import detect_sentiment_columns, prepare_causal_sentiment_features


def _parse_int_list(value: str, *, name: str) -> list[int]:
    out: list[int] = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid integer in --{name}: {token}") from exc
    if not out:
        raise ValueError(f"--{name} must contain at least one integer")
    return out


def _resolve_paths(data_glob: str) -> list[str]:
    paths = sorted(glob.glob(data_glob))
    if not paths and data_glob.startswith("data/"):
        paths = sorted(glob.glob("." + data_glob))
    return paths


def _split_train_index(n_rows: int) -> int:
    if n_rows < 96:
        raise ValueError(f"Too few rows ({n_rows}) for train/test diagnostics split")
    if n_rows >= 360:
        return 300
    train_end = max(64, int(n_rows * 0.7))
    train_end = min(train_end, n_rows - 24)
    if train_end <= 0 or train_end >= n_rows:
        raise ValueError(f"Failed to build train/test split for n_rows={n_rows}")
    return int(train_end)


def _build_base_features(raw: pd.DataFrame, *, time_col: str, horizon: int) -> pd.DataFrame:
    feat = build_feature_frame(raw, config=FeatureConfig(lookbacks=(20, 60), ewma_span=20, roc_period=5), time_col=time_col)
    feat = add_targets(feat, horizons=(horizon,))
    feat = make_lag_features(feat, source_col="return_1d", max_lag=20)
    return feat


def _build_regime_features(raw: pd.DataFrame, *, time_col: str, horizon: int) -> pd.DataFrame:
    feat = build_feature_frame(
        raw,
        config=FeatureConfig(
            lookbacks=(20, 60),
            ewma_span=20,
            roc_period=5,
            use_regime_features=True,
            regime_lookbacks=(20, 60),
        ),
        time_col=time_col,
    )
    feat = add_targets(feat, horizons=(horizon,))
    feat = make_lag_features(feat, source_col="return_1d", max_lag=20)
    return feat


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sentiment diagnostics (correlation, dimensionality profile, PCA ablation)")
    parser.add_argument("--data-glob", type=str, default=".data/hourly/*_timeseries_hourly.csv")
    parser.add_argument("--include-assets", type=str, default="")
    parser.add_argument("--exclude-assets", type=str, default="")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 5, 20])
    parser.add_argument("--lags", type=str, default="0,1,2,3,5,7")
    parser.add_argument("--pca-components", type=str, default="3,5,10,15")
    parser.add_argument("--output-dir", type=str, default="results/paper/sentiment_diagnostics")
    args = parser.parse_args()

    lags = _parse_int_list(args.lags, name="lags")
    pca_components = _parse_int_list(args.pca_components, name="pca-components")

    data_paths = _resolve_paths(args.data_glob)
    if not data_paths:
        raise ValueError(f"No files matched --data-glob pattern: {args.data_glob}")
    data_paths = filter_asset_paths(
        data_paths,
        include_assets=parse_asset_list(args.include_assets),
        exclude_assets=parse_asset_list(args.exclude_assets),
    )
    if not data_paths:
        raise ValueError("No assets left after include/exclude filtering")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_col = f"target_ret_{args.horizon}d"
    lag_cols = [f"return_1d_lag_{k}" for k in range(1, 21)]
    extra_cols = ["dow_sin", "dow_cos", "month_sin", "month_cos", "ewma_ret", "ewma_roc", "ret_mean_20", "ret_std_20"]
    regime_cols = ["realized_vol_20", "vol_z_20", "vol_regime_20", "realized_vol_60", "vol_z_60", "vol_regime_60"]

    corr_frames: list[pd.DataFrame] = []
    dim_frames: list[pd.DataFrame] = []
    pca_frames: list[pd.DataFrame] = []
    group_frames: list[pd.DataFrame] = []

    processed_assets: list[str] = []

    for data_path in data_paths:
        asset = asset_name_from_path(data_path)
        raw = load_market_data(data_path, time_col="time")

        features = _build_base_features(raw, time_col="time", horizon=args.horizon)
        features_regime = _build_regime_features(raw, time_col="time", horizon=args.horizon)

        financial_cols = [c for c in lag_cols + extra_cols if c in features.columns]
        exclude_cols = set(financial_cols).union({"log_price", "return_1d"})
        exclude_cols.update({c for c in features.columns if c.startswith("target_")})
        raw_sentiment_cols = detect_sentiment_columns(features, exclude_cols=exclude_cols)
        features_model, sentiment_cols, _ = prepare_causal_sentiment_features(
            features,
            raw_sentiment_cols,
            lag=1,
            min_non_null_ratio=0.2,
            fill_method="ffill_zero",
        )

        corr_df = compute_sentiment_target_correlations(features, raw_sentiment_cols, target_col, lags=lags)
        if not corr_df.empty:
            corr_df.insert(0, "asset", asset)
            corr_frames.append(corr_df)

        model_cols = [*financial_cols, *sentiment_cols, target_col]
        model_frame = features_model[model_cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        train_end = _split_train_index(len(model_frame))

        X = model_frame[financial_cols + sentiment_cols].to_numpy(dtype=float)
        y = model_frame[target_col].to_numpy(dtype=float)
        X_train, X_test = X[:train_end], X[train_end:]
        y_train, y_test = y[:train_end], y[train_end:]

        sentiment_idx = list(range(len(financial_cols), len(financial_cols) + len(sentiment_cols)))
        pca_df = run_pca_sentiment_ablation(
            X_train,
            X_test,
            y_train,
            y_test,
            sentiment_col_indices=sentiment_idx,
            n_components_list=pca_components,
        )
        pca_df.insert(0, "asset", asset)
        pca_frames.append(pca_df)

        regime_count = sum(1 for c in regime_cols if c in features_regime.columns)
        feature_counts = {
            "financial_only": len(financial_cols),
            "financial_plus_sentiment": len(financial_cols) + len(sentiment_cols),
            "financial_plus_sentiment_plus_regime": len(financial_cols) + len(sentiment_cols) + regime_count,
        }
        dim_df = compute_dimensionality_profile(feature_counts, n_train_samples=train_end)
        dim_df.insert(0, "asset", asset)
        dim_frames.append(dim_df)

        group_df = compute_feature_group_summary(
            features_model,
            sentiment_cols=sentiment_cols,
            financial_cols=financial_cols,
            target_col=target_col,
        )
        group_df.insert(0, "asset", asset)
        group_frames.append(group_df)

        processed_assets.append(asset)

    corr_out = pd.concat(corr_frames, ignore_index=True) if corr_frames else pd.DataFrame()
    dim_out = pd.concat(dim_frames, ignore_index=True) if dim_frames else pd.DataFrame()
    pca_out = pd.concat(pca_frames, ignore_index=True) if pca_frames else pd.DataFrame()
    group_out = pd.concat(group_frames, ignore_index=True) if group_frames else pd.DataFrame()

    corr_path = output_dir / "sentiment_correlation_by_lag.csv"
    dim_path = output_dir / "dimensionality_profile.csv"
    pca_path = output_dir / "pca_sentiment_ablation.csv"
    group_path = output_dir / "feature_group_summary.csv"

    corr_out.to_csv(corr_path, index=False)
    dim_out.to_csv(dim_path, index=False)
    pca_out.to_csv(pca_path, index=False)
    group_out.to_csv(group_path, index=False)

    metadata = {
        "script": "run_sentiment_diagnostics.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_glob": args.data_glob,
        "include_assets": sorted(parse_asset_list(args.include_assets)),
        "exclude_assets": sorted(parse_asset_list(args.exclude_assets)),
        "processed_assets": processed_assets,
        "horizon": int(args.horizon),
        "target_col": target_col,
        "lags": lags,
        "pca_components": pca_components,
        "outputs": {
            "sentiment_correlation_by_lag": str(corr_path),
            "dimensionality_profile": str(dim_path),
            "pca_sentiment_ablation": str(pca_path),
            "feature_group_summary": str(group_path),
        },
    }
    save_metadata_json(output_dir / "metadata.json", metadata)

    print(f"Saved: {corr_path}")
    print(f"Saved: {dim_path}")
    print(f"Saved: {pca_path}")
    print(f"Saved: {group_path}")
    print(f"Saved: {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
