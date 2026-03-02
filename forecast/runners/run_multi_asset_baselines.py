from __future__ import annotations

import argparse
import glob
import json
import subprocess
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forecast.pipeline.assets import asset_name_from_path, filter_asset_paths, parse_asset_list
from forecast.pipeline.data_pipeline import (
    FeatureConfig,
    add_targets,
    build_feature_frame,
    load_market_data,
    make_lag_features,
)
from forecast.pipeline.sentiment import detect_sentiment_columns, prepare_causal_sentiment_features


def _build_runner_cmd(
    data_path: str,
    output_path: Path,
    args: argparse.Namespace,
    min_train_size: int,
    test_size: int,
    step_size: int,
) -> list[str]:
    cmd = [
        "python",
        "forecast/runners/run_baselines.py",
        "--data-path",
        data_path,
        "--time-col",
        args.time_col,
        "--horizon",
        str(args.horizon),
        "--max-lag",
        str(args.max_lag),
        "--min-train-size",
        str(min_train_size),
        "--test-size",
        str(test_size),
        "--step-size",
        str(step_size),
        "--sentiment-lag",
        str(args.sentiment_lag),
        "--sentiment-min-non-null-ratio",
        str(args.sentiment_min_non_null_ratio),
        "--regime-lookbacks",
        args.regime_lookbacks,
        "--output",
        str(output_path),
    ]
    if args.use_regime_features:
        cmd.append("--use-regime-features")
    if args.tune_linear_alpha:
        cmd.extend(["--tune-linear-alpha", "--alpha-grid", args.alpha_grid, "--cv-splits", str(args.cv_splits), "--cv-embargo", str(args.cv_embargo), "--cv-label-horizon", str(args.cv_label_horizon)])
    if args.run_paired_sentiment:
        cmd.append("--run-paired-sentiment")
    elif args.use_sentiment:
        cmd.append("--use-sentiment")
    if args.no_arima:
        cmd.append("--no-arima")
    if args.no_xgboost:
        cmd.append("--no-xgboost")
    if args.save_predictions:
        cmd.append("--save-predictions")
    return cmd


def _load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _estimate_usable_rows(data_path: str, args: argparse.Namespace, use_sentiment: bool) -> int:
    raw = load_market_data(data_path, time_col=args.time_col)
    features = build_feature_frame(
        raw,
        config=FeatureConfig(lookbacks=(20, 60), ewma_span=20, roc_period=5),
        time_col=args.time_col,
    )
    features = add_targets(features, horizons=(1, 5, 20))
    features = make_lag_features(features, source_col="return_1d", max_lag=args.max_lag)

    target_col = f"target_ret_{args.horizon}d"
    lag_cols = [f"return_1d_lag_{k}" for k in range(1, args.max_lag + 1)]
    extra_cols = ["dow_sin", "dow_cos", "month_sin", "month_cos", "ewma_ret", "ewma_roc", "ret_mean_20", "ret_std_20"]
    feature_cols = lag_cols + [c for c in extra_cols if c in features.columns]

    if use_sentiment:
        raw_sentiment_cols = detect_sentiment_columns(
            features,
            exclude_cols=set(feature_cols).union({target_col, "log_price", "return_1d"}),
        )
        features, sentiment_cols, _ = prepare_causal_sentiment_features(
            features,
            raw_sentiment_cols,
            lag=args.sentiment_lag,
            min_non_null_ratio=args.sentiment_min_non_null_ratio,
            fill_method="ffill_zero",
        )
        feature_cols = feature_cols + sentiment_cols

    required_cols = [args.time_col, target_col, *feature_cols]
    features = features[required_cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return int(len(features))


def _resolve_split_params(data_path: str, args: argparse.Namespace, run_with_sentiment: bool) -> tuple[int, int, int]:
    step = int(args.step_size)
    if step <= 0:
        raise ValueError("step_size must be positive")

    if not args.auto_adjust_splits:
        return args.min_train_size, args.test_size, step

    n = _estimate_usable_rows(data_path, args, use_sentiment=run_with_sentiment)
    if n < 60:
        raise ValueError(f"Too few rows ({n}) for walk-forward modeling")

    min_train = min(args.min_train_size, max(40, int(n * 0.7)))
    test = min(args.test_size, max(10, int(n * 0.15)))

    if min_train + test > n:
        test = max(10, min(test, n // 4))
        min_train = max(40, n - test)

    return int(min_train), int(test), int(step)


def _collect_for_mode(
    summary_path: Path,
    asset: str,
    data_path: str,
    mode: str,
) -> tuple[pd.DataFrame, dict]:
    summary = pd.read_csv(summary_path)
    summary = summary.copy()
    summary.insert(0, "asset", asset)
    summary.insert(1, "data_path", data_path)
    summary.insert(2, "mode", mode)

    metadata_path = summary_path.with_name(summary_path.stem + "_metadata.json")
    metadata = _load_metadata(metadata_path)
    metadata_row = {
        "asset": asset,
        "data_path": data_path,
        "mode": mode,
        "target_space": metadata.get("target_space"),
        "horizon_days": metadata.get("horizon_days"),
        "feature_mode": metadata.get("feature_mode"),
        "sentiment_lag": metadata.get("sentiment_lag"),
        "raw_rows": metadata.get("raw_rows"),
        "model_rows": metadata.get("model_rows"),
        "time_start": metadata.get("time_start"),
        "time_end": metadata.get("time_end"),
        "inferred_frequency": metadata.get("inferred_frequency"),
        "split_method": metadata.get("split_method"),
    }
    return summary, metadata_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline benchmark across multiple crypto assets")
    parser.add_argument("--data-glob", type=str, default="data/*_timeseries.csv")
    parser.add_argument("--time-col", type=str, default="time")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 5, 20])
    parser.add_argument("--max-lag", type=int, default=20)
    parser.add_argument("--min-train-size", type=int, default=365)
    parser.add_argument("--test-size", type=int, default=30)
    parser.add_argument("--step-size", type=int, default=30)
    parser.add_argument("--sentiment-lag", type=int, default=1)
    parser.add_argument("--sentiment-min-non-null-ratio", type=float, default=0.2)
    parser.add_argument("--use-regime-features", action="store_true")
    parser.add_argument("--regime-lookbacks", type=str, default="20,60")
    parser.add_argument("--use-sentiment", action="store_true")
    parser.add_argument("--run-paired-sentiment", action="store_true")
    parser.add_argument("--no-arima", action="store_true")
    parser.add_argument("--no-xgboost", action="store_true")
    parser.add_argument("--tune-linear-alpha", action="store_true")
    parser.add_argument("--alpha-grid", type=str, default="0.01,0.1,1.0,10.0")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-embargo", type=int, default=5)
    parser.add_argument("--cv-label-horizon", type=int, default=1)
    parser.add_argument(
        "--auto-adjust-splits",
        action="store_true",
        help="Auto-adjust min-train/test/step per asset based on available rows",
    )
    parser.add_argument(
        "--skip-failed-assets",
        action="store_true",
        help="Skip assets that fail to run and continue with remaining assets",
    )
    parser.add_argument("--limit-assets", type=int, default=0, help="Optional cap for quick runs; 0 means all discovered assets")
    parser.add_argument(
        "--include-assets",
        type=str,
        default="",
        help="Comma-separated asset list to include (e.g. btc,eth). Empty means include all discovered assets.",
    )
    parser.add_argument(
        "--exclude-assets",
        type=str,
        default="",
        help="Comma-separated asset list to exclude.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Persist and aggregate per-fold prediction/quantile traces across assets",
    )
    parser.add_argument("--output", type=str, default="results/multi_asset_baselines_summary.csv")
    args = parser.parse_args()

    data_paths = sorted(glob.glob(args.data_glob))
    if not data_paths and args.data_glob.startswith("data/"):
        alt_glob = "." + args.data_glob
        data_paths = sorted(glob.glob(alt_glob))
    if not data_paths:
        raise ValueError(f"No files matched --data-glob pattern: {args.data_glob}")
    data_paths = filter_asset_paths(
        data_paths,
        include_assets=parse_asset_list(args.include_assets),
        exclude_assets=parse_asset_list(args.exclude_assets),
    )
    if not data_paths:
        raise ValueError("No assets left after include/exclude filtering")
    if args.limit_assets > 0:
        data_paths = data_paths[: args.limit_assets]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[pd.DataFrame] = []
    metadata_rows: list[dict] = []
    failed_assets: list[dict[str, str]] = []
    prediction_frames: list[pd.DataFrame] = []
    quantile_frames: list[pd.DataFrame] = []

    for data_path in data_paths:
        asset = asset_name_from_path(data_path)
        asset_base = output_path.with_name(f"{output_path.stem}_{asset}{output_path.suffix}")
        try:
            min_train_size, test_size, step_size = _resolve_split_params(
                data_path,
                args,
                run_with_sentiment=(args.run_paired_sentiment or args.use_sentiment),
            )
        except Exception as exc:
            if args.skip_failed_assets:
                failed_assets.append({"asset": asset, "data_path": data_path, "error": str(exc)})
                print(f"Skipping failed asset={asset}: {exc}")
                continue
            raise

        cmd = _build_runner_cmd(
            data_path=data_path,
            output_path=asset_base,
            args=args,
            min_train_size=min_train_size,
            test_size=test_size,
            step_size=step_size,
        )
        print(
            f"Running asset={asset} file={data_path} "
            f"(min_train={min_train_size}, test={test_size}, step={step_size})"
        )
        try:
            subprocess.run(cmd, check=True)
        except Exception as exc:
            if args.skip_failed_assets:
                failed_assets.append({"asset": asset, "data_path": data_path, "error": str(exc)})
                print(f"Skipping failed asset={asset}: {exc}")
                continue
            raise

        if args.run_paired_sentiment:
            no_sent = asset_base.with_name(asset_base.stem + "_no_sent" + asset_base.suffix)
            with_sent = asset_base.with_name(asset_base.stem + "_with_sent" + asset_base.suffix)
            no_df, no_meta = _collect_for_mode(no_sent, asset=asset, data_path=data_path, mode="no_sentiment")
            ws_df, ws_meta = _collect_for_mode(with_sent, asset=asset, data_path=data_path, mode="with_sentiment")
            all_rows.extend([no_df, ws_df])
            metadata_rows.extend([no_meta, ws_meta])
            if args.save_predictions:
                for mode_base in (no_sent, with_sent):
                    pred_path = mode_base.with_name(mode_base.stem + "_predictions.csv")
                    quant_path = mode_base.with_name(mode_base.stem + "_quantiles.csv")
                    if pred_path.exists():
                        prediction_frames.append(pd.read_csv(pred_path))
                    if quant_path.exists():
                        quantile_frames.append(pd.read_csv(quant_path))
        else:
            mode = "with_sentiment" if args.use_sentiment else "no_sentiment"
            df, meta = _collect_for_mode(asset_base, asset=asset, data_path=data_path, mode=mode)
            all_rows.append(df)
            metadata_rows.append(meta)
            if args.save_predictions:
                pred_path = asset_base.with_name(asset_base.stem + "_predictions.csv")
                quant_path = asset_base.with_name(asset_base.stem + "_quantiles.csv")
                if pred_path.exists():
                    prediction_frames.append(pd.read_csv(pred_path))
                if quant_path.exists():
                    quantile_frames.append(pd.read_csv(quant_path))

    if not all_rows:
        raise ValueError("No successful asset runs were produced.")

    summary_long = pd.concat(all_rows, ignore_index=True)
    summary_long.to_csv(output_path, index=False)

    rank_df = summary_long.copy()
    rank_df["mae_rank_within_asset_mode"] = rank_df.groupby(["asset", "mode"])["mae_mean"].rank(method="min")
    rank_table = (
        rank_df.groupby(["mode", "model"], as_index=False)
        .agg(
            avg_mae_mean=("mae_mean", "mean"),
            avg_rank=("mae_rank_within_asset_mode", "mean"),
            n_asset_evals=("asset", "count"),
        )
        .sort_values(["mode", "avg_rank", "avg_mae_mean"], ascending=[True, True, True])
    )
    rank_path = output_path.with_name(output_path.stem + "_ranks.csv")
    rank_table.to_csv(rank_path, index=False)
    if args.save_predictions and prediction_frames:
        pred_out = output_path.with_name(output_path.stem + "_predictions.csv")
        pd.concat(prediction_frames, ignore_index=True).to_csv(pred_out, index=False)
        print(f"Saved aggregated predictions: {pred_out}")
    if args.save_predictions and quantile_frames:
        quant_out = output_path.with_name(output_path.stem + "_quantiles.csv")
        pd.concat(quantile_frames, ignore_index=True).to_csv(quant_out, index=False)
        print(f"Saved aggregated quantiles: {quant_out}")

    audit_df = pd.DataFrame(metadata_rows)
    unique_target_spaces = sorted(set(audit_df["target_space"].dropna().tolist()))
    audit_df["consistent_target_space_all"] = len(unique_target_spaces) == 1
    audit_df["expected_target_space"] = unique_target_spaces[0] if unique_target_spaces else "unknown"
    if failed_assets:
        failed_df = pd.DataFrame(failed_assets)
        failed_path = output_path.with_name(output_path.stem + "_failed_assets.csv")
        failed_df.to_csv(failed_path, index=False)
        print(f"Saved failed asset log: {failed_path}")
    audit_path = output_path.with_name(output_path.stem + "_comparability_audit.csv")
    audit_df.to_csv(audit_path, index=False)

    if len(unique_target_spaces) != 1:
        raise ValueError(
            "Comparability check failed: multiple target spaces detected: "
            f"{unique_target_spaces}. Inspect {audit_path}."
        )

    print(f"Saved multi-asset summary: {output_path}")
    print(f"Saved model ranking table: {rank_path}")
    print(f"Saved comparability audit: {audit_path}")


if __name__ == "__main__":
    main()
