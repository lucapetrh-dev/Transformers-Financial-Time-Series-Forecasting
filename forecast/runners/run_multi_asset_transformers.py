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
from forecast.pipeline.data_pipeline import FeatureConfig, add_targets, build_feature_frame, load_market_data
from forecast.pipeline.sentiment import detect_sentiment_columns, prepare_causal_sentiment_features
from forecast.pipeline.transformers import build_sliding_windows

def _build_cmd(
    *,
    data_path: str,
    output_path: Path,
    args: argparse.Namespace,
    min_train_size: int,
    test_size: int,
    step_size: int,
) -> list[str]:
    cmd = [
        "python",
        "forecast/runners/run_transformers.py",
        "--data-path",
        data_path,
        "--time-col",
        args.time_col,
        "--horizon",
        str(args.horizon),
        "--lookback",
        str(args.lookback),
        "--min-train-size",
        str(min_train_size),
        "--test-size",
        str(test_size),
        "--step-size",
        str(step_size),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--models",
        args.models,
        "--probabilistic-mode",
        args.probabilistic_mode,
        "--quantiles",
        args.quantiles,
        "--point-loss",
        args.point_loss,
        "--seed",
        str(args.seed),
        "--sentiment-lag",
        str(args.sentiment_lag),
        "--sentiment-min-non-null-ratio",
        str(args.sentiment_min_non_null_ratio),
        "--regime-lookbacks",
        args.regime_lookbacks,
        "--output",
        str(output_path),
    ]
    if args.objective_track:
        cmd.extend(["--objective-track", args.objective_track])
    if args.use_regime_features:
        cmd.append("--use-regime-features")
    if args.run_paired_sentiment:
        cmd.append("--run-paired-sentiment")
    elif args.use_sentiment:
        cmd.append("--use-sentiment")
    if args.save_predictions:
        cmd.append("--save-predictions")
    if args.save_history:
        cmd.append("--save-history")
    return cmd


def _load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _estimate_usable_sequences(data_path: str, args: argparse.Namespace, use_sentiment: bool) -> int:
    raw = load_market_data(data_path, time_col=args.time_col)
    features = build_feature_frame(
        raw,
        config=FeatureConfig(
            lookbacks=(20, 60),
            ewma_span=20,
            roc_period=5,
            use_regime_features=args.use_regime_features,
            regime_lookbacks=tuple(int(v.strip()) for v in args.regime_lookbacks.split(",") if v.strip()),
        ),
        time_col=args.time_col,
    )
    features = add_targets(features, horizons=(1, 5, 20))
    target_col = f"target_ret_{args.horizon}d"

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
    if args.use_regime_features:
        for lb in tuple(int(v.strip()) for v in args.regime_lookbacks.split(",") if v.strip()):
            feature_cols.extend([f"realized_vol_{lb}", f"vol_z_{lb}", f"vol_regime_{lb}"])

    if use_sentiment:
        raw_sent_cols = detect_sentiment_columns(features, exclude_cols=set(feature_cols).union({target_col, "log_price"}))
        features, sent_cols, _ = prepare_causal_sentiment_features(
            features,
            raw_sent_cols,
            lag=args.sentiment_lag,
            min_non_null_ratio=args.sentiment_min_non_null_ratio,
            fill_method="ffill_zero",
        )
        feature_cols = feature_cols + sent_cols

    frame = features[[args.time_col, *feature_cols, target_col]].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if frame.empty:
        return 0
    X_seq, _ = build_sliding_windows(frame, feature_cols=feature_cols, target_col=target_col, lookback=args.lookback)
    return int(len(X_seq))


def _resolve_split_params(data_path: str, args: argparse.Namespace, run_with_sentiment: bool) -> tuple[int, int, int]:
    step = int(args.step_size)
    if step <= 0:
        raise ValueError("step_size must be positive")

    if not args.auto_adjust_splits:
        return args.min_train_size, args.test_size, step

    n_seq = _estimate_usable_sequences(data_path, args, use_sentiment=run_with_sentiment)
    if n_seq < 80:
        raise ValueError(f"Too few sequences ({n_seq}) for transformer evaluation")

    min_train = min(args.min_train_size, max(128, int(n_seq * 0.7)))
    test = min(args.test_size, max(16, int(n_seq * 0.15)))
    if min_train + test > n_seq:
        test = max(16, min(test, n_seq // 4))
        min_train = max(128, n_seq - test)
    return int(min_train), int(test), int(step)


def _collect_for_mode(summary_path: Path, asset: str, data_path: str, mode: str) -> tuple[pd.DataFrame, dict]:
    summary = pd.read_csv(summary_path)
    summary = summary.copy()
    summary.insert(0, "asset", asset)
    summary.insert(1, "data_path", data_path)
    summary.insert(2, "mode", mode)

    metadata = _load_metadata(summary_path.with_name(summary_path.stem + "_metadata.json"))
    meta_row = {
        "asset": asset,
        "data_path": data_path,
        "mode": mode,
        "target_space": metadata.get("target_space"),
        "horizon_days": metadata.get("horizon_days"),
        "feature_mode": metadata.get("feature_mode"),
        "sentiment_lag": metadata.get("sentiment_lag"),
        "probabilistic_mode": metadata.get("probabilistic_mode"),
        "objective_tracks": ";".join(metadata.get("objective_tracks", [])) if isinstance(metadata.get("objective_tracks"), list) else metadata.get("objective_tracks"),
        "point_loss": metadata.get("point_loss"),
        "raw_rows": metadata.get("raw_rows"),
        "model_rows": metadata.get("model_rows"),
        "sequence_rows": metadata.get("sequence_rows"),
        "time_start": metadata.get("time_start"),
        "time_end": metadata.get("time_end"),
        "inferred_frequency": metadata.get("inferred_frequency"),
        "split_method": metadata.get("split_method"),
    }
    return summary, meta_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run transformer-family benchmark across multiple assets")
    parser.add_argument("--data-glob", type=str, default=".data/hourly/*_timeseries_hourly.csv")
    parser.add_argument("--time-col", type=str, default="time")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 5, 20])
    parser.add_argument("--lookback", type=int, default=64)
    parser.add_argument("--min-train-size", type=int, default=365)
    parser.add_argument("--test-size", type=int, default=60)
    parser.add_argument("--step-size", type=int, default=180)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--models", type=str, default="dlinear_like,tcn_like,patchtst_like,itransformer_like")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probabilistic-mode", type=str, default="quantile", choices=["quantile", "gaussian_proxy"])
    parser.add_argument("--quantiles", type=str, default="0.1,0.5,0.9")
    parser.add_argument("--objective-track", type=str, default=None, choices=["point", "quantile", "both"])
    parser.add_argument("--point-loss", type=str, default="mse", choices=["mse", "mae"])
    parser.add_argument("--use-regime-features", action="store_true")
    parser.add_argument("--regime-lookbacks", type=str, default="20,60")
    parser.add_argument("--use-sentiment", action="store_true")
    parser.add_argument("--run-paired-sentiment", action="store_true")
    parser.add_argument("--sentiment-lag", type=int, default=1)
    parser.add_argument("--sentiment-min-non-null-ratio", type=float, default=0.2)
    parser.add_argument("--auto-adjust-splits", action="store_true")
    parser.add_argument("--skip-failed-assets", action="store_true")
    parser.add_argument("--limit-assets", type=int, default=0)
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
    parser.add_argument(
        "--save-history",
        action="store_true",
        help="Persist and aggregate epoch-level training traces across assets",
    )
    parser.add_argument("--output", type=str, default="results/multi_asset/multi_asset_transformers_summary.csv")
    args = parser.parse_args()

    data_paths = sorted(glob.glob(args.data_glob))
    if not data_paths and args.data_glob.startswith("data/"):
        data_paths = sorted(glob.glob("." + args.data_glob))
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
    history_frames: list[pd.DataFrame] = []

    for data_path in data_paths:
        asset = asset_name_from_path(data_path)
        asset_base = output_path.with_name(f"{output_path.stem}_{asset}{output_path.suffix}")
        try:
            min_train, test, step = _resolve_split_params(
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

        cmd = _build_cmd(
            data_path=data_path,
            output_path=asset_base,
            args=args,
            min_train_size=min_train,
            test_size=test,
            step_size=step,
        )
        print(f"Running transformers asset={asset} (min_train={min_train}, test={test}, step={step})")
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
            if args.save_predictions or args.save_history:
                for mode_base in (no_sent, with_sent):
                    pred_path = mode_base.with_name(mode_base.stem + "_predictions.csv")
                    quant_path = mode_base.with_name(mode_base.stem + "_quantiles.csv")
                    hist_path = mode_base.with_name(mode_base.stem + "_history.csv")
                    if args.save_predictions and pred_path.exists():
                        prediction_frames.append(pd.read_csv(pred_path))
                    if args.save_predictions and quant_path.exists():
                        quantile_frames.append(pd.read_csv(quant_path))
                    if args.save_history and hist_path.exists():
                        history_frames.append(pd.read_csv(hist_path))
        else:
            mode = "with_sentiment" if args.use_sentiment else "no_sentiment"
            df, meta = _collect_for_mode(asset_base, asset=asset, data_path=data_path, mode=mode)
            all_rows.append(df)
            metadata_rows.append(meta)
            if args.save_predictions or args.save_history:
                pred_path = asset_base.with_name(asset_base.stem + "_predictions.csv")
                quant_path = asset_base.with_name(asset_base.stem + "_quantiles.csv")
                hist_path = asset_base.with_name(asset_base.stem + "_history.csv")
                if args.save_predictions and pred_path.exists():
                    prediction_frames.append(pd.read_csv(pred_path))
                if args.save_predictions and quant_path.exists():
                    quantile_frames.append(pd.read_csv(quant_path))
                if args.save_history and hist_path.exists():
                    history_frames.append(pd.read_csv(hist_path))

    if not all_rows:
        raise ValueError("No successful transformer asset runs were produced")

    summary_long = pd.concat(all_rows, ignore_index=True)
    summary_long.to_csv(output_path, index=False)

    rank_df = summary_long.copy()
    rank_group_cols = ["asset", "mode"] + (["objective_track"] if "objective_track" in rank_df.columns else [])
    rank_df["mae_rank_within_asset_mode"] = rank_df.groupby(rank_group_cols)["mae_mean"].rank(method="min")
    out_group_cols = ["mode"] + (["objective_track"] if "objective_track" in rank_df.columns else []) + ["model"]
    ranks = (
        rank_df.groupby(out_group_cols, as_index=False)
        .agg(
            avg_mae_mean=("mae_mean", "mean"),
            avg_rank=("mae_rank_within_asset_mode", "mean"),
            n_asset_evals=("asset", "count"),
        )
        .sort_values([*out_group_cols[:-1], "avg_rank", "avg_mae_mean"], ascending=[True] * (len(out_group_cols) - 1) + [True, True])
    )
    rank_path = output_path.with_name(output_path.stem + "_ranks.csv")
    ranks.to_csv(rank_path, index=False)
    if args.save_predictions and prediction_frames:
        pred_out = output_path.with_name(output_path.stem + "_predictions.csv")
        pd.concat(prediction_frames, ignore_index=True).to_csv(pred_out, index=False)
        print(f"Saved aggregated predictions: {pred_out}")
    if args.save_predictions and quantile_frames:
        quant_out = output_path.with_name(output_path.stem + "_quantiles.csv")
        pd.concat(quantile_frames, ignore_index=True).to_csv(quant_out, index=False)
        print(f"Saved aggregated quantiles: {quant_out}")
    if args.save_history and history_frames:
        hist_out = output_path.with_name(output_path.stem + "_history.csv")
        pd.concat(history_frames, ignore_index=True).to_csv(hist_out, index=False)
        print(f"Saved aggregated history: {hist_out}")

    audit_df = pd.DataFrame(metadata_rows)
    unique_target_spaces = sorted(set(audit_df["target_space"].dropna().tolist()))
    audit_df["consistent_target_space_all"] = len(unique_target_spaces) == 1
    audit_df["expected_target_space"] = unique_target_spaces[0] if unique_target_spaces else "unknown"
    audit_path = output_path.with_name(output_path.stem + "_comparability_audit.csv")
    audit_df.to_csv(audit_path, index=False)

    if failed_assets:
        failed_path = output_path.with_name(output_path.stem + "_failed_assets.csv")
        pd.DataFrame(failed_assets).to_csv(failed_path, index=False)
        print(f"Saved failed asset log: {failed_path}")

    if len(unique_target_spaces) != 1:
        raise ValueError(
            "Comparability check failed: multiple target spaces detected: "
            f"{unique_target_spaces}. Inspect {audit_path}."
        )

    print(f"Saved multi-asset transformers summary: {output_path}")
    print(f"Saved ranking table: {rank_path}")
    print(f"Saved comparability audit: {audit_path}")


if __name__ == "__main__":
    main()
