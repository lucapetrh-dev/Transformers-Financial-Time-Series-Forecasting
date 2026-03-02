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


def _parse_model_overrides(value: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if not value:
        return out
    for item in value.split(","):
        part = item.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid --model-overrides token (expected key=value): {part}")
        k, v = part.split("=", 1)
        out[k.strip().lower()] = v.strip()
    return out


def _load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _estimate_usable_rows(data_path: str, time_col: str, horizon: int) -> int:
    raw = load_market_data(data_path, time_col=time_col)
    features = build_feature_frame(
        raw,
        config=FeatureConfig(lookbacks=(20, 60), ewma_span=20, roc_period=5),
        time_col=time_col,
    )
    features = add_targets(features, horizons=(1, 5, 20))
    target_col = f"target_ret_{horizon}d"
    keep = [time_col, "log_price", "return_1d", target_col]
    frame = features[keep].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return int(len(frame))


def _resolve_split_params(
    *,
    data_path: str,
    args: argparse.Namespace,
) -> tuple[int, int, int]:
    step = int(args.step_size)
    if step <= 0:
        raise ValueError("step_size must be positive")
    if not args.auto_adjust_splits:
        return args.min_train_size, args.test_size, step

    n = _estimate_usable_rows(data_path, time_col=args.time_col, horizon=args.horizon)
    if n < 60:
        raise ValueError(f"Too few rows ({n}) for walk-forward modeling")

    min_train = min(args.min_train_size, max(40, int(n * 0.7)))
    test = min(args.test_size, max(10, int(n * 0.15)))
    if min_train + test > n:
        test = max(10, min(test, n // 4))
        min_train = max(40, n - test)
    return int(min_train), int(test), int(step)


def _build_single_cmd(
    *,
    data_path: str,
    output_path: Path,
    args: argparse.Namespace,
    model_key: str,
    model_id_override: str | None,
    min_train: int,
    test_size: int,
    step_size: int,
) -> list[str]:
    cmd = [
        "python",
        "forecast/runners/run_foundation_zero_shot.py",
        "--data-path",
        data_path,
        "--time-col",
        args.time_col,
        "--horizon",
        str(args.horizon),
        "--context-length",
        str(args.context_length),
        "--min-train-size",
        str(min_train),
        "--test-size",
        str(test_size),
        "--step-size",
        str(step_size),
        "--batch-size",
        str(args.batch_size),
        "--model-key",
        model_key,
        "--output",
        str(output_path),
    ]
    if model_id_override:
        cmd.extend(["--model-id", model_id_override])
    if args.save_predictions:
        cmd.append("--save-predictions")
    if args.paper_strict:
        cmd.append("--paper-strict")
    return cmd


def _collect_asset_model(
    *,
    summary_path: Path,
    asset: str,
    data_path: str,
) -> tuple[pd.DataFrame, dict]:
    summary = pd.read_csv(summary_path).copy()
    summary.insert(0, "asset", asset)
    summary.insert(1, "data_path", data_path)
    summary.insert(2, "mode", "no_sentiment")
    summary.insert(3, "objective_track", "point")

    metadata = _load_metadata(summary_path.with_name(summary_path.stem + "_metadata.json"))
    meta_row = {
        "asset": asset,
        "data_path": data_path,
        "mode": "no_sentiment",
        "objective_track": "point",
        "target_space": metadata.get("target_space"),
        "horizon_days": metadata.get("horizon_days"),
        "feature_mode": metadata.get("feature_mode"),
        "sentiment_lag": metadata.get("sentiment_lag"),
        "backend_status": metadata.get("backend_status"),
        "backend_note": metadata.get("backend_note"),
        "raw_rows": metadata.get("raw_rows"),
        "model_rows": metadata.get("model_rows"),
        "time_start": metadata.get("time_start"),
        "time_end": metadata.get("time_end"),
        "inferred_frequency": metadata.get("inferred_frequency"),
        "split_method": metadata.get("split_method"),
        "model_key": metadata.get("model_key"),
        "model_id": metadata.get("model_id"),
    }
    return summary, meta_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run foundation-model benchmark across multiple assets")
    parser.add_argument("--data-glob", type=str, default=".data/hourly/*_timeseries_hourly.csv")
    parser.add_argument("--time-col", type=str, default="time")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 5, 20])
    parser.add_argument("--context-length", type=int, default=120)
    parser.add_argument("--min-train-size", type=int, default=365)
    parser.add_argument("--test-size", type=int, default=30)
    parser.add_argument("--step-size", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--models", type=str, default="chronos2")
    parser.add_argument("--model-overrides", type=str, default="")
    parser.add_argument("--auto-adjust-splits", action="store_true")
    parser.add_argument("--skip-failed-assets", action="store_true")
    parser.add_argument("--limit-assets", type=int, default=0)
    parser.add_argument("--include-assets", type=str, default="")
    parser.add_argument("--exclude-assets", type=str, default="")
    parser.add_argument(
        "--paper-strict",
        action="store_true",
        help="Fail if any selected foundation adapter is non-native or if any asset run fails.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Persist and aggregate per-fold prediction/quantile traces across assets",
    )
    parser.add_argument("--output", type=str, default="results/multi_asset_foundation_summary.csv")
    args = parser.parse_args()

    if args.paper_strict and args.skip_failed_assets:
        raise ValueError("--paper-strict cannot be combined with --skip-failed-assets")

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

    model_keys = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    if not model_keys:
        raise ValueError("No foundation models requested")
    model_overrides = _parse_model_overrides(args.model_overrides)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[pd.DataFrame] = []
    metadata_rows: list[dict] = []
    failed_rows: list[dict[str, str]] = []
    prediction_frames: list[pd.DataFrame] = []
    quantile_frames: list[pd.DataFrame] = []

    for data_path in data_paths:
        asset = asset_name_from_path(data_path)
        try:
            min_train, test_size, step_size = _resolve_split_params(data_path=data_path, args=args)
        except Exception as exc:
            if args.skip_failed_assets:
                failed_rows.append({"asset": asset, "data_path": data_path, "model_key": "*", "error": str(exc)})
                print(f"Skipping failed asset={asset}: {exc}")
                continue
            raise

        for model_key in model_keys:
            model_id_override = model_overrides.get(model_key)
            run_output = output_path.with_name(f"{output_path.stem}_{asset}_{model_key}{output_path.suffix}")
            cmd = _build_single_cmd(
                data_path=data_path,
                output_path=run_output,
                args=args,
                model_key=model_key,
                model_id_override=model_id_override,
                min_train=min_train,
                test_size=test_size,
                step_size=step_size,
            )
            print(
                f"Running foundation asset={asset} model={model_key} "
                f"(min_train={min_train}, test={test_size}, step={step_size})"
            )
            try:
                subprocess.run(cmd, check=True)
            except Exception as exc:
                if args.skip_failed_assets:
                    failed_rows.append(
                        {"asset": asset, "data_path": data_path, "model_key": model_key, "error": str(exc)}
                    )
                    print(f"Skipping failed asset={asset} model={model_key}: {exc}")
                    continue
                raise

            summary_df, meta_row = _collect_asset_model(summary_path=run_output, asset=asset, data_path=data_path)
            all_rows.append(summary_df)
            metadata_rows.append(meta_row)
            if args.save_predictions:
                pred_path = run_output.with_name(run_output.stem + "_predictions.csv")
                quant_path = run_output.with_name(run_output.stem + "_quantiles.csv")
                if pred_path.exists():
                    prediction_frames.append(pd.read_csv(pred_path))
                if quant_path.exists():
                    quantile_frames.append(pd.read_csv(quant_path))

    if not all_rows:
        raise ValueError("No successful foundation asset runs were produced")

    summary_long = pd.concat(all_rows, ignore_index=True)
    summary_long.to_csv(output_path, index=False)

    rank_df = summary_long.copy()
    rank_df["mae_rank_within_asset_mode"] = rank_df.groupby(["asset", "mode", "objective_track"])["mae_mean"].rank(method="min")
    rank_table = (
        rank_df.groupby(["mode", "objective_track", "model"], as_index=False)
        .agg(
            avg_mae_mean=("mae_mean", "mean"),
            avg_rank=("mae_rank_within_asset_mode", "mean"),
            n_asset_evals=("asset", "count"),
        )
        .sort_values(["mode", "objective_track", "avg_rank", "avg_mae_mean"], ascending=[True, True, True, True])
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
    audit_path = output_path.with_name(output_path.stem + "_comparability_audit.csv")
    audit_df.to_csv(audit_path, index=False)

    if failed_rows:
        failed_path = output_path.with_name(output_path.stem + "_failed_assets.csv")
        pd.DataFrame(failed_rows).to_csv(failed_path, index=False)
        print(f"Saved failed asset log: {failed_path}")

    if len(unique_target_spaces) != 1:
        raise ValueError(
            "Comparability check failed: multiple target spaces detected: "
            f"{unique_target_spaces}. Inspect {audit_path}."
        )

    print(f"Saved multi-asset foundation summary: {output_path}")
    print(f"Saved ranking table: {rank_path}")
    print(f"Saved comparability audit: {audit_path}")


if __name__ == "__main__":
    main()

