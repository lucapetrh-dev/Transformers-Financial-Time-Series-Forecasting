from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forecast.pipeline.data_pipeline import FeatureConfig, add_targets, build_feature_frame, load_market_data
from forecast.pipeline.experiment_contract import fold_manifest_rows, infer_frequency_label, save_metadata_json
from forecast.pipeline.metrics import (
    backtest_metrics,
    cost_sensitivity_metrics,
    crps_gaussian,
    directional_accuracy,
    diebold_mariano_test,
    interval_coverage,
    interval_width,
    mae,
    pinball_loss,
    rmse,
    weighted_interval_score,
)
from forecast.pipeline.quality import save_preprocessing_quality_artifacts
from forecast.pipeline.splits import walk_forward_splits


def _infer_asset_label(path: str | Path) -> str:
    name = Path(path).stem.lower()
    name = name.replace("_lunarcrush_timeseries_hourly", "")
    name = name.replace("_lunarcrash_timeseries_hourly", "")
    name = name.replace("_timeseries_hourly", "")
    name = name.replace("_timeseries", "")
    return name


def _append_prediction_records(
    rows_pred: list[dict[str, float | str | int]],
    rows_quant: list[dict[str, float | str | int]],
    *,
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    fold: int,
    model: str,
    mode: str,
    asset: str,
) -> None:
    for ts, yt, yp, ql, qm, qu in zip(
        timestamps.astype(str).tolist(),
        y_true.tolist(),
        y_pred.tolist(),
        q10.tolist(),
        q50.tolist(),
        q90.tolist(),
    ):
        rows_pred.append(
            {
                "timestamp": ts,
                "y_true": float(yt),
                "y_pred": float(yp),
                "fold": int(fold),
                "model": model,
                "mode": mode,
                "asset": asset,
            }
        )
        rows_quant.append(
            {
                "timestamp": ts,
                "y_true": float(yt),
                "q10": float(ql),
                "q50": float(qm),
                "q90": float(qu),
                "fold": int(fold),
                "model": model,
                "mode": mode,
                "asset": asset,
            }
        )


def evaluate_model(
    y_true: np.ndarray,
    pred_mean: np.ndarray,
    pred_q10: np.ndarray,
    pred_q50: np.ndarray,
    pred_q90: np.ndarray,
    n_trials: int,
) -> dict[str, float]:
    sigma = np.maximum((pred_q90 - pred_q10) / (2.0 * 1.28155), 1e-6)

    metrics = {
        "mae": mae(y_true, pred_mean),
        "rmse": rmse(y_true, pred_mean),
        "directional_accuracy": directional_accuracy(y_true, pred_mean),
        "pinball_q10": pinball_loss(y_true, pred_q10, q=0.1),
        "pinball_q50": pinball_loss(y_true, pred_q50, q=0.5),
        "pinball_q90": pinball_loss(y_true, pred_q90, q=0.9),
        "crps_gaussian": crps_gaussian(y_true, pred_mean, sigma),
        "coverage_80": interval_coverage(y_true, pred_q10, pred_q90),
        "interval_width_80": interval_width(pred_q10, pred_q90),
        "wis_80": weighted_interval_score(y_true, pred_q10, pred_q90, alpha=0.2),
    }
    metrics.update(backtest_metrics(y_true, pred_mean, cost_bps=5.0))
    metrics.update(cost_sensitivity_metrics(y_true, pred_mean, cost_bps_list=(0.0, 5.0, 10.0, 20.0, 50.0), n_trials=n_trials))
    return metrics


def _to_input(context_1d: np.ndarray) -> np.ndarray:
    # Chronos-2 expects [n_variates, history_length] per series when using list input.
    return context_1d.reshape(1, -1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run zero-shot Chronos-2 walk-forward evaluation")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--time-col", type=str, default="time")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 5, 20])
    parser.add_argument("--context-length", type=int, default=120)
    parser.add_argument("--min-train-size", type=int, default=365)
    parser.add_argument("--test-size", type=int, default=30)
    parser.add_argument("--step-size", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--model-id", type=str, default="amazon/chronos-2")
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Export standardized per-fold prediction and quantile traces for downstream figure generation",
    )
    parser.add_argument("--output", type=str, default="results/chronos2_zero_shot_summary.csv")
    args = parser.parse_args()

    try:
        from chronos import Chronos2Pipeline
    except Exception as exc:
        raise ImportError("chronos package is required. Install with: pip install chronos-forecasting") from exc

    raw = load_market_data(args.data_path, time_col=args.time_col)
    raw_rows = len(raw)
    asset_label = _infer_asset_label(args.data_path)
    features = build_feature_frame(raw, config=FeatureConfig(lookbacks=(20, 60), ewma_span=20, roc_period=5), time_col=args.time_col)
    features = add_targets(features, horizons=(1, 5, 20))

    target_col = f"target_ret_{args.horizon}d"
    required_cols = [args.time_col, "log_price", "return_1d", target_col]
    frame_before_sentiment = features[required_cols].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    frame_after_sentiment = frame_before_sentiment.copy()
    frame_before_dropna = frame_after_sentiment.copy()
    frame = frame_before_dropna.dropna().reset_index(drop=True)
    timestamps = pd.to_datetime(frame[args.time_col], errors="coerce")

    if len(frame) < args.min_train_size + args.test_size:
        raise ValueError("Not enough samples for requested split settings")

    print(f"Loading Chronos-2 model: {args.model_id}")
    pipeline = Chronos2Pipeline.from_pretrained(args.model_id, device_map="cpu")

    rows: list[dict[str, float | str | int]] = []
    fold_manifest: list[dict] = []
    all_y_true: list[float] = []
    all_pred: list[float] = []
    all_rw: list[float] = []
    prediction_rows: list[dict[str, float | str | int]] = []
    quantile_rows: list[dict[str, float | str | int]] = []

    log_prices = frame["log_price"].to_numpy(dtype=float)
    returns_1d = frame["return_1d"].to_numpy(dtype=float)
    target = frame[target_col].to_numpy(dtype=float)

    for fold, (train_idx, test_idx) in enumerate(
        walk_forward_splits(
            n_samples=len(frame),
            min_train_size=args.min_train_size,
            test_size=args.test_size,
            step_size=args.step_size,
            expanding=True,
        )
    ):
        fold_manifest.append(
            fold_manifest_rows(
                fold_id=fold,
                mode="walk_forward",
                train_idx=train_idx,
                eval_idx=test_idx,
                timestamps=timestamps,
            )
        )
        inputs = []
        baselines = []
        y_test = target[test_idx]
        ts_test = timestamps.iloc[test_idx].reset_index(drop=True)

        for idx in test_idx:
            start = max(0, idx - args.context_length + 1)
            context = log_prices[start : idx + 1]
            inputs.append(_to_input(context))
            baselines.append(args.horizon * returns_1d[idx])

        q_list, mean_list = pipeline.predict_quantiles(
            inputs,
            prediction_length=args.horizon,
            quantile_levels=[0.1, 0.5, 0.9],
            batch_size=args.batch_size,
            context_length=args.context_length,
        )

        pred_q10 = []
        pred_q50 = []
        pred_q90 = []
        pred_mean = []

        for series_input, q_fcst, mean_fcst in zip(inputs, q_list, mean_list):
            last_obs = float(series_input[0, -1])
            q_last = q_fcst[0, args.horizon - 1, :].detach().cpu().numpy()
            mean_last = float(mean_fcst[0, args.horizon - 1].detach().cpu().item())

            pred_q10.append(float(q_last[0] - last_obs))
            pred_q50.append(float(q_last[1] - last_obs))
            pred_q90.append(float(q_last[2] - last_obs))
            pred_mean.append(float(mean_last - last_obs))

        pred_q10_arr = np.asarray(pred_q10, dtype=float)
        pred_q50_arr = np.asarray(pred_q50, dtype=float)
        pred_q90_arr = np.asarray(pred_q90, dtype=float)
        pred_mean_arr = np.asarray(pred_mean, dtype=float)
        rw_arr = np.asarray(baselines, dtype=float)

        metrics = evaluate_model(
            y_test,
            pred_mean_arr,
            pred_q10_arr,
            pred_q50_arr,
            pred_q90_arr,
            n_trials=2,
        )
        rows.append({"fold": fold, "model": "chronos2_zero_shot", **metrics})
        if args.save_predictions:
            _append_prediction_records(
                prediction_rows,
                quantile_rows,
                timestamps=ts_test,
                y_true=y_test,
                y_pred=pred_mean_arr,
                q10=pred_q10_arr,
                q50=pred_q50_arr,
                q90=pred_q90_arr,
                fold=fold,
                model="chronos2_zero_shot",
                mode="no_sentiment",
                asset=asset_label,
            )

        rw_metrics = evaluate_model(
            y_test,
            rw_arr,
            rw_arr,
            rw_arr,
            rw_arr,
            n_trials=2,
        )
        rows.append({"fold": fold, "model": "random_walk_scaled", **rw_metrics})
        if args.save_predictions:
            _append_prediction_records(
                prediction_rows,
                quantile_rows,
                timestamps=ts_test,
                y_true=y_test,
                y_pred=rw_arr,
                q10=rw_arr,
                q50=rw_arr,
                q90=rw_arr,
                fold=fold,
                model="random_walk_scaled",
                mode="no_sentiment",
                asset=asset_label,
            )

        all_y_true.extend(y_test.tolist())
        all_pred.extend(pred_mean_arr.tolist())
        all_rw.extend(rw_arr.tolist())

    results = pd.DataFrame(rows)
    if results.empty:
        raise ValueError("No walk-forward folds produced results for Chronos-2")

    summary = results.groupby("model").agg(["mean", "std"])
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    summary = summary.reset_index()

    dm = diebold_mariano_test(
        np.asarray(all_y_true, dtype=float),
        np.asarray(all_pred, dtype=float),
        np.asarray(all_rw, dtype=float),
        power=2,
        horizon=args.horizon,
    )
    summary["dm_vs_rw_stat"] = np.nan
    summary["dm_vs_rw_pvalue"] = np.nan
    summary.loc[summary["model"] == "chronos2_zero_shot", "dm_vs_rw_stat"] = dm["dm_stat"]
    summary.loc[summary["model"] == "chronos2_zero_shot", "dm_vs_rw_pvalue"] = dm["p_value"]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    folds_path = output_path.with_name(output_path.stem + "_folds.csv")
    results.to_csv(folds_path, index=False)
    fold_manifest_path = output_path.with_name(output_path.stem + "_fold_manifest.csv")
    pd.DataFrame(fold_manifest).to_csv(fold_manifest_path, index=False)
    predictions_path = None
    quantiles_path = None
    if args.save_predictions and prediction_rows:
        predictions_path = output_path.with_name(output_path.stem + "_predictions.csv")
        quantiles_path = output_path.with_name(output_path.stem + "_quantiles.csv")
        pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
        pd.DataFrame(quantile_rows).to_csv(quantiles_path, index=False)
        print(f"Saved prediction traces: {predictions_path}")
        print(f"Saved quantile traces: {quantiles_path}")

    quality_summary_path, missingness_stages_path = save_preprocessing_quality_artifacts(
        output_path,
        raw_df=raw,
        frame_before_sentiment=frame_before_sentiment,
        frame_after_sentiment=frame_after_sentiment,
        frame_before_dropna=frame_before_dropna,
        frame_after_dropna=frame,
        time_col=args.time_col,
        target_col=target_col,
        feature_cols=["log_price", "return_1d"],
        sentiment_cols=[],
        sentiment_lag=0,
    )

    metadata = {
        "script": "run_chronos2_zero_shot.py",
        "data_path": args.data_path,
        "time_col": args.time_col,
        "target_space": "log_return",
        "target_col": target_col,
        "direction_target_col": f"target_dir_{args.horizon}d",
        "horizon_days": args.horizon,
        "feature_mode": "financial_only",
        "sentiment_feature_count": 0,
        "sentiment_columns": [],
        "sentiment_lag": 0,
        "causal_alignment_rule": "n/a",
        "imputation_policy": "dropna_only_no_value_imputation",
        "raw_rows": raw_rows,
        "model_rows": len(frame),
        "feature_rows_pre_sentiment_lag": len(frame_before_sentiment),
        "feature_rows_post_sentiment_lag": len(frame_after_sentiment),
        "model_rows_pre_dropna": len(frame_before_dropna),
        "time_start": str(timestamps.min()),
        "time_end": str(timestamps.max()),
        "inferred_frequency": infer_frequency_label(timestamps),
        "split_method": "walk_forward_expanding",
        "min_train_size": args.min_train_size,
        "test_size": args.test_size,
        "step_size": args.step_size,
        "context_length": args.context_length,
        "model_id": args.model_id,
        "asset": asset_label,
        "mode": "no_sentiment",
        "predictions_path": str(predictions_path) if predictions_path else None,
        "quantiles_path": str(quantiles_path) if quantiles_path else None,
        "prediction_rows": len(prediction_rows) if prediction_rows else 0,
        "quantile_rows": len(quantile_rows) if quantile_rows else 0,
        "data_quality_summary_path": str(quality_summary_path),
        "missingness_stages_path": str(missingness_stages_path),
    }
    metadata_path = output_path.with_name(output_path.stem + "_metadata.json")
    save_metadata_json(metadata_path, metadata)

    print(f"Saved summary: {output_path}")
    print(f"Saved fold metrics: {folds_path}")
    print(f"Saved fold manifest: {fold_manifest_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    main()
