from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forecast.pipeline.baselines import (
    LinearLagBaseline,
    RandomWalkReturnBaseline,
    XGBoostLagBaseline,
    has_xgboost,
    try_arima_forecast,
)
from forecast.pipeline.data_pipeline import (
    FeatureConfig,
    add_targets,
    build_feature_frame,
    load_market_data,
    make_lag_features,
)
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
from forecast.pipeline.sentiment import (
    build_sentiment_comparison_table,
    detect_sentiment_columns,
    prepare_causal_sentiment_features,
)
from forecast.pipeline.quality import save_preprocessing_quality_artifacts
from forecast.pipeline.splits import walk_forward_splits
from forecast.pipeline.tuning import PurgedCVConfig, tune_ridge_alpha_purged_cv


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def _infer_asset_label(path: str | Path) -> str:
    name = Path(path).stem.lower()
    name = name.replace("_lunarcrush_timeseries_hourly", "")
    name = name.replace("_lunarcrash_timeseries_hourly", "")
    name = name.replace("_timeseries_hourly", "")
    name = name.replace("_timeseries", "")
    return name


def _gaussian_q10_q90(y_pred: np.ndarray, train_resid_std: float) -> tuple[np.ndarray, np.ndarray]:
    sigma = np.full_like(y_pred, fill_value=max(train_resid_std, 1e-6), dtype=float)
    return y_pred - 1.28155 * sigma, y_pred + 1.28155 * sigma


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
    y_pred: np.ndarray,
    train_resid_std: float,
    n_trials: int,
) -> dict[str, float]:
    sigma = np.full_like(y_pred, fill_value=max(train_resid_std, 1e-6), dtype=float)
    q10 = y_pred - 1.28155 * sigma
    q90 = y_pred + 1.28155 * sigma

    metrics = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "pinball_q10": pinball_loss(y_true, q10, q=0.1),
        "pinball_q50": pinball_loss(y_true, y_pred, q=0.5),
        "pinball_q90": pinball_loss(y_true, q90, q=0.9),
        "crps_gaussian": crps_gaussian(y_true, y_pred, sigma),
        "coverage_80": interval_coverage(y_true, q10, q90),
        "interval_width_80": interval_width(q10, q90),
        "wis_80": weighted_interval_score(y_true, q10, q90, alpha=0.2),
    }
    metrics.update(backtest_metrics(y_true, y_pred, cost_bps=5.0))
    metrics.update(cost_sensitivity_metrics(y_true, y_pred, cost_bps_list=(0.0, 5.0, 10.0, 20.0, 50.0), n_trials=n_trials))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-aware walk-forward baseline evaluation on returns")
    parser.add_argument("--data-path", type=str, required=True, help="Path to market CSV")
    parser.add_argument("--time-col", type=str, default="time")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 5, 20])
    parser.add_argument("--max-lag", type=int, default=20)
    parser.add_argument("--min-train-size", type=int, default=365)
    parser.add_argument("--test-size", type=int, default=30)
    parser.add_argument("--step-size", type=int, default=30)
    parser.add_argument(
        "--tune-linear-alpha",
        action="store_true",
        help="Tune linear ridge alpha with purged CV on each training fold",
    )
    parser.add_argument(
        "--alpha-grid",
        type=str,
        default="0.01,0.1,1.0,10.0",
        help="Comma-separated alpha candidates for ridge tuning",
    )
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-embargo", type=int, default=5)
    parser.add_argument("--cv-label-horizon", type=int, default=1)
    parser.add_argument("--use-regime-features", action="store_true", help="Enable volatility regime features")
    parser.add_argument("--regime-lookbacks", type=str, default="20,60", help="Comma-separated lookbacks for regime features")
    parser.add_argument(
        "--use-sentiment",
        action="store_true",
        help="Include sentiment/social features in addition to financial features",
    )
    parser.add_argument(
        "--sentiment-lag",
        type=int,
        default=1,
        help="Causal lag applied to sentiment features (1 = previous period only)",
    )
    parser.add_argument(
        "--sentiment-min-non-null-ratio",
        type=float,
        default=0.2,
        help="Minimum non-null ratio for sentiment columns before causal imputation",
    )
    parser.add_argument(
        "--run-paired-sentiment",
        action="store_true",
        help="Run both no-sentiment and with-sentiment experiments and produce a comparison table",
    )
    parser.add_argument(
        "--no-arima",
        action="store_true",
        help="Disable ARIMA baseline (useful for faster paired runs)",
    )
    parser.add_argument(
        "--no-xgboost",
        action="store_true",
        help="Disable XGBoost baseline even when xgboost is installed",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Export standardized per-fold prediction and quantile traces for downstream figure generation",
    )
    parser.add_argument("--output", type=str, default="results/baselines_summary.csv")
    args = parser.parse_args()

    if args.run_paired_sentiment:
        base_output = Path(args.output)
        no_sent_output = base_output.with_name(base_output.stem + "_no_sent" + base_output.suffix)
        with_sent_output = base_output.with_name(base_output.stem + "_with_sent" + base_output.suffix)

        _run_single(args, use_sentiment=False, output_path=no_sent_output)
        _run_single(args, use_sentiment=True, output_path=with_sent_output)

        comparison_path = base_output.with_name(base_output.stem + "_sentiment_comparison.csv")
        build_sentiment_comparison_table(no_sent_output, with_sent_output, comparison_path)
        print(f"Saved sentiment comparison: {comparison_path}")
        return

    _run_single(args, use_sentiment=args.use_sentiment, output_path=Path(args.output))


def _run_single(args, use_sentiment: bool, output_path: Path) -> None:
    raw = load_market_data(args.data_path, time_col=args.time_col)
    raw_rows = len(raw)
    regime_lookbacks = _parse_int_tuple(args.regime_lookbacks)
    features = build_feature_frame(
        raw,
        config=FeatureConfig(
            lookbacks=(20, 60),
            ewma_span=20,
            roc_period=5,
            use_regime_features=args.use_regime_features,
            regime_lookbacks=regime_lookbacks,
        ),
        time_col=args.time_col,
    )
    features = add_targets(features, horizons=(1, 5, 20))
    features = make_lag_features(features, source_col="return_1d", max_lag=args.max_lag)
    features_pre_sentiment = features.copy()

    target_col = f"target_ret_{args.horizon}d"
    lag_cols = [f"return_1d_lag_{k}" for k in range(1, args.max_lag + 1)]
    extra_cols = ["dow_sin", "dow_cos", "month_sin", "month_cos", "ewma_ret", "ewma_roc", "ret_mean_20", "ret_std_20"]
    if args.use_regime_features:
        for lb in regime_lookbacks:
            extra_cols.extend([f"realized_vol_{lb}", f"vol_z_{lb}", f"vol_regime_{lb}"])
    feature_cols = lag_cols + [c for c in extra_cols if c in features.columns]

    sentiment_cols: list[str] = []
    dropped_sentiment_cols: list[str] = []
    if use_sentiment:
        raw_sentiment_cols = detect_sentiment_columns(
            features,
            exclude_cols=set(feature_cols).union({target_col, "log_price", "return_1d"}),
        )
        features, sentiment_cols, dropped_sentiment_cols = prepare_causal_sentiment_features(
            features,
            raw_sentiment_cols,
            lag=args.sentiment_lag,
            min_non_null_ratio=args.sentiment_min_non_null_ratio,
            fill_method="ffill_zero",
        )
        feature_cols = feature_cols + sentiment_cols
    features_post_sentiment = features.copy()

    required_cols = [args.time_col, target_col, *feature_cols]
    frame_before_sentiment = features_pre_sentiment[required_cols].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    frame_after_sentiment = features_post_sentiment[required_cols].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    frame_before_dropna = frame_after_sentiment.copy()
    features = frame_before_dropna.dropna().reset_index(drop=True)
    if features.empty:
        raise ValueError("No rows left after feature selection/dropna. Check feature coverage for chosen sentiment mode.")

    mode_name = "with_sentiment" if use_sentiment else "no_sentiment"
    asset_label = _infer_asset_label(args.data_path)
    print(
        f"Running mode: {mode_name}; features={len(feature_cols)}; "
        f"sentiment_features={len(sentiment_cols)}; dropped_sentiment_features={len(dropped_sentiment_cols)}"
    )

    timestamps = pd.to_datetime(features[args.time_col], errors="coerce")
    X = features[feature_cols].to_numpy(dtype=float)
    y = features[target_col].to_numpy(dtype=float)

    all_rows: list[dict[str, float | str | int]] = []
    fold_manifest: list[dict] = []
    all_preds_linear: list[float] = []
    all_preds_rw: list[float] = []
    all_preds_xgb: list[float] = []
    all_y_true: list[float] = []
    all_cv_rows: list[pd.DataFrame] = []
    prediction_rows: list[dict[str, float | str | int]] = []
    quantile_rows: list[dict[str, float | str | int]] = []
    alpha_grid = [float(v.strip()) for v in args.alpha_grid.split(",") if v.strip()]
    use_xgboost = has_xgboost() and (not args.no_xgboost)
    n_trials = 4 if use_xgboost else 3

    for fold, (train_idx, test_idx) in enumerate(
        walk_forward_splits(
            n_samples=len(features),
            min_train_size=args.min_train_size,
            test_size=args.test_size,
            step_size=args.step_size,
            expanding=True,
        )
    ):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        ts_test = timestamps.iloc[test_idx].reset_index(drop=True)
        fold_manifest.append(
            fold_manifest_rows(
                fold_id=fold,
                mode="walk_forward",
                train_idx=train_idx,
                eval_idx=test_idx,
                timestamps=timestamps,
            )
        )

        linear_alpha = 1.0
        if args.tune_linear_alpha:
            linear_alpha, cv_df = tune_ridge_alpha_purged_cv(
                X_train,
                y_train,
                alpha_grid=alpha_grid,
                cv_config=PurgedCVConfig(
                    n_splits=args.cv_splits,
                    embargo=args.cv_embargo,
                    label_horizon=args.cv_label_horizon,
                ),
                objective="mae",
            )
            cv_df = cv_df.copy()
            cv_df["walk_forward_fold"] = fold
            all_cv_rows.append(cv_df)

        rw = RandomWalkReturnBaseline().fit(X_train, y_train)
        lr = LinearLagBaseline(alpha=linear_alpha).fit(X_train, y_train)

        pred_rw = rw.predict(X_test)
        pred_lr = lr.predict(X_test)

        train_resid_std_rw = float(np.std(y_train - rw.predict(X_train), ddof=1)) if len(y_train) > 1 else 1e-6
        train_resid_std_lr = float(np.std(y_train - lr.predict(X_train), ddof=1)) if len(y_train) > 1 else 1e-6

        rw_metrics = evaluate_model(y_test, pred_rw, train_resid_std_rw, n_trials=n_trials)
        lr_metrics = evaluate_model(y_test, pred_lr, train_resid_std_lr, n_trials=n_trials)

        all_rows.append({"fold": fold, "model": "random_walk", "linear_alpha": np.nan, **rw_metrics})
        all_rows.append({"fold": fold, "model": "linear_ridge", "linear_alpha": linear_alpha, **lr_metrics})
        if args.save_predictions:
            rw_q10, rw_q90 = _gaussian_q10_q90(pred_rw, train_resid_std_rw)
            lr_q10, lr_q90 = _gaussian_q10_q90(pred_lr, train_resid_std_lr)
            _append_prediction_records(
                prediction_rows,
                quantile_rows,
                timestamps=ts_test,
                y_true=y_test,
                y_pred=pred_rw,
                q10=rw_q10,
                q50=pred_rw,
                q90=rw_q90,
                fold=fold,
                model="random_walk",
                mode=mode_name,
                asset=asset_label,
            )
            _append_prediction_records(
                prediction_rows,
                quantile_rows,
                timestamps=ts_test,
                y_true=y_test,
                y_pred=pred_lr,
                q10=lr_q10,
                q50=pred_lr,
                q90=lr_q90,
                fold=fold,
                model="linear_ridge",
                mode=mode_name,
                asset=asset_label,
            )

        if use_xgboost:
            xgb = XGBoostLagBaseline(random_state=42 + fold).fit(X_train, y_train)
            pred_xgb = xgb.predict(X_test)
            train_resid_std_xgb = float(np.std(y_train - xgb.predict(X_train), ddof=1)) if len(y_train) > 1 else 1e-6
            xgb_metrics = evaluate_model(y_test, pred_xgb, train_resid_std_xgb, n_trials=n_trials)
            all_rows.append({"fold": fold, "model": "xgboost", "linear_alpha": np.nan, **xgb_metrics})
            all_preds_xgb.extend(pred_xgb.tolist())
            if args.save_predictions:
                xgb_q10, xgb_q90 = _gaussian_q10_q90(pred_xgb, train_resid_std_xgb)
                _append_prediction_records(
                    prediction_rows,
                    quantile_rows,
                    timestamps=ts_test,
                    y_true=y_test,
                    y_pred=pred_xgb,
                    q10=xgb_q10,
                    q50=pred_xgb,
                    q90=xgb_q90,
                    fold=fold,
                    model="xgboost",
                    mode=mode_name,
                    asset=asset_label,
                )

        arima_pred = None if args.no_arima else try_arima_forecast(train_y=y_train, test_y=y_test, order=(1, 0, 1))
        if arima_pred is not None:
            arima_std = float(np.std(y_train, ddof=1)) if len(y_train) > 1 else 1e-6
            arima_metrics = evaluate_model(y_test, arima_pred, arima_std, n_trials=n_trials)
            all_rows.append({"fold": fold, "model": "arima_101", "linear_alpha": np.nan, **arima_metrics})
            if args.save_predictions:
                arima_q10, arima_q90 = _gaussian_q10_q90(arima_pred, arima_std)
                _append_prediction_records(
                    prediction_rows,
                    quantile_rows,
                    timestamps=ts_test,
                    y_true=y_test,
                    y_pred=arima_pred,
                    q10=arima_q10,
                    q50=arima_pred,
                    q90=arima_q90,
                    fold=fold,
                    model="arima_101",
                    mode=mode_name,
                    asset=asset_label,
                )

        all_preds_linear.extend(pred_lr.tolist())
        all_preds_rw.extend(pred_rw.tolist())
        all_y_true.extend(y_test.tolist())

    results = pd.DataFrame(all_rows)
    if results.empty:
        raise ValueError(
            "No walk-forward folds were generated. "
            "Reduce --min-train-size or --test-size for the selected dataset."
        )
    summary = results.groupby("model").agg(["mean", "std"])
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    summary = summary.reset_index()

    y_true_arr = np.asarray(all_y_true, dtype=float)
    dm = diebold_mariano_test(y_true_arr, np.asarray(all_preds_linear), np.asarray(all_preds_rw), power=2, horizon=args.horizon)
    summary["dm_vs_rw_stat"] = np.nan
    summary["dm_vs_rw_pvalue"] = np.nan
    summary.loc[summary["model"] == "linear_ridge", "dm_vs_rw_stat"] = dm["dm_stat"]
    summary.loc[summary["model"] == "linear_ridge", "dm_vs_rw_pvalue"] = dm["p_value"]
    if use_xgboost and all_preds_xgb:
        dm_xgb = diebold_mariano_test(
            y_true_arr,
            np.asarray(all_preds_xgb),
            np.asarray(all_preds_rw),
            power=2,
            horizon=args.horizon,
        )
        summary.loc[summary["model"] == "xgboost", "dm_vs_rw_stat"] = dm_xgb["dm_stat"]
        summary.loc[summary["model"] == "xgboost", "dm_vs_rw_pvalue"] = dm_xgb["p_value"]

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

    if all_cv_rows:
        cv_path = output_path.with_name(output_path.stem + "_cv.csv")
        pd.concat(all_cv_rows, ignore_index=True).to_csv(cv_path, index=False)
        print(f"Saved CV details: {cv_path}")

    quality_summary_path, missingness_stages_path = save_preprocessing_quality_artifacts(
        output_path,
        raw_df=raw,
        frame_before_sentiment=frame_before_sentiment,
        frame_after_sentiment=frame_after_sentiment,
        frame_before_dropna=frame_before_dropna,
        frame_after_dropna=features,
        time_col=args.time_col,
        target_col=target_col,
        feature_cols=feature_cols,
        sentiment_cols=sentiment_cols,
        sentiment_lag=args.sentiment_lag,
    )

    metadata = {
        "script": "run_baselines.py",
        "data_path": args.data_path,
        "time_col": args.time_col,
        "target_space": "log_return",
        "target_col": target_col,
        "direction_target_col": f"target_dir_{args.horizon}d",
        "horizon_days": args.horizon,
        "feature_mode": "financial_plus_sentiment" if use_sentiment else "financial_only",
        "use_regime_features": bool(args.use_regime_features),
        "regime_lookbacks": list(regime_lookbacks),
        "sentiment_feature_count": len(sentiment_cols),
        "sentiment_columns": sentiment_cols,
        "dropped_sentiment_feature_count": len(dropped_sentiment_cols),
        "dropped_sentiment_columns": dropped_sentiment_cols,
        "sentiment_min_non_null_ratio": args.sentiment_min_non_null_ratio if use_sentiment else None,
        "sentiment_lag": args.sentiment_lag if use_sentiment else 0,
        "causal_alignment_rule": "sentiment columns shifted by sentiment_lag periods",
        "imputation_policy": "sentiment_ffill_then_zero_fill; other features dropna_only",
        "raw_rows": raw_rows,
        "model_rows": len(features),
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
        "purged_cv_tuning_enabled": bool(args.tune_linear_alpha),
        "cv_splits": args.cv_splits,
        "cv_embargo": args.cv_embargo,
        "cv_label_horizon": args.cv_label_horizon,
        "asset": asset_label,
        "mode": mode_name,
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
    main()
