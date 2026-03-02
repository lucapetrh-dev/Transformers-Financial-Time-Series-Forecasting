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
from forecast.pipeline.stability import PermutationImportanceConfig, permutation_importance_mae
from forecast.pipeline.splits import purged_kfold_splits
from forecast.pipeline.tuning import PurgedCVConfig, tune_ridge_alpha_purged_cv


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, resid_std: float, n_trials: int) -> dict[str, float]:
    sigma = np.full_like(y_pred, fill_value=max(resid_std, 1e-6), dtype=float)
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
    parser = argparse.ArgumentParser(description="Run CPCV/Purged CV model report on returns")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--time-col", type=str, default="time")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 5, 20])
    parser.add_argument("--max-lag", type=int, default=20)
    parser.add_argument("--n-splits", type=int, default=6)
    parser.add_argument("--embargo", type=int, default=5)
    parser.add_argument("--label-horizon", type=int, default=1)
    parser.add_argument("--use-regime-features", action="store_true", help="Enable volatility regime features")
    parser.add_argument("--regime-lookbacks", type=str, default="20,60", help="Comma-separated lookbacks for regime features")
    parser.add_argument("--no-xgboost", action="store_true")
    parser.add_argument(
        "--no-arima",
        action="store_true",
        help="Disable ARIMA baseline (useful for faster paired runs)",
    )
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
        help="Run both no-sentiment and with-sentiment CPCV reports and produce a comparison table",
    )
    parser.add_argument(
        "--feature-stability-report",
        action="store_true",
        help="Compute nested permutation-importance stability report on each outer CPCV fold (linear_ridge)",
    )
    parser.add_argument("--stability-repeats", type=int, default=3)
    parser.add_argument("--tune-linear-alpha", action="store_true")
    parser.add_argument("--alpha-grid", type=str, default="0.01,0.1,1.0,10.0")
    parser.add_argument("--output", type=str, default="results/cpcv_summary.csv")
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
    else:
        sentiment_cols = []
    features_post_sentiment = features.copy()

    required_cols = [args.time_col, target_col, *feature_cols]
    frame_before_sentiment = features_pre_sentiment[required_cols].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    frame_after_sentiment = features_post_sentiment[required_cols].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    frame_before_dropna = frame_after_sentiment.copy()
    features = frame_before_dropna.dropna().reset_index(drop=True)
    if features.empty:
        raise ValueError("No rows left after feature selection/dropna for CPCV run.")

    mode_name = "with_sentiment" if use_sentiment else "no_sentiment"
    print(
        f"Running CPCV mode: {mode_name}; features={len(feature_cols)}; "
        f"sentiment_features={len(sentiment_cols)}; dropped_sentiment_features={len(dropped_sentiment_cols)}"
    )

    timestamps = pd.to_datetime(features[args.time_col], errors="coerce")
    X = features[feature_cols].to_numpy(dtype=float)
    y = features[target_col].to_numpy(dtype=float)

    use_xgboost = has_xgboost() and (not args.no_xgboost)
    n_trials = 4 if use_xgboost else 3
    alpha_grid = [float(v.strip()) for v in args.alpha_grid.split(",") if v.strip()]

    rows: list[dict[str, float | str | int]] = []
    tune_rows: list[pd.DataFrame] = []
    stability_rows: list[pd.DataFrame] = []
    fold_manifest: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(
        purged_kfold_splits(
            n_samples=len(y),
            n_splits=args.n_splits,
            embargo=args.embargo,
            label_horizon=args.label_horizon,
        )
    ):
        fold_manifest.append(
            fold_manifest_rows(
                fold_id=fold,
                mode="purged_kfold",
                train_idx=list(train_idx),
                eval_idx=list(val_idx),
                timestamps=timestamps,
                embargo=args.embargo,
                label_horizon=args.label_horizon,
            )
        )
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        if len(X_train) == 0 or len(X_val) == 0:
            continue

        linear_alpha = 1.0
        if args.tune_linear_alpha:
            linear_alpha, cv_df = tune_ridge_alpha_purged_cv(
                X_train,
                y_train,
                alpha_grid=alpha_grid,
                cv_config=PurgedCVConfig(
                    n_splits=min(5, max(2, args.n_splits - 1)),
                    embargo=args.embargo,
                    label_horizon=args.label_horizon,
                ),
                objective="mae",
            )
            cv_df = cv_df.copy()
            cv_df["outer_fold"] = fold
            tune_rows.append(cv_df)

        rw = RandomWalkReturnBaseline().fit(X_train, y_train)
        pred_rw = rw.predict(X_val)
        rw_std = float(np.std(y_train - rw.predict(X_train), ddof=1)) if len(y_train) > 1 else 1e-6
        rows.append({"fold": fold, "model": "random_walk", "linear_alpha": np.nan, **evaluate_model(y_val, pred_rw, rw_std, n_trials=n_trials)})

        lr = LinearLagBaseline(alpha=linear_alpha).fit(X_train, y_train)
        pred_lr = lr.predict(X_val)
        lr_std = float(np.std(y_train - lr.predict(X_train), ddof=1)) if len(y_train) > 1 else 1e-6
        rows.append({"fold": fold, "model": "linear_ridge", "linear_alpha": linear_alpha, **evaluate_model(y_val, pred_lr, lr_std, n_trials=n_trials)})
        if args.feature_stability_report:
            stab_df = permutation_importance_mae(
                predict_fn=lr.predict,
                X_val=X_val,
                y_val=y_val,
                feature_names=feature_cols,
                config=PermutationImportanceConfig(
                    n_repeats=max(1, int(args.stability_repeats)),
                    random_state=1000 + fold,
                ),
            )
            stab_df["fold"] = fold
            stab_df["model"] = "linear_ridge"
            stability_rows.append(stab_df)

        if use_xgboost:
            xgb = XGBoostLagBaseline(random_state=100 + fold).fit(X_train, y_train)
            pred_xgb = xgb.predict(X_val)
            xgb_std = float(np.std(y_train - xgb.predict(X_train), ddof=1)) if len(y_train) > 1 else 1e-6
            rows.append({"fold": fold, "model": "xgboost", "linear_alpha": np.nan, **evaluate_model(y_val, pred_xgb, xgb_std, n_trials=n_trials)})

        arima_pred = None if args.no_arima else try_arima_forecast(train_y=y_train, test_y=y_val, order=(1, 0, 1))
        if arima_pred is not None:
            arima_std = float(np.std(y_train, ddof=1)) if len(y_train) > 1 else 1e-6
            rows.append({"fold": fold, "model": "arima_101", "linear_alpha": np.nan, **evaluate_model(y_val, arima_pred, arima_std, n_trials=n_trials)})

    results = pd.DataFrame(rows)
    if results.empty:
        raise ValueError("No CPCV folds produced results. Check split settings.")

    summary = results.groupby("model").agg(["mean", "std"])
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    summary = summary.reset_index().sort_values("mae_mean", ascending=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    folds_path = output_path.with_name(output_path.stem + "_folds.csv")
    results.to_csv(folds_path, index=False)
    fold_manifest_path = output_path.with_name(output_path.stem + "_fold_manifest.csv")
    pd.DataFrame(fold_manifest).to_csv(fold_manifest_path, index=False)

    if tune_rows:
        tune_path = output_path.with_name(output_path.stem + "_tuning.csv")
        pd.concat(tune_rows, ignore_index=True).to_csv(tune_path, index=False)
        print(f"Saved tuning details: {tune_path}")

    if stability_rows:
        stability_folds_path = output_path.with_name(output_path.stem + "_feature_importance_folds.csv")
        stability_folds = pd.concat(stability_rows, ignore_index=True)
        stability_folds.to_csv(stability_folds_path, index=False)

        stability_summary = (
            stability_folds.groupby("feature", as_index=False)
            .agg(
                importance_mean=("importance_mean", "mean"),
                importance_std=("importance_mean", "std"),
                importance_abs_mean=("importance_abs_mean", "mean"),
                fold_coverage=("fold", "nunique"),
            )
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )
        stability_summary["rank_by_mean_importance"] = np.arange(1, len(stability_summary) + 1)
        stability_summary_path = output_path.with_name(output_path.stem + "_feature_importance_summary.csv")
        stability_summary.to_csv(stability_summary_path, index=False)
        print(f"Saved feature-importance folds: {stability_folds_path}")
        print(f"Saved feature-importance summary: {stability_summary_path}")
    else:
        stability_folds_path = None
        stability_summary_path = None

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
        "script": "run_cpcv_report.py",
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
        "split_method": "purged_kfold",
        "n_splits": args.n_splits,
        "embargo": args.embargo,
        "label_horizon": args.label_horizon,
        "purged_cv_tuning_enabled": bool(args.tune_linear_alpha),
        "feature_stability_report_enabled": bool(args.feature_stability_report),
        "stability_repeats": int(args.stability_repeats),
        "feature_importance_folds_path": str(stability_folds_path) if stability_folds_path else None,
        "feature_importance_summary_path": str(stability_summary_path) if stability_summary_path else None,
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
