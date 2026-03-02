from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

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
from forecast.pipeline.sentiment import detect_sentiment_columns, prepare_causal_sentiment_features
from forecast.pipeline.splits import walk_forward_splits
from forecast.pipeline.transformer_official_backends import (
    CanonicalBackendConfig,
    build_canonical_model,
    canonical_model_name,
    config_provenance,
    parse_canonical_models,
)
from forecast.pipeline.transformers import (
    SequenceStandardizer,
    TrainConfig,
    build_sliding_windows,
    predict_model,
    train_model,
)


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(v.strip()) for v in value.split(",") if v.strip())


def _resolve_objective_tracks(objective_track: str) -> tuple[str, ...]:
    if objective_track == "both":
        return ("point", "quantile")
    if objective_track not in {"point", "quantile"}:
        raise ValueError("objective_track must be one of: point, quantile, both")
    return (objective_track,)


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
    objective_track: str,
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
                "objective_track": objective_track,
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
                "objective_track": objective_track,
            }
        )


def _append_history_records(
    rows_hist: list[dict[str, float | str | int]],
    *,
    history_rows: list[dict[str, float | int]],
    fold: int,
    model: str,
    mode: str,
    asset: str,
    objective_track: str,
) -> None:
    for row in history_rows:
        rows_hist.append(
            {
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "val_loss": float(row["val_loss"]),
                "fold": int(fold),
                "model": model,
                "mode": mode,
                "asset": asset,
                "objective_track": objective_track,
            }
        )


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_resid_std: float,
    n_trials: int,
    pred_q10: np.ndarray | None = None,
    pred_q50: np.ndarray | None = None,
    pred_q90: np.ndarray | None = None,
) -> dict[str, float]:
    if pred_q10 is None or pred_q50 is None or pred_q90 is None:
        sigma = np.full_like(y_pred, fill_value=max(train_resid_std, 1e-6), dtype=float)
        q10 = y_pred - 1.28155 * sigma
        q90 = y_pred + 1.28155 * sigma
        q50 = y_pred
    else:
        q10 = pred_q10
        q50 = pred_q50
        q90 = pred_q90
        sigma = np.maximum((q90 - q10) / (2.0 * 1.28155), 1e-6)

    metrics = {
        "mae": mae(y_true, q50),
        "rmse": rmse(y_true, q50),
        "directional_accuracy": directional_accuracy(y_true, q50),
        "pinball_q10": pinball_loss(y_true, q10, q=0.1),
        "pinball_q50": pinball_loss(y_true, q50, q=0.5),
        "pinball_q90": pinball_loss(y_true, q90, q=0.9),
        "crps_gaussian": crps_gaussian(y_true, q50, sigma),
        "coverage_80": interval_coverage(y_true, q10, q90),
        "interval_width_80": interval_width(q10, q90),
        "wis_80": weighted_interval_score(y_true, q10, q90, alpha=0.2),
    }
    metrics.update(backtest_metrics(y_true, q50, cost_bps=5.0))
    metrics.update(cost_sensitivity_metrics(y_true, q50, cost_bps_list=(0.0, 5.0, 10.0, 20.0, 50.0), n_trials=n_trials))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run canonical transformer benchmark (official-backend proxy API)")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--time-col", type=str, default="time")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 5, 20])
    parser.add_argument("--lookback", type=int, default=64)
    parser.add_argument("--min-train-size", type=int, default=365)
    parser.add_argument("--test-size", type=int, default=60)
    parser.add_argument("--step-size", type=int, default=180)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--models", type=str, default="itransformer,patchtst")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patch-len", type=int, default=0)
    parser.add_argument("--stride", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--objective-track", type=str, default="point", choices=["point", "quantile", "both"])
    parser.add_argument("--point-loss", type=str, default="mse", choices=["mse", "mae"])
    parser.add_argument("--quantiles", type=str, default="0.1,0.5,0.9")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-regime-features", action="store_true")
    parser.add_argument("--regime-lookbacks", type=str, default="20,60")
    parser.add_argument("--use-sentiment", action="store_true")
    parser.add_argument("--sentiment-lag", type=int, default=1)
    parser.add_argument("--sentiment-min-non-null-ratio", type=float, default=0.2)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--save-history", action="store_true")
    parser.add_argument("--output", type=str, default="results/transformers_canonical_summary.csv")
    args = parser.parse_args()

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
    features_pre_sentiment = features.copy()

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
        for lb in regime_lookbacks:
            feature_cols.extend([f"realized_vol_{lb}", f"vol_z_{lb}", f"vol_regime_{lb}"])

    dropped_sentiment_cols: list[str] = []
    if args.use_sentiment:
        raw_sentiment_cols = detect_sentiment_columns(
            features,
            exclude_cols=set(feature_cols).union({target_col, "log_price"}),
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

    required_cols = [args.time_col, *feature_cols, target_col]
    frame_before_sentiment = features_pre_sentiment[required_cols].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    frame_after_sentiment = features_post_sentiment[required_cols].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    frame_before_dropna = frame_after_sentiment.copy()
    frame = frame_before_dropna.dropna().reset_index(drop=True)
    if frame.empty:
        raise ValueError("No rows left after preprocessing")

    mode_name = "with_sentiment" if args.use_sentiment else "no_sentiment"
    asset_label = _infer_asset_label(args.data_path)

    X_seq, y = build_sliding_windows(frame, feature_cols=feature_cols, target_col=target_col, lookback=args.lookback)
    seq_timestamps = pd.to_datetime(frame[args.time_col], errors="coerce").iloc[args.lookback - 1 :].reset_index(drop=True)
    if len(y) < args.min_train_size + args.test_size:
        raise ValueError("Not enough samples for requested split settings")

    model_families = parse_canonical_models(args.models)
    objective_tracks = _resolve_objective_tracks(args.objective_track)
    quantiles = _parse_float_tuple(args.quantiles)
    if "quantile" in objective_tracks:
        if set(round(q, 4) for q in quantiles) != {0.1, 0.5, 0.9}:
            raise ValueError("--quantiles must be exactly 0.1,0.5,0.9")
        q10_idx = quantiles.index(0.1)
        q50_idx = quantiles.index(0.5)
        q90_idx = quantiles.index(0.9)
    else:
        q10_idx = q50_idx = q90_idx = -1

    canonical_cfg = {
        model_family: CanonicalBackendConfig(
            model_family=model_family,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            dropout=args.dropout,
            lr=args.lr,
            patch_len=args.patch_len if args.patch_len > 0 else None,
            stride=args.stride if args.stride > 0 else None,
        ).validated()
        for model_family in model_families
    }
    model_names = [canonical_model_name(m) for m in model_families]
    provenance_by_model = {canonical_model_name(m): config_provenance(cfg) for m, cfg in canonical_cfg.items()}

    n_trials = len(model_names) + 1
    ret_idx = feature_cols.index("return_1d")
    all_rows: list[dict[str, float | str | int]] = []
    fold_manifest: list[dict] = []
    preds_by_track_model: dict[tuple[str, str], list[float]] = {
        (track, model): []
        for track in objective_tracks
        for model in [*model_names, "random_walk_sequence"]
    }
    y_all_by_track: dict[str, list[float]] = {track: [] for track in objective_tracks}
    prediction_rows: list[dict[str, float | str | int]] = []
    quantile_rows: list[dict[str, float | str | int]] = []
    history_rows_all: list[dict[str, float | str | int]] = []

    for fold, (train_idx, test_idx) in enumerate(
        walk_forward_splits(
            n_samples=len(y),
            min_train_size=args.min_train_size,
            test_size=args.test_size,
            step_size=args.step_size,
            expanding=True,
        )
    ):
        X_train_raw, y_train = X_seq[train_idx], y[train_idx]
        X_test_raw, y_test = X_seq[test_idx], y[test_idx]
        ts_test = seq_timestamps.iloc[test_idx].reset_index(drop=True)
        fold_manifest.append(
            fold_manifest_rows(
                fold_id=fold,
                mode="walk_forward",
                train_idx=train_idx,
                eval_idx=test_idx,
                timestamps=seq_timestamps,
            )
        )

        scaler = SequenceStandardizer().fit(X_train_raw)
        X_train = scaler.transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        val_size = max(32, int(len(X_train) * 0.2))
        if len(X_train) - val_size < 32:
            val_size = max(16, len(X_train) // 5)
        if len(X_train) - val_size < 16:
            continue

        X_subtrain, y_subtrain = X_train[:-val_size], y_train[:-val_size]
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]
        pred_rw = X_test_raw[:, -1, ret_idx]
        rw_train_pred = X_train_raw[:, -1, ret_idx]
        rw_resid_std = float(np.std(y_train - rw_train_pred, ddof=1)) if len(y_train) > 1 else 1e-6

        for objective_track in objective_tracks:
            rw_metrics = evaluate_model(y_test, pred_rw, rw_resid_std, n_trials=n_trials)
            all_rows.append({"fold": fold, "model": "random_walk_sequence", "objective_track": objective_track, **rw_metrics})
            preds_by_track_model[(objective_track, "random_walk_sequence")].extend(pred_rw.tolist())

            if args.save_predictions:
                rw_q10, rw_q90 = _gaussian_q10_q90(pred_rw, rw_resid_std)
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
                    model="random_walk_sequence",
                    mode=mode_name,
                    asset=asset_label,
                    objective_track=objective_track,
                )

            for model_family in model_families:
                cfg = canonical_cfg[model_family]
                model_name = canonical_model_name(model_family)
                output_dim = len(quantiles) if objective_track == "quantile" else 1
                model = build_canonical_model(
                    cfg,
                    lookback=args.lookback,
                    n_features=X_train.shape[-1],
                    output_dim=output_dim,
                )
                train_cfg = TrainConfig(
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=cfg.lr,
                    weight_decay=1e-4,
                    patience=3,
                    loss=args.point_loss if objective_track == "point" else "mse",
                    seed=args.seed + fold + (10_000 if objective_track == "quantile" else 0),
                    quantiles=quantiles if objective_track == "quantile" else None,
                    track_history=args.save_history,
                )
                model = train_model(model, X_subtrain, y_subtrain, X_val, y_val, train_cfg)

                if args.save_history:
                    _append_history_records(
                        history_rows_all,
                        history_rows=getattr(model, "_training_history", []),
                        fold=fold,
                        model=model_name,
                        mode=mode_name,
                        asset=asset_label,
                        objective_track=objective_track,
                    )

                raw_pred = predict_model(model, X_test)
                if objective_track == "quantile":
                    if raw_pred.ndim != 2 or raw_pred.shape[1] != len(quantiles):
                        raise ValueError("Quantile mode expected [N, 3] model predictions")
                    pred_q10 = raw_pred[:, q10_idx]
                    pred_q50 = raw_pred[:, q50_idx]
                    pred_q90 = raw_pred[:, q90_idx]
                    pred = pred_q50
                else:
                    pred = raw_pred

                train_pred = predict_model(model, X_subtrain)
                if objective_track == "quantile" and train_pred.ndim == 2:
                    resid_base = train_pred[:, q50_idx]
                else:
                    resid_base = train_pred
                resid_std = float(np.std(y_subtrain - resid_base, ddof=1)) if len(y_subtrain) > 1 else 1e-6

                if objective_track == "quantile":
                    metrics = evaluate_model(
                        y_test,
                        pred,
                        resid_std,
                        n_trials=n_trials,
                        pred_q10=pred_q10,
                        pred_q50=pred_q50,
                        pred_q90=pred_q90,
                    )
                    q10 = pred_q10
                    q50 = pred_q50
                    q90 = pred_q90
                else:
                    metrics = evaluate_model(y_test, pred, resid_std, n_trials=n_trials)
                    q10, q90 = _gaussian_q10_q90(pred, resid_std)
                    q50 = pred

                row = {"fold": fold, "model": model_name, "objective_track": objective_track, **metrics}
                row.update(provenance_by_model[model_name])
                all_rows.append(row)
                preds_by_track_model[(objective_track, model_name)].extend(pred.tolist())

                if args.save_predictions:
                    _append_prediction_records(
                        prediction_rows,
                        quantile_rows,
                        timestamps=ts_test,
                        y_true=y_test,
                        y_pred=pred,
                        q10=q10,
                        q50=q50,
                        q90=q90,
                        fold=fold,
                        model=model_name,
                        mode=mode_name,
                        asset=asset_label,
                        objective_track=objective_track,
                    )

            y_all_by_track[objective_track].extend(y_test.tolist())

    results = pd.DataFrame(all_rows)
    if results.empty:
        raise ValueError("No folds produced canonical transformer metrics")

    metric_cols = [
        c
        for c in results.columns
        if c not in {"model", "objective_track"}
        and pd.api.types.is_numeric_dtype(results[c])
    ]
    summary = results.groupby(["model", "objective_track"])[metric_cols].agg(["mean", "std"])
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    summary = summary.reset_index()
    summary["dm_vs_rw_stat"] = np.nan
    summary["dm_vs_rw_pvalue"] = np.nan

    for objective_track in objective_tracks:
        y_arr = np.asarray(y_all_by_track[objective_track], dtype=float)
        rw_arr = np.asarray(preds_by_track_model[(objective_track, "random_walk_sequence")], dtype=float)
        for model_name in model_names:
            pred_arr = np.asarray(preds_by_track_model[(objective_track, model_name)], dtype=float)
            dm = diebold_mariano_test(y_arr, pred_arr, rw_arr, power=2, horizon=args.horizon)
            mask = (summary["model"] == model_name) & (summary["objective_track"] == objective_track)
            summary.loc[mask, "dm_vs_rw_stat"] = dm["dm_stat"]
            summary.loc[mask, "dm_vs_rw_pvalue"] = dm["p_value"]

    for model_name in model_names:
        for key, value in provenance_by_model[model_name].items():
            summary.loc[summary["model"] == model_name, key] = value

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    folds_path = output_path.with_name(output_path.stem + "_folds.csv")
    results.to_csv(folds_path, index=False)
    fold_manifest_path = output_path.with_name(output_path.stem + "_fold_manifest.csv")
    pd.DataFrame(fold_manifest).to_csv(fold_manifest_path, index=False)

    predictions_path = None
    quantiles_path = None
    history_path = None
    if args.save_predictions and prediction_rows:
        predictions_path = output_path.with_name(output_path.stem + "_predictions.csv")
        quantiles_path = output_path.with_name(output_path.stem + "_quantiles.csv")
        pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
        pd.DataFrame(quantile_rows).to_csv(quantiles_path, index=False)
    if args.save_history and history_rows_all:
        history_path = output_path.with_name(output_path.stem + "_history.csv")
        pd.DataFrame(history_rows_all).to_csv(history_path, index=False)

    quality_summary_path, missingness_stages_path = save_preprocessing_quality_artifacts(
        output_path,
        raw_df=raw,
        frame_before_sentiment=frame_before_sentiment,
        frame_after_sentiment=frame_after_sentiment,
        frame_before_dropna=frame_before_dropna,
        frame_after_dropna=frame,
        time_col=args.time_col,
        target_col=target_col,
        feature_cols=feature_cols,
        sentiment_cols=sentiment_cols,
        sentiment_lag=args.sentiment_lag,
    )

    metadata = {
        "script": "run_transformers_canonical.py",
        "data_path": args.data_path,
        "time_col": args.time_col,
        "target_space": "log_return",
        "target_col": target_col,
        "direction_target_col": f"target_dir_{args.horizon}d",
        "horizon_days": args.horizon,
        "lookback": args.lookback,
        "feature_mode": "financial_plus_sentiment" if args.use_sentiment else "financial_only",
        "use_regime_features": bool(args.use_regime_features),
        "regime_lookbacks": list(regime_lookbacks),
        "sentiment_feature_count": len(sentiment_cols),
        "sentiment_columns": sentiment_cols,
        "dropped_sentiment_feature_count": len(dropped_sentiment_cols),
        "dropped_sentiment_columns": dropped_sentiment_cols,
        "sentiment_min_non_null_ratio": args.sentiment_min_non_null_ratio if args.use_sentiment else None,
        "sentiment_lag": args.sentiment_lag if args.use_sentiment else 0,
        "raw_rows": raw_rows,
        "model_rows": len(frame),
        "sequence_rows": len(y),
        "time_start": str(seq_timestamps.min()),
        "time_end": str(seq_timestamps.max()),
        "inferred_frequency": infer_frequency_label(seq_timestamps),
        "split_method": "walk_forward_expanding",
        "min_train_size": args.min_train_size,
        "test_size": args.test_size,
        "step_size": args.step_size,
        "models": model_names,
        "model_families": model_families,
        "objective_tracks": list(objective_tracks),
        "point_loss": args.point_loss,
        "quantiles": list(quantiles) if "quantile" in objective_tracks else None,
        "canonical_config": {m: provenance_by_model[canonical_model_name(m)] for m in model_families},
        "asset": asset_label,
        "mode": mode_name,
        "predictions_path": str(predictions_path) if predictions_path else None,
        "quantiles_path": str(quantiles_path) if quantiles_path else None,
        "history_path": str(history_path) if history_path else None,
        "prediction_rows": len(prediction_rows) if prediction_rows else 0,
        "quantile_rows": len(quantile_rows) if quantile_rows else 0,
        "history_rows": len(history_rows_all) if history_rows_all else 0,
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
