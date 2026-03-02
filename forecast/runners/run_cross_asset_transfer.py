from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forecast.pipeline.data_pipeline import FeatureConfig, add_targets, build_feature_frame, load_market_data
from forecast.pipeline.experiment_contract import infer_frequency_label, save_metadata_json
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
from forecast.pipeline.sentiment import detect_sentiment_columns, prepare_causal_sentiment_features
from forecast.pipeline.transformers import (
    DLinearLikeRegressor,
    ITransformerLikeRegressor,
    PatchTSTLikeRegressor,
    SequenceStandardizer,
    TCNLikeRegressor,
    TrainConfig,
    build_sliding_windows,
    predict_model,
    train_model,
)


def _infer_asset_label(path: str | Path) -> str:
    name = Path(path).stem.lower()
    name = name.replace("_lunarcrush_timeseries_hourly", "")
    name = name.replace("_lunarcrash_timeseries_hourly", "")
    name = name.replace("_timeseries_hourly", "")
    name = name.replace("_timeseries", "")
    return name


def _build_model(model_name: str, lookback: int, n_features: int):
    if model_name == "dlinear_like":
        return DLinearLikeRegressor(lookback=lookback, n_features=n_features, moving_avg=max(3, lookback // 12))
    if model_name == "tcn_like":
        return TCNLikeRegressor(lookback=lookback, n_features=n_features, channels=(64, 64, 64), kernel_size=3, dropout=0.1)
    if model_name == "patchtst_like":
        return PatchTSTLikeRegressor(
            lookback=lookback,
            n_features=n_features,
            patch_len=max(4, lookback // 10),
            stride=max(2, lookback // 20),
            d_model=64,
            n_heads=4,
            n_layers=2,
            dropout=0.1,
        )
    if model_name == "itransformer_like":
        return ITransformerLikeRegressor(lookback=lookback, n_features=n_features, d_model=64, n_heads=4, n_layers=2, dropout=0.1)
    raise ValueError(f"Unknown model: {model_name}")


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, train_resid_std: float, n_trials: int) -> dict[str, float]:
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


def _prepare_features(path: str, time_col: str) -> pd.DataFrame:
    raw = load_market_data(path, time_col=time_col)
    feat = build_feature_frame(raw, config=FeatureConfig(lookbacks=(20, 60), ewma_span=20, roc_period=5), time_col=time_col)
    feat = add_targets(feat, horizons=(1, 5, 20))
    return feat


def _split_target_indices(n: int) -> tuple[int, int]:
    train_end = max(128, int(n * 0.6))
    val_end = max(train_end + 32, int(n * 0.8))
    val_end = min(val_end, n - 32)
    if train_end >= val_end:
        raise ValueError("Not enough target samples for transfer split")
    return train_end, val_end


def _split_source_indices(n: int) -> int:
    train_end = max(256, int(n * 0.85))
    train_end = min(train_end, n - 32)
    if train_end <= 0:
        raise ValueError("Not enough source samples for pretraining split")
    return train_end


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cross-asset transfer protocol (source->target) on return forecasting")
    parser.add_argument("--source-data-path", type=str, required=True)
    parser.add_argument("--target-data-path", type=str, required=True)
    parser.add_argument("--time-col", type=str, default="time")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 5, 20])
    parser.add_argument("--lookback", type=int, default=64)
    parser.add_argument("--model", type=str, default="dlinear_like")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-sentiment", action="store_true")
    parser.add_argument("--sentiment-lag", type=int, default=1)
    parser.add_argument("--sentiment-min-non-null-ratio", type=float, default=0.2)
    parser.add_argument("--output", type=str, default="results/paper/transfer_summary.csv")
    args = parser.parse_args()
    mode_name = "with_sentiment" if args.use_sentiment else "no_sentiment"
    source_asset = _infer_asset_label(args.source_data_path)
    target_asset = _infer_asset_label(args.target_data_path)

    target_col = f"target_ret_{args.horizon}d"
    base_feature_cols = [
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

    source_feat = _prepare_features(args.source_data_path, time_col=args.time_col)
    target_feat = _prepare_features(args.target_data_path, time_col=args.time_col)

    source_sent_cols: list[str] = []
    target_sent_cols: list[str] = []
    dropped_source_sent_cols: list[str] = []
    dropped_target_sent_cols: list[str] = []
    if args.use_sentiment:
        source_sent_cols = detect_sentiment_columns(source_feat, exclude_cols=set(base_feature_cols).union({target_col, "log_price"}))
        target_sent_cols = detect_sentiment_columns(target_feat, exclude_cols=set(base_feature_cols).union({target_col, "log_price"}))
        common_sent = sorted(set(source_sent_cols).intersection(set(target_sent_cols)))
        source_feat, source_common_kept, dropped_source_sent_cols = prepare_causal_sentiment_features(
            source_feat,
            common_sent,
            lag=args.sentiment_lag,
            min_non_null_ratio=args.sentiment_min_non_null_ratio,
            fill_method="ffill_zero",
        )
        target_feat, target_common_kept, dropped_target_sent_cols = prepare_causal_sentiment_features(
            target_feat,
            common_sent,
            lag=args.sentiment_lag,
            min_non_null_ratio=args.sentiment_min_non_null_ratio,
            fill_method="ffill_zero",
        )
        common_sent = sorted(set(source_common_kept).intersection(set(target_common_kept)))
    else:
        common_sent = []

    feature_cols = base_feature_cols + common_sent
    src_frame = source_feat[[args.time_col, *feature_cols, target_col]].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    tgt_frame = target_feat[[args.time_col, *feature_cols, target_col]].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    X_src, y_src = build_sliding_windows(src_frame, feature_cols=feature_cols, target_col=target_col, lookback=args.lookback)
    X_tgt, y_tgt = build_sliding_windows(tgt_frame, feature_cols=feature_cols, target_col=target_col, lookback=args.lookback)
    ts_tgt = pd.to_datetime(tgt_frame[args.time_col], errors="coerce").iloc[args.lookback - 1 :].reset_index(drop=True)

    if len(X_src) < 320 or len(X_tgt) < 256:
        raise ValueError("Not enough source/target sequences for transfer experiment")

    src_train_end = _split_source_indices(len(X_src))
    tgt_train_end, tgt_val_end = _split_target_indices(len(X_tgt))

    X_src_train, y_src_train = X_src[:src_train_end], y_src[:src_train_end]
    X_src_val, y_src_val = X_src[src_train_end:], y_src[src_train_end:]

    X_tgt_train_raw, y_tgt_train = X_tgt[:tgt_train_end], y_tgt[:tgt_train_end]
    X_tgt_val_raw, y_tgt_val = X_tgt[tgt_train_end:tgt_val_end], y_tgt[tgt_train_end:tgt_val_end]
    X_tgt_test_raw, y_tgt_test = X_tgt[tgt_val_end:], y_tgt[tgt_val_end:]
    ts_tgt_test = ts_tgt.iloc[tgt_val_end:].reset_index(drop=True)

    # Source pretraining
    src_scaler = SequenceStandardizer().fit(X_src_train)
    X_src_train_s = src_scaler.transform(X_src_train)
    X_src_val_s = src_scaler.transform(X_src_val)
    pretrained = _build_model(args.model, lookback=args.lookback, n_features=X_src_train_s.shape[-1])
    pre_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=1e-4,
        patience=3,
        loss="huber",
        seed=args.seed,
    )
    pretrained = train_model(pretrained, X_src_train_s, y_src_train, X_src_val_s, y_src_val, pre_cfg)

    # Zero-shot source->target using source scaler
    X_tgt_test_src_s = src_scaler.transform(X_tgt_test_raw)
    pred_zero_shot = predict_model(pretrained, X_tgt_test_src_s)
    src_train_pred = predict_model(pretrained, X_src_train_s)
    src_resid_std = float(np.std(y_src_train - src_train_pred, ddof=1)) if len(y_src_train) > 1 else 1e-6

    # Fine-tune on target using target scaler
    tgt_scaler = SequenceStandardizer().fit(X_tgt_train_raw)
    X_tgt_train_s = tgt_scaler.transform(X_tgt_train_raw)
    X_tgt_val_s = tgt_scaler.transform(X_tgt_val_raw)
    X_tgt_test_s = tgt_scaler.transform(X_tgt_test_raw)

    fine_model = copy.deepcopy(pretrained)
    fine_cfg = TrainConfig(
        epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=1e-4,
        patience=3,
        loss="huber",
        seed=args.seed + 1,
    )
    fine_model = train_model(fine_model, X_tgt_train_s, y_tgt_train, X_tgt_val_s, y_tgt_val, fine_cfg)
    pred_finetune = predict_model(fine_model, X_tgt_test_s)
    fine_train_pred = predict_model(fine_model, X_tgt_train_s)
    fine_resid_std = float(np.std(y_tgt_train - fine_train_pred, ddof=1)) if len(y_tgt_train) > 1 else 1e-6

    # Target-only training
    target_only_model = _build_model(args.model, lookback=args.lookback, n_features=X_tgt_train_s.shape[-1])
    tgt_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=1e-4,
        patience=3,
        loss="huber",
        seed=args.seed + 2,
    )
    target_only_model = train_model(target_only_model, X_tgt_train_s, y_tgt_train, X_tgt_val_s, y_tgt_val, tgt_cfg)
    pred_target_only = predict_model(target_only_model, X_tgt_test_s)
    tgt_train_pred = predict_model(target_only_model, X_tgt_train_s)
    tgt_resid_std = float(np.std(y_tgt_train - tgt_train_pred, ddof=1)) if len(y_tgt_train) > 1 else 1e-6

    # Random walk sequence baseline
    ret_idx = feature_cols.index("return_1d")
    pred_rw = X_tgt_test_raw[:, -1, ret_idx]
    rw_train_pred = X_tgt_train_raw[:, -1, ret_idx]
    rw_resid_std = float(np.std(y_tgt_train - rw_train_pred, ddof=1)) if len(y_tgt_train) > 1 else 1e-6

    n_trials = 4
    rows = [
        {"mode": "source_zero_shot", **evaluate_model(y_tgt_test, pred_zero_shot, src_resid_std, n_trials=n_trials)},
        {"mode": "source_finetuned", **evaluate_model(y_tgt_test, pred_finetune, fine_resid_std, n_trials=n_trials)},
        {"mode": "target_only", **evaluate_model(y_tgt_test, pred_target_only, tgt_resid_std, n_trials=n_trials)},
        {"mode": "random_walk_sequence", **evaluate_model(y_tgt_test, pred_rw, rw_resid_std, n_trials=n_trials)},
    ]

    summary = pd.DataFrame(rows)
    dm_rows: list[dict[str, float | str]] = []
    for compare_mode, pred in {
        "source_zero_shot": pred_zero_shot,
        "source_finetuned": pred_finetune,
        "random_walk_sequence": pred_rw,
    }.items():
        dm = diebold_mariano_test(y_tgt_test, pred, pred_target_only, power=2, horizon=args.horizon)
        dm_rows.append({"mode": compare_mode, "dm_vs_target_only_stat": dm["dm_stat"], "dm_vs_target_only_pvalue": dm["p_value"]})
    dm_df = pd.DataFrame(dm_rows)
    summary = summary.merge(dm_df, on="mode", how="left")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    pred_df = pd.DataFrame(
        {
            "time": ts_tgt_test.astype(str),
            "y_true": y_tgt_test,
            "pred_source_zero_shot": pred_zero_shot,
            "pred_source_finetuned": pred_finetune,
            "pred_target_only": pred_target_only,
            "pred_random_walk_sequence": pred_rw,
        }
    )
    preds_path = output_path.with_name(output_path.stem + "_predictions.csv")
    pred_df.to_csv(preds_path, index=False)
    pred_long_rows: list[dict[str, float | str | int]] = []
    for model_name, pred_arr in {
        "source_zero_shot": pred_zero_shot,
        "source_finetuned": pred_finetune,
        "target_only": pred_target_only,
        "random_walk_sequence": pred_rw,
    }.items():
        for ts, yt, yp in zip(ts_tgt_test.astype(str).tolist(), y_tgt_test.tolist(), pred_arr.tolist()):
            pred_long_rows.append(
                {
                    "timestamp": ts,
                    "y_true": float(yt),
                    "y_pred": float(yp),
                    "fold": 0,
                    "model": model_name,
                    "mode": mode_name,
                    "asset": target_asset,
                    "source_asset": source_asset,
                }
            )
    preds_long_path = output_path.with_name(output_path.stem + "_predictions_long.csv")
    pd.DataFrame(pred_long_rows).to_csv(preds_long_path, index=False)

    metadata = {
        "script": "run_cross_asset_transfer.py",
        "source_data_path": args.source_data_path,
        "target_data_path": args.target_data_path,
        "time_col": args.time_col,
        "target_space": "log_return",
        "target_col": target_col,
        "direction_target_col": f"target_dir_{args.horizon}d",
        "horizon_days": args.horizon,
        "lookback": args.lookback,
        "model": args.model,
        "use_sentiment": bool(args.use_sentiment),
        "common_sentiment_columns": common_sent,
        "dropped_source_sentiment_columns": dropped_source_sent_cols,
        "dropped_target_sentiment_columns": dropped_target_sent_cols,
        "sentiment_min_non_null_ratio": args.sentiment_min_non_null_ratio if args.use_sentiment else None,
        "sentiment_lag": args.sentiment_lag if args.use_sentiment else 0,
        "feature_cols": feature_cols,
        "source_sequences": int(len(X_src)),
        "target_sequences": int(len(X_tgt)),
        "source_train_end": int(src_train_end),
        "target_train_end": int(tgt_train_end),
        "target_val_end": int(tgt_val_end),
        "target_test_size": int(len(y_tgt_test)),
        "target_test_time_start": str(ts_tgt_test.min()),
        "target_test_time_end": str(ts_tgt_test.max()),
        "target_inferred_frequency": infer_frequency_label(ts_tgt),
        "transfer_protocol": "pretrain_source_then_zero_shot_or_finetune_on_target",
        "predictions_path": str(preds_path),
        "predictions_long_path": str(preds_long_path),
        "source_asset": source_asset,
        "target_asset": target_asset,
        "mode": mode_name,
    }
    metadata_path = output_path.with_name(output_path.stem + "_metadata.json")
    save_metadata_json(metadata_path, metadata)

    print(f"Saved transfer summary: {output_path}")
    print(f"Saved transfer predictions: {preds_path}")
    print(f"Saved transfer predictions (long): {preds_long_path}")
    print(f"Saved transfer metadata: {metadata_path}")


if __name__ == "__main__":
    main()
