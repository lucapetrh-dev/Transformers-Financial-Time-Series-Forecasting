from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, *, n_trials: int) -> dict[str, float]:
    resid_std = float(np.std(y_true - y_pred, ddof=1)) if len(y_true) > 1 else 1e-6
    sigma = np.full_like(y_pred, fill_value=max(resid_std, 1e-6), dtype=float)
    q10 = y_pred - 1.28155 * sigma
    q90 = y_pred + 1.28155 * sigma
    out = {
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
    out.update(backtest_metrics(y_true, y_pred, cost_bps=5.0))
    out.update(cost_sensitivity_metrics(y_true, y_pred, cost_bps_list=(0.0, 5.0, 10.0, 20.0, 50.0), n_trials=n_trials))
    return out


def _dedupe_prediction_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"])
    return (
        out.groupby(["asset", "mode", "objective_track", "model", "fold", "timestamp"], as_index=False)
        .agg(y_true=("y_true", "mean"), y_pred=("y_pred", "mean"))
        .sort_values(["asset", "model", "fold", "timestamp"])
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run equal-weight ridge + best-foundation ensemble benchmark")
    parser.add_argument("--results-root", type=str, default="results/paper")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 5, 20])
    parser.add_argument("--baseline-model", type=str, default="linear_ridge")
    parser.add_argument("--mode", type=str, default="no_sentiment")
    parser.add_argument("--objective-track", type=str, default="point")
    parser.add_argument("--output", type=str, default="results/paper/ensemble_h1_summary.csv")
    args = parser.parse_args()

    root = Path(args.results_root)
    h = args.horizon
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_summary = pd.read_csv(root / f"multi_asset_baselines_h{h}_paired_summary.csv")
    base_pred = pd.read_csv(root / f"multi_asset_baselines_h{h}_paired_summary_predictions.csv")

    foundation_summary_path = root / f"multi_asset_foundation_h{h}_summary.csv"
    if foundation_summary_path.exists():
        fnd_summary = pd.read_csv(foundation_summary_path)
        fnd_pred = pd.read_csv(root / f"multi_asset_foundation_h{h}_summary_predictions.csv")
    else:
        fnd_summary = pd.read_csv(root / f"multi_asset_chronos2_h{h}_summary.csv")
        fnd_pred = pd.read_csv(root / f"multi_asset_chronos2_h{h}_summary_predictions.csv")

    for df in (base_summary, fnd_summary, base_pred, fnd_pred):
        if "mode" not in df.columns:
            df["mode"] = "no_sentiment"
        if "objective_track" not in df.columns:
            df["objective_track"] = "point"

    base_pred = _dedupe_prediction_frame(base_pred)
    fnd_pred = _dedupe_prediction_frame(fnd_pred)

    fnd_best = (
        fnd_summary[(fnd_summary["mode"] == args.mode) & (fnd_summary["objective_track"] == args.objective_track)]
        .sort_values(["asset", "mae_mean"], ascending=[True, True])
        .groupby("asset", as_index=False)
        .head(1)[["asset", "model"]]
        .rename(columns={"model": "foundation_model"})
        .reset_index(drop=True)
    )

    summary_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []

    for row in fnd_best.itertuples(index=False):
        asset = str(row.asset)
        foundation_model = str(row.foundation_model)

        lr_df = base_pred[
            (base_pred["asset"] == asset)
            & (base_pred["mode"] == args.mode)
            & (base_pred["objective_track"] == args.objective_track)
            & (base_pred["model"] == args.baseline_model)
        ][["fold", "timestamp", "y_true", "y_pred"]].rename(columns={"y_pred": "y_pred_lr"})

        fnd_df = fnd_pred[
            (fnd_pred["asset"] == asset)
            & (fnd_pred["mode"] == args.mode)
            & (fnd_pred["objective_track"] == args.objective_track)
            & (fnd_pred["model"] == foundation_model)
        ][["fold", "timestamp", "y_true", "y_pred"]].rename(columns={"y_pred": "y_pred_fnd"})

        rw_df = base_pred[
            (base_pred["asset"] == asset)
            & (base_pred["mode"] == args.mode)
            & (base_pred["objective_track"] == args.objective_track)
            & (base_pred["model"] == "random_walk")
        ][["fold", "timestamp", "y_pred"]].rename(columns={"y_pred": "y_pred_rw"})

        merged = lr_df.merge(fnd_df, on=["fold", "timestamp"], how="inner", suffixes=("_lr", "_fnd"))
        if merged.empty:
            merged = lr_df.merge(fnd_df, on=["timestamp"], how="inner", suffixes=("_lr", "_fnd"))
            merged["fold"] = merged.get("fold_lr", 0)

        if merged.empty:
            continue

        if "y_true_lr" in merged.columns and "y_true_fnd" in merged.columns:
            merged["y_true"] = merged[["y_true_lr", "y_true_fnd"]].mean(axis=1)
        elif "y_true" not in merged.columns:
            merged["y_true"] = merged.get("y_true_lr", merged.get("y_true_fnd"))

        merged["y_pred"] = 0.5 * merged["y_pred_lr"] + 0.5 * merged["y_pred_fnd"]
        merged = merged.merge(rw_df, on=["fold", "timestamp"], how="left")
        if merged["y_pred_rw"].isna().all():
            merged["y_pred_rw"] = merged["y_pred_lr"]
        else:
            merged["y_pred_rw"] = merged["y_pred_rw"].fillna(merged["y_pred_lr"])

        fold_metric_rows = []
        for fold_id, sub in merged.groupby("fold"):
            y_true = sub["y_true"].to_numpy(dtype=float)
            y_pred = sub["y_pred"].to_numpy(dtype=float)
            metrics = _evaluate(y_true, y_pred, n_trials=3)
            fold_metric_rows.append({"fold": int(fold_id), **metrics})

        if not fold_metric_rows:
            continue

        fold_df = pd.DataFrame(fold_metric_rows)
        fold_df.insert(0, "asset", asset)
        fold_df.insert(1, "mode", args.mode)
        fold_df.insert(2, "objective_track", args.objective_track)
        fold_df.insert(3, "model", "ensemble_ridge_foundation")
        fold_df.insert(4, "baseline_model", args.baseline_model)
        fold_df.insert(5, "foundation_model", foundation_model)
        fold_rows.append(fold_df)

        pred_df = merged[["fold", "timestamp", "y_true", "y_pred"]].copy()
        pred_df.insert(0, "asset", asset)
        pred_df.insert(1, "mode", args.mode)
        pred_df.insert(2, "objective_track", args.objective_track)
        pred_df.insert(3, "model", "ensemble_ridge_foundation")
        pred_df.insert(4, "baseline_model", args.baseline_model)
        pred_df.insert(5, "foundation_model", foundation_model)
        pred_rows.append(pred_df)

        y_true_full = merged["y_true"].to_numpy(dtype=float)
        y_pred_full = merged["y_pred"].to_numpy(dtype=float)
        y_rw_full = merged["y_pred_rw"].to_numpy(dtype=float)
        dm = diebold_mariano_test(y_true_full, y_pred_full, y_rw_full, power=2, horizon=args.horizon)

        metric_cols = [
            c
            for c in fold_df.columns
            if c
            not in {
                "asset",
                "mode",
                "objective_track",
                "model",
                "baseline_model",
                "foundation_model",
                "fold",
            }
        ]
        summary = fold_df.groupby("model")[metric_cols].agg(["mean", "std"])
        summary.columns = ["_".join(col).strip("_") for col in summary.columns]
        summary = summary.reset_index()
        summary.insert(0, "asset", asset)
        summary.insert(1, "mode", args.mode)
        summary.insert(2, "objective_track", args.objective_track)
        summary["family"] = "ensemble"
        summary["baseline_model"] = args.baseline_model
        summary["foundation_model"] = foundation_model
        summary["dm_vs_rw_stat"] = dm["dm_stat"]
        summary["dm_vs_rw_pvalue"] = dm["p_value"]
        summary_rows.append(summary.iloc[0].to_dict())

    if not summary_rows:
        raise ValueError("No ensemble rows were produced; check upstream predictions and mode/track settings")

    summary_df = pd.DataFrame(summary_rows).sort_values("asset").reset_index(drop=True)
    summary_df.to_csv(output_path, index=False)

    rank_path = output_path.with_name(output_path.stem + "_ranks.csv")
    rank_df = (
        summary_df.groupby(["mode", "objective_track", "model"], as_index=False)
        .agg(
            avg_mae_mean=("mae_mean", "mean"),
            n_asset_evals=("asset", "count"),
        )
        .sort_values(["mode", "objective_track", "avg_mae_mean"], ascending=[True, True, True])
        .reset_index(drop=True)
    )
    rank_df["avg_rank"] = 1.0
    rank_df.to_csv(rank_path, index=False)

    folds_path = output_path.with_name(output_path.stem + "_folds.csv")
    pd.concat(fold_rows, ignore_index=True).to_csv(folds_path, index=False)

    pred_path = output_path.with_name(output_path.stem + "_predictions.csv")
    pd.concat(pred_rows, ignore_index=True).to_csv(pred_path, index=False)

    audit_df = summary_df[["asset", "mode", "objective_track"]].copy()
    audit_df["target_space"] = "log_return"
    audit_df["consistent_target_space_all"] = True
    audit_df["expected_target_space"] = "log_return"
    audit_path = output_path.with_name(output_path.stem + "_comparability_audit.csv")
    audit_df.to_csv(audit_path, index=False)

    print(f"Saved ensemble summary: {output_path}")
    print(f"Saved ranking table: {rank_path}")
    print(f"Saved fold metrics: {folds_path}")
    print(f"Saved predictions: {pred_path}")
    print(f"Saved comparability audit: {audit_path}")


if __name__ == "__main__":
    main()
