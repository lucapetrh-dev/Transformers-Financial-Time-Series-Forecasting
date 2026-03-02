from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forecast.pipeline.metrics import backtest_metrics, deflated_sharpe_ratio, strategy_returns
from forecast.pipeline.statistical_inference import (
    exact_binomial_directional_test,
    model_confidence_set,
    moving_block_bootstrap_ci,
)


def _load_predictions(results_root: Path, horizon: int) -> pd.DataFrame:
    paths = [
        results_root / f"multi_asset_baselines_h{horizon}_paired_summary_predictions.csv",
        results_root / f"multi_asset_transformers_h{horizon}_paired_summary_predictions.csv",
    ]
    foundation_pred = results_root / f"multi_asset_foundation_h{horizon}_summary_predictions.csv"
    if foundation_pred.exists():
        paths.append(foundation_pred)
    else:
        paths.append(results_root / f"multi_asset_chronos2_h{horizon}_summary_predictions.csv")

    frames = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing predictions file: {p}")
        frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    if "mode" not in df.columns:
        df["mode"] = "no_sentiment"
    else:
        df["mode"] = df["mode"].fillna("no_sentiment")
    if "objective_track" not in df.columns:
        df["objective_track"] = "point"
    else:
        df["objective_track"] = df["objective_track"].fillna("point")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values(["asset", "mode", "objective_track", "model", "timestamp"]).reset_index(drop=True)
    return df


def _bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bootstrap: int,
    block_size: int,
    seed: int,
) -> dict[str, float]:
    metric_fns = {
        "mae": lambda yt, yp: float(np.mean(np.abs(yt - yp))),
        "rmse": lambda yt, yp: float(np.sqrt(np.mean((yt - yp) ** 2))),
        "directional_accuracy": lambda yt, yp: float(np.mean(np.sign(yt) == np.sign(yp))),
        "sharpe_5bps": lambda yt, yp: float(backtest_metrics(yt, yp, cost_bps=5.0)["sharpe"]),
        "dsr": lambda yt, yp: float(deflated_sharpe_ratio(strategy_returns(yt, yp, cost_bps=5.0)[0], n_trials=1)),
    }

    out: dict[str, float] = {}
    for metric_name, fn in metric_fns.items():
        ci = moving_block_bootstrap_ci(
            y_true,
            y_pred,
            metric_fn=fn,
            n_bootstrap=n_bootstrap,
            block_size=block_size,
            alpha=0.05,
            seed=seed,
        )
        out[metric_name] = ci["estimate"]
        out[f"{metric_name}_ci95_low"] = ci["ci_low"]
        out[f"{metric_name}_ci95_high"] = ci["ci_high"]
    return out


def _overlap_components(pivot: pd.DataFrame, *, min_overlap: int = 2) -> list[list[str]]:
    models = [str(c) for c in pivot.columns.tolist()]
    if len(models) < 2:
        return []

    present_idx = {m: set(pivot.index[pivot[m].notna()].tolist()) for m in models}
    adjacency: dict[str, set[str]] = {m: set() for m in models}
    for i, m1 in enumerate(models):
        for m2 in models[i + 1 :]:
            if len(present_idx[m1].intersection(present_idx[m2])) >= min_overlap:
                adjacency[m1].add(m2)
                adjacency[m2].add(m1)

    components: list[list[str]] = []
    visited: set[str] = set()
    for start in models:
        if start in visited:
            continue
        stack = [start]
        comp: list[str] = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.append(cur)
            for nxt in adjacency[cur]:
                if nxt not in visited:
                    stack.append(nxt)
        if len(comp) >= 2:
            components.append(sorted(comp))
    components.sort(key=lambda c: (-len(c), c))
    return components


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute bootstrap CIs, directional binomial tests, and MCS tables")
    parser.add_argument("--results-root", type=str, default="results/paper")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 5, 20])
    parser.add_argument("--output-prefix", type=str, default="PAPER_H1")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-block-size", type=int, default=20)
    parser.add_argument("--mcs-bootstrap-samples", type=int, default=500)
    parser.add_argument("--mcs-block-size", type=int, default=20)
    parser.add_argument("--mcs-alpha", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.results_root)
    pred_df = _load_predictions(root, args.horizon)

    ci_rows: list[dict[str, object]] = []
    binom_rows: list[dict[str, object]] = []
    mcs_rows: list[dict[str, object]] = []

    group_cols = ["asset", "mode", "objective_track", "model"]
    for keys, sub in pred_df.groupby(group_cols, dropna=False):
        asset, mode, objective_track, model = keys
        series = (
            sub[["timestamp", "y_true", "y_pred"]]
            .sort_values("timestamp")
            .groupby("timestamp", as_index=False)
            .mean(numeric_only=True)
        )
        y_true = series["y_true"].to_numpy(dtype=float)
        y_pred = series["y_pred"].to_numpy(dtype=float)
        if len(y_true) < 2:
            continue

        ci = _bootstrap_metrics(
            y_true,
            y_pred,
            n_bootstrap=args.bootstrap_samples,
            block_size=args.bootstrap_block_size,
            seed=args.seed,
        )
        ci_rows.append(
            {
                "asset": asset,
                "mode": mode,
                "objective_track": objective_track,
                "model": model,
                "n_obs": int(len(y_true)),
                **ci,
            }
        )

        b = exact_binomial_directional_test(y_true, y_pred, p_null=0.5)
        binom_rows.append(
            {
                "asset": asset,
                "mode": mode,
                "objective_track": objective_track,
                "model": model,
                "n_obs": int(b["n_obs"]),
                "n_success": int(b["n_success"]),
                "hit_rate": float(b["hit_rate"]),
                "p_value": float(b["p_value"]),
            }
        )

    mcs_group_cols = ["asset", "mode", "objective_track"]
    for (asset, mode, objective_track), sub in pred_df.groupby(mcs_group_cols, dropna=False):
        aligned = (
            sub[["timestamp", "model", "y_true", "y_pred"]]
            .sort_values("timestamp")
            .groupby(["timestamp", "model"], as_index=False)
            .mean(numeric_only=True)
        )
        aligned["sq_error"] = (aligned["y_true"] - aligned["y_pred"]) ** 2
        pivot = aligned.pivot_table(index="timestamp", columns="model", values="sq_error", aggfunc="mean")
        pivot = pivot.dropna(axis=1, how="all")
        if pivot.shape[1] < 2:
            continue
        components = _overlap_components(pivot, min_overlap=2)
        for comp_idx, comp_models in enumerate(components, start=1):
            comp_pivot = pivot[comp_models].dropna(axis=0, how="any")
            if comp_pivot.shape[1] < 2 or comp_pivot.shape[0] < 2:
                continue
            mcs = model_confidence_set(
                comp_pivot,
                alpha=args.mcs_alpha,
                n_bootstrap=args.mcs_bootstrap_samples,
                block_size=args.mcs_block_size,
                seed=args.seed,
            )
            scope = f"component_{comp_idx}"
            model_set = ",".join(comp_models)
            for row in mcs:
                mcs_rows.append(
                    {
                        "asset": asset,
                        "mode": mode,
                        "objective_track": objective_track,
                        "comparison_scope": scope,
                        "comparison_models": model_set,
                        "model": row.model,
                        "mcs_member": bool(row.mcs_member),
                        "elimination_rank": row.elimination_rank,
                        "p_value_vs_best": row.p_value_vs_best,
                        "mean_loss": row.mean_loss,
                    }
                )

    ci_df = pd.DataFrame(ci_rows).sort_values(group_cols).reset_index(drop=True)
    binom_df = pd.DataFrame(binom_rows).sort_values(group_cols).reset_index(drop=True)
    mcs_df = pd.DataFrame(mcs_rows).sort_values(group_cols).reset_index(drop=True)

    ci_path = root / f"{args.output_prefix}_metric_ci95.csv"
    binom_path = root / f"{args.output_prefix}_directional_binomial.csv"
    mcs_path = root / f"{args.output_prefix}_mcs.csv"
    ci_df.to_csv(ci_path, index=False)
    binom_df.to_csv(binom_path, index=False)
    mcs_df.to_csv(mcs_path, index=False)

    print(f"Saved metric CI table: {ci_path}")
    print(f"Saved directional binomial table: {binom_path}")
    print(f"Saved MCS table: {mcs_path}")


if __name__ == "__main__":
    main()
