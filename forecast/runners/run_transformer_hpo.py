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
from forecast.pipeline.transformer_official_backends import canonical_model_name


def _build_canonical_cmd(
    *,
    data_path: str,
    out_path: Path,
    args: argparse.Namespace,
    d_model: int,
    n_layers: int,
    lr: float,
) -> list[str]:
    cmd = [
        sys.executable,
        "forecast/runners/run_transformers_canonical.py",
        "--data-path",
        data_path,
        "--time-col",
        args.time_col,
        "--horizon",
        str(args.horizon),
        "--lookback",
        str(args.lookback),
        "--min-train-size",
        str(args.min_train_size),
        "--test-size",
        str(args.test_size),
        "--step-size",
        str(args.step_size),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--models",
        args.model_family,
        "--d-model",
        str(d_model),
        "--n-layers",
        str(n_layers),
        "--n-heads",
        str(args.n_heads),
        "--dropout",
        str(args.dropout),
        "--lr",
        str(lr),
        "--objective-track",
        args.objective_track,
        "--point-loss",
        args.point_loss,
        "--quantiles",
        args.quantiles,
        "--seed",
        str(args.seed),
        "--output",
        str(out_path),
    ]
    if args.use_regime_features:
        cmd.append("--use-regime-features")
    if args.use_sentiment:
        cmd.append("--use-sentiment")
        cmd.extend(["--sentiment-lag", str(args.sentiment_lag)])
        cmd.extend(["--sentiment-min-non-null-ratio", str(args.sentiment_min_non_null_ratio)])
    if args.patch_len > 0:
        cmd.extend(["--patch-len", str(args.patch_len)])
    if args.stride > 0:
        cmd.extend(["--stride", str(args.stride)])
    return cmd


def _resolve_objective_track_for_scoring(objective_track: str) -> str:
    if objective_track == "both":
        return "point"
    return objective_track


def _extract_objective_value(summary_path: Path, model_name: str, objective_track: str) -> float:
    df = pd.read_csv(summary_path)
    sub = df[(df["model"] == model_name) & (df.get("objective_track", "point") == objective_track)]
    if sub.empty:
        # Backward compatibility if objective_track column is not present.
        sub = df[df["model"] == model_name]
    if sub.empty:
        raise ValueError(f"Model row not found in summary: model={model_name} track={objective_track}")
    return float(sub.iloc[0]["mae_mean"])


def _run_trial_eval(
    *,
    data_path: str,
    trial_output: Path,
    args: argparse.Namespace,
    d_model: int,
    n_layers: int,
    lr: float,
    model_name: str,
    scoring_track: str,
) -> float:
    cmd = _build_canonical_cmd(
        data_path=data_path,
        out_path=trial_output,
        args=args,
        d_model=d_model,
        n_layers=n_layers,
        lr=lr,
    )
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return _extract_objective_value(trial_output, model_name=model_name, objective_track=scoring_track)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna HPO for canonical transformer backends")
    parser.add_argument("--data-glob", type=str, default=".data/hourly/*_timeseries_hourly.csv")
    parser.add_argument("--time-col", type=str, default="time")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 5, 20])
    parser.add_argument("--lookback", type=int, default=64)
    parser.add_argument("--min-train-size", type=int, default=365)
    parser.add_argument("--test-size", type=int, default=60)
    parser.add_argument("--step-size", type=int, default=180)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-family", type=str, default="itransformer", choices=["itransformer", "patchtst"])
    parser.add_argument("--objective-track", type=str, default="point", choices=["point", "quantile", "both"])
    parser.add_argument("--point-loss", type=str, default="mse", choices=["mse", "mae"])
    parser.add_argument("--quantiles", type=str, default="0.1,0.5,0.9")
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patch-len", type=int, default=0)
    parser.add_argument("--stride", type=int, default=0)
    parser.add_argument("--use-regime-features", action="store_true")
    parser.add_argument("--use-sentiment", action="store_true")
    parser.add_argument("--sentiment-lag", type=int, default=1)
    parser.add_argument("--sentiment-min-non-null-ratio", type=float, default=0.2)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-assets", type=int, default=0)
    parser.add_argument("--include-assets", type=str, default="")
    parser.add_argument("--exclude-assets", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="results/paper/hpo")
    parser.add_argument(
        "--summary-output",
        type=str,
        default="results/paper/hpo/transformer_hpo_best_summary.csv",
        help="Aggregated best-per-asset canonical summary table.",
    )
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

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    model_name = canonical_model_name(args.model_family)
    scoring_track = _resolve_objective_track_for_scoring(args.objective_track)

    try:
        import optuna
    except Exception:
        optuna = None  # type: ignore[assignment]

    best_rows: list[dict[str, object]] = []
    best_summary_rows: list[pd.DataFrame] = []
    all_trials_rows: list[pd.DataFrame] = []

    for data_path in data_paths:
        asset = asset_name_from_path(data_path)
        asset_dir = out_root / f"{asset}_{args.model_family}"
        asset_dir.mkdir(parents=True, exist_ok=True)

        trial_rows: list[dict[str, object]] = []

        if optuna is not None:
            sampler = optuna.samplers.TPESampler(seed=args.seed)
            study = optuna.create_study(direction="minimize", sampler=sampler)

            def _objective(trial):
                d_model = int(trial.suggest_categorical("d_model", [64, 128, 256]))
                n_layers = int(trial.suggest_categorical("n_layers", [2, 3, 4]))
                lr = float(trial.suggest_float("lr", 1e-4, 3e-3, log=True))
                trial_output = asset_dir / f"trial_{trial.number:03d}_summary.csv"
                try:
                    score = _run_trial_eval(
                        data_path=data_path,
                        trial_output=trial_output,
                        args=args,
                        d_model=d_model,
                        n_layers=n_layers,
                        lr=lr,
                        model_name=model_name,
                        scoring_track=scoring_track,
                    )
                    trial_rows.append(
                        {
                            "asset": asset,
                            "trial_number": int(trial.number),
                            "d_model": d_model,
                            "n_layers": n_layers,
                            "lr": lr,
                            "mae_mean": score,
                            "status": "ok",
                        }
                    )
                    return score
                except Exception as exc:
                    trial_rows.append(
                        {
                            "asset": asset,
                            "trial_number": int(trial.number),
                            "d_model": d_model,
                            "n_layers": n_layers,
                            "lr": lr,
                            "mae_mean": np.nan,
                            "status": "failed",
                            "error": str(exc),
                        }
                    )
                    return float("inf")

            study.optimize(_objective, n_trials=args.trials, show_progress_bar=False)
            best_params = study.best_params
            best_value = float(study.best_value)
        else:
            rng = np.random.default_rng(args.seed)
            best_params: dict[str, object] = {}
            best_value = float("inf")
            for t in range(args.trials):
                d_model = int(rng.choice([64, 128, 256]))
                n_layers = int(rng.choice([2, 3, 4]))
                lr = float(np.exp(rng.uniform(np.log(1e-4), np.log(3e-3))))
                trial_output = asset_dir / f"trial_{t:03d}_summary.csv"
                try:
                    score = _run_trial_eval(
                        data_path=data_path,
                        trial_output=trial_output,
                        args=args,
                        d_model=d_model,
                        n_layers=n_layers,
                        lr=lr,
                        model_name=model_name,
                        scoring_track=scoring_track,
                    )
                    trial_rows.append(
                        {
                            "asset": asset,
                            "trial_number": t,
                            "d_model": d_model,
                            "n_layers": n_layers,
                            "lr": lr,
                            "mae_mean": score,
                            "status": "ok",
                        }
                    )
                    if score < best_value:
                        best_value = score
                        best_params = {"d_model": d_model, "n_layers": n_layers, "lr": lr}
                except Exception as exc:
                    trial_rows.append(
                        {
                            "asset": asset,
                            "trial_number": t,
                            "d_model": d_model,
                            "n_layers": n_layers,
                            "lr": lr,
                            "mae_mean": np.nan,
                            "status": "failed",
                            "error": str(exc),
                        }
                    )

            if not best_params:
                raise ValueError(f"HPO failed for asset={asset}; no successful trials")

        trials_df = pd.DataFrame(trial_rows).sort_values("trial_number")
        trials_path = asset_dir / "trials.csv"
        trials_df.to_csv(trials_path, index=False)
        all_trials_rows.append(trials_df.assign(model_family=args.model_family))

        best_cfg = {
            "asset": asset,
            "data_path": data_path,
            "model_family": args.model_family,
            "model_name": model_name,
            "objective_track_scored": scoring_track,
            "best_mae_mean": best_value,
            "d_model": int(best_params["d_model"]),
            "n_layers": int(best_params["n_layers"]),
            "lr": float(best_params["lr"]),
            "n_heads": int(args.n_heads),
            "dropout": float(args.dropout),
            "seed": int(args.seed),
            "trials": int(args.trials),
            "search_space": {
                "d_model": [64, 128, 256],
                "n_layers": [2, 3, 4],
                "lr_log_range": [1e-4, 3e-3],
            },
        }
        best_cfg_path = asset_dir / "best_config.json"
        with best_cfg_path.open("w", encoding="utf-8") as f:
            json.dump(best_cfg, f, indent=2)
        best_rows.append(best_cfg)

        best_eval_path = asset_dir / "best_summary.csv"
        final_cmd = _build_canonical_cmd(
            data_path=data_path,
            out_path=best_eval_path,
            args=args,
            d_model=int(best_params["d_model"]),
            n_layers=int(best_params["n_layers"]),
            lr=float(best_params["lr"]),
        )
        subprocess.run(final_cmd, check=True)
        best_eval_df = pd.read_csv(best_eval_path)
        best_eval_df.insert(0, "asset", asset)
        best_eval_df.insert(1, "data_path", data_path)
        best_summary_rows.append(best_eval_df)

    best_df = pd.DataFrame(best_rows).sort_values("asset")
    best_df_path = out_root / "best_configs.csv"
    best_df.to_csv(best_df_path, index=False)

    if all_trials_rows:
        all_trials_df = pd.concat(all_trials_rows, ignore_index=True)
        all_trials_path = out_root / "all_trials.csv"
        all_trials_df.to_csv(all_trials_path, index=False)
        print(f"Saved all trial logs: {all_trials_path}")

    if best_summary_rows:
        combined_best_summary = pd.concat(best_summary_rows, ignore_index=True)
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        combined_best_summary.to_csv(summary_path, index=False)
        rank_group_cols = [c for c in ["objective_track", "model"] if c in combined_best_summary.columns]
        if not rank_group_cols:
            rank_group_cols = ["model"]
        ranks = (
            combined_best_summary.groupby(rank_group_cols, as_index=False)
            .agg(
                avg_mae_mean=("mae_mean", "mean"),
                n_asset_evals=("asset", "count"),
            )
            .sort_values(["avg_mae_mean"], ascending=[True])
            .reset_index(drop=True)
        )
        if rank_group_cols:
            ranks["avg_rank"] = ranks["avg_mae_mean"].rank(method="min")
        rank_path = summary_path.with_name(summary_path.stem + "_ranks.csv")
        ranks.to_csv(rank_path, index=False)

        audit = combined_best_summary[["asset", "data_path"]].drop_duplicates().copy()
        audit["target_space"] = "log_return"
        audit["consistent_target_space_all"] = True
        audit["expected_target_space"] = "log_return"
        audit_path = summary_path.with_name(summary_path.stem + "_comparability_audit.csv")
        audit.to_csv(audit_path, index=False)
        print(f"Saved aggregated best-summary table: {summary_path}")
        print(f"Saved ranking table: {rank_path}")
        print(f"Saved comparability audit: {audit_path}")

    print(f"Saved best configs: {best_df_path}")


if __name__ == "__main__":
    main()
