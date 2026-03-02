from __future__ import annotations

import argparse
import glob
import json
import subprocess
from pathlib import Path
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forecast.pipeline.assets import asset_name_from_path, filter_asset_paths, parse_asset_list

def _load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _mode_specs(regime_lookbacks: str) -> list[tuple[str, list[str]]]:
    return [
        ("financial_only", []),
        ("financial_plus_sentiment", ["--use-sentiment"]),
        (
            "financial_plus_sentiment_plus_regime",
            ["--use-sentiment", "--use-regime-features", "--regime-lookbacks", regime_lookbacks],
        ),
    ]


def _resolve_glob(data_glob: str) -> list[str]:
    paths = sorted(glob.glob(data_glob))
    if not paths and data_glob.startswith("data/"):
        paths = sorted(glob.glob("." + data_glob))
    return paths


def _build_cmd(
    *,
    data_path: str,
    output_path: Path,
    args: argparse.Namespace,
    mode_flags: list[str],
) -> list[str]:
    cmd = [
        "python",
        "forecast/runners/run_cpcv_report.py",
        "--data-path",
        data_path,
        "--time-col",
        args.time_col,
        "--horizon",
        str(args.horizon),
        "--max-lag",
        str(args.max_lag),
        "--n-splits",
        str(args.n_splits),
        "--embargo",
        str(args.embargo),
        "--label-horizon",
        str(args.label_horizon),
        "--sentiment-lag",
        str(args.sentiment_lag),
        "--output",
        str(output_path),
    ]
    if args.no_arima:
        cmd.append("--no-arima")
    if args.no_xgboost:
        cmd.append("--no-xgboost")
    cmd.extend(mode_flags)
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run feature ablations across assets/modes and aggregate results")
    parser.add_argument("--data-glob", type=str, default="data/*_timeseries.csv")
    parser.add_argument("--time-col", type=str, default="time")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 5, 20])
    parser.add_argument("--max-lag", type=int, default=20)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--embargo", type=int, default=5)
    parser.add_argument("--label-horizon", type=int, default=1)
    parser.add_argument("--sentiment-lag", type=int, default=1)
    parser.add_argument("--regime-lookbacks", type=str, default="20,60")
    parser.add_argument("--no-arima", action="store_true", default=True)
    parser.add_argument("--no-xgboost", action="store_true", default=True)
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
    parser.add_argument("--output", type=str, default="results/multi_asset/feature_ablation_summary.csv")
    args = parser.parse_args()

    data_paths = _resolve_glob(args.data_glob)
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

    rows_long: list[pd.DataFrame] = []
    metadata_rows: list[dict] = []
    failed_rows: list[dict] = []

    for data_path in data_paths:
        asset = asset_name_from_path(data_path)
        for mode_name, mode_flags in _mode_specs(args.regime_lookbacks):
            run_out = output_path.with_name(f"{output_path.stem}_{asset}_{mode_name}{output_path.suffix}")
            cmd = _build_cmd(data_path=data_path, output_path=run_out, args=args, mode_flags=mode_flags)
            print(f"Running ablation asset={asset} mode={mode_name}")
            try:
                subprocess.run(cmd, check=True)
            except Exception as exc:
                if args.skip_failed_assets:
                    failed_rows.append({"asset": asset, "mode": mode_name, "data_path": data_path, "error": str(exc)})
                    print(f"Skipping failed ablation run asset={asset} mode={mode_name}: {exc}")
                    continue
                raise

            summary = pd.read_csv(run_out)
            summary = summary.copy()
            summary.insert(0, "asset", asset)
            summary.insert(1, "mode", mode_name)
            summary.insert(2, "data_path", data_path)
            rows_long.append(summary)

            meta = _load_metadata(run_out.with_name(run_out.stem + "_metadata.json"))
            metadata_rows.append(
                {
                    "asset": asset,
                    "mode": mode_name,
                    "data_path": data_path,
                    "target_space": meta.get("target_space"),
                    "horizon_days": meta.get("horizon_days"),
                    "feature_mode": meta.get("feature_mode"),
                    "use_regime_features": meta.get("use_regime_features"),
                    "sentiment_feature_count": meta.get("sentiment_feature_count"),
                    "time_start": meta.get("time_start"),
                    "time_end": meta.get("time_end"),
                }
            )

    if not rows_long:
        raise ValueError("No successful ablation runs were produced.")

    long_df = pd.concat(rows_long, ignore_index=True)
    long_df.to_csv(output_path, index=False)

    best_df = (
        long_df.sort_values(["asset", "mode", "mae_mean"], ascending=[True, True, True])
        .groupby(["asset", "mode"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_path = output_path.with_name(output_path.stem + "_best.csv")
    best_df.to_csv(best_path, index=False)

    aggregate_df = (
        long_df.groupby(["mode", "model"], as_index=False)
        .agg(
            avg_mae_mean=("mae_mean", "mean"),
            avg_rmse_mean=("rmse_mean", "mean"),
            avg_directional_accuracy_mean=("directional_accuracy_mean", "mean"),
            n_asset_evals=("asset", "count"),
        )
        .sort_values(["mode", "avg_mae_mean"], ascending=[True, True])
    )
    aggregate_path = output_path.with_name(output_path.stem + "_aggregate.csv")
    aggregate_df.to_csv(aggregate_path, index=False)

    rank_df = long_df.copy()
    rank_df["mae_rank_within_asset_mode"] = rank_df.groupby(["asset", "mode"])["mae_mean"].rank(method="min")
    rank_table = (
        rank_df.groupby(["mode", "model"], as_index=False)
        .agg(
            avg_mae_mean=("mae_mean", "mean"),
            avg_rank=("mae_rank_within_asset_mode", "mean"),
            n_asset_evals=("asset", "count"),
        )
        .sort_values(["mode", "avg_rank", "avg_mae_mean"], ascending=[True, True, True])
    )
    rank_path = output_path.with_name(output_path.stem + "_ranks.csv")
    rank_table.to_csv(rank_path, index=False)

    mode_ref = "financial_only"
    # Build simple mode delta from best model MAE by mode (asset-level), relative to financial_only best MAE.
    best_by_mode = best_df[["asset", "mode", "mae_mean", "directional_accuracy_mean"]]
    ref = best_by_mode[best_by_mode["mode"] == mode_ref][["asset", "mae_mean", "directional_accuracy_mean"]]
    ref = ref.rename(columns={"mae_mean": "mae_mean_ref_financial_only", "directional_accuracy_mean": "directional_accuracy_ref_financial_only"})
    deltas = best_by_mode.merge(ref, on="asset", how="left")
    deltas["delta_mae_vs_financial_only"] = deltas["mae_mean"] - deltas["mae_mean_ref_financial_only"]
    deltas["delta_directional_accuracy_vs_financial_only"] = (
        deltas["directional_accuracy_mean"] - deltas["directional_accuracy_ref_financial_only"]
    )
    delta_path = output_path.with_name(output_path.stem + "_mode_deltas.csv")
    deltas.to_csv(delta_path, index=False)

    audit_df = pd.DataFrame(metadata_rows)
    unique_target_spaces = sorted(set(audit_df["target_space"].dropna().tolist()))
    audit_df["consistent_target_space_all"] = len(unique_target_spaces) == 1
    audit_df["expected_target_space"] = unique_target_spaces[0] if unique_target_spaces else "unknown"
    audit_path = output_path.with_name(output_path.stem + "_comparability_audit.csv")
    audit_df.to_csv(audit_path, index=False)
    if len(unique_target_spaces) != 1:
        raise ValueError(
            "Comparability check failed: multiple target spaces detected: "
            f"{unique_target_spaces}. Inspect {audit_path}."
        )

    if failed_rows:
        failed_path = output_path.with_name(output_path.stem + "_failed_runs.csv")
        pd.DataFrame(failed_rows).to_csv(failed_path, index=False)
        print(f"Saved failed run log: {failed_path}")

    print(f"Saved ablation long table: {output_path}")
    print(f"Saved ablation best table: {best_path}")
    print(f"Saved ablation aggregate table: {aggregate_path}")
    print(f"Saved ablation rank table: {rank_path}")
    print(f"Saved ablation mode deltas: {delta_path}")
    print(f"Saved comparability audit: {audit_path}")


if __name__ == "__main__":
    main()
