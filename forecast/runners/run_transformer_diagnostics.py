from __future__ import annotations

import argparse
import glob
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forecast.analysis.transformer_diagnostics import (
    analyze_hpo_sensitivity,
    analyze_training_convergence,
    compute_residual_autocorrelation,
    estimate_return_predictability,
)
from forecast.pipeline.assets import asset_name_from_path, filter_asset_paths, parse_asset_list
from forecast.pipeline.data_pipeline import FeatureConfig, build_feature_frame, load_market_data
from forecast.pipeline.experiment_contract import save_metadata_json


def _resolve_paths(data_glob: str) -> list[str]:
    paths = sorted(glob.glob(data_glob))
    if not paths and data_glob.startswith("data/"):
        paths = sorted(glob.glob("." + data_glob))
    return paths


def _load_transformer_predictions(results_root: Path) -> pd.DataFrame:
    agg = results_root / "multi_asset_transformers_h1_paired_summary_predictions.csv"
    if agg.exists():
        return pd.read_csv(agg)

    parts = sorted(results_root.glob("multi_asset_transformers_h1_paired_summary_*_predictions.csv"))
    if not parts:
        raise FileNotFoundError("No transformer prediction files found")
    return pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)


def _load_transformer_history(results_root: Path) -> pd.DataFrame:
    agg = results_root / "multi_asset_transformers_h1_paired_summary_history.csv"
    if agg.exists():
        return pd.read_csv(agg)

    parts = sorted(results_root.glob("multi_asset_transformers_h1_paired_summary_*_history.csv"))
    if not parts:
        raise FileNotFoundError("No transformer history files found")
    return pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)


def _load_hpo_trials(hpo_dir: Path) -> pd.DataFrame:
    all_trials = hpo_dir / "all_trials.csv"
    if all_trials.exists():
        return pd.read_csv(all_trials)

    trial_files = sorted(hpo_dir.rglob("trials.csv"))
    if not trial_files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p) for p in trial_files], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run transformer diagnostics (residual ACF, convergence, return predictability, HPO sensitivity)")
    parser.add_argument("--results-root", type=str, default="results/paper")
    parser.add_argument("--data-glob", type=str, default=".data/hourly/*_timeseries_hourly.csv")
    parser.add_argument("--include-assets", type=str, default="")
    parser.add_argument("--exclude-assets", type=str, default="")
    parser.add_argument("--hpo-dir", type=str, default="results/paper/hpo")
    parser.add_argument("--output-dir", type=str, default="results/paper/transformer_diagnostics")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Residual ACF/PACF from prediction traces
    preds = _load_transformer_predictions(results_root)
    if "objective_track" in preds.columns:
        preds = preds[preds["objective_track"] == "point"].copy()

    group_cols = [c for c in ["asset", "model", "mode", "fold", "objective_track"] if c in preds.columns]
    residual_frames: list[pd.DataFrame] = []
    for key, grp in preds.groupby(group_cols, dropna=False):
        key_vals = key if isinstance(key, tuple) else (key,)
        meta = {col: key_vals[idx] for idx, col in enumerate(group_cols)}

        if "timestamp" in grp.columns:
            grp = grp.assign(timestamp=pd.to_datetime(grp["timestamp"], errors="coerce")).sort_values("timestamp")

        y_true = pd.to_numeric(grp["y_true"], errors="coerce").to_numpy(dtype=float)
        y_pred = pd.to_numeric(grp["y_pred"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if int(mask.sum()) < 10:
            continue

        acf_df = compute_residual_autocorrelation(y_true[mask], y_pred[mask], max_lags=40)
        if acf_df.empty:
            continue
        acf_df.insert(0, "n_obs", int(mask.sum()))
        for col, val in meta.items():
            acf_df.insert(0, col, val)
        residual_frames.append(acf_df)

    residual_out = pd.concat(residual_frames, ignore_index=True) if residual_frames else pd.DataFrame()
    residual_path = output_dir / "residual_acf_analysis.csv"
    residual_out.to_csv(residual_path, index=False)

    # 2) Convergence summaries from history files
    history_df = _load_transformer_history(results_root)
    if "objective_track" in history_df.columns:
        history_df = history_df[history_df["objective_track"] == "point"].copy()
    conv_out = analyze_training_convergence(history_df)
    conv_path = output_dir / "training_convergence.csv"
    conv_out.to_csv(conv_path, index=False)

    # 3) Return predictability diagnostics per asset
    data_paths = _resolve_paths(args.data_glob)
    if not data_paths:
        raise ValueError(f"No files matched --data-glob pattern: {args.data_glob}")
    data_paths = filter_asset_paths(
        data_paths,
        include_assets=parse_asset_list(args.include_assets),
        exclude_assets=parse_asset_list(args.exclude_assets),
    )
    if not data_paths:
        raise ValueError("No assets left after include/exclude filtering")

    rp_rows: list[dict[str, float | int | str]] = []
    for data_path in data_paths:
        asset = asset_name_from_path(data_path)
        raw = load_market_data(data_path, time_col="time")
        feat = build_feature_frame(raw, config=FeatureConfig(lookbacks=(20, 60), ewma_span=20, roc_period=5), time_col="time")
        returns = pd.to_numeric(feat["return_1d"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)

        diag = estimate_return_predictability(returns, max_lags=20)
        acf_table = diag["acf_table"]
        vr_table = diag["variance_ratios"]
        vr_map = {int(r.q): float(r.variance_ratio) for r in vr_table.itertuples(index=False)}

        for row in acf_table.itertuples(index=False):
            rp_rows.append(
                {
                    "asset": asset,
                    "lag": int(row.lag),
                    "acf": float(row.acf),
                    "ljung_box_stat": float(diag["ljung_box_stat"]),
                    "ljung_box_p": float(diag["ljung_box_p"]),
                    "vr_q2": vr_map.get(2, np.nan),
                    "vr_q5": vr_map.get(5, np.nan),
                    "vr_q10": vr_map.get(10, np.nan),
                    "vr_q20": vr_map.get(20, np.nan),
                }
            )

    rp_out = pd.DataFrame(rp_rows)
    rp_path = output_dir / "return_predictability.csv"
    rp_out.to_csv(rp_path, index=False)

    # 4) Optional HPO sensitivity
    hpo_path: Path | None = None
    hpo_dir = Path(args.hpo_dir)
    if hpo_dir.exists():
        trials_df = _load_hpo_trials(hpo_dir)
        if not trials_df.empty:
            hpo_tables = analyze_hpo_sensitivity(trials_df)
            hpo_frames: list[pd.DataFrame] = []
            for name, table in hpo_tables.items():
                if table.empty:
                    continue
                t = table.copy()
                t.insert(0, "analysis_type", name)
                hpo_frames.append(t)
            if hpo_frames:
                hpo_out = pd.concat(hpo_frames, ignore_index=True)
                hpo_path = output_dir / "hpo_sensitivity.csv"
                hpo_out.to_csv(hpo_path, index=False)

    metadata = {
        "script": "run_transformer_diagnostics.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_root": str(results_root),
        "data_glob": args.data_glob,
        "include_assets": sorted(parse_asset_list(args.include_assets)),
        "exclude_assets": sorted(parse_asset_list(args.exclude_assets)),
        "hpo_dir": str(hpo_dir),
        "outputs": {
            "residual_acf_analysis": str(residual_path),
            "training_convergence": str(conv_path),
            "return_predictability": str(rp_path),
            "hpo_sensitivity": str(hpo_path) if hpo_path is not None else None,
        },
    }
    save_metadata_json(output_dir / "metadata.json", metadata)

    print(f"Saved: {residual_path}")
    print(f"Saved: {conv_path}")
    print(f"Saved: {rp_path}")
    if hpo_path is not None:
        print(f"Saved: {hpo_path}")
    print(f"Saved: {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
