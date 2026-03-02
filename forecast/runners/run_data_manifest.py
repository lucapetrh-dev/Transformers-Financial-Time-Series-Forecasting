from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forecast.pipeline.assets import asset_name_from_path, filter_asset_paths, parse_asset_list
from forecast.pipeline.data_pipeline import _aggregation_rule, _find_price_column, load_market_data
from forecast.pipeline.experiment_contract import infer_frequency_label
from forecast.pipeline.sentiment import detect_sentiment_columns


def _resolve_glob(data_glob: str) -> list[str]:
    paths = sorted(glob.glob(data_glob))
    if not paths and data_glob.startswith("data/"):
        paths = sorted(glob.glob("." + data_glob))
    return paths


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_provenance(path: Path) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing provenance config: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    default = dict(raw.get("default", {}))
    assets = {str(k).lower(): dict(v) for k, v in dict(raw.get("assets", {})).items()}
    return default, assets


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dataset manifest and intraday aggregation policy report")
    parser.add_argument("--data-glob", type=str, default=".data/hourly/*_timeseries_hourly.csv")
    parser.add_argument("--time-col", type=str, default="time")
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
    parser.add_argument(
        "--provenance-config",
        type=str,
        default="forecast/config/data_provenance.json",
        help="JSON provenance config with source provider and timestamp semantics.",
    )
    parser.add_argument("--output", type=str, default="results/paper/data_manifest.csv")
    args = parser.parse_args()

    paths = _resolve_glob(args.data_glob)
    if not paths:
        raise ValueError(f"No files matched --data-glob pattern: {args.data_glob}")
    paths = filter_asset_paths(
        paths,
        include_assets=parse_asset_list(args.include_assets),
        exclude_assets=parse_asset_list(args.exclude_assets),
    )
    if not paths:
        raise ValueError("No assets left after include/exclude filtering")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    default_prov, asset_prov = _load_provenance(Path(args.provenance_config))

    manifest_rows: list[dict[str, object]] = []
    policy_rows: list[dict[str, object]] = []

    for path in paths:
        asset = asset_name_from_path(path)
        file_path = Path(path)
        raw = pd.read_csv(path)
        if args.time_col not in raw.columns:
            raise ValueError(f"Missing time column '{args.time_col}' in {path}")

        raw[args.time_col] = pd.to_datetime(raw[args.time_col], errors="coerce")
        raw = raw.dropna(subset=[args.time_col]).sort_values(args.time_col).reset_index(drop=True)

        processed = load_market_data(path, time_col=args.time_col)
        price_col = _find_price_column(processed)
        sent_cols = detect_sentiment_columns(processed, exclude_cols={"return_1d", "log_price"})
        provenance = dict(default_prov)
        provenance.update(asset_prov.get(asset, {}))

        manifest_rows.append(
            {
                "asset": asset,
                "data_path": path,
                "raw_rows": int(len(raw)),
                "processed_rows": int(len(processed)),
                "raw_time_start": str(raw[args.time_col].min()),
                "raw_time_end": str(raw[args.time_col].max()),
                "processed_time_start": str(processed[args.time_col].min()),
                "processed_time_end": str(processed[args.time_col].max()),
                "raw_frequency": infer_frequency_label(raw[args.time_col]),
                "processed_frequency": infer_frequency_label(processed[args.time_col]),
                "intraday_detected": bool((raw[args.time_col].dt.floor("D") != raw[args.time_col]).any()),
                "price_column_selected": price_col,
                "raw_column_count": int(raw.shape[1]),
                "processed_column_count": int(processed.shape[1]),
                "sentiment_column_count": int(len(sent_cols)),
                "sha256": _sha256_file(file_path),
                "file_size_bytes": int(file_path.stat().st_size),
                "source_provider": provenance.get("source_provider"),
                "provider_timezone": provenance.get("provider_timezone"),
                "daily_cutoff_time": provenance.get("daily_cutoff_time"),
                "snapshot_date": provenance.get("snapshot_date"),
            }
        )

        for col in raw.columns:
            if col == args.time_col:
                continue
            policy_rows.append(
                {
                    "asset": asset,
                    "column": col,
                    "aggregation_rule_when_intraday": _aggregation_rule(col),
                }
            )

    manifest_df = pd.DataFrame(manifest_rows).sort_values("asset").reset_index(drop=True)
    manifest_df.to_csv(output_path, index=False)

    policy_path = output_path.with_name(output_path.stem + "_aggregation_policy.csv")
    pd.DataFrame(policy_rows).sort_values(["asset", "column"]).to_csv(policy_path, index=False)

    print(f"Saved data manifest: {output_path}")
    print(f"Saved aggregation policy: {policy_path}")


if __name__ == "__main__":
    main()
