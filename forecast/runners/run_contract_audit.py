from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forecast.pipeline.paper_contract import contract_asset_table, load_paper_contract


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and materialize paper experiment contract")
    parser.add_argument(
        "--contract-path",
        type=str,
        default="forecast/config/paper_experiment.json",
        help="Path to canonical paper experiment config",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/paper/contract_asset_manifest.csv",
        help="Output CSV path for contract asset manifest",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default="results/paper/data_manifest.csv",
        help="Data manifest CSV to validate for provenance requirements.",
    )
    parser.add_argument(
        "--require-provenance",
        action="store_true",
        help="Require provenance/hash columns to be present and non-null for contract assets.",
    )
    args = parser.parse_args()

    contract = load_paper_contract(args.contract_path)
    rows = contract_asset_table(contract)
    df = pd.DataFrame(rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    missing = df.loc[~df["exists"]]
    if not missing.empty:
        missing_assets = ", ".join(missing["asset"].astype(str).tolist())
        raise ValueError(f"Contract validation failed; missing data for assets: {missing_assets}")

    if args.require_provenance:
        manifest_path = Path(args.manifest_path)
        if not manifest_path.exists():
            raise ValueError(f"Provenance validation failed; missing manifest file: {manifest_path}")
        manifest = pd.read_csv(manifest_path)
        required_cols = [
            "asset",
            "sha256",
            "file_size_bytes",
            "source_provider",
            "provider_timezone",
            "daily_cutoff_time",
            "snapshot_date",
        ]
        missing_cols = [c for c in required_cols if c not in manifest.columns]
        if missing_cols:
            raise ValueError(f"Provenance validation failed; missing manifest columns: {', '.join(missing_cols)}")

        contract_assets = set(df["asset"].astype(str).str.lower().tolist())
        man = manifest.copy()
        man["asset"] = man["asset"].astype(str).str.lower()
        man = man[man["asset"].isin(contract_assets)]
        if set(man["asset"].tolist()) != contract_assets:
            missing_asset_rows = sorted(contract_assets.difference(set(man["asset"].tolist())))
            raise ValueError(
                "Provenance validation failed; manifest missing contract assets: "
                f"{', '.join(missing_asset_rows)}"
            )

        null_mask = man[required_cols].isna().any(axis=1)
        if bool(null_mask.any()):
            bad_assets = ", ".join(sorted(man.loc[null_mask, "asset"].astype(str).unique().tolist()))
            raise ValueError(f"Provenance validation failed; null provenance fields for assets: {bad_assets}")
        if (man["sha256"].astype(str).str.len() < 64).any():
            raise ValueError("Provenance validation failed; sha256 values look malformed")

    print(f"Validated contract: {args.contract_path}")
    print(f"Saved asset manifest: {output_path}")
    print(f"Assets: {', '.join(df['asset'].tolist())}")


if __name__ == "__main__":
    main()
