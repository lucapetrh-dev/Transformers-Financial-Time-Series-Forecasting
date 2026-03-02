from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forecast.pipeline.paper_validity import (
    validate_asset_universe,
    validate_metadata_paths,
    validate_no_fallback_rows,
)


EXPECTED_ASSETS = {"btc", "eth", "ada", "doge", "xmr", "xrp"}


def _required_for_model_run(summary_path: Path) -> list[Path]:
    return [
        summary_path.with_name(summary_path.stem + "_folds.csv"),
        summary_path.with_name(summary_path.stem + "_fold_manifest.csv"),
        summary_path.with_name(summary_path.stem + "_metadata.json"),
        summary_path.with_name(summary_path.stem + "_data_quality_summary.csv"),
        summary_path.with_name(summary_path.stem + "_missingness_stages.csv"),
    ]


def _required_for_aggregate(summary_path: Path) -> list[Path]:
    stem = summary_path.stem
    if stem.startswith("feature_ablation_") and stem.endswith("_summary"):
        return [
            summary_path.with_name(stem + "_best.csv"),
            summary_path.with_name(stem + "_aggregate.csv"),
            summary_path.with_name(stem + "_mode_deltas.csv"),
            summary_path.with_name(stem + "_comparability_audit.csv"),
        ]
    return [
        summary_path.with_name(stem + "_ranks.csv"),
        summary_path.with_name(stem + "_comparability_audit.csv"),
    ]


def _is_model_run_summary(summary_path: Path) -> bool:
    return summary_path.with_name(summary_path.stem + "_metadata.json").exists()


def _is_paper_aggregate(summary_path: Path) -> bool:
    name = summary_path.name
    return name in {
        "multi_asset_baselines_h1_paired_summary.csv",
        "multi_asset_transformers_h1_paired_summary.csv",
        "multi_asset_chronos2_h1_summary.csv",
        "multi_asset_foundation_h1_summary.csv",
        "feature_ablation_h1_summary.csv",
    }


def _validate_comparability_audit(summary_path: Path) -> list[str]:
    errors: list[str] = []
    audit_path = summary_path.with_name(summary_path.stem + "_comparability_audit.csv")
    if not audit_path.exists():
        errors.append(f"Missing comparability audit: {audit_path}")
        return errors

    try:
        audit_df = pd.read_csv(audit_path)
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"Failed to read comparability audit {audit_path}: {exc}")
        return errors

    if "target_space" in audit_df.columns:
        bad = audit_df["target_space"].astype(str).str.lower() != "log_return"
        if bool(bad.any()):
            errors.append(f"Non-log_return target_space rows in {audit_path}")
    elif "expected_target_space" in audit_df.columns:
        bad = audit_df["expected_target_space"].astype(str).str.lower() != "log_return"
        if bool(bad.any()):
            errors.append(f"Non-log_return expected_target_space rows in {audit_path}")
    else:
        errors.append(f"Missing target-space columns in comparability audit: {audit_path}")

    return errors


def _validate_aggregate_asset_universe(summary_path: Path) -> list[str]:
    errors: list[str] = []
    try:
        df = pd.read_csv(summary_path)
        validate_asset_universe(df, EXPECTED_ASSETS)
    except Exception as exc:
        errors.append(str(exc))
    return errors


def _validate_aggregate_fallback_rows(summary_path: Path) -> list[str]:
    errors: list[str] = []
    try:
        df = pd.read_csv(summary_path)
        validate_no_fallback_rows(df, model_col="model", note_col="backend_note")
    except Exception as exc:
        errors.append(str(exc))
    return errors


def _validate_aggregate_metadata_paths(summary_path: Path) -> list[str]:
    errors: list[str] = []
    pattern = summary_path.with_name(summary_path.stem + "_*_metadata.json")
    metadata_paths = sorted(summary_path.parent.glob(pattern.name))
    for meta in metadata_paths:
        try:
            validate_metadata_paths(meta)
        except Exception as exc:
            errors.append(str(exc))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit reproducibility artifact completeness for experiment outputs")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="results/reproducibility")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    excluded_suffix_markers = (
        "_data_quality_summary.csv",
        "_feature_importance_summary.csv",
        "_stationarity_summary.csv",
        "_aggregate.csv",
        "_best.csv",
        "_mode_deltas.csv",
    )
    summary_files = []
    for p in results_root.rglob("*_summary.csv"):
        rel = p.relative_to(results_root)
        if "archive_flat" in rel.parts or "reproducibility" in rel.parts:
            continue
        if any(p.name.endswith(marker) for marker in excluded_suffix_markers):
            continue
        summary_files.append(p)
    summary_files = sorted(summary_files)
    if not summary_files:
        raise ValueError(f"No *_summary.csv files found under {results_root}")

    is_paper_root = "paper" in results_root.parts or results_root.name == "paper"
    rows: list[dict[str, object]] = []
    for summary_path in summary_files:
        is_model = _is_model_run_summary(summary_path)
        required = _required_for_model_run(summary_path) if is_model else _required_for_aggregate(summary_path)
        missing = [str(p) for p in required if not p.exists()]

        strict_errors: list[str] = []
        if args.strict:
            if is_model:
                metadata_path = summary_path.with_name(summary_path.stem + "_metadata.json")
                try:
                    validate_metadata_paths(metadata_path)
                except Exception as exc:
                    strict_errors.append(str(exc))
            elif is_paper_root and _is_paper_aggregate(summary_path):
                strict_errors.extend(_validate_aggregate_asset_universe(summary_path))
                strict_errors.extend(_validate_aggregate_fallback_rows(summary_path))
                strict_errors.extend(_validate_comparability_audit(summary_path))
                strict_errors.extend(_validate_aggregate_metadata_paths(summary_path))

        rows.append(
            {
                "summary_path": str(summary_path),
                "summary_type": "model_run" if is_model else "aggregate",
                "required_count": len(required),
                "present_count": len(required) - len(missing),
                "all_required_present": len(missing) == 0,
                "strict_pass": len(strict_errors) == 0,
                "missing_paths": ";".join(missing),
                "strict_errors": " | ".join(strict_errors),
            }
        )

    audit_df = pd.DataFrame(rows).sort_values(["summary_type", "summary_path"])
    audit_path = output_dir / "reproducibility_audit.csv"
    audit_df.to_csv(audit_path, index=False)

    fail_count = int((~audit_df["all_required_present"]).sum())
    strict_fail_count = int((~audit_df["strict_pass"]).sum())
    paper_strict_pass = bool(args.strict and is_paper_root and fail_count == 0 and strict_fail_count == 0)
    overall = {
        "results_root": str(results_root),
        "summary_files_found": int(len(summary_files)),
        "model_run_summaries": int((audit_df["summary_type"] == "model_run").sum()),
        "aggregate_summaries": int((audit_df["summary_type"] == "aggregate").sum()),
        "pass_count": int(audit_df["all_required_present"].sum()),
        "fail_count": fail_count,
        "strict_pass_count": int(audit_df["strict_pass"].sum()),
        "strict_fail_count": strict_fail_count,
        "pass_rate": float(audit_df["all_required_present"].mean()),
        "strict_mode": bool(args.strict),
        "paper_strict_pass": paper_strict_pass,
        "audit_csv": str(audit_path),
    }
    summary_path = output_dir / "reproducibility_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    print(f"Saved reproducibility audit CSV: {audit_path}")
    print(f"Saved reproducibility summary JSON: {summary_path}")
    print(f"Pass rate: {overall['pass_rate']:.3f} ({overall['pass_count']}/{overall['summary_files_found']})")
    if args.strict:
        print(
            "Strict pass rate: "
            f"{overall['strict_pass_count'] / max(1, overall['summary_files_found']):.3f} "
            f"({overall['strict_pass_count']}/{overall['summary_files_found']})"
        )

    if args.strict and (overall["fail_count"] > 0 or overall["strict_fail_count"] > 0):
        failing_required = audit_df.loc[~audit_df["all_required_present"], ["summary_path", "missing_paths"]]
        failing_strict = audit_df.loc[~audit_df["strict_pass"], ["summary_path", "strict_errors"]]
        if not failing_required.empty:
            print("Required-path failures:")
            for _, row in failing_required.iterrows():
                print(f"- {row['summary_path']}: {row['missing_paths']}")
        if not failing_strict.empty:
            print("Strict-validation failures:")
            for _, row in failing_strict.iterrows():
                print(f"- {row['summary_path']}: {row['strict_errors']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
