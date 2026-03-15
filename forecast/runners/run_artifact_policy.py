from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path


DETAILED_SUFFIXES = (
    "_folds.csv",
    "_fold_manifest.csv",
    "_metadata.json",
    "_data_quality_summary.csv",
    "_missingness_stages.csv",
    "_history.csv",
    "_quantiles.csv",
)

ACTIVE_KEEP_FILENAMES = {
    "multi_asset_baselines_h1_paired_summary.csv",
    "multi_asset_baselines_h1_paired_summary_ranks.csv",
    "multi_asset_baselines_h1_paired_summary_comparability_audit.csv",
    "multi_asset_baselines_h1_paired_summary_predictions.csv",
    "multi_asset_transformers_h1_paired_summary.csv",
    "multi_asset_transformers_h1_paired_summary_ranks.csv",
    "multi_asset_transformers_h1_paired_summary_comparability_audit.csv",
    "multi_asset_transformers_h1_paired_summary_predictions.csv",
    "multi_asset_chronos2_h1_summary.csv",
    "multi_asset_chronos2_h1_summary_ranks.csv",
    "multi_asset_chronos2_h1_summary_comparability_audit.csv",
    "multi_asset_chronos2_h1_summary_predictions.csv",
    "multi_asset_foundation_h1_summary.csv",
    "multi_asset_foundation_h1_summary_ranks.csv",
    "multi_asset_foundation_h1_summary_comparability_audit.csv",
    "multi_asset_foundation_h1_summary_predictions.csv",
    "feature_ablation_h1_summary.csv",
    "feature_ablation_h1_summary_best.csv",
    "feature_ablation_h1_summary_aggregate.csv",
    "feature_ablation_h1_summary_mode_deltas.csv",
    "feature_ablation_h1_summary_comparability_audit.csv",
    "contract_asset_manifest.csv",
    "data_manifest.csv",
    "data_manifest_aggregation_policy.csv",
    "PAPER_H1_RESULTS_SUMMARY.md",
    "PAPER_H1_best_by_asset_mode.csv",
    "PAPER_H1_best_by_asset.csv",
    "PAPER_H1_sentiment_delta.csv",
    "PAPER_H1_family_performance.csv",
    "PAPER_H1_ablation_mode_wins.csv",
    "PAPER_H1_naive_reference.csv",
    "PAPER_H1_skill_vs_zero.csv",
    "PAPER_H1_metric_ci95.csv",
    "PAPER_H1_directional_binomial.csv",
    "PAPER_H1_mcs.csv",
}


def _archive_target(root: Path, archive_dir: Path, src: Path) -> Path:
    rel = src.relative_to(root)
    return archive_dir / rel


def _is_under_archive(path: Path, archive_dir: Path) -> bool:
    try:
        path.relative_to(archive_dir)
        return True
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Move detailed paper artifacts into timestamped archive directory")
    parser.add_argument("--results-root", type=str, default="results/paper")
    parser.add_argument("--archive-base", type=str, default="results/paper/_archive_detailed")
    parser.add_argument("--timestamp", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.results_root)
    if not root.exists():
        raise FileNotFoundError(f"Missing results root: {root}")

    timestamp = args.timestamp.strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_dir = Path(args.archive_base) / timestamp
    archive_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    moved_bytes = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if _is_under_archive(p, archive_dir):
            continue
        if p.parent.name == "overleaf_thesis":
            continue
        if p.name in ACTIVE_KEEP_FILENAMES:
            continue

        should_move = p.name.endswith(DETAILED_SUFFIXES)
        if p.name.endswith("_summary.csv") and p.name not in ACTIVE_KEEP_FILENAMES:
            should_move = True
        if "_summary_" in p.name and p.name.endswith(".csv") and p.name not in ACTIVE_KEEP_FILENAMES:
            should_move = True
        if not should_move:
            continue

        dst = _archive_target(root, archive_dir, p)
        dst.parent.mkdir(parents=True, exist_ok=True)
        moved += 1
        moved_bytes += int(p.stat().st_size)
        if not args.dry_run:
            shutil.move(str(p), str(dst))

    print(f"Archive directory: {archive_dir}")
    print(f"Detailed files moved: {moved}")
    print(f"Bytes moved: {moved_bytes}")
    if args.dry_run:
        print("Dry run only; no files moved.")


if __name__ == "__main__":
    main()
