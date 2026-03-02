from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_artifact_policy_moves_detailed_files_to_archive(tmp_path: Path) -> None:
    root = tmp_path / "paper"
    root.mkdir(parents=True, exist_ok=True)

    keep_file = root / "PAPER_H1_best_by_asset.csv"
    keep_file.write_text("asset,model\nbtc,linear_ridge\n", encoding="utf-8")
    detailed_file = root / "multi_asset_transformers_h1_paired_summary_btc_no_sent_folds.csv"
    detailed_file.write_text("fold,mae\n0,0.1\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "forecast/runners/run_artifact_policy.py",
            "--results-root",
            str(root),
            "--archive-base",
            str(root / "_archive_detailed"),
            "--timestamp",
            "TEST",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    archived = root / "_archive_detailed" / "TEST" / "multi_asset_transformers_h1_paired_summary_btc_no_sent_folds.csv"
    assert archived.exists()
    assert keep_file.exists()
    assert not detailed_file.exists()

