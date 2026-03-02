from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _write_model_run_bundle(root: Path, *, predictions_path: str | None, quantiles_path: str | None) -> None:
    summary = root / "sample_summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"model": "linear_ridge", "mae_mean": 0.1}]).to_csv(summary, index=False)
    pd.DataFrame([{"fold": 0}]).to_csv(root / "sample_summary_folds.csv", index=False)
    pd.DataFrame([{"fold": 0}]).to_csv(root / "sample_summary_fold_manifest.csv", index=False)
    pd.DataFrame([{"stage": "after_dropna"}]).to_csv(root / "sample_summary_data_quality_summary.csv", index=False)
    pd.DataFrame([{"stage": "after_dropna"}]).to_csv(root / "sample_summary_missingness_stages.csv", index=False)

    payload = {
        "predictions_path": predictions_path,
        "quantiles_path": quantiles_path,
        "history_path": None,
        "data_quality_summary_path": str(root / "sample_summary_data_quality_summary.csv"),
        "missingness_stages_path": str(root / "sample_summary_missingness_stages.csv"),
    }
    with (root / "sample_summary_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def _run_audit(results_root: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "forecast/runners/run_reproducibility_audit.py",
            "--results-root",
            str(results_root),
            "--output-dir",
            str(output_dir),
            "--strict",
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_repro_audit_strict_fails_on_missing_metadata_paths(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    run_root = results_root / "btc" / "sample"
    run_root.mkdir(parents=True, exist_ok=True)
    _write_model_run_bundle(run_root, predictions_path="missing_predictions.csv", quantiles_path=None)

    proc = _run_audit(results_root, tmp_path / "audit")
    assert proc.returncode != 0
    assert "does not exist" in proc.stdout.lower() or "does not exist" in proc.stderr.lower()


def test_repro_audit_strict_fails_on_truth_prefix_references(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    run_root = results_root / "btc" / "sample"
    run_root.mkdir(parents=True, exist_ok=True)
    _write_model_run_bundle(run_root, predictions_path="truth_predictions.csv", quantiles_path=None)

    proc = _run_audit(results_root, tmp_path / "audit")
    assert proc.returncode != 0
    assert "truth_" in proc.stdout.lower() or "truth_" in proc.stderr.lower()


def test_repro_audit_strict_passes_when_metadata_paths_resolve(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    run_root = results_root / "btc" / "sample"
    run_root.mkdir(parents=True, exist_ok=True)

    pred = run_root / "sample_summary_predictions.csv"
    quant = run_root / "sample_summary_quantiles.csv"
    pd.DataFrame([{"x": 1}]).to_csv(pred, index=False)
    pd.DataFrame([{"x": 1}]).to_csv(quant, index=False)
    _write_model_run_bundle(run_root, predictions_path=str(pred), quantiles_path=str(quant))

    proc = _run_audit(results_root, tmp_path / "audit")
    assert proc.returncode == 0, proc.stderr

    payload = json.loads((tmp_path / "audit" / "reproducibility_summary.json").read_text(encoding="utf-8"))
    assert payload["fail_count"] == 0
    assert payload["strict_fail_count"] == 0

