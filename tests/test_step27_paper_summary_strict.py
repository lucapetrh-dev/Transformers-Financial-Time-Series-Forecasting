from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ASSETS = ["btc", "eth", "ada", "doge", "xmr", "xrp"]


def _rows_for_assets(model: str, *, with_track: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for asset in ASSETS:
        row: dict[str, object] = {
            "asset": asset,
            "mode": "no_sentiment",
            "model": model,
            "mae_mean": 0.01,
            "rmse_mean": 0.02,
            "directional_accuracy_mean": 0.5,
            "sharpe_5bps_mean": 0.0,
            "dsr_mean": 0.0,
        }
        if with_track:
            row["objective_track"] = "point"
        rows.append(row)
    return rows


def _write_minimal_inputs(root: Path, *, include_transformer_track: bool = True, fallback_row: bool = False) -> None:
    base = pd.DataFrame(
        _rows_for_assets("linear_ridge", with_track=False) + _rows_for_assets("random_walk", with_track=False)
    )
    trf = pd.DataFrame(
        _rows_for_assets("patchtst_like", with_track=include_transformer_track)
        + _rows_for_assets("itransformer_like", with_track=include_transformer_track)
        + _rows_for_assets("random_walk_sequence", with_track=include_transformer_track)
    )
    fnd = pd.DataFrame(
        _rows_for_assets("chronos2_zero_shot", with_track=False) + _rows_for_assets("random_walk_scaled", with_track=False)
    )
    if fallback_row:
        fnd.loc[0, "backend_status"] = "fallback_persistence"

    abl = pd.DataFrame(
        [{"asset": asset, "mode": "financial_only", "model": "dummy", "mae_mean": 0.01} for asset in ASSETS]
    )

    base.to_csv(root / "multi_asset_baselines_h1_paired_summary.csv", index=False)
    trf.to_csv(root / "multi_asset_transformers_h1_paired_summary.csv", index=False)
    fnd.to_csv(root / "multi_asset_chronos2_h1_summary.csv", index=False)
    abl.to_csv(root / "feature_ablation_h1_summary_best.csv", index=False)


def _write_inference(root: Path, prefix: str = "TEST_H1") -> None:
    ci = pd.DataFrame(
        [
            {
                "asset": asset,
                "mode": "no_sentiment",
                "objective_track": "point",
                "model": "linear_ridge",
                "mae_ci95_low": 0.009,
                "mae_ci95_high": 0.011,
            }
            for asset in ASSETS
        ]
    )
    binom = pd.DataFrame(
        [
            {
                "asset": asset,
                "mode": "no_sentiment",
                "objective_track": "point",
                "model": "linear_ridge",
                "n_obs": 10,
                "n_success": 5,
                "hit_rate": 0.5,
                "p_value": 1.0,
            }
            for asset in ASSETS
        ]
    )
    mcs = pd.DataFrame(
        [
            {
                "asset": asset,
                "mode": "no_sentiment",
                "objective_track": "point",
                "model": "linear_ridge",
                "mcs_member": True,
                "elimination_rank": None,
                "p_value_vs_best": None,
                "mean_loss": 0.01,
            }
            for asset in ASSETS
        ]
    )
    ci.to_csv(root / f"{prefix}_metric_ci95.csv", index=False)
    binom.to_csv(root / f"{prefix}_directional_binomial.csv", index=False)
    mcs.to_csv(root / f"{prefix}_mcs.csv", index=False)


def _run_summary(root: Path, prefix: str = "TEST_H1") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "forecast/runners/run_paper_summary.py",
            "--results-root",
            str(root),
            "--horizon",
            "1",
            "--output-prefix",
            prefix,
            "--strict",
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_paper_summary_strict_fails_when_inference_missing(tmp_path: Path) -> None:
    _write_minimal_inputs(tmp_path)
    proc = _run_summary(tmp_path)
    assert proc.returncode != 0
    assert "inference" in proc.stderr.lower() or "missing required inference file" in proc.stderr.lower()


def test_paper_summary_strict_fails_without_objective_track(tmp_path: Path) -> None:
    _write_minimal_inputs(tmp_path, include_transformer_track=False)
    _write_inference(tmp_path)
    proc = _run_summary(tmp_path)
    assert proc.returncode != 0
    assert "objective_track" in proc.stderr


def test_paper_summary_strict_fails_on_fallback_rows(tmp_path: Path) -> None:
    _write_minimal_inputs(tmp_path, include_transformer_track=True, fallback_row=True)
    _write_inference(tmp_path)
    proc = _run_summary(tmp_path)
    assert proc.returncode != 0
    assert "backend_status" in proc.stderr.lower() or "fallback" in proc.stderr.lower()


def test_paper_summary_strict_passes_on_valid_inputs(tmp_path: Path) -> None:
    _write_minimal_inputs(tmp_path, include_transformer_track=True, fallback_row=False)
    _write_inference(tmp_path)
    proc = _run_summary(tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "TEST_H1_best_by_asset.csv").exists()
    assert (tmp_path / "TEST_H1_RESULTS_SUMMARY.md").exists()
