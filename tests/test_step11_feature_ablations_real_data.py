from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd


def test_feature_ablation_runner_real_data(tmp_path: Path) -> None:
    out_path = tmp_path / "feature_ablation_summary.csv"
    cmd = [
        "python",
        "forecast/runners/run_feature_ablations.py",
        "--data-glob",
        "data/*_timeseries.csv",
        "--limit-assets",
        "2",
        "--horizon",
        "1",
        "--n-splits",
        "4",
        "--embargo",
        "5",
        "--label-horizon",
        "1",
        "--no-arima",
        "--no-xgboost",
        "--output",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)

    assert out_path.exists()
    best_path = out_path.with_name(out_path.stem + "_best.csv")
    aggregate_path = out_path.with_name(out_path.stem + "_aggregate.csv")
    deltas_path = out_path.with_name(out_path.stem + "_mode_deltas.csv")
    audit_path = out_path.with_name(out_path.stem + "_comparability_audit.csv")
    assert best_path.exists()
    assert aggregate_path.exists()
    assert deltas_path.exists()
    assert audit_path.exists()

    summary = pd.read_csv(out_path)
    assert not summary.empty
    assert {"asset", "mode", "model", "mae_mean"}.issubset(summary.columns)
    expected_modes = {"financial_only", "financial_plus_sentiment", "financial_plus_sentiment_plus_regime"}
    assert expected_modes.issubset(set(summary["mode"].unique().tolist()))

    audit = pd.read_csv(audit_path)
    assert not audit.empty
    assert audit["consistent_target_space_all"].all()
