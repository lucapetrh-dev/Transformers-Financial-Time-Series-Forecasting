from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd


def test_reproducibility_audit_runner_real_data(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    out_summary = results_root / "btc" / "baselines" / "baselines_summary.csv"
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    run_cmd = [
        "python",
        "forecast/runners/run_baselines.py",
        "--data-path",
        "data/btc_timeseries.csv",
        "--horizon",
        "1",
        "--min-train-size",
        "500",
        "--test-size",
        "60",
        "--step-size",
        "120",
        "--no-arima",
        "--no-xgboost",
        "--output",
        str(out_summary),
    ]
    subprocess.run(run_cmd, check=True)

    audit_dir = tmp_path / "reproducibility"
    audit_cmd = [
        "python",
        "forecast/runners/run_reproducibility_audit.py",
        "--results-root",
        str(results_root),
        "--output-dir",
        str(audit_dir),
        "--strict",
    ]
    subprocess.run(audit_cmd, check=True)

    audit_csv = audit_dir / "reproducibility_audit.csv"
    summary_json = audit_dir / "reproducibility_summary.json"
    assert audit_csv.exists()
    assert summary_json.exists()

    df = pd.read_csv(audit_csv)
    assert not df.empty
    assert df["all_required_present"].all()

    with summary_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["fail_count"] == 0
    assert payload["pass_rate"] == 1.0
