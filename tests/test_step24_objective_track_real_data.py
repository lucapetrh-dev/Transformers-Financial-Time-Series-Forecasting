from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd


def test_transformers_objective_track_both_real_data(tmp_path: Path) -> None:
    out_path = tmp_path / "transformers_summary.csv"
    cmd = [
        "python",
        "forecast/runners/run_transformers.py",
        "--data-path",
        ".data/hourly/btc_lunarcrush_timeseries_hourly.csv",
        "--time-col",
        "time",
        "--horizon",
        "1",
        "--lookback",
        "48",
        "--min-train-size",
        "300",
        "--test-size",
        "60",
        "--step-size",
        "240",
        "--epochs",
        "1",
        "--batch-size",
        "64",
        "--lr",
        "1e-3",
        "--models",
        "patchtst_like",
        "--objective-track",
        "both",
        "--point-loss",
        "mse",
        "--output",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)

    assert out_path.exists()
    summary = pd.read_csv(out_path)
    assert not summary.empty
    assert "objective_track" in summary.columns
    assert {"point", "quantile"}.issubset(set(summary["objective_track"].astype(str).tolist()))
