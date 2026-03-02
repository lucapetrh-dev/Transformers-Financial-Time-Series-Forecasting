from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd


def test_transformer_runner_quantile_mode_real_data(tmp_path: Path) -> None:
    out_path = tmp_path / "transformers_quantile_summary.csv"
    cmd = [
        "python",
        "forecast/runners/run_transformers.py",
        "--data-path",
        ".data/hourly/btc_lunarcrush_timeseries_hourly.csv",
        "--horizon",
        "1",
        "--lookback",
        "48",
        "--min-train-size",
        "300",
        "--test-size",
        "60",
        "--step-size",
        "180",
        "--epochs",
        "2",
        "--batch-size",
        "64",
        "--models",
        "dlinear_like",
        "--probabilistic-mode",
        "quantile",
        "--quantiles",
        "0.1,0.5,0.9",
        "--output",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)

    assert out_path.exists()
    summary = pd.read_csv(out_path)
    assert not summary.empty
    assert {"model", "pinball_q10_mean", "pinball_q50_mean", "pinball_q90_mean", "coverage_80_mean", "wis_80_mean"}.issubset(summary.columns)
    assert "dlinear_like" in set(summary["model"].tolist())

    metadata_path = out_path.with_name(out_path.stem + "_metadata.json")
    assert metadata_path.exists()
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    assert metadata["probabilistic_mode"] == "quantile"
    assert metadata["quantiles"] == [0.1, 0.5, 0.9]
