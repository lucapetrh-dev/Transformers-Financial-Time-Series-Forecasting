from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd


def test_statistical_inference_runner_real_data(tmp_path: Path) -> None:
    prefix = "TEST_H1_INFER"
    cmd = [
        "python",
        "forecast/runners/run_statistical_inference.py",
        "--results-root",
        "results/paper",
        "--horizon",
        "1",
        "--output-prefix",
        prefix,
        "--bootstrap-samples",
        "20",
        "--bootstrap-block-size",
        "10",
        "--mcs-bootstrap-samples",
        "20",
        "--mcs-block-size",
        "10",
        "--seed",
        "42",
    ]
    subprocess.run(cmd, check=True)

    root = Path("results/paper")
    ci_path = root / f"{prefix}_metric_ci95.csv"
    binom_path = root / f"{prefix}_directional_binomial.csv"
    mcs_path = root / f"{prefix}_mcs.csv"
    for p in (ci_path, binom_path, mcs_path):
        assert p.exists(), f"Missing inference output: {p}"

    ci = pd.read_csv(ci_path)
    binom = pd.read_csv(binom_path)
    mcs = pd.read_csv(mcs_path)
    assert not ci.empty
    assert not binom.empty
    assert not mcs.empty
    assert {"mae_ci95_low", "mae_ci95_high", "rmse_ci95_low", "rmse_ci95_high"}.issubset(ci.columns)
    assert {"p_value", "hit_rate", "n_success", "n_obs"}.issubset(binom.columns)
    assert {"mcs_member", "model", "asset", "mode"}.issubset(mcs.columns)
