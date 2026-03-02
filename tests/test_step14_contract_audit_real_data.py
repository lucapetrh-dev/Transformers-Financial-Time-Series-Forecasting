from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

from forecast.pipeline.paper_contract import load_paper_contract


def test_paper_contract_loads_and_has_expected_assets() -> None:
    contract = load_paper_contract("forecast/config/paper_experiment.json")
    assert contract.target_space == "log_return"
    assert set(contract.target_horizons_days) == {1, 5, 20}
    assets = {a.asset for a in contract.assets}
    assert assets == {"btc", "eth", "ada", "doge", "xmr", "xrp"}


def test_contract_audit_runner_real_data(tmp_path: Path) -> None:
    out_path = tmp_path / "contract_manifest.csv"
    manifest_path = tmp_path / "data_manifest.csv"
    manifest_cmd = [
        "python",
        "forecast/runners/run_data_manifest.py",
        "--data-glob",
        ".data/hourly/*_timeseries_hourly.csv",
        "--include-assets",
        "btc,eth,ada,doge,xmr,xrp",
        "--output",
        str(manifest_path),
    ]
    subprocess.run(manifest_cmd, check=True)

    cmd = [
        "python",
        "forecast/runners/run_contract_audit.py",
        "--contract-path",
        "forecast/config/paper_experiment.json",
        "--manifest-path",
        str(manifest_path),
        "--require-provenance",
        "--output",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)

    assert out_path.exists()
    df = pd.read_csv(out_path)
    assert not df.empty
    assert set(df["asset"].tolist()) == {"btc", "eth", "ada", "doge", "xmr", "xrp"}
    assert df["exists"].all()
