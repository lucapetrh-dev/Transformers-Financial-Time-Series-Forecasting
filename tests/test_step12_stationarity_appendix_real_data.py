from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd


def test_stationarity_appendix_runner_real_data(tmp_path: Path) -> None:
    out_dir = tmp_path / "stationarity_appendix"
    cmd = [
        "python",
        "forecast/runners/run_stationarity_appendix.py",
        "--data-path",
        "data/btc_timeseries.csv",
        "--time-col",
        "time",
        "--output-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)

    summary_path = out_dir / "btc_stationarity_summary.csv"
    tests_path = out_dir / "btc_stationarity_tests.csv"
    rolling_plot = out_dir / "btc_rolling_diagnostics.png"
    acf_pacf_plot = out_dir / "btc_acf_pacf.png"
    fft_plot = out_dir / "btc_fft.png"
    assert summary_path.exists()
    assert tests_path.exists()
    assert rolling_plot.exists()
    assert acf_pacf_plot.exists()
    assert fft_plot.exists()

    tests = pd.read_csv(tests_path)
    assert not tests.empty
    assert {"series", "test", "statistic", "p_value", "stationary_at_5pct"}.issubset(tests.columns)
    assert {"log_price", "return_1d"}.issubset(set(tests["series"].unique()))
    assert {"adf", "kpss"}.issubset(set(tests["test"].unique()))


def test_stationarity_appendix_notebook_exists() -> None:
    nb_path = Path(".docs/appendix_stationarity_notebook.ipynb")
    assert nb_path.exists()
