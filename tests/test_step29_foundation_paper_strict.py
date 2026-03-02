from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from forecast.pipeline.foundation_adapters import build_foundation_adapter


def _write_synthetic_market_csv(path: Path, n_rows: int = 140) -> None:
    ts = pd.date_range("2021-01-01", periods=n_rows, freq="D")
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, size=n_rows))
    df = pd.DataFrame({"time": ts, "close": close})
    df.to_csv(path, index=False)


def test_foundation_paper_strict_runs_native_moirai(tmp_path: Path) -> None:
    data_path = tmp_path / "synthetic.csv"
    _write_synthetic_market_csv(data_path)

    proc = subprocess.run(
        [
            sys.executable,
            "forecast/runners/run_foundation_zero_shot.py",
            "--data-path",
            str(data_path),
            "--model-key",
            "moirai",
            "--horizon",
            "1",
            "--min-train-size",
            "30",
            "--test-size",
            "10",
            "--step-size",
            "20",
            "--paper-strict",
            "--output",
            str(tmp_path / "out.csv"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_foundation_native_adapter_paths_are_available() -> None:
    expected = {
        "chronos2": "chronos2_zero_shot",
        "timesfm": "timesfm_zero_shot",
        "moirai": "moirai_zero_shot",
        "lagllama": "lagllama_zero_shot",
    }
    for key, model_name in expected.items():
        adapter = build_foundation_adapter(model_key=key)
        assert str(adapter.backend_status).lower() == "native"
        assert str(adapter.model_name) == model_name
