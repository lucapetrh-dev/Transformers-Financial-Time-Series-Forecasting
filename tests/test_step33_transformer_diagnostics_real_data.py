from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_transformer_diagnostics_runner_real_data(tmp_path: Path) -> None:
    out_dir = tmp_path / "transformer_diagnostics"
    cmd = [
        sys.executable,
        "forecast/runners/run_transformer_diagnostics.py",
        "--results-root",
        "results/paper",
        "--data-glob",
        ".data/hourly/*_timeseries_hourly.csv",
        "--include-assets",
        "btc",
        "--output-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)

    residual_path = out_dir / "residual_acf_analysis.csv"
    convergence_path = out_dir / "training_convergence.csv"
    predictability_path = out_dir / "return_predictability.csv"
    metadata_path = out_dir / "metadata.json"

    for path in [residual_path, convergence_path, predictability_path, metadata_path]:
        assert path.exists(), f"Missing output: {path}"

    residual_df = pd.read_csv(residual_path)
    conv_df = pd.read_csv(convergence_path)
    pred_df = pd.read_csv(predictability_path)

    assert not residual_df.empty
    assert not conv_df.empty
    assert not pred_df.empty

    assert {"asset", "model", "mode", "lag", "acf", "pacf", "acf_ci_upper", "acf_ci_lower"}.issubset(residual_df.columns)
    assert {"asset", "model", "mode", "fold", "final_train_loss", "final_val_loss", "best_val_loss", "best_epoch", "total_epochs", "train_val_gap"}.issubset(conv_df.columns)
    assert {"asset", "lag", "acf", "ljung_box_stat", "ljung_box_p", "vr_q2", "vr_q5", "vr_q10", "vr_q20"}.issubset(pred_df.columns)

    assert set(pred_df["asset"].astype(str).str.lower()) == {"btc"}
