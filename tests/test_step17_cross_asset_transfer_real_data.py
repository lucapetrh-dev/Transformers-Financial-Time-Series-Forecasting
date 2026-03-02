from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd


def test_cross_asset_transfer_runner_real_data(tmp_path: Path) -> None:
    out_path = tmp_path / "transfer_summary.csv"
    cmd = [
        "python",
        "forecast/runners/run_cross_asset_transfer.py",
        "--source-data-path",
        ".data/hourly/btc_lunarcrush_timeseries_hourly.csv",
        "--target-data-path",
        ".data/hourly/eth_lunarcrush_timeseries_hourly.csv",
        "--horizon",
        "1",
        "--lookback",
        "48",
        "--model",
        "dlinear_like",
        "--epochs",
        "2",
        "--finetune-epochs",
        "1",
        "--batch-size",
        "64",
        "--output",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)

    assert out_path.exists()
    preds_path = out_path.with_name(out_path.stem + "_predictions.csv")
    metadata_path = out_path.with_name(out_path.stem + "_metadata.json")
    assert preds_path.exists()
    assert metadata_path.exists()

    summary = pd.read_csv(out_path)
    assert not summary.empty
    expected_modes = {"source_zero_shot", "source_finetuned", "target_only", "random_walk_sequence"}
    assert expected_modes.issubset(set(summary["mode"].tolist()))
    assert {"mae", "rmse", "directional_accuracy", "crps_gaussian", "sharpe_5bps", "dsr"}.issubset(summary.columns)

    preds = pd.read_csv(preds_path)
    assert not preds.empty
    assert {"time", "y_true", "pred_source_zero_shot", "pred_source_finetuned", "pred_target_only"}.issubset(preds.columns)

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    assert metadata["target_space"] == "log_return"
    assert metadata["transfer_protocol"] == "pretrain_source_then_zero_shot_or_finetune_on_target"
