from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from forecast.pipeline.data_pipeline import load_market_data


def test_hourly_to_daily_aggregation_ohlc_real_btc_data() -> None:
    path = ".data/hourly/btc_lunarcrush_timeseries_hourly.csv"
    raw = pd.read_csv(path, usecols=["time", "open", "high", "low", "close"])
    raw["time"] = pd.to_datetime(raw["time"], errors="coerce")
    raw = raw.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    raw["day"] = raw["time"].dt.floor("D")

    counts = raw.groupby("day").size()
    candidate_days = counts[counts > 3]
    assert not candidate_days.empty
    day = candidate_days.index[0]
    intra = raw.loc[raw["day"] == day].copy()
    intra = intra.sort_values("time")

    expected_open = float(intra["open"].iloc[0])
    expected_high = float(intra["high"].max())
    expected_low = float(intra["low"].min())
    expected_close = float(intra["close"].iloc[-1])

    daily = load_market_data(path, time_col="time")
    daily["day"] = pd.to_datetime(daily["time"], errors="coerce").dt.floor("D")
    row = daily.loc[daily["day"] == day]
    assert len(row) == 1

    got = row.iloc[0]
    assert np.isclose(float(got["open"]), expected_open)
    assert np.isclose(float(got["high"]), expected_high)
    assert np.isclose(float(got["low"]), expected_low)
    assert np.isclose(float(got["close"]), expected_close)


def test_data_manifest_runner_real_data(tmp_path: Path) -> None:
    out_path = tmp_path / "data_manifest.csv"
    cmd = [
        "python",
        "forecast/runners/run_data_manifest.py",
        "--data-glob",
        ".data/hourly/*_timeseries_hourly.csv",
        "--include-assets",
        "btc,eth,ada,doge,xmr,xrp",
        "--output",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)

    assert out_path.exists()
    policy_path = out_path.with_name(out_path.stem + "_aggregation_policy.csv")
    assert policy_path.exists()

    manifest = pd.read_csv(out_path)
    assert not manifest.empty
    assert {"asset", "raw_rows", "processed_rows", "intraday_detected", "price_column_selected"}.issubset(manifest.columns)
    assert set(manifest["asset"].tolist()) == {"btc", "eth", "ada", "doge", "xmr", "xrp"}
    assert manifest["intraday_detected"].any()
    assert bool(manifest.loc[manifest["asset"] == "btc", "intraday_detected"].iloc[0]) is True

    policy = pd.read_csv(policy_path)
    assert not policy.empty
    assert {"asset", "column", "aggregation_rule_when_intraday"}.issubset(policy.columns)
