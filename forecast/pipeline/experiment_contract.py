from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def infer_frequency_label(ts: pd.Series) -> str:
    if len(ts) < 3:
        return "unknown"

    dt = pd.to_datetime(ts, errors="coerce").dropna().sort_values().diff().dropna()
    if dt.empty:
        return "unknown"

    median_seconds = float(dt.dt.total_seconds().median())
    if median_seconds <= 3600:
        return "hourly_or_faster"
    if median_seconds <= 6 * 3600:
        return "multi_hour"
    if median_seconds <= 36 * 3600:
        return "daily"
    if median_seconds <= 9 * 24 * 3600:
        return "weekly"
    return "coarser_than_weekly"


def save_metadata_json(path: str | Path, metadata: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)


def fold_manifest_rows(
    fold_id: int,
    mode: str,
    train_idx: list[int],
    eval_idx: list[int],
    timestamps: pd.Series,
    embargo: int | None = None,
    label_horizon: int | None = None,
) -> dict:
    ts = pd.to_datetime(timestamps, errors="coerce")
    row = {
        "fold": fold_id,
        "mode": mode,
        "train_size": len(train_idx),
        "eval_size": len(eval_idx),
        "train_start": ts.iloc[train_idx[0]] if train_idx else pd.NaT,
        "train_end": ts.iloc[train_idx[-1]] if train_idx else pd.NaT,
        "eval_start": ts.iloc[eval_idx[0]] if eval_idx else pd.NaT,
        "eval_end": ts.iloc[eval_idx[-1]] if eval_idx else pd.NaT,
    }
    if embargo is not None:
        row["embargo"] = embargo
    if label_horizon is not None:
        row["label_horizon"] = label_horizon
    return row
