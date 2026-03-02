from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SENTIMENT_KEYWORDS = (
    "sentiment",
    "tweet",
    "tweets",
    "reddit",
    "social",
    "news",
    "url_shares",
    "galaxy",
    "correlation_rank",
    "price_score",
    "social_impact",
)


def detect_sentiment_columns(
    df: pd.DataFrame,
    exclude_cols: set[str] | None = None,
) -> list[str]:
    if exclude_cols is None:
        exclude_cols = set()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected: list[str] = []
    for col in numeric_cols:
        col_l = col.lower()
        if col in exclude_cols:
            continue
        if col_l.startswith("target_ret_"):
            continue
        if any(key in col_l for key in SENTIMENT_KEYWORDS):
            selected.append(col)

    return sorted(set(selected))


def apply_causal_sentiment_lag(
    df: pd.DataFrame,
    sentiment_cols: list[str],
    lag: int = 1,
) -> pd.DataFrame:
    """
    Shift sentiment columns backward in time so row t only uses information
    available strictly before (or at a chosen lag from) prediction time.
    """
    if lag < 0:
        raise ValueError("lag must be >= 0")
    if not sentiment_cols or lag == 0:
        return df

    out = df.copy()
    for col in sentiment_cols:
        out[col] = out[col].shift(lag)
    return out


def prepare_causal_sentiment_features(
    df: pd.DataFrame,
    sentiment_cols: list[str],
    *,
    lag: int = 1,
    min_non_null_ratio: float = 0.2,
    fill_method: str = "ffill_zero",
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Prepare sentiment features for modeling in a leakage-safe way.

    Steps:
    1) Apply causal lag (no future information).
    2) Drop sentiment columns with insufficient coverage.
    3) Apply causal imputation on kept sentiment columns.
    """
    if not sentiment_cols:
        return df, [], []
    if min_non_null_ratio < 0.0 or min_non_null_ratio > 1.0:
        raise ValueError("min_non_null_ratio must be in [0, 1]")

    out = apply_causal_sentiment_lag(df, sentiment_cols, lag=lag)
    out = out.copy()

    coverage = (
        out[sentiment_cols]
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .mean()
    )
    usable_cols = [col for col in sentiment_cols if float(coverage.get(col, 0.0)) >= float(min_non_null_ratio)]
    dropped_cols = [col for col in sentiment_cols if col not in usable_cols]

    if usable_cols:
        out[usable_cols] = out[usable_cols].replace([np.inf, -np.inf], np.nan)
        if fill_method == "ffill_zero":
            # Causal fill: carry last available value forward, then fill early gaps with 0.
            out[usable_cols] = out[usable_cols].ffill().fillna(0.0)
        elif fill_method == "none":
            pass
        else:
            raise ValueError(f"Unknown fill_method: {fill_method}")

    return out, usable_cols, dropped_cols


def build_sentiment_comparison_table(
    no_sent_summary_path: str | Path,
    with_sent_summary_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    no_df = pd.read_csv(no_sent_summary_path)
    with_df = pd.read_csv(with_sent_summary_path)

    merged = no_df.merge(with_df, on="model", suffixes=("_no_sent", "_with_sent"), how="inner")
    if merged.empty:
        raise ValueError("No overlapping models found between no-sentiment and with-sentiment summaries")

    numeric_base_cols = [c for c in no_df.columns if c != "model" and pd.api.types.is_numeric_dtype(no_df[c])]

    for base_col in numeric_base_cols:
        left = f"{base_col}_no_sent"
        right = f"{base_col}_with_sent"
        if left in merged.columns and right in merged.columns:
            merged[f"{base_col}_delta_with_minus_no"] = merged[right] - merged[left]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged
