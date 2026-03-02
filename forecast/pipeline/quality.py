from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _col_role(col: str, time_col: str, target_col: str, feature_cols: list[str], sentiment_cols: list[str]) -> str:
    if col == time_col:
        return "time"
    if col == target_col:
        return "target"
    if col in sentiment_cols:
        return "sentiment"
    if col in feature_cols:
        return "feature"
    return "other"


def _stage_profile(
    stage_name: str,
    frame: pd.DataFrame,
    *,
    time_col: str,
    target_col: str,
    feature_cols: list[str],
    sentiment_cols: list[str],
    imputation_policy: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    n_rows = int(len(frame))
    for col in frame.columns:
        series = frame[col]
        missing_count = int(series.isna().sum())
        missing_pct = (100.0 * missing_count / n_rows) if n_rows > 0 else np.nan
        rows.append(
            {
                "stage": stage_name,
                "column": col,
                "column_role": _col_role(col, time_col=time_col, target_col=target_col, feature_cols=feature_cols, sentiment_cols=sentiment_cols),
                "dtype": str(series.dtype),
                "row_count": n_rows,
                "missing_count": missing_count,
                "missing_pct": missing_pct,
                "non_missing_count": n_rows - missing_count,
                "non_missing_pct": (100.0 - missing_pct) if n_rows > 0 else np.nan,
                "imputation_policy": imputation_policy,
                # Current pipeline is dropna-only (no value imputation).
                "imputed_pct": 0.0,
            }
        )
    return pd.DataFrame(rows)


def save_preprocessing_quality_artifacts(
    output_path: str | Path,
    *,
    raw_df: pd.DataFrame,
    frame_before_sentiment: pd.DataFrame,
    frame_after_sentiment: pd.DataFrame,
    frame_before_dropna: pd.DataFrame,
    frame_after_dropna: pd.DataFrame,
    time_col: str,
    target_col: str,
    feature_cols: list[str],
    sentiment_cols: list[str],
    sentiment_lag: int,
    imputation_policy: str = "dropna_only_no_value_imputation",
) -> tuple[Path, Path]:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    raw_profile = _stage_profile(
        "raw_loaded",
        raw_df,
        time_col=time_col,
        target_col=target_col,
        feature_cols=feature_cols,
        sentiment_cols=sentiment_cols,
        imputation_policy=imputation_policy,
    )
    pre_sent_profile = _stage_profile(
        "feature_frame_pre_sentiment_lag",
        frame_before_sentiment,
        time_col=time_col,
        target_col=target_col,
        feature_cols=feature_cols,
        sentiment_cols=sentiment_cols,
        imputation_policy=imputation_policy,
    )
    post_sent_profile = _stage_profile(
        "feature_frame_post_sentiment_lag",
        frame_after_sentiment,
        time_col=time_col,
        target_col=target_col,
        feature_cols=feature_cols,
        sentiment_cols=sentiment_cols,
        imputation_policy=imputation_policy,
    )
    pre_dropna_profile = _stage_profile(
        "modeling_frame_pre_dropna",
        frame_before_dropna,
        time_col=time_col,
        target_col=target_col,
        feature_cols=feature_cols,
        sentiment_cols=sentiment_cols,
        imputation_policy=imputation_policy,
    )
    post_dropna_profile = _stage_profile(
        "modeling_frame_post_dropna",
        frame_after_dropna,
        time_col=time_col,
        target_col=target_col,
        feature_cols=feature_cols,
        sentiment_cols=sentiment_cols,
        imputation_policy=imputation_policy,
    )

    stage_df = pd.concat(
        [raw_profile, pre_sent_profile, post_sent_profile, pre_dropna_profile, post_dropna_profile],
        ignore_index=True,
    )
    stages_path = out.with_name(out.stem + "_missingness_stages.csv")
    stage_df.to_csv(stages_path, index=False)

    pre_complete = int(frame_before_sentiment.dropna().shape[0])
    post_complete = int(frame_after_sentiment.dropna().shape[0])
    rows_lost_from_sentiment_lag = max(0, pre_complete - post_complete)

    summary_row = {
        "raw_rows": int(len(raw_df)),
        "feature_rows_pre_sentiment_lag": int(len(frame_before_sentiment)),
        "feature_rows_post_sentiment_lag": int(len(frame_after_sentiment)),
        "model_rows_pre_dropna": int(len(frame_before_dropna)),
        "model_rows_post_dropna": int(len(frame_after_dropna)),
        "rows_dropped_by_dropna": max(0, int(len(frame_before_dropna) - len(frame_after_dropna))),
        "rows_dropped_by_dropna_pct": (100.0 * max(0, int(len(frame_before_dropna) - len(frame_after_dropna))) / int(len(frame_before_dropna)))
        if len(frame_before_dropna) > 0
        else np.nan,
        "rows_lost_from_sentiment_lag": rows_lost_from_sentiment_lag,
        "rows_lost_from_sentiment_lag_pct": (100.0 * rows_lost_from_sentiment_lag / pre_complete) if pre_complete > 0 else np.nan,
        "sentiment_feature_count": int(len(sentiment_cols)),
        "sentiment_lag_effective": int(sentiment_lag if sentiment_cols else 0),
        "imputation_policy": imputation_policy,
        "missing_cells_pre_dropna": int(frame_before_dropna.isna().sum().sum()),
        "missing_cells_post_dropna": int(frame_after_dropna.isna().sum().sum()),
        "feature_count_total": int(len(feature_cols)),
        "sentiment_columns": ";".join(sorted(sentiment_cols)),
    }
    summary_df = pd.DataFrame([summary_row])
    summary_path = out.with_name(out.stem + "_data_quality_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    return summary_path, stages_path
