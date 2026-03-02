from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ADJ_CLOSE_CANDIDATES = ["adjusted_close", "adj_close", "Adj Close", "adjclose", "close"]


@dataclass
class FeatureConfig:
    lookbacks: tuple[int, ...] = (20, 60)
    ewma_span: int = 20
    roc_period: int = 5
    use_regime_features: bool = False
    regime_lookbacks: tuple[int, ...] = (20, 60)
    regime_z_low: float = -0.5
    regime_z_high: float = 0.5


def _find_price_column(df: pd.DataFrame) -> str:
    existing = {c.lower(): c for c in df.columns}
    for candidate in ADJ_CLOSE_CANDIDATES:
        key = candidate.lower()
        if key in existing:
            return existing[key]
    raise ValueError("No close/adjusted close column found.")


def _resolve_data_path(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p

    if not p.is_absolute():
        parts = p.parts
        if parts and parts[0] == "data":
            alt = Path(".data").joinpath(*parts[1:])
            if alt.exists():
                return alt

    raise FileNotFoundError(f"Data file not found: {path}")


def _aggregation_rule(column_name: str) -> str:
    col = column_name.lower()

    if col in {"open", "open_price"} or col.startswith("open_"):
        return "first"
    if col in {"high", "high_price"} or col.startswith("high_"):
        return "max"
    if col in {"low", "low_price"} or col.startswith("low_"):
        return "min"
    if col in {c.lower() for c in ADJ_CLOSE_CANDIDATES} or col in {"close", "close_price"}:
        return "last"

    if "market_cap" in col or "circulating_supply" in col or col.endswith("_supply"):
        return "last"

    # Count-like event columns are daily totals.
    if any(
        key in col
        for key in (
            "tweets",
            "tweet_retweets",
            "tweet_replies",
            "tweet_favorites",
            "reddit_posts",
            "reddit_comments",
            "news",
            "url_shares",
            "unique_url_shares",
            "social_contributors",
            "social_volume",
        )
    ):
        return "sum"

    if col == "volume":
        return "sum"

    # Score/sentiment/rank-like columns are averaged intraday.
    if any(
        key in col
        for key in (
            "sentiment",
            "score",
            "rank",
            "dominance",
            "impact",
            "correlation",
            "volatility",
            "galaxy",
        )
    ):
        return "mean"

    # Default policy keeps end-of-day state.
    return "last"


def _aggregate_intraday_to_daily(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    if df.empty:
        return df

    out_rows: list[dict[str, object]] = []
    non_time_cols = [c for c in df.columns if c != time_col]

    grouped = df.groupby(df[time_col].dt.floor("D"), sort=True)
    for day, group in grouped:
        row: dict[str, object] = {time_col: day}
        group = group.sort_values(time_col)

        for col in non_time_cols:
            series = group[col]
            if series.isna().all():
                row[col] = np.nan
                continue

            if not pd.api.types.is_numeric_dtype(series):
                row[col] = series.dropna().iloc[-1]
                continue

            rule = _aggregation_rule(col)
            if rule == "first":
                row[col] = series.dropna().iloc[0]
            elif rule == "last":
                row[col] = series.dropna().iloc[-1]
            elif rule == "max":
                row[col] = float(series.max())
            elif rule == "min":
                row[col] = float(series.min())
            elif rule == "mean":
                row[col] = float(series.mean())
            elif rule == "sum":
                row[col] = float(series.sum())
            else:
                row[col] = series.dropna().iloc[-1]

        out_rows.append(row)

    out = pd.DataFrame(out_rows).sort_values(time_col).reset_index(drop=True)
    return out


def load_market_data(path: str, time_col: str = "time") -> pd.DataFrame:
    resolved_path = _resolve_data_path(path)
    df = pd.read_csv(resolved_path)
    if time_col not in df.columns:
        raise ValueError(f"Missing time column '{time_col}' in {path}")

    df[time_col] = pd.to_datetime(df[time_col], utc=False, errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).drop_duplicates(subset=[time_col], keep="last")

    # If intraday rows exist, collapse to daily observations with column-aware causal aggregation.
    if (df[time_col].dt.floor("D") != df[time_col]).any():
        df = _aggregate_intraday_to_daily(df, time_col=time_col)
        df = df.dropna(how="all").reset_index(drop=True)

    return df


def build_feature_frame(df: pd.DataFrame, config: FeatureConfig | None = None, time_col: str = "time") -> pd.DataFrame:
    if config is None:
        config = FeatureConfig()

    out = df.copy()
    price_col = _find_price_column(out)
    out = out.sort_values(time_col).reset_index(drop=True)

    out["log_price"] = np.log(out[price_col].clip(lower=1e-12))
    out["return_1d"] = out["log_price"].diff()

    dow = out[time_col].dt.dayofweek
    month = out[time_col].dt.month
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    for lb in config.lookbacks:
        out[f"ret_mean_{lb}"] = out["return_1d"].rolling(lb, min_periods=lb).mean()
        out[f"ret_std_{lb}"] = out["return_1d"].rolling(lb, min_periods=lb).std()

    out["ewma_ret"] = out["return_1d"].ewm(span=config.ewma_span, adjust=False).mean()
    out["ewma_roc"] = out["ewma_ret"].pct_change(config.roc_period)

    if config.use_regime_features:
        for lb in config.regime_lookbacks:
            vol_col = f"realized_vol_{lb}"
            vol = out["return_1d"].rolling(lb, min_periods=lb).std()
            vol_mean = vol.rolling(lb, min_periods=lb).mean()
            vol_std = vol.rolling(lb, min_periods=lb).std()
            vol_z = (vol - vol_mean) / (vol_std + 1e-12)
            regime = np.where(vol_z <= config.regime_z_low, 0.0, np.where(vol_z >= config.regime_z_high, 2.0, 1.0))

            out[vol_col] = vol
            out[f"vol_z_{lb}"] = vol_z
            out[f"vol_regime_{lb}"] = regime

    return out


def add_targets(
    df: pd.DataFrame,
    horizons: Iterable[int] = (1, 5, 20),
    add_direction: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    for h in horizons:
        ret_col = f"target_ret_{h}d"
        out[ret_col] = out["log_price"].shift(-h) - out["log_price"]
        if add_direction:
            out[f"target_dir_{h}d"] = (out[ret_col] > 0.0).astype(float)
    return out


def make_lag_features(df: pd.DataFrame, source_col: str = "return_1d", max_lag: int = 20) -> pd.DataFrame:
    out = df.copy()
    for lag in range(1, max_lag + 1):
        out[f"{source_col}_lag_{lag}"] = out[source_col].shift(lag)
    return out


def drop_na_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan).dropna(axis=0).reset_index(drop=True)
