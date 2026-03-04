from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf


def compute_residual_autocorrelation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_lags: int = 40,
) -> pd.DataFrame:
    """Compute residual ACF/PACF with 95% ACF confidence intervals."""
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    resid = yt - yp
    resid = resid[np.isfinite(resid)]

    if len(resid) < 5:
        return pd.DataFrame(columns=["lag", "acf", "pacf", "acf_ci_upper", "acf_ci_lower"])

    nlags = max(1, min(int(max_lags), len(resid) - 1))
    acf_vals, acf_ci = acf(resid, nlags=nlags, alpha=0.05, fft=True)

    pacf_nlags = min(nlags, max(1, len(resid) // 2 - 1))
    pacf_vals = pacf(resid, nlags=pacf_nlags, method="ywm")
    pacf_full = np.full(nlags + 1, np.nan, dtype=float)
    pacf_full[: len(pacf_vals)] = pacf_vals

    out = pd.DataFrame(
        {
            "lag": np.arange(nlags + 1, dtype=int),
            "acf": acf_vals,
            "pacf": pacf_full,
            "acf_ci_upper": acf_ci[:, 1],
            "acf_ci_lower": acf_ci[:, 0],
        }
    )
    return out


def analyze_training_convergence(history_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize training convergence from epoch-level history logs."""
    required = {"epoch", "train_loss", "val_loss"}
    if not required.issubset(set(history_df.columns)):
        raise ValueError(f"history_df must include columns: {sorted(required)}")

    group_cols = [c for c in ["asset", "model", "mode", "fold"] if c in history_df.columns]
    if not group_cols:
        raise ValueError("history_df must include at least one of asset/model/mode/fold")

    extra_cols = [c for c in ["objective_track"] if c in history_df.columns]
    all_group_cols = group_cols + extra_cols

    rows: list[dict[str, float | int | str]] = []
    for group_key, grp in history_df.groupby(all_group_cols, dropna=False):
        group_vals = group_key if isinstance(group_key, tuple) else (group_key,)
        row = {col: group_vals[idx] for idx, col in enumerate(all_group_cols)}

        g = grp.sort_values("epoch")
        final = g.iloc[-1]
        best_idx = g["val_loss"].idxmin()
        best_row = g.loc[best_idx]

        row.update(
            {
                "final_train_loss": float(final["train_loss"]),
                "final_val_loss": float(final["val_loss"]),
                "best_val_loss": float(best_row["val_loss"]),
                "best_epoch": int(best_row["epoch"]),
                "total_epochs": int(g["epoch"].nunique()),
                "train_val_gap": float(final["val_loss"] - final["train_loss"]),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(all_group_cols).reset_index(drop=True)


def _variance_ratio(returns: np.ndarray, q: int) -> float:
    if q <= 1 or len(returns) <= q:
        return np.nan
    var_1 = float(np.var(returns, ddof=1))
    if not np.isfinite(var_1) or var_1 <= 0.0:
        return np.nan

    q_ret = pd.Series(returns).rolling(q).sum().dropna().to_numpy(dtype=float)
    if len(q_ret) < 2:
        return np.nan
    var_q = float(np.var(q_ret, ddof=1))
    return float(var_q / (q * var_1))


def estimate_return_predictability(
    returns: np.ndarray,
    max_lags: int = 20,
) -> dict[str, float | pd.DataFrame]:
    """Estimate serial predictability via ACF, Ljung-Box, and variance-ratio checks."""
    r = np.asarray(returns, dtype=float).reshape(-1)
    r = r[np.isfinite(r)]
    if len(r) < 10:
        raise ValueError("Need at least 10 return observations for predictability diagnostics")

    nlags = max(1, min(int(max_lags), len(r) - 1))
    acf_vals = acf(r, nlags=nlags, fft=True)
    acf_table = pd.DataFrame({"lag": np.arange(1, nlags + 1, dtype=int), "acf": acf_vals[1 : nlags + 1]})

    lb_lag = max(1, min(nlags, len(r) // 4))
    lb = acorr_ljungbox(r, lags=[lb_lag], return_df=True)
    ljung_box_stat = float(lb["lb_stat"].iloc[0])
    ljung_box_p = float(lb["lb_pvalue"].iloc[0])

    variance_ratios = pd.DataFrame(
        {
            "q": [2, 5, 10, 20],
            "variance_ratio": [_variance_ratio(r, q) for q in [2, 5, 10, 20]],
        }
    )

    return {
        "acf_table": acf_table,
        "ljung_box_stat": ljung_box_stat,
        "ljung_box_p": ljung_box_p,
        "variance_ratios": variance_ratios,
    }


def analyze_hpo_sensitivity(trials_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Aggregate HPO trials by architecture dimensions and learning-rate bins."""
    required = {"d_model", "n_layers", "lr", "mae_mean"}
    missing = required.difference(trials_df.columns)
    if missing:
        raise ValueError(f"trials_df missing required columns: {sorted(missing)}")

    df = trials_df.copy()
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "ok"]

    for col in ["d_model", "n_layers", "lr", "mae_mean"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["d_model", "n_layers", "lr", "mae_mean"])

    if df.empty:
        empty = pd.DataFrame()
        return {
            "d_model_sensitivity": empty,
            "n_layers_sensitivity": empty,
            "lr_sensitivity": empty,
            "interaction_matrix": empty,
        }

    d_model_sens = (
        df.groupby("d_model", as_index=False)
        .agg(avg_mae=("mae_mean", "mean"), median_mae=("mae_mean", "median"), std_mae=("mae_mean", "std"), n_trials=("mae_mean", "count"))
        .sort_values("d_model")
        .reset_index(drop=True)
    )

    n_layers_sens = (
        df.groupby("n_layers", as_index=False)
        .agg(avg_mae=("mae_mean", "mean"), median_mae=("mae_mean", "median"), std_mae=("mae_mean", "std"), n_trials=("mae_mean", "count"))
        .sort_values("n_layers")
        .reset_index(drop=True)
    )

    bins = [0.0, 3e-4, 8e-4, 2e-3, np.inf]
    labels = ["<=3e-4", "3e-4_to_8e-4", "8e-4_to_2e-3", ">2e-3"]
    df["lr_bin"] = pd.cut(df["lr"], bins=bins, labels=labels, include_lowest=True)
    lr_sens = (
        df.groupby("lr_bin", as_index=False)
        .agg(avg_mae=("mae_mean", "mean"), median_mae=("mae_mean", "median"), std_mae=("mae_mean", "std"), n_trials=("mae_mean", "count"))
        .sort_values("lr_bin")
        .reset_index(drop=True)
    )

    interaction = (
        df.pivot_table(index="n_layers", columns="d_model", values="mae_mean", aggfunc="mean")
        .sort_index()
        .sort_index(axis=1)
    )
    interaction = interaction.rename_axis(index="n_layers", columns="d_model").reset_index()

    return {
        "d_model_sensitivity": d_model_sens,
        "n_layers_sensitivity": n_layers_sens,
        "lr_sensitivity": lr_sens,
        "interaction_matrix": interaction,
    }
