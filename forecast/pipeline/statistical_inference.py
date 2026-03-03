from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import binomtest


MetricFn = Callable[[np.ndarray, np.ndarray], float]


def moving_block_bootstrap_indices(
    n: int,
    *,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive")
    b = max(1, min(block_size, n))
    starts = rng.integers(0, n, size=int(np.ceil(n / b)))
    idx: list[int] = []
    for s in starts.tolist():
        for k in range(b):
            idx.append((s + k) % n)
            if len(idx) >= n:
                return np.asarray(idx[:n], dtype=int)
    return np.asarray(idx[:n], dtype=int)


def moving_block_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric_fn: MetricFn,
    n_bootstrap: int = 1000,
    block_size: int = 20,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have matching length")
    if len(y_true) < 2:
        value = metric_fn(y_true, y_pred)
        return {"estimate": float(value), "ci_low": float(value), "ci_high": float(value)}

    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_bootstrap):
        idx = moving_block_bootstrap_indices(len(y_true), block_size=block_size, rng=rng)
        stats.append(metric_fn(y_true[idx], y_pred[idx]))

    est = metric_fn(y_true, y_pred)
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1.0 - alpha / 2))
    return {"estimate": float(est), "ci_low": lo, "ci_high": hi}


def exact_binomial_directional_test(y_true: np.ndarray, y_pred: np.ndarray, p_null: float = 0.5) -> dict[str, float]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have matching length")
    if len(y_true) == 0:
        return {"n_obs": 0.0, "n_success": 0.0, "hit_rate": np.nan, "p_value": np.nan}
    true_dir = np.where(y_true >= 0.0, 1, -1)
    pred_dir = np.where(y_pred >= 0.0, 1, -1)
    success = int(np.sum(true_dir == pred_dir))
    n = int(len(y_true))
    hit_rate = float(success / n)
    p_value = float(binomtest(success, n=n, p=p_null, alternative="two-sided").pvalue)
    return {"n_obs": float(n), "n_success": float(success), "hit_rate": hit_rate, "p_value": p_value}


@dataclass
class MCSResult:
    asset: str
    mode: str
    model: str
    mcs_member: bool
    elimination_rank: int | None
    p_value_vs_best: float | None
    mean_loss: float


def model_confidence_set(
    loss_df: pd.DataFrame,
    *,
    alpha: float = 0.10,
    n_bootstrap: int = 500,
    block_size: int = 20,
    seed: int = 42,
) -> list[MCSResult]:
    """
    Approximate MCS elimination procedure.

    The implementation follows the practical elimination spirit of Hansen-Lunde-Nason
    by iteratively dropping models that are significantly worse than the current best
    under moving-block bootstrap on the loss differential.
    """
    if loss_df.empty:
        return []
    if loss_df.shape[1] < 2:
        only_model = loss_df.columns[0]
        return [
            MCSResult(
                asset="",
                mode="",
                model=str(only_model),
                mcs_member=True,
                elimination_rank=None,
                p_value_vs_best=None,
                mean_loss=float(loss_df.iloc[:, 0].mean()),
            )
        ]

    rng = np.random.default_rng(seed)
    active = list(loss_df.columns)
    elimination: dict[str, tuple[int, float]] = {}
    rank = 1

    while len(active) > 1:
        means = loss_df[active].mean(axis=0).sort_values()
        best = str(means.index[0])
        removable: list[tuple[str, float, float]] = []

        for model in active:
            if model == best:
                continue
            diff = (loss_df[model] - loss_df[best]).to_numpy(dtype=float)
            diff_mean = float(np.mean(diff))
            if diff_mean <= 0:
                continue
            boot_means = []
            for _ in range(n_bootstrap):
                idx = moving_block_bootstrap_indices(len(diff), block_size=block_size, rng=rng)
                boot_means.append(float(np.mean(diff[idx])))
            # One-sided p-value for "not worse than best" (H0: mean(diff) <= 0).
            p_value = float(np.mean(np.asarray(boot_means) <= 0.0))
            if p_value < alpha:
                removable.append((model, p_value, diff_mean))

        if not removable:
            break

        # Remove the model with largest observed excess loss among significant candidates.
        removable.sort(key=lambda x: x[2], reverse=True)
        remove_model, p_val, _ = removable[0]
        elimination[remove_model] = (rank, p_val)
        active.remove(remove_model)
        rank += 1

    out: list[MCSResult] = []
    mean_loss_all = loss_df.mean(axis=0)
    for model in loss_df.columns:
        elim = elimination.get(str(model))
        out.append(
            MCSResult(
                asset="",
                mode="",
                model=str(model),
                mcs_member=str(model) in active,
                elimination_rank=elim[0] if elim else None,
                p_value_vs_best=elim[1] if elim else None,
                mean_loss=float(mean_loss_all[str(model)]),
            )
        )
    return out
