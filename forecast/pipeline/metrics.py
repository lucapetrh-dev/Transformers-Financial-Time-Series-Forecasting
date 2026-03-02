from __future__ import annotations

from math import erf, sqrt

import numpy as np
from scipy.stats import kurtosis, norm, skew


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def pinball_loss(y_true: np.ndarray, y_pred_q: np.ndarray, q: float) -> float:
    err = y_true - y_pred_q
    return float(np.mean(np.maximum(q * err, (q - 1.0) * err)))


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(erf)(z / np.sqrt(2.0)))


def crps_gaussian(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    sigma = np.maximum(sigma, 1e-8)
    z = (y_true - mu) / sigma
    crps = sigma * (z * (2.0 * _norm_cdf(z) - 1.0) + 2.0 * _norm_pdf(z) - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))


def interval_coverage(y_true: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> float:
    inside = (y_true >= q_lo) & (y_true <= q_hi)
    return float(np.mean(inside))


def interval_width(q_lo: np.ndarray, q_hi: np.ndarray) -> float:
    return float(np.mean(q_hi - q_lo))


def weighted_interval_score(
    y_true: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    alpha: float = 0.2,
) -> float:
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must be in (0, 1)")

    width = q_hi - q_lo
    below = np.maximum(q_lo - y_true, 0.0)
    above = np.maximum(y_true - q_hi, 0.0)
    wis = width + (2.0 / alpha) * below + (2.0 / alpha) * above
    return float(np.mean(wis))


def backtest_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_bps: float = 5.0,
    annualization: int = 252,
) -> dict[str, float]:
    strategy_rets, turnover = strategy_returns(y_true, y_pred, cost_bps=cost_bps)

    mean_ret = float(np.mean(strategy_rets))
    std_ret = float(np.std(strategy_rets, ddof=1)) if len(strategy_rets) > 1 else 0.0
    downside = np.minimum(strategy_rets, 0.0)
    downside_std = float(np.std(downside, ddof=1)) if len(strategy_rets) > 1 else 0.0

    sharpe = 0.0 if std_ret == 0 else (mean_ret / std_ret) * np.sqrt(annualization)
    sortino = 0.0 if downside_std == 0 else (mean_ret / downside_std) * np.sqrt(annualization)

    equity_curve = np.cumprod(1.0 + strategy_rets)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / np.maximum(peak, 1e-12)
    max_dd = float(np.min(drawdown))

    return {
        "mean_strategy_return": mean_ret,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_dd,
        "turnover": float(np.mean(turnover)),
    }


def strategy_returns(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_bps: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    # Simple sign strategy: long if forecast > 0 else short.
    position = np.where(y_pred >= 0.0, 1.0, -1.0)
    turnover = np.abs(np.diff(position, prepend=position[0]))
    costs = turnover * (cost_bps / 10000.0)
    strategy_rets = position * y_true - costs
    return strategy_rets, turnover


def deflated_sharpe_ratio(
    strategy_rets: np.ndarray,
    n_trials: int = 1,
) -> float:
    T = len(strategy_rets)
    if T < 3:
        return float("nan")

    mean_ret = float(np.mean(strategy_rets))
    std_ret = float(np.std(strategy_rets, ddof=1))
    if std_ret == 0.0:
        return 0.0

    sr = mean_ret / std_ret
    g3 = float(skew(strategy_rets, bias=False))
    g4 = float(kurtosis(strategy_rets, fisher=False, bias=False))

    denom_term = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * (sr**2)
    denom_term = max(denom_term, 1e-12)

    if n_trials <= 1:
        z = sr * np.sqrt(max(T - 1, 1)) / np.sqrt(denom_term)
        return float(norm.cdf(z))

    euler_gamma = 0.5772156649
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    sr_std = np.sqrt(denom_term / max(T - 1, 1))
    sr_star = sr_std * ((1.0 - euler_gamma) * z1 + euler_gamma * z2)

    z = (sr - sr_star) * np.sqrt(max(T - 1, 1)) / np.sqrt(denom_term)
    return float(norm.cdf(z))


def cost_sensitivity_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_bps_list: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0),
    annualization: int = 252,
    n_trials: int = 1,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for cost_bps in cost_bps_list:
        metrics = backtest_metrics(y_true, y_pred, cost_bps=cost_bps, annualization=annualization)
        cost_key = f"{int(cost_bps)}bps"
        out[f"mean_strategy_return_{cost_key}"] = metrics["mean_strategy_return"]
        out[f"sharpe_{cost_key}"] = metrics["sharpe"]
        out[f"sortino_{cost_key}"] = metrics["sortino"]
        out[f"max_drawdown_{cost_key}"] = metrics["max_drawdown"]
        out[f"turnover_{cost_key}"] = metrics["turnover"]

    base_cost = 5.0 if 5.0 in cost_bps_list else cost_bps_list[0]
    base_rets, _ = strategy_returns(y_true, y_pred, cost_bps=base_cost)
    out["dsr"] = deflated_sharpe_ratio(base_rets, n_trials=n_trials)
    return out


def diebold_mariano_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    power: int = 2,
    horizon: int = 1,
) -> dict[str, float]:
    err_a = np.abs(y_true - pred_a) ** power
    err_b = np.abs(y_true - pred_b) ** power
    d = err_a - err_b

    T = len(d)
    if T < 3:
        return {"dm_stat": np.nan, "p_value": np.nan}

    d_mean = np.mean(d)

    gamma0 = np.var(d, ddof=1)
    var_d = gamma0
    max_lag = max(0, horizon - 1)
    for lag in range(1, max_lag + 1):
        cov = np.cov(d[lag:], d[:-lag], ddof=1)[0, 1]
        var_d += 2.0 * cov

    var_d = max(var_d, 1e-12)
    dm_stat = d_mean / np.sqrt(var_d / T)

    p_value = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(dm_stat) / sqrt(2.0))))
    return {"dm_stat": float(dm_stat), "p_value": float(p_value)}
