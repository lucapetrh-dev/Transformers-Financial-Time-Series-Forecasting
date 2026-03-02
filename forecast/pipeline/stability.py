from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecast.pipeline.metrics import mae


@dataclass
class PermutationImportanceConfig:
    n_repeats: int = 3
    random_state: int = 42


def permutation_importance_mae(
    *,
    predict_fn,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    config: PermutationImportanceConfig | None = None,
) -> pd.DataFrame:
    if config is None:
        config = PermutationImportanceConfig()
    if X_val.ndim != 2:
        raise ValueError("X_val must be 2D")
    if X_val.shape[1] != len(feature_names):
        raise ValueError("feature_names length must match X_val.shape[1]")
    if len(y_val) != X_val.shape[0]:
        raise ValueError("y_val length must match X_val rows")

    baseline_pred = np.asarray(predict_fn(X_val), dtype=float).reshape(-1)
    baseline_mae = mae(y_val, baseline_pred)

    rng = np.random.default_rng(config.random_state)
    rows: list[dict[str, float | str | int]] = []

    for j, feature in enumerate(feature_names):
        deltas: list[float] = []
        for _ in range(config.n_repeats):
            Xp = np.array(X_val, copy=True)
            perm = rng.permutation(Xp.shape[0])
            Xp[:, j] = Xp[perm, j]
            pred_perm = np.asarray(predict_fn(Xp), dtype=float).reshape(-1)
            perm_mae = mae(y_val, pred_perm)
            deltas.append(float(perm_mae - baseline_mae))

        deltas_arr = np.asarray(deltas, dtype=float)
        rows.append(
            {
                "feature": feature,
                "baseline_mae": float(baseline_mae),
                "importance_mean": float(np.mean(deltas_arr)),
                "importance_std": float(np.std(deltas_arr, ddof=1)) if len(deltas_arr) > 1 else 0.0,
                "importance_abs_mean": float(np.mean(np.abs(deltas_arr))),
                "n_repeats": int(config.n_repeats),
            }
        )

    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)
