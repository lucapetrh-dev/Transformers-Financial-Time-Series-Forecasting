from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .metrics import mae, rmse
from .splits import purged_kfold_splits


@dataclass
class PurgedCVConfig:
    n_splits: int = 5
    embargo: int = 5
    label_horizon: int = 1


def tune_ridge_alpha_purged_cv(
    X: np.ndarray,
    y: np.ndarray,
    alpha_grid: list[float],
    cv_config: PurgedCVConfig | None = None,
    objective: str = "mae",
) -> tuple[float, pd.DataFrame]:
    if cv_config is None:
        cv_config = PurgedCVConfig()

    if objective not in {"mae", "rmse"}:
        raise ValueError("objective must be either 'mae' or 'rmse'")

    rows: list[dict[str, float | int]] = []
    scorer = mae if objective == "mae" else rmse

    for alpha in alpha_grid:
        fold_scores: list[float] = []
        for fold_id, (train_idx, val_idx) in enumerate(
            purged_kfold_splits(
                n_samples=len(y),
                n_splits=cv_config.n_splits,
                embargo=cv_config.embargo,
                label_horizon=cv_config.label_horizon,
            )
        ):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            model = Ridge(alpha=alpha)
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_val_s)
            score = scorer(y_val, y_pred)
            fold_scores.append(score)
            rows.append({"alpha": alpha, "fold": fold_id, "score": score})

        rows.append(
            {
                "alpha": alpha,
                "fold": -1,
                "score": float(np.mean(fold_scores)),
            }
        )

    cv_df = pd.DataFrame(rows)
    summary = cv_df[cv_df["fold"] == -1].sort_values("score", ascending=True)
    best_alpha = float(summary.iloc[0]["alpha"])
    return best_alpha, cv_df
