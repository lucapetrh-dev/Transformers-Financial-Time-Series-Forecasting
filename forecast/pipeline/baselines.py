from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class RandomWalkReturnBaseline:
    """Persistence baseline: predict next return as current return."""

    def fit(self, X: np.ndarray, y: np.ndarray):
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # X is expected to contain lagged returns where lag_1 is first column.
        return X[:, 0]


class LinearLagBaseline:
    def __init__(self, alpha: float = 1.0):
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs)


class XGBoostLagBaseline:
    def __init__(
        self,
        n_estimators: int = 400,
        learning_rate: float = 0.03,
        max_depth: int = 4,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        random_state: int = 42,
    ):
        try:
            from xgboost import XGBRegressor
        except Exception as exc:
            raise ImportError("xgboost is required for XGBoostLagBaseline") from exc

        self.model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


def has_xgboost() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except Exception:
        return False


def try_arima_forecast(train_y: np.ndarray, test_y: np.ndarray, order: tuple[int, int, int] = (1, 0, 1)) -> np.ndarray | None:
    """Rolling one-step ARIMA predictions. Returns None if statsmodels is unavailable."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except Exception:
        return None

    history = list(train_y)
    preds = []
    for actual in test_y:
        model = ARIMA(history, order=order)
        fit = model.fit()
        pred = float(fit.forecast(steps=1)[0])
        preds.append(pred)
        history.append(float(actual))

    return np.asarray(preds)
