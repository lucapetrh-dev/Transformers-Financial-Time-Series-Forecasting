from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from forecast.pipeline.metrics import directional_accuracy, mae, rmse


DEFAULT_LAGS = [0, 1, 2, 3, 5, 7]
DEFAULT_PCA_COMPONENTS = [3, 5, 10, 15]


def compute_sentiment_target_correlations(
    df: pd.DataFrame,
    sentiment_cols: list[str],
    target_col: str,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Compute Pearson and Spearman correlations by sentiment feature and lag."""
    if lags is None:
        lags = DEFAULT_LAGS

    y = pd.to_numeric(df[target_col], errors="coerce")
    rows: list[dict[str, float | int | str]] = []

    for feature in sentiment_cols:
        x_raw = pd.to_numeric(df[feature], errors="coerce")
        for lag in lags:
            if lag < 0:
                raise ValueError("lags must be >= 0")
            x = x_raw.shift(int(lag))
            pair = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
            n_obs = int(len(pair))

            pearson_r = np.nan
            pearson_p = np.nan
            spearman_r = np.nan
            spearman_p = np.nan

            if n_obs >= 3 and pair["x"].nunique() > 1 and pair["y"].nunique() > 1:
                pearson_r, pearson_p = pearsonr(pair["x"].to_numpy(dtype=float), pair["y"].to_numpy(dtype=float))
                spearman_r, spearman_p = spearmanr(pair["x"].to_numpy(dtype=float), pair["y"].to_numpy(dtype=float))

            rows.append(
                {
                    "feature": feature,
                    "lag": int(lag),
                    "pearson_r": float(pearson_r) if pd.notna(pearson_r) else np.nan,
                    "pearson_p": float(pearson_p) if pd.notna(pearson_p) else np.nan,
                    "spearman_r": float(spearman_r) if pd.notna(spearman_r) else np.nan,
                    "spearman_p": float(spearman_p) if pd.notna(spearman_p) else np.nan,
                    "n_obs": n_obs,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["feature", "lag", "pearson_r", "pearson_p", "spearman_r", "spearman_p", "n_obs"])
    return out.sort_values(["feature", "lag"]).reset_index(drop=True)


def compute_dimensionality_profile(
    feature_counts_by_mode: dict[str, int],
    n_train_samples: int,
) -> pd.DataFrame:
    """Build a p/n profile table for each feature mode."""
    if n_train_samples <= 0:
        raise ValueError("n_train_samples must be > 0")

    rows = []
    for mode, n_features in feature_counts_by_mode.items():
        n_features_int = int(n_features)
        rows.append(
            {
                "mode": str(mode),
                "n_features": n_features_int,
                "n_train_samples": int(n_train_samples),
                "p_over_n": float(n_features_int / float(n_train_samples)),
            }
        )
    return pd.DataFrame(rows).sort_values("mode").reset_index(drop=True)


def _evaluate_ridge(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    return {
        "mae": mae(y_test, y_pred),
        "rmse": rmse(y_test, y_pred),
        "dir_acc": directional_accuracy(y_test, y_pred),
    }


def run_pca_sentiment_ablation(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    sentiment_col_indices: list[int],
    n_components_list: list[int] | None = None,
) -> pd.DataFrame:
    """Run Ridge comparisons with full sentiment, no sentiment, and PCA-compressed sentiment."""
    if n_components_list is None:
        n_components_list = DEFAULT_PCA_COMPONENTS

    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    y_test = np.asarray(y_test, dtype=float)

    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("X_train and X_test must be 2D arrays")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("X_train and X_test must have the same number of columns")
    if len(y_train) != len(X_train) or len(y_test) != len(X_test):
        raise ValueError("Target lengths must match feature rows")

    n_features = X_train.shape[1]
    sent_idx = sorted({int(i) for i in sentiment_col_indices if 0 <= int(i) < n_features})
    nonsent_idx = [i for i in range(n_features) if i not in sent_idx]

    rows: list[dict[str, float | int | str]] = []

    full_metrics = _evaluate_ridge(X_train, X_test, y_train, y_test)
    rows.append(
        {
            "mode": "full_sentiment",
            "n_pca_components": int(len(sent_idx)),
            "n_total_features": int(n_features),
            **full_metrics,
        }
    )

    if nonsent_idx:
        no_sent_metrics = _evaluate_ridge(X_train[:, nonsent_idx], X_test[:, nonsent_idx], y_train, y_test)
        no_sent_n_features = len(nonsent_idx)
    else:
        no_sent_metrics = full_metrics
        no_sent_n_features = n_features

    rows.append(
        {
            "mode": "no_sentiment",
            "n_pca_components": 0,
            "n_total_features": int(no_sent_n_features),
            **no_sent_metrics,
        }
    )

    if sent_idx:
        Xs_train = X_train[:, sent_idx]
        Xs_test = X_test[:, sent_idx]
        scaler = StandardScaler().fit(Xs_train)
        Xs_train_scaled = scaler.transform(Xs_train)
        Xs_test_scaled = scaler.transform(Xs_test)

        for requested in n_components_list:
            k = int(requested)
            if k <= 0:
                continue
            k_eff = min(k, Xs_train_scaled.shape[1], max(1, Xs_train_scaled.shape[0] - 1))
            if k_eff <= 0:
                continue
            pca = PCA(n_components=k_eff, random_state=42)
            X_train_pca = pca.fit_transform(Xs_train_scaled)
            X_test_pca = pca.transform(Xs_test_scaled)

            if nonsent_idx:
                X_train_new = np.hstack([X_train[:, nonsent_idx], X_train_pca])
                X_test_new = np.hstack([X_test[:, nonsent_idx], X_test_pca])
            else:
                X_train_new = X_train_pca
                X_test_new = X_test_pca

            metrics = _evaluate_ridge(X_train_new, X_test_new, y_train, y_test)
            rows.append(
                {
                    "mode": "sentiment_pca",
                    "n_pca_components": int(k_eff),
                    "n_total_features": int(X_train_new.shape[1]),
                    **metrics,
                }
            )

    out = pd.DataFrame(rows)
    order = {"no_sentiment": 0, "sentiment_pca": 1, "full_sentiment": 2}
    out["_order"] = out["mode"].map(order).fillna(99)
    out = out.sort_values(["_order", "n_pca_components"]).drop(columns=["_order"]).reset_index(drop=True)
    return out


def _feature_group(col: str) -> str:
    c = col.lower()
    if "tweet" in c:
        return "tweet"
    if "reddit" in c:
        return "reddit"
    if "social" in c or "news" in c or "url_shares" in c:
        return "social"
    return "other_sentiment"


def compute_feature_group_summary(
    df: pd.DataFrame,
    sentiment_cols: list[str],
    financial_cols: list[str],
    target_col: str,
) -> pd.DataFrame:
    """Summarize absolute target correlations by feature group."""
    y = pd.to_numeric(df[target_col], errors="coerce")

    groups: dict[str, list[str]] = {"financial": [c for c in financial_cols if c in df.columns]}
    for col in sentiment_cols:
        if col not in df.columns:
            continue
        g = _feature_group(col)
        groups.setdefault(g, []).append(col)

    rows: list[dict[str, float | int | str]] = []
    for group_name, cols in groups.items():
        abs_corrs: list[float] = []
        for col in cols:
            x = pd.to_numeric(df[col], errors="coerce")
            pair = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) < 3 or pair["x"].nunique() <= 1 or pair["y"].nunique() <= 1:
                continue
            corr = pair["x"].corr(pair["y"])
            if pd.notna(corr):
                abs_corrs.append(float(abs(corr)))

        rows.append(
            {
                "group": group_name,
                "n_features": int(len(cols)),
                "mean_abs_corr": float(np.mean(abs_corrs)) if abs_corrs else np.nan,
                "max_abs_corr": float(np.max(abs_corrs)) if abs_corrs else np.nan,
                "std_corr": float(np.std(abs_corrs, ddof=1)) if len(abs_corrs) > 1 else (0.0 if len(abs_corrs) == 1 else np.nan),
            }
        )

    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
