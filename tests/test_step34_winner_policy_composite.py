from __future__ import annotations

import pandas as pd

from forecast.analysis.paper_figures_v2 import _choose_best_available_model
from forecast.runners.run_paper_summary import _best_by_asset_mode


def _winner(df: pd.DataFrame, *, policy: str = "composite_rank", exclude: set[str] | None = None) -> str:
    out = _best_by_asset_mode(
        df,
        winner_policy=policy,
        winner_exclude_models=exclude or set(),
        winner_metrics=["mae_mean", "rmse_mean", "directional_accuracy_mean", "sharpe_5bps_mean"],
        winner_tiebreak="mae_mean",
    )
    return str(out.iloc[0]["model"])


def test_composite_rank_can_differ_from_mae_winner() -> None:
    df = pd.DataFrame(
        [
            {
                "asset": "btc",
                "mode": "no_sentiment",
                "objective_track": "point",
                "model": "mae_best",
                "mae_mean": 0.010,
                "rmse_mean": 0.040,
                "directional_accuracy_mean": 0.49,
                "sharpe_5bps_mean": 0.00,
            },
            {
                "asset": "btc",
                "mode": "no_sentiment",
                "objective_track": "point",
                "model": "composite_best",
                "mae_mean": 0.012,
                "rmse_mean": 0.020,
                "directional_accuracy_mean": 0.62,
                "sharpe_5bps_mean": 0.15,
            },
            {
                "asset": "btc",
                "mode": "no_sentiment",
                "objective_track": "point",
                "model": "middle",
                "mae_mean": 0.011,
                "rmse_mean": 0.030,
                "directional_accuracy_mean": 0.55,
                "sharpe_5bps_mean": 0.05,
            },
        ]
    )

    assert _winner(df, policy="mae") == "mae_best"
    assert _winner(df, policy="composite_rank") == "composite_best"


def test_winner_exclusion_blocks_zero_forecast_from_winner() -> None:
    df = pd.DataFrame(
        [
            {
                "asset": "eth",
                "mode": "no_sentiment",
                "objective_track": "point",
                "model": "zero_forecast",
                "mae_mean": 0.010,
                "rmse_mean": 0.020,
                "directional_accuracy_mean": 0.50,
                "sharpe_5bps_mean": 0.00,
            },
            {
                "asset": "eth",
                "mode": "no_sentiment",
                "objective_track": "point",
                "model": "linear_ridge",
                "mae_mean": 0.011,
                "rmse_mean": 0.021,
                "directional_accuracy_mean": 0.52,
                "sharpe_5bps_mean": 0.01,
            },
        ]
    )

    assert _winner(df, policy="mae") == "zero_forecast"
    assert _winner(df, policy="mae", exclude={"zero_forecast"}) == "linear_ridge"


def test_tie_break_is_deterministic_for_equal_composite_scores() -> None:
    df = pd.DataFrame(
        [
            {
                "asset": "xrp",
                "mode": "no_sentiment",
                "objective_track": "point",
                "model": "beta_model",
                "mae_mean": 0.020,
                "rmse_mean": 0.030,
                "directional_accuracy_mean": 0.51,
                "sharpe_5bps_mean": 0.02,
            },
            {
                "asset": "xrp",
                "mode": "no_sentiment",
                "objective_track": "point",
                "model": "alpha_model",
                "mae_mean": 0.020,
                "rmse_mean": 0.030,
                "directional_accuracy_mean": 0.51,
                "sharpe_5bps_mean": 0.02,
            },
        ]
    )

    assert _winner(df, policy="composite_rank") == "alpha_model"


def test_figure_winner_selection_excludes_zero_forecast() -> None:
    summary = pd.DataFrame(
        [
            {
                "asset": "btc",
                "mode": "no_sentiment",
                "objective_track": "point",
                "model": "zero_forecast",
                "mae_mean": 0.01,
                "rmse_mean": 0.02,
                "directional_accuracy_mean": 0.50,
                "sharpe_5bps_mean": 0.0,
            },
            {
                "asset": "btc",
                "mode": "no_sentiment",
                "objective_track": "point",
                "model": "linear_ridge",
                "mae_mean": 0.011,
                "rmse_mean": 0.021,
                "directional_accuracy_mean": 0.52,
                "sharpe_5bps_mean": 0.01,
            },
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "asset": "btc",
                "mode": "no_sentiment",
                "objective_track": "point",
                "model": "zero_forecast",
                "timestamp": "2023-01-01",
                "y_pred": 0.0,
            },
            {
                "asset": "btc",
                "mode": "no_sentiment",
                "objective_track": "point",
                "model": "linear_ridge",
                "timestamp": "2023-01-01",
                "y_pred": 0.001,
            },
        ]
    )
    chosen = _choose_best_available_model(
        summary,
        predictions,
        "btc",
        winner_policy="mae",
        winner_exclude_models={"zero_forecast"},
        winner_metrics=["mae_mean", "rmse_mean", "directional_accuracy_mean", "sharpe_5bps_mean"],
    )
    assert chosen == "linear_ridge"
