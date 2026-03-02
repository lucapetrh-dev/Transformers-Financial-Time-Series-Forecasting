"""Leakage-aware forecasting utilities for the manuscript upgrade."""

from .data_pipeline import load_market_data, build_feature_frame, add_targets
from .assets import asset_name_from_path, parse_asset_list, filter_asset_paths
from .splits import walk_forward_splits, purged_kfold_splits
from .tuning import PurgedCVConfig, tune_ridge_alpha_purged_cv
from .baselines import XGBoostLagBaseline, has_xgboost
from .experiment_contract import infer_frequency_label, save_metadata_json, fold_manifest_rows
from .sentiment import (
    detect_sentiment_columns,
    build_sentiment_comparison_table,
    apply_causal_sentiment_lag,
)
from .transformers import (
    ITransformerLikeRegressor,
    PatchTSTLikeRegressor,
    SequenceStandardizer,
    TrainConfig,
    build_sliding_windows,
    predict_model,
    train_model,
)
from .transformer_official_backends import (
    CANONICAL_MODEL_FAMILIES,
    CanonicalBackendConfig,
    build_canonical_model,
    canonical_model_name,
    config_provenance,
    parse_canonical_models,
)
from .statistical_inference import (
    exact_binomial_directional_test,
    model_confidence_set,
    moving_block_bootstrap_ci,
    moving_block_bootstrap_indices,
)
from .metrics import (
    mae,
    rmse,
    directional_accuracy,
    pinball_loss,
    crps_gaussian,
    interval_coverage,
    interval_width,
    weighted_interval_score,
    backtest_metrics,
    cost_sensitivity_metrics,
    deflated_sharpe_ratio,
    diebold_mariano_test,
)
