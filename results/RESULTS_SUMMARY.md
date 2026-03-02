# Results Summary

Generated: 2026-03-02 (strict hardening state, native multi-foundation rerun completed).

This folder tracks active, reproducible, six-asset paper outputs under strict validation gates.

## Canonical Scope

- Assets: `btc, eth, ada, doge, xmr, xrp`
- Target space: `log_return`
- Horizon: `h=1`
- AAVE-era and temporary outputs were removed.
- Active paper families:
`linear_ridge/random_walk`,
`patchtst_like/itransformer_like/random_walk_sequence`,
`chronos2_zero_shot/moirai_zero_shot/lagllama_zero_shot/random_walk_scaled`.

## Canonical Files

- Main paper summary: `results/paper/PAPER_H1_RESULTS_SUMMARY.md`
- Best-by-asset table: `results/paper/PAPER_H1_best_by_asset.csv`
- Family performance table: `results/paper/PAPER_H1_family_performance.csv`
- Sentiment delta table: `results/paper/PAPER_H1_sentiment_delta.csv`
- Ablation winners: `results/paper/PAPER_H1_ablation_mode_wins.csv`

## Core Benchmark Inputs

- Baselines: `results/paper/multi_asset_baselines_h1_paired_summary.csv`
- Transformers: `results/paper/multi_asset_transformers_h1_paired_summary.csv`
- Foundation (unified): `results/paper/multi_asset_foundation_h1_summary.csv`
- Chronos-2 compatibility wrapper output: `results/paper/multi_asset_chronos2_h1_summary.csv`
- Feature ablation best: `results/paper/feature_ablation_h1_summary_best.csv`

## Strict Policy

- Paper summary generation runs in strict mode by default and fails on:
`missing objective_track`, missing/empty inference tables, fallback rows, and asset-universe mismatch.
- Reproducibility audit strict mode validates metadata path integrity and rejects deprecated `truth_*` references.
- Detailed traces are archived under `results/paper/_archive_detailed/<timestamp>/` via:
`forecast/runners/run_artifact_policy.py`.

## Current Validation Status

- Inference tables are present and non-empty:
`results/paper/PAPER_H1_metric_ci95.csv`,
`results/paper/PAPER_H1_directional_binomial.csv`,
`results/paper/PAPER_H1_mcs.csv`.
- Inference row counts:
`metric_ci95=120`, `directional_binomial=120`, `mcs=120`.
- MCS now reports overlap-aware comparison scopes (`comparison_scope`) when model timestamp panels are disconnected.
- Strict paper summary generation passes:
`python forecast/runners/run_paper_summary.py --results-root results/paper --horizon 1 --output-prefix PAPER_H1 --strict`
- Strict reproducibility audit passes with `paper_strict_pass: true`:
`results/paper/reproducibility/reproducibility_summary.json`.
- Canonical summary cardinalities:
`baselines=24 rows`, `transformers=72 rows`, `foundation=36 rows`, `chronos2=12 rows`, `ablations=36 rows`, each on exactly six assets.
- Note: TimesFM is excluded from the active paper benchmark.

## Deterministic Runbook

- Full strict local orchestration entrypoint:
`python forecast/runners/run_precolab_paper_pipeline.py --horizon 1 --assets btc,eth,ada,doge,xmr,xrp`
- Archive policy applied after rerun:
`results/paper/_archive_detailed/20260302_precolab_hardening/`
- Native multi-foundation detailed traces archive:
`results/paper/_archive_detailed/20260302_post_native_foundation/`
