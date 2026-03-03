# Final Results Summary

Generated: 2026-03-03 (strict local rerun, native multi-foundation included).

## Canonical Scope

- Assets: `btc, eth, ada, doge, xmr, xrp`
- Target: one-day `log_return` (`h=1`)
- Active families (point track): baselines, compact transformers, native foundation models (`chronos2_zero_shot`, `timesfm_zero_shot`, `moirai_zero_shot`, `lagllama_zero_shot`, `random_walk_scaled`)

## Key Benchmark Outcomes (No Sentiment, Point Track)

- Family-level averages (`results/paper/PAPER_H1_family_performance.csv`):
`zero_forecast` MAE `0.03098` (lowest reference), `timesfm_zero_shot` MAE `0.03170` (strongest foundation), `linear_ridge` MAE `0.03310`.
- Best-by-asset rows (`results/paper/PAPER_H1_best_by_asset_mode.csv`):
ADA `timesfm_zero_shot`; BTC `zero_forecast`; DOGE `zero_forecast`; ETH `zero_forecast`; XMR `timesfm_zero_shot`; XRP `timesfm_zero_shot`.
- Sentiment deltas (`results/paper/PAPER_H1_sentiment_delta.csv`, family means):
baselines `+0.00679` MAE, transformers `+0.00963` MAE (degradation in both).

## Inference and Reproducibility Status

- Inference outputs present and non-empty:
`results/paper/PAPER_H1_metric_ci95.csv`,
`results/paper/PAPER_H1_directional_binomial.csv`,
`results/paper/PAPER_H1_mcs.csv`.
- Inference row counts:
`metric_ci95=150`, `directional_binomial=150`, `mcs=150`.
- Directional sample sizes:
baseline/foundation rows use `n_obs=660`; transformer sequence rows use `n_obs=600`.
- Strict paper summary passes:
`python forecast/runners/run_paper_summary.py --strict`
- Strict reproducibility audit passes:
`results/paper/reproducibility/reproducibility_summary.json` has `paper_strict_pass: true`.
- Transformer history file exists:
`results/paper/multi_asset_transformers_h1_paired_summary_history.csv` (`max_epoch=20`).

## Manuscript Status

- Updated thesis source:
`results/paper/overleaf_thesis/main.tex`
- Mirrored upload source:
`results/paper/overleaf_upload/main.tex`
- Current local compile:
`results/paper/overleaf_thesis/main.pdf` is `30` pages.

## Still Missing / Open Items

- None blocking for the manuscript target: page cap (<=30) and LaTeX warning cleanup are both satisfied in the local build.
