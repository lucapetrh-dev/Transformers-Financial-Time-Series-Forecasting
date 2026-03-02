# Transformers for Financial Time Series Forecasting in Cryptocurrency Markets

This repository now runs the thesis pipeline with leakage-aware return forecasting, paired sentiment evaluation, modern model baselines, multi-foundation zero-shot benchmarking, canonical transformer/HPO tooling, and paper-ready artifacts.

## Current Repository Layout

```text
forecast/
  analysis/     appendix diagnostics (stationarity/FFT/ACF/PACF)
  config/       frozen experiment contract
  pipeline/     reusable modules (features, splits, metrics, models)
  runners/      executable experiment scripts
tests/          real-data regression tests
.data/          source datasets (hourly crypto CSVs)
results/        generated outputs
.docs/          manuscript plans and notes
.legacy/        archived old notebooks/files (gitignored)
```

## Environment

Python 3.10+ recommended.

Install core dependencies:

```bash
pip install numpy pandas scipy scikit-learn torch matplotlib seaborn
```

Install optional packages used by some runners:

```bash
pip install statsmodels xgboost chronos-forecasting
```

Install native multi-foundation backends (TimesFM, Moirai, Lag-Llama):

```bash
pip install "timesfm[torch]" uni2ts gluonts lightning
pip install "git+https://github.com/time-series-foundation-models/lag-llama.git"
pip install "huggingface_hub<1.0"
```

## Data Requirements

All runners expect CSV files with:

- timestamp column (default: `time`)
- price column (one of: `adjusted_close`, `adj_close`, `Adj Close`, `adjclose`, `close`)

Main dataset used in this repo:

- `.data/hourly/ada_lunarcrush_timeseries_hourly.csv`
- `.data/hourly/btc_lunarcrush_timeseries_hourly.csv`
- `.data/hourly/doge_lunarcrush_timeseries_hourly.csv`
- `.data/hourly/eth_lunarcrush_timeseries_hourly.csv`
- `.data/hourly/xmr_lunarcrush_timeseries_hourly.csv`
- `.data/hourly/xrp_lunarcrush_timeseries_hourly.csv`

Paper benchmark active asset universe:

- `btc,eth,ada,doge,xmr,xrp`

## Frozen Experiment Contract

- target: `target_ret_{1,5,20}d` (log-return space)
- split protocol: walk-forward for main benchmark, purged/CPCV for ablations
- sentiment handling: causal lag + coverage filter + causal `ffill` then zero-fill
- comparability: same target space across all multi-asset tables
- reproducibility artifacts generated for each run

## Run Order (Paper Pipeline)

### 0) Validate contract + data manifest

```bash
python forecast/runners/run_data_manifest.py \
  --data-glob '.data/hourly/*_timeseries_hourly.csv' \
  --include-assets btc,eth,ada,doge,xmr,xrp \
  --provenance-config forecast/config/data_provenance.json \
  --output results/paper/data_manifest.csv

python forecast/runners/run_contract_audit.py \
  --contract-path forecast/config/paper_experiment.json \
  --manifest-path results/paper/data_manifest.csv \
  --require-provenance \
  --output results/paper/contract_asset_manifest.csv
```

### 1) Multi-asset baselines (paired no-sent vs sentiment)

```bash
python forecast/runners/run_multi_asset_baselines.py \
  --data-glob '.data/hourly/*_timeseries_hourly.csv' \
  --include-assets btc,eth,ada,doge,xmr,xrp \
  --horizon 1 \
  --min-train-size 300 \
  --test-size 60 \
  --step-size 240 \
  --auto-adjust-splits \
  --skip-failed-assets \
  --run-paired-sentiment \
  --no-arima \
  --no-xgboost \
  --output results/paper/multi_asset_baselines_h1_paired_summary.csv
```

### 2) Multi-asset transformer family (paired)

```bash
python forecast/runners/run_multi_asset_transformers.py \
  --data-glob '.data/hourly/*_timeseries_hourly.csv' \
  --include-assets btc,eth,ada,doge,xmr,xrp \
  --horizon 1 \
  --lookback 64 \
  --min-train-size 300 \
  --test-size 60 \
  --step-size 240 \
  --epochs 2 \
  --models patchtst_like,itransformer_like \
  --objective-track both \
  --point-loss mse \
  --probabilistic-mode quantile \
  --quantiles 0.1,0.5,0.9 \
  --auto-adjust-splits \
  --skip-failed-assets \
  --run-paired-sentiment \
  --output results/paper/multi_asset_transformers_h1_paired_summary.csv
```

### 3) Multi-foundation zero-shot (Chronos-2, TimesFM, Moirai, Lag-Llama)

```bash
python forecast/runners/run_multi_asset_foundation.py \
  --data-glob '.data/hourly/*_timeseries_hourly.csv' \
  --include-assets btc,eth,ada,doge,xmr,xrp \
  --horizon 1 \
  --context-length 64 \
  --min-train-size 300 \
  --test-size 60 \
  --step-size 240 \
  --models chronos2,timesfm,moirai,lagllama \
  --disable-fallback-adapters \
  --auto-adjust-splits \
  --skip-failed-assets \
  --output results/paper/multi_asset_foundation_h1_summary.csv
```

Note: the first native run downloads large checkpoints from HuggingFace and can take significant time.

Compatibility wrapper (Chronos-only) is still available:

```bash
python forecast/runners/run_multi_asset_chronos2.py \
  --data-glob '.data/hourly/*_timeseries_hourly.csv' \
  --include-assets btc,eth,ada,doge,xmr,xrp \
  --output results/paper/multi_asset_chronos2_h1_summary.csv
```

### 4) Feature ablations (financial / +sentiment / +regime)

```bash
python forecast/runners/run_feature_ablations.py \
  --data-glob '.data/hourly/*_timeseries_hourly.csv' \
  --include-assets btc,eth,ada,doge,xmr,xrp \
  --horizon 1 \
  --n-splits 5 \
  --embargo 5 \
  --label-horizon 1 \
  --sentiment-lag 1 \
  --regime-lookbacks 20,60 \
  --no-arima \
  --no-xgboost \
  --skip-failed-assets \
  --output results/paper/feature_ablation_h1_summary.csv
```

### 5) Cross-asset transfer (BTC -> ETH)

```bash
python forecast/runners/run_cross_asset_transfer.py \
  --source-data-path .data/hourly/btc_lunarcrush_timeseries_hourly.csv \
  --target-data-path .data/hourly/eth_lunarcrush_timeseries_hourly.csv \
  --horizon 1 \
  --lookback 64 \
  --model patchtst_like \
  --epochs 2 \
  --finetune-epochs 1 \
  --use-sentiment \
  --sentiment-lag 1 \
  --sentiment-min-non-null-ratio 0.2 \
  --output results/paper/transfer_btc_eth_patchtst_with_sent.csv
```

### 6) Appendix diagnostics (run for each asset)

```bash
for f in \
  .data/hourly/btc_lunarcrush_timeseries_hourly.csv \
  .data/hourly/eth_lunarcrush_timeseries_hourly.csv \
  .data/hourly/ada_lunarcrush_timeseries_hourly.csv \
  .data/hourly/doge_lunarcrush_timeseries_hourly.csv \
  .data/hourly/xmr_lunarcrush_timeseries_hourly.csv \
  .data/hourly/xrp_lunarcrush_timeseries_hourly.csv; do
  python forecast/runners/run_stationarity_appendix.py \
    --data-path "$f" \
    --time-col time \
    --output-dir results/paper/appendix/stationarity
done
```

### 7) Reproducibility audit

```bash
python forecast/runners/run_reproducibility_audit.py \
  --results-root results/paper \
  --output-dir results/paper/reproducibility
```

### 8) Canonical transformer HPO (Optuna)

```bash
python forecast/runners/run_transformer_hpo.py \
  --data-glob '.data/hourly/*_timeseries_hourly.csv' \
  --include-assets btc,eth,ada,doge,xmr,xrp \
  --model-family itransformer \
  --objective-track point \
  --point-loss mse \
  --trials 50 \
  --output-dir results/paper/hpo \
  --summary-output results/paper/hpo/transformer_hpo_best_summary.csv
```

### 9) Ensemble benchmark (ridge + best foundation)

```bash
python forecast/runners/run_ensemble_benchmark.py \
  --results-root results/paper \
  --horizon 1 \
  --output results/paper/ensemble_h1_summary.csv
```

### 10) Statistical inference (bootstrap CI, binomial, MCS)

```bash
python forecast/runners/run_statistical_inference.py \
  --results-root results/paper \
  --horizon 1 \
  --output-prefix PAPER_H1
```

### 11) Generate paper-ready consolidated tables

```bash
python forecast/runners/run_paper_summary.py \
  --results-root results/paper \
  --horizon 1 \
  --output-prefix PAPER_H1
```

## Where To Read Results

Main files for manuscript writing:

- `results/paper/PAPER_H1_RESULTS_SUMMARY.md`
- `results/paper/PAPER_H1_best_by_asset.csv`
- `results/paper/PAPER_H1_best_by_asset_mode.csv`
- `results/paper/PAPER_H1_family_performance.csv`
- `results/paper/PAPER_H1_sentiment_delta.csv`
- `results/paper/PAPER_H1_ablation_mode_wins.csv`
- `results/paper/PAPER_H1_metric_ci95.csv`
- `results/paper/PAPER_H1_directional_binomial.csv`
- `results/paper/PAPER_H1_mcs.csv`

Core experiment outputs:

- `results/paper/multi_asset_baselines_h1_paired_summary.csv`
- `results/paper/multi_asset_transformers_h1_paired_summary.csv`
- `results/paper/multi_asset_foundation_h1_summary.csv`
- `results/paper/multi_asset_chronos2_h1_summary.csv` (compatibility output)
- `results/paper/ensemble_h1_summary.csv`
- `results/paper/feature_ablation_h1_summary.csv`
- `results/paper/transfer_btc_eth_patchtst_with_sent.csv`
- `results/paper/reproducibility/reproducibility_summary.json`

## Tests

Run full real-data regression tests:

```bash
pytest -q tests
```
