# Results Layout

Active, strict paper layout:

```text
results/
  RESULTS_SUMMARY.md
  appendix/
  paper/
    PAPER_H1_RESULTS_SUMMARY.md
    PAPER_H1_best_by_asset.csv
    PAPER_H1_best_by_asset_mode.csv
    PAPER_H1_family_performance.csv
    PAPER_H1_sentiment_delta.csv
    PAPER_H1_ablation_mode_wins.csv
    PAPER_H1_metric_ci95.csv
    PAPER_H1_directional_binomial.csv
    PAPER_H1_mcs.csv
    multi_asset_baselines_h1_paired_summary*.csv
    multi_asset_transformers_h1_paired_summary*.csv
    multi_asset_chronos2_h1_summary*.csv
    multi_asset_foundation_h1_summary*.csv
    feature_ablation_h1_summary*.csv
    contract_asset_manifest.csv
    data_manifest.csv
    data_manifest_aggregation_policy.csv
    reproducibility/
    _archive_detailed/
```

The canonical entry point for paper review is:

- `results/paper/PAPER_H1_RESULTS_SUMMARY.md`

Detailed traces (folds, metadata, quantiles, histories, missingness stages) are archived under:

- `results/paper/_archive_detailed/<timestamp>/`

Deterministic strict rerun command:

- `python forecast/runners/run_precolab_paper_pipeline.py --horizon 1 --assets btc,eth,ada,doge,xmr,xrp`
