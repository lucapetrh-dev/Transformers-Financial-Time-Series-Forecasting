from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class FigureRecord:
    figure_id: str
    filename: str
    title: str
    source_files: str


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def _asset_short(asset: str) -> str:
    if not isinstance(asset, str):
        return str(asset)
    return asset.split("_")[0].upper()


def _save(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _records_to_csv(records: list[FigureRecord], output_path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "figure_id": r.figure_id,
                "filename": r.filename,
                "title": r.title,
                "source_files": r.source_files,
            }
            for r in records
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def _plot_data_coverage(data_manifest: pd.DataFrame, out_dir: Path) -> FigureRecord:
    df = data_manifest.copy()
    df["asset_short"] = df["asset"].map(_asset_short)
    df = df.sort_values("processed_rows", ascending=False)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    bars = ax1.bar(x, df["processed_rows"], color="#3B82F6", label="Processed rows (daily)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["asset_short"])
    ax1.set_ylabel("Processed rows")
    ax1.set_title("Data Coverage by Asset (Post-Processing)")

    ax2 = ax1.twinx()
    ax2.plot(x, df["sentiment_column_count"], color="#F97316", marker="o", linewidth=2, label="Sentiment columns")
    ax2.set_ylabel("Sentiment feature columns")

    for b in bars:
        height = b.get_height()
        ax1.annotate(
            f"{int(height)}",
            xy=(b.get_x() + b.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    out = out_dir / "fig_01_data_coverage.png"
    _save(fig, out)
    return FigureRecord(
        figure_id="fig_01",
        filename=out.name,
        title="Data coverage and sentiment column availability by asset",
        source_files="data_manifest.csv",
    )


def _plot_data_timespan(data_manifest: pd.DataFrame, out_dir: Path) -> FigureRecord:
    df = data_manifest.copy()
    df["asset_short"] = df["asset"].map(_asset_short)
    df["start"] = pd.to_datetime(df["processed_time_start"], errors="coerce")
    df["end"] = pd.to_datetime(df["processed_time_end"], errors="coerce")
    df = df.dropna(subset=["start", "end"]).sort_values("start")

    fig, ax = plt.subplots(figsize=(11, 5))
    y = np.arange(len(df))
    for i, row in enumerate(df.itertuples(index=False)):
        ax.hlines(i, row.start, row.end, color="#0EA5A4", linewidth=6)
        ax.plot([row.start, row.end], [i, i], "o", color="#115E59", markersize=4)

    ax.set_yticks(y)
    ax.set_yticklabels(df["asset_short"])
    ax.set_xlabel("Date")
    ax.set_title("Processed Date Span by Asset")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="x", alpha=0.25)

    out = out_dir / "fig_02_data_timespan.png"
    _save(fig, out)
    return FigureRecord(
        figure_id="fig_02",
        filename=out.name,
        title="Processed date span per asset",
        source_files="data_manifest.csv",
    )


def _plot_btc_walkforward_timeline(
    baseline_manifest: pd.DataFrame,
    transformer_manifest: pd.DataFrame,
    out_dir: Path,
) -> FigureRecord:
    b = baseline_manifest.copy()
    t = transformer_manifest.copy()
    for df in (b, t):
        df["train_start"] = pd.to_datetime(df["train_start"], errors="coerce")
        df["train_end"] = pd.to_datetime(df["train_end"], errors="coerce")
        df["eval_start"] = pd.to_datetime(df["eval_start"], errors="coerce")
        df["eval_end"] = pd.to_datetime(df["eval_end"], errors="coerce")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    panels = [("Baselines", b, axes[0]), ("Transformers", t, axes[1])]
    for title, df, ax in panels:
        for row in df.itertuples(index=False):
            y = int(row.fold)
            ax.hlines(y, row.train_start, row.train_end, color="#2563EB", linewidth=4)
            ax.hlines(y, row.eval_start, row.eval_end, color="#EA580C", linewidth=4)
        ax.set_ylabel("Fold")
        ax.set_title(f"BTC Walk-Forward Splits ({title})")
        ax.grid(axis="x", alpha=0.25)

    axes[1].set_xlabel("Date")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    out = out_dir / "fig_03_btc_walkforward_timeline.png"
    _save(fig, out)
    return FigureRecord(
        figure_id="fig_03",
        filename=out.name,
        title="BTC walk-forward fold timeline for baseline and transformer runs",
        source_files=(
            "multi_asset_baselines_h1_paired_summary_btc_no_sent_fold_manifest.csv;"
            "multi_asset_transformers_h1_paired_summary_btc_no_sent_fold_manifest.csv"
        ),
    )


def _plot_stationarity_moments(stationarity_summaries: pd.DataFrame, out_dir: Path) -> FigureRecord:
    df = stationarity_summaries.copy()
    df["asset"] = df["asset"].str.upper()
    df = df.sort_values("asset")
    metrics = [
        ("return_mean", "Return Mean"),
        ("return_std", "Return Std"),
        ("return_skew", "Return Skewness"),
        ("return_kurtosis", "Return Kurtosis"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    for ax, (col, label) in zip(axes, metrics):
        ax.bar(df["asset"], df[col], color="#64748B")
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.2)

    out = out_dir / "fig_04_stationarity_moments.png"
    _save(fig, out)
    return FigureRecord(
        figure_id="fig_04",
        filename=out.name,
        title="Distributional moments of return series used in stationarity diagnostics",
        source_files="appendix/stationarity/*_stationarity_summary.csv",
    )


def _plot_stationarity_tests(stationarity_tests: pd.DataFrame, out_dir: Path) -> FigureRecord:
    df = stationarity_tests.copy()
    df = df[df["series"] == "return_1d"].copy()
    df["asset"] = df["asset"].str.upper()
    pivot = df.pivot_table(index="asset", columns="test", values="p_value", aggfunc="mean").sort_index()
    if pivot.empty:
        pivot = pd.DataFrame(index=["N/A"], columns=["adf", "kpss"], data=[[np.nan, np.nan]])

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Stationarity Test p-values on Return Series (ADF/KPSS)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            txt = "-" if pd.isna(val) else f"{val:.3f}"
            ax.text(j, i, txt, ha="center", va="center", color="white", fontsize=8)

    out = out_dir / "fig_05_stationarity_tests_return_pvalues.png"
    _save(fig, out)
    return FigureRecord(
        figure_id="fig_05",
        filename=out.name,
        title="ADF/KPSS p-value heatmap for 1-day returns",
        source_files="appendix/stationarity/*_stationarity_tests.csv",
    )


def _plot_best_model_by_asset(best_by_asset: pd.DataFrame, out_dir: Path) -> FigureRecord:
    df = best_by_asset.copy().sort_values("asset")
    df["asset"] = df["asset"].str.upper()
    colors = {"baselines": "#2563EB", "transformers": "#7C3AED", "chronos2_zero_shot": "#059669"}
    bar_colors = [colors.get(f, "#6B7280") for f in df["family"]]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(df["asset"], df["mae_mean"], color=bar_colors)
    ax.set_title("Best Model per Asset (MAE, No-Sentiment)")
    ax.set_ylabel("MAE")
    ax.grid(axis="y", alpha=0.25)
    for b, model in zip(bars, df["model"]):
        ax.annotate(
            model,
            xy=(b.get_x() + b.get_width() / 2, b.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )

    out = out_dir / "fig_06_best_model_mae_by_asset.png"
    _save(fig, out)
    return FigureRecord(
        figure_id="fig_06",
        filename=out.name,
        title="Best model by asset under common MAE scale",
        source_files="PAPER_H1_best_by_asset.csv",
    )


def _plot_family_performance(family_perf: pd.DataFrame, out_dir: Path) -> FigureRecord:
    df = family_perf.copy()
    df["label"] = df["family"] + ":" + df["model"] + ":" + df["mode"]
    df = df.sort_values("avg_mae_mean", ascending=True)
    x = np.arange(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(x, df["avg_mae_mean"], color="#2563EB")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["label"], rotation=75, ha="right", fontsize=8)
    axes[0].set_title("Average MAE by Family/Model/Mode")
    axes[0].set_ylabel("MAE")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, df["avg_directional_accuracy_mean"], color="#16A34A")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["label"], rotation=75, ha="right", fontsize=8)
    axes[1].set_title("Average Directional Accuracy by Family/Model/Mode")
    axes[1].set_ylabel("Directional Accuracy")
    axes[1].grid(axis="y", alpha=0.25)

    out = out_dir / "fig_07_family_performance.png"
    _save(fig, out)
    return FigureRecord(
        figure_id="fig_07",
        filename=out.name,
        title="Family-level comparative performance across modes",
        source_files="PAPER_H1_family_performance.csv",
    )


def _plot_sentiment_delta_distribution(sent_delta: pd.DataFrame, out_dir: Path) -> FigureRecord:
    df = sent_delta.copy()
    metrics = [
        ("delta_mae_mean_with_minus_no", "Delta MAE"),
        ("delta_rmse_mean_with_minus_no", "Delta RMSE"),
        ("delta_directional_accuracy_mean_with_minus_no", "Delta Directional Accuracy"),
    ]
    families = sorted(df["family"].dropna().unique().tolist())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (col, title) in zip(axes, metrics):
        vals = [df.loc[df["family"] == f, col].dropna().to_numpy(dtype=float) for f in families]
        ax.boxplot(vals, labels=families, showmeans=True)
        ax.axhline(0.0, color="#DC2626", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.2)

    out = out_dir / "fig_08_sentiment_delta_distributions.png"
    _save(fig, out)
    return FigureRecord(
        figure_id="fig_08",
        filename=out.name,
        title="Distribution of performance deltas from adding sentiment features",
        source_files="PAPER_H1_sentiment_delta.csv",
    )


def _plot_sentiment_delta_heatmap(sent_delta: pd.DataFrame, out_dir: Path) -> FigureRecord:
    df = sent_delta.copy()
    summary = (
        df.groupby(["asset", "family"], as_index=False)["delta_mae_mean_with_minus_no"]
        .mean()
        .pivot(index="asset", columns="family", values="delta_mae_mean_with_minus_no")
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(summary.to_numpy(dtype=float), cmap="coolwarm", aspect="auto")
    ax.set_xticks(np.arange(len(summary.columns)))
    ax.set_xticklabels(summary.columns, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(summary.index)))
    ax.set_yticklabels([a.upper() for a in summary.index])
    ax.set_title("Average Delta MAE (With Sentiment - No Sentiment)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(summary.shape[0]):
        for j in range(summary.shape[1]):
            v = summary.iloc[i, j]
            txt = "-" if pd.isna(v) else f"{v:+.3f}"
            ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=8)

    out = out_dir / "fig_09_sentiment_delta_heatmap.png"
    _save(fig, out)
    return FigureRecord(
        figure_id="fig_09",
        filename=out.name,
        title="Asset/family heatmap for sentiment-induced MAE changes",
        source_files="PAPER_H1_sentiment_delta.csv",
    )


def _plot_ablation_deltas(ablation_delta: pd.DataFrame, out_dir: Path) -> FigureRecord:
    df = ablation_delta.copy()
    df = df[df["mode"] != "financial_only"].copy()
    if df.empty:
        raise ValueError("Ablation delta table has no non-financial-only rows")

    mode_agg = df.groupby("mode", as_index=False).agg(
        avg_delta_mae=("delta_mae_vs_financial_only", "mean"),
        avg_delta_da=("delta_directional_accuracy_vs_financial_only", "mean"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(mode_agg["mode"], mode_agg["avg_delta_mae"], color="#DC2626")
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_title("Ablation Delta MAE vs Financial-Only")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(mode_agg["mode"], mode_agg["avg_delta_da"], color="#2563EB")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_title("Ablation Delta Directional Accuracy vs Financial-Only")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.2)

    out = out_dir / "fig_10_ablation_mode_deltas.png"
    _save(fig, out)
    return FigureRecord(
        figure_id="fig_10",
        filename=out.name,
        title="Ablation deltas relative to financial-only feature mode",
        source_files="feature_ablation_h1_summary_mode_deltas.csv",
    )


def _plot_ablation_wins(ablation_wins: pd.DataFrame, out_dir: Path) -> FigureRecord:
    df = ablation_wins.copy().sort_values("n_assets_won", ascending=False)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    x = np.arange(len(df))
    ax1.bar(x, df["n_assets_won"], color="#0EA5A4")
    ax1.set_ylabel("Number of assets won")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["mode"], rotation=20, ha="right")
    ax1.set_title("Ablation Mode Wins (CPCV)")

    ax2 = ax1.twinx()
    ax2.plot(x, df["avg_winning_mae"], color="#9333EA", marker="o", linewidth=2)
    ax2.set_ylabel("Average winning MAE")

    out = out_dir / "fig_11_ablation_mode_wins.png"
    _save(fig, out)
    return FigureRecord(
        figure_id="fig_11",
        filename=out.name,
        title="Winning frequency and MAE for each ablation mode",
        source_files="PAPER_H1_ablation_mode_wins.csv",
    )


def _plot_transfer_metrics(no_sent: pd.DataFrame, with_sent: pd.DataFrame, out_dir: Path) -> FigureRecord:
    metrics = ["mae", "rmse", "directional_accuracy", "sharpe_5bps", "dsr"]
    a = no_sent[["mode", *metrics]].copy()
    a["setup"] = "no_sentiment"
    b = with_sent[["mode", *metrics]].copy()
    b["setup"] = "with_sentiment"
    df = pd.concat([a, b], ignore_index=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    pairs = [("mae", "MAE"), ("rmse", "RMSE"), ("directional_accuracy", "Directional Accuracy"), ("dsr", "DSR")]
    for ax, (col, title) in zip(axes, pairs):
        for setup, color in [("no_sentiment", "#2563EB"), ("with_sentiment", "#DC2626")]:
            sub = df[df["setup"] == setup].copy()
            ax.plot(sub["mode"], sub[col], marker="o", linewidth=2, label=setup, color=color)
        ax.set_title(f"Transfer {title}")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend()

    out = out_dir / "fig_12_transfer_metrics.png"
    _save(fig, out)
    return FigureRecord(
        figure_id="fig_12",
        filename=out.name,
        title="BTC->ETH transfer metrics in no-sentiment and with-sentiment settings",
        source_files="transfer_btc_eth_patchtst_no_sent.csv;transfer_btc_eth_patchtst_with_sent.csv",
    )


def _plot_transfer_predictions(no_sent_pred: pd.DataFrame, with_sent_pred: pd.DataFrame, out_dir: Path) -> FigureRecord:
    n_tail = 120
    a = no_sent_pred.copy().tail(n_tail)
    b = with_sent_pred.copy().tail(n_tail)
    a["time"] = pd.to_datetime(a["time"], errors="coerce")
    b["time"] = pd.to_datetime(b["time"], errors="coerce")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(a["time"], a["y_true"], color="black", linewidth=1.5, label="true")
    axes[0].plot(a["time"], a["pred_source_zero_shot"], color="#2563EB", linewidth=1.2, label="source_zero_shot")
    axes[0].plot(a["time"], a["pred_target_only"], color="#EA580C", linewidth=1.2, label="target_only")
    axes[0].set_title("Transfer Predictions (No Sentiment, last 120 points)")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.2)

    axes[1].plot(b["time"], b["y_true"], color="black", linewidth=1.5, label="true")
    axes[1].plot(b["time"], b["pred_source_zero_shot"], color="#DC2626", linewidth=1.2, label="source_zero_shot")
    axes[1].plot(b["time"], b["pred_target_only"], color="#16A34A", linewidth=1.2, label="target_only")
    axes[1].set_title("Transfer Predictions (With Sentiment, last 120 points)")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.2)

    out = out_dir / "fig_13_transfer_prediction_traces.png"
    _save(fig, out)
    return FigureRecord(
        figure_id="fig_13",
        filename=out.name,
        title="Prediction traces for BTC->ETH transfer (no-sentiment vs with-sentiment)",
        source_files=(
            "transfer_btc_eth_patchtst_no_sent_predictions.csv;"
            "transfer_btc_eth_patchtst_with_sent_predictions.csv"
        ),
    )


def _plot_cost_sensitivity(best_by_asset: pd.DataFrame, out_dir: Path) -> FigureRecord:
    df = best_by_asset.copy()
    df["asset"] = df["asset"].str.upper()
    cols = ["sharpe_0bps_mean", "sharpe_5bps_mean", "sharpe_10bps_mean", "sharpe_20bps_mean"]
    long = df[["asset", *cols]].melt(id_vars=["asset"], var_name="cost", value_name="sharpe")
    long["cost"] = long["cost"].str.extract(r"sharpe_(\d+)bps_mean")[0].astype(int)

    fig, ax = plt.subplots(figsize=(10, 5))
    for asset in sorted(long["asset"].unique()):
        sub = long[long["asset"] == asset].sort_values("cost")
        ax.plot(sub["cost"], sub["sharpe"], marker="o", linewidth=1.5, label=asset)
    ax.set_xlabel("Transaction cost (bps)")
    ax.set_ylabel("Sharpe")
    ax.set_title("Cost Sensitivity of Best-Per-Asset Models")
    ax.grid(alpha=0.2)
    ax.legend(ncol=4, fontsize=8, loc="lower left")

    out = out_dir / "fig_14_cost_sensitivity_sharpe.png"
    _save(fig, out)
    return FigureRecord(
        figure_id="fig_14",
        filename=out.name,
        title="Sharpe sensitivity to transaction costs for best-per-asset models",
        source_files="PAPER_H1_best_by_asset.csv",
    )


def generate_paper_figures(results_root: str | Path, output_dir: str | Path) -> list[FigureRecord]:
    root = Path(results_root)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_manifest = _load_csv(root / "data_manifest.csv")
    best_by_asset = _load_csv(root / "PAPER_H1_best_by_asset.csv")
    family_perf = _load_csv(root / "PAPER_H1_family_performance.csv")
    sent_delta = _load_csv(root / "PAPER_H1_sentiment_delta.csv")
    ablation_delta = _load_csv(root / "feature_ablation_h1_summary_mode_deltas.csv")
    ablation_wins = _load_csv(root / "PAPER_H1_ablation_mode_wins.csv")
    transfer_no_sent = _load_csv(root / "transfer_btc_eth_patchtst_no_sent.csv")
    transfer_with_sent = _load_csv(root / "transfer_btc_eth_patchtst_with_sent.csv")
    transfer_no_sent_pred = _load_csv(root / "transfer_btc_eth_patchtst_no_sent_predictions.csv")
    transfer_with_sent_pred = _load_csv(root / "transfer_btc_eth_patchtst_with_sent_predictions.csv")
    btc_base_fold_manifest = _load_csv(root / "multi_asset_baselines_h1_paired_summary_btc_no_sent_fold_manifest.csv")
    btc_trf_fold_manifest = _load_csv(root / "multi_asset_transformers_h1_paired_summary_btc_no_sent_fold_manifest.csv")

    st_sum_paths = sorted((root / "appendix" / "stationarity").glob("*_stationarity_summary.csv"))
    st_test_paths = sorted((root / "appendix" / "stationarity").glob("*_stationarity_tests.csv"))
    if not st_sum_paths or not st_test_paths:
        raise FileNotFoundError("Missing stationarity appendix CSV files")
    stationarity_summaries = pd.concat([pd.read_csv(p) for p in st_sum_paths], ignore_index=True)
    stationarity_tests = pd.concat([pd.read_csv(p) for p in st_test_paths], ignore_index=True)

    records = [
        _plot_data_coverage(data_manifest, out_dir),
        _plot_data_timespan(data_manifest, out_dir),
        _plot_btc_walkforward_timeline(btc_base_fold_manifest, btc_trf_fold_manifest, out_dir),
        _plot_stationarity_moments(stationarity_summaries, out_dir),
        _plot_stationarity_tests(stationarity_tests, out_dir),
        _plot_best_model_by_asset(best_by_asset, out_dir),
        _plot_family_performance(family_perf, out_dir),
        _plot_sentiment_delta_distribution(sent_delta, out_dir),
        _plot_sentiment_delta_heatmap(sent_delta, out_dir),
        _plot_ablation_deltas(ablation_delta, out_dir),
        _plot_ablation_wins(ablation_wins, out_dir),
        _plot_transfer_metrics(transfer_no_sent, transfer_with_sent, out_dir),
        _plot_transfer_predictions(transfer_no_sent_pred, transfer_with_sent_pred, out_dir),
        _plot_cost_sensitivity(best_by_asset, out_dir),
    ]

    _records_to_csv(records, out_dir / "FIGURES_MANIFEST.csv")
    return records
