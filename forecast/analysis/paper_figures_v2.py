from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class FigureRecord:
    figure_id: str
    filename: str
    title: str
    section: str
    question: str
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
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _records_to_csv(records: list[FigureRecord], output_path: Path) -> None:
    rows = [
        {
            "figure_id": r.figure_id,
            "filename": r.filename,
            "title": r.title,
            "section": r.section,
            "question": r.question,
            "source_files": r.source_files,
        }
        for r in records
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _record(
    records: list[FigureRecord],
    figure_id: str,
    filename: str,
    title: str,
    section: str,
    question: str,
    source_files: str,
) -> None:
    records.append(
        FigureRecord(
            figure_id=figure_id,
            filename=filename,
            title=title,
            section=section,
            question=question,
            source_files=source_files,
        )
    )


def _save_and_record(
    records: list[FigureRecord],
    fig: plt.Figure,
    out_path: Path,
    figure_id: str,
    title: str,
    section: str,
    question: str,
    source_files: str,
) -> None:
    _save(fig, out_path)
    _record(records, figure_id, out_path.name, title, section, question, source_files)


def _copy_and_record(
    records: list[FigureRecord],
    src_path: Path,
    out_path: Path,
    figure_id: str,
    title: str,
    section: str,
    question: str,
    source_files: str,
) -> None:
    if not src_path.exists():
        raise FileNotFoundError(f"Missing source image: {src_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, out_path)
    _record(records, figure_id, out_path.name, title, section, question, source_files)


def _family_color(model_name: str) -> str:
    m = model_name.lower()
    if "chronos" in m:
        return "#059669"
    if "random_walk" in m or "linear" in m or "xgboost" in m or "arima" in m:
        return "#2563EB"
    return "#7C3AED"


DEFAULT_FIGURE_WINNER_POLICY = "composite_rank"
DEFAULT_FIGURE_WINNER_METRICS = ["mae_mean", "rmse_mean", "directional_accuracy_mean", "sharpe_5bps_mean"]
LOWER_IS_BETTER_METRICS = {"mae_mean", "rmse_mean"}
HIGHER_IS_BETTER_METRICS = {"directional_accuracy_mean", "sharpe_5bps_mean"}


def _load_stationarity_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    st_root = root / "appendix" / "stationarity"
    summary_paths = sorted(st_root.glob("*_stationarity_summary.csv"))
    tests_paths = sorted(st_root.glob("*_stationarity_tests.csv"))
    if not summary_paths or not tests_paths:
        raise FileNotFoundError(f"Missing stationarity CSV files under: {st_root}")
    summaries = pd.concat([pd.read_csv(p) for p in summary_paths], ignore_index=True)
    tests = pd.concat([pd.read_csv(p) for p in tests_paths], ignore_index=True)
    return summaries, tests


def _build_missingness_summary(root: Path) -> pd.DataFrame:
    rows = []
    pattern = "multi_asset_baselines_h1_paired_summary_*_with_sent_data_quality_summary.csv"
    for path in sorted(root.glob(pattern)):
        asset = path.name.replace("multi_asset_baselines_h1_paired_summary_", "").replace("_with_sent_data_quality_summary.csv", "")
        df = pd.read_csv(path)
        if df.empty:
            continue
        r = df.iloc[0].to_dict()
        r["asset"] = asset
        rows.append(r)
    if not rows:
        raise FileNotFoundError("No data quality summary files found for missingness figure")
    return pd.DataFrame(rows)


def _choose_best_model(
    summary_df: pd.DataFrame,
    asset: str,
    mode: str = "no_sentiment",
    objective_track: str = "point",
    winner_policy: str = DEFAULT_FIGURE_WINNER_POLICY,
    winner_exclude_models: set[str] | None = None,
    winner_metrics: list[str] | None = None,
) -> str:
    sub = summary_df[(summary_df["asset"] == asset) & (summary_df["mode"] == mode)].copy()
    if "objective_track" in sub.columns:
        sub = sub[sub["objective_track"].fillna("point") == objective_track]
    if sub.empty:
        raise ValueError(f"No rows for asset={asset}, mode={mode}")
    ranked = _rank_for_winner_policy(
        sub,
        winner_policy=winner_policy,
        winner_exclude_models=winner_exclude_models or set(),
        winner_metrics=winner_metrics or list(DEFAULT_FIGURE_WINNER_METRICS),
    )
    return str(ranked.iloc[0]["model"])


def _rank_for_winner_policy(
    df: pd.DataFrame,
    *,
    winner_policy: str,
    winner_exclude_models: set[str],
    winner_metrics: list[str],
) -> pd.DataFrame:
    candidates = df.copy()
    if "model" in candidates.columns and winner_exclude_models:
        filtered = candidates[~candidates["model"].astype(str).isin(winner_exclude_models)].copy()
        if not filtered.empty:
            candidates = filtered

    if winner_policy == "composite_rank":
        n = len(candidates)
        if n <= 1:
            candidates["selection_score"] = 1.0
        else:
            aligned_scores: list[pd.Series] = []
            for metric in winner_metrics:
                if metric not in candidates.columns:
                    continue
                if metric not in LOWER_IS_BETTER_METRICS and metric not in HIGHER_IS_BETTER_METRICS:
                    continue
                vals = pd.to_numeric(candidates[metric], errors="coerce")
                if vals.notna().sum() == 0:
                    continue
                if metric in LOWER_IS_BETTER_METRICS:
                    rank = vals.rank(method="average", ascending=True, na_option="bottom")
                else:
                    rank = vals.rank(method="average", ascending=False, na_option="bottom")
                aligned_scores.append((1.0 - (rank - 1.0) / max(n - 1, 1)).fillna(0.0))
            if aligned_scores:
                candidates["selection_score"] = pd.concat(aligned_scores, axis=1).mean(axis=1)
            else:
                candidates["selection_score"] = np.nan
        sort_cols = ["selection_score", "mae_mean", "directional_accuracy_mean", "model"]
        sort_asc = [False, True, False, True]
    else:
        candidates["selection_score"] = pd.to_numeric(candidates.get("mae_mean", np.nan), errors="coerce")
        sort_cols = ["mae_mean", "directional_accuracy_mean", "model"]
        sort_asc = [True, False, True]

    sort_cols = [c for c in sort_cols if c in candidates.columns]
    sort_asc = sort_asc[: len(sort_cols)]
    return candidates.sort_values(sort_cols, ascending=sort_asc, na_position="last")


def _choose_best_available_model(
    summary_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    asset: str,
    mode: str = "no_sentiment",
    objective_track: str = "point",
    winner_policy: str = DEFAULT_FIGURE_WINNER_POLICY,
    winner_exclude_models: set[str] | None = None,
    winner_metrics: list[str] | None = None,
) -> str:
    sub = summary_df[(summary_df["asset"] == asset) & (summary_df["mode"] == mode)].copy()
    if "objective_track" in sub.columns:
        sub = sub[sub["objective_track"].fillna("point") == objective_track]
    if sub.empty:
        raise ValueError(f"No rows for asset={asset}, mode={mode}")

    pred = predictions_df[(predictions_df["asset"] == asset) & (predictions_df["mode"] == mode)].copy()
    if "objective_track" in pred.columns:
        pred = pred[pred["objective_track"].fillna("point") == objective_track]
    available_models = set(pred["model"].astype(str).unique().tolist())

    ranked = _rank_for_winner_policy(
        sub,
        winner_policy=winner_policy,
        winner_exclude_models=winner_exclude_models or set(),
        winner_metrics=winner_metrics or list(DEFAULT_FIGURE_WINNER_METRICS),
    )
    for row in ranked.itertuples(index=False):
        model = str(getattr(row, "model"))
        if model in available_models:
            return model
    return str(ranked.iloc[0]["model"])


def _prediction_trace(
    pred_all: pd.DataFrame,
    asset: str,
    models: list[tuple[str, str]],
    out_path: Path,
    title: str,
    records: list[FigureRecord],
    figure_id: str,
    foundation_predictions_source: str,
) -> None:
    df = pred_all[(pred_all["asset"] == asset) & (pred_all["mode"] == "no_sentiment")].copy()
    if "objective_track" in df.columns:
        df = df[df["objective_track"].fillna("point") == "point"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    tail_n = 260
    fig, ax = plt.subplots(figsize=(12, 4.8))

    truth = (
        df[["timestamp", "y_true"]]
        .groupby("timestamp", as_index=False)["y_true"]
        .mean()
        .sort_values("timestamp")
        .tail(tail_n)
    )
    if truth.empty:
        return
    full_idx = pd.date_range(truth["timestamp"].min(), truth["timestamp"].max(), freq="D")
    truth = truth.set_index("timestamp").reindex(full_idx).rename_axis("timestamp").reset_index()
    ax.plot(truth["timestamp"], truth["y_true"], color="black", linewidth=1.8, label="y_true")

    role_colors = {
        "baseline": "#2563EB",
        "transformer": "#F97316",
        "foundation": "#059669",
    }
    for role, model in models:
        sub = (
            df[df["model"] == model][["timestamp", "y_pred"]]
            .groupby("timestamp", as_index=False)["y_pred"]
            .mean()
            .sort_values("timestamp")
        )
        if sub.empty:
            continue
        sub = sub.set_index("timestamp").reindex(full_idx).rename_axis("timestamp").reset_index()
        ax.plot(
            sub["timestamp"],
            sub["y_pred"],
            linewidth=1.4,
            label=f"{role}: {model}",
            color=role_colors.get(role, _family_color(model)),
        )

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("1-day log-return")
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="upper left", fontsize=8)

    _save_and_record(
        records,
        fig,
        out_path,
        figure_id,
        title,
        "5. Results and Findings",
        "How do top baseline/transformer/foundation models track realized returns over time?",
        "multi_asset_baselines_h1_paired_summary_predictions.csv;"
        "multi_asset_transformers_h1_paired_summary_predictions.csv;"
        f"{foundation_predictions_source}",
    )


def generate_paper_figures_v2(results_root: str | Path, output_dir: str | Path) -> list[FigureRecord]:
    root = Path(results_root)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_manifest = _load_csv(root / "data_manifest.csv")
    base_summary = _load_csv(root / "multi_asset_baselines_h1_paired_summary.csv")
    trf_summary = _load_csv(root / "multi_asset_transformers_h1_paired_summary.csv")
    benchmark_assets = sorted(base_summary["asset"].dropna().astype(str).unique().tolist())
    foundation_summary_path = root / "multi_asset_foundation_h1_summary.csv"
    if foundation_summary_path.exists():
        chrn_summary = _load_csv(foundation_summary_path)
        foundation_prefix = "multi_asset_foundation_h1_summary"
    else:
        chrn_summary = _load_csv(root / "multi_asset_chronos2_h1_summary.csv")
        foundation_prefix = "multi_asset_chronos2_h1_summary"
    foundation_predictions_source = f"{foundation_prefix}_predictions.csv"
    if "mode" not in chrn_summary.columns:
        chrn_summary["mode"] = "no_sentiment"
    if "objective_track" not in base_summary.columns:
        base_summary["objective_track"] = "point"
    if "objective_track" not in trf_summary.columns:
        trf_summary["objective_track"] = "point"
    if "objective_track" not in chrn_summary.columns:
        chrn_summary["objective_track"] = "point"

    best_by_asset = _load_csv(root / "PAPER_H1_best_by_asset.csv")
    family_perf = _load_csv(root / "PAPER_H1_family_performance.csv")
    sent_delta = _load_csv(root / "PAPER_H1_sentiment_delta.csv")
    ablation_delta = _load_csv(root / "feature_ablation_h1_summary_mode_deltas.csv")
    ablation_wins = _load_csv(root / "PAPER_H1_ablation_mode_wins.csv")
    skill_vs_zero_path = root / "PAPER_H1_skill_vs_zero.csv"
    skill_vs_zero = pd.read_csv(skill_vs_zero_path) if skill_vs_zero_path.exists() else pd.DataFrame()

    winner_policy = DEFAULT_FIGURE_WINNER_POLICY
    if "selection_policy" in best_by_asset.columns and best_by_asset["selection_policy"].notna().any():
        winner_policy = str(best_by_asset["selection_policy"].dropna().iloc[0])
    winner_excluded_models: set[str] = {"zero_forecast"}
    if "selection_excluded_models" in best_by_asset.columns and best_by_asset["selection_excluded_models"].notna().any():
        raw = str(best_by_asset["selection_excluded_models"].dropna().iloc[0]).strip()
        if raw:
            winner_excluded_models = {p.strip() for p in raw.split(",") if p.strip()}

    base_pred = _load_csv(root / "multi_asset_baselines_h1_paired_summary_predictions.csv")
    trf_pred = _load_csv(root / "multi_asset_transformers_h1_paired_summary_predictions.csv")
    chrn_pred = _load_csv(root / f"{foundation_prefix}_predictions.csv")
    if "objective_track" not in base_pred.columns:
        base_pred["objective_track"] = "point"
    if "objective_track" not in trf_pred.columns:
        trf_pred["objective_track"] = "point"
    if "objective_track" not in chrn_pred.columns:
        chrn_pred["objective_track"] = "point"
    pred_all = pd.concat([base_pred, trf_pred, chrn_pred], ignore_index=True)

    base_quant = _load_csv(root / "multi_asset_baselines_h1_paired_summary_quantiles.csv")
    trf_quant = _load_csv(root / "multi_asset_transformers_h1_paired_summary_quantiles.csv")
    chrn_quant = _load_csv(root / f"{foundation_prefix}_quantiles.csv")
    if "objective_track" not in base_quant.columns:
        base_quant["objective_track"] = "point"
    if "objective_track" not in trf_quant.columns:
        trf_quant["objective_track"] = "point"
    if "objective_track" not in chrn_quant.columns:
        chrn_quant["objective_track"] = "point"
    quant_all = pd.concat([base_quant, trf_quant, chrn_quant], ignore_index=True)

    trf_history = _load_csv(root / "multi_asset_transformers_h1_paired_summary_history.csv")

    transfer_no_sent = _load_csv(root / "transfer_btc_eth_patchtst_no_sent.csv")
    transfer_with_sent = _load_csv(root / "transfer_btc_eth_patchtst_with_sent.csv")
    transfer_no_sent_pred = _load_csv(root / "transfer_btc_eth_patchtst_no_sent_predictions.csv")
    transfer_with_sent_pred = _load_csv(root / "transfer_btc_eth_patchtst_with_sent_predictions.csv")

    stationarity_summaries, stationarity_tests = _load_stationarity_tables(root)
    missingness_summary = _build_missingness_summary(root)

    records: list[FigureRecord] = []

    # FIG 01: Coverage and sentiment columns
    dm = data_manifest.copy().sort_values("processed_rows", ascending=False)
    dm["asset_short"] = dm["asset"].map(_asset_short)
    fig, ax1 = plt.subplots(figsize=(10.5, 4.5))
    x = np.arange(len(dm))
    ax1.bar(x, dm["processed_rows"], color="#1D4ED8", alpha=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(dm["asset_short"])
    ax1.set_ylabel("Processed rows")
    ax1.grid(axis="y", alpha=0.2)
    ax2 = ax1.twinx()
    ax2.plot(x, dm["sentiment_column_count"], color="#EA580C", marker="o", linewidth=2)
    ax2.set_ylabel("Sentiment column count")
    ax1.set_title("Asset Coverage and Sentiment Availability")
    _save_and_record(
        records,
        fig,
        out_dir / "fig_v2_01_asset_coverage.png",
        "fig_v2_01",
        "Asset coverage and sentiment column availability",
        "3. Methodology",
        "Do all assets have enough processed observations and sentiment coverage?",
        "data_manifest.csv",
    )

    # FIG 02: Timespan by asset
    dm2 = data_manifest.copy()
    dm2["asset_short"] = dm2["asset"].map(_asset_short)
    dm2["start"] = pd.to_datetime(dm2["processed_time_start"], errors="coerce")
    dm2["end"] = pd.to_datetime(dm2["processed_time_end"], errors="coerce")
    dm2 = dm2.dropna(subset=["start", "end"]).sort_values("start")
    fig, ax = plt.subplots(figsize=(11.2, 4.6))
    y = np.arange(len(dm2))
    for i, row in enumerate(dm2.itertuples(index=False)):
        ax.hlines(i, row.start, row.end, color="#0F766E", linewidth=6)
        ax.plot([row.start, row.end], [i, i], "o", color="#115E59", markersize=4)
    ax.set_yticks(y)
    ax.set_yticklabels(dm2["asset_short"])
    ax.set_xlabel("Date")
    ax.set_title("Processed Date Span by Asset")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="x", alpha=0.25)
    _save_and_record(
        records,
        fig,
        out_dir / "fig_v2_02_timespan.png",
        "fig_v2_02",
        "Processed date span by asset",
        "3. Methodology",
        "How balanced is temporal coverage across assets?",
        "data_manifest.csv",
    )

    # FIG 03: Row retention raw->processed
    dm3 = data_manifest.copy()
    dm3["asset_short"] = dm3["asset"].map(_asset_short)
    dm3["retention_pct"] = 100.0 * dm3["processed_rows"] / dm3["raw_rows"].replace(0, np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(len(dm3))
    axes[0].bar(x - 0.2, dm3["raw_rows"], width=0.4, label="raw", color="#94A3B8")
    axes[0].bar(x + 0.2, dm3["processed_rows"], width=0.4, label="processed", color="#2563EB")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(dm3["asset_short"], rotation=0)
    axes[0].set_title("Rows: Raw vs Processed")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(x, dm3["retention_pct"], color="#16A34A")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(dm3["asset_short"], rotation=0)
    axes[1].set_ylim(0, 105)
    axes[1].set_title("Retention Percentage")
    axes[1].set_ylabel("% retained")
    axes[1].grid(axis="y", alpha=0.2)
    _save_and_record(
        records,
        fig,
        out_dir / "fig_v2_03_row_retention.png",
        "fig_v2_03",
        "Raw-to-processed row retention by asset",
        "3. Methodology",
        "Where does preprocessing reduce sample size the most?",
        "data_manifest.csv",
    )

    # FIG 04: Missingness and row-drop pipeline summary
    miss = missingness_summary.copy().sort_values("asset")
    miss["asset"] = miss["asset"].str.upper()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    axes[0].bar(miss["asset"], miss["rows_dropped_by_dropna_pct"], color="#DC2626")
    axes[0].set_title("Rows Dropped by DropNA (%)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(miss["asset"], miss["rows_lost_from_sentiment_lag_pct"], color="#EA580C")
    axes[1].set_title("Rows Lost from Sentiment Lag (%)")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.2)

    axes[2].bar(miss["asset"], miss["missing_cells_pre_dropna"], color="#475569", label="pre-dropna")
    axes[2].bar(miss["asset"], miss["missing_cells_post_dropna"], color="#0EA5E9", label="post-dropna")
    axes[2].set_title("Missing Cells Pre/Post DropNA")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].legend(fontsize=8)
    axes[2].grid(axis="y", alpha=0.2)

    _save_and_record(
        records,
        fig,
        out_dir / "fig_v2_04_missingness_pipeline.png",
        "fig_v2_04",
        "Causal missingness pipeline impact by asset",
        "3. Methodology",
        "How much data is lost by strict causal lagging and NA filtering?",
        "multi_asset_baselines_h1_paired_summary_*_with_sent_data_quality_summary.csv",
    )

    # FIG 05: BTC walk-forward timelines
    bman = _load_csv(root / "multi_asset_baselines_h1_paired_summary_btc_no_sent_fold_manifest.csv")
    tman = _load_csv(root / "multi_asset_transformers_h1_paired_summary_btc_no_sent_fold_manifest.csv")
    for df in (bman, tman):
        df["train_start"] = pd.to_datetime(df["train_start"], errors="coerce")
        df["train_end"] = pd.to_datetime(df["train_end"], errors="coerce")
        df["eval_start"] = pd.to_datetime(df["eval_start"], errors="coerce")
        df["eval_end"] = pd.to_datetime(df["eval_end"], errors="coerce")
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    for title, df, ax in [("Baselines", bman, axes[0]), ("Transformers", tman, axes[1])]:
        for row in df.itertuples(index=False):
            yv = int(row.fold)
            ax.hlines(yv, row.train_start, row.train_end, color="#2563EB", linewidth=4)
            ax.hlines(yv, row.eval_start, row.eval_end, color="#EA580C", linewidth=4)
        ax.set_title(f"BTC Walk-Forward Splits ({title})")
        ax.set_ylabel("Fold")
        ax.grid(axis="x", alpha=0.2)
    axes[1].set_xlabel("Date")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _save_and_record(
        records,
        fig,
        out_dir / "fig_v2_05_btc_walkforward_timeline.png",
        "fig_v2_05",
        "BTC walk-forward train/eval windows for baseline and transformer blocks",
        "4. Model Training and Evaluation",
        "Are the rolling evaluation windows strictly chronological and comparable across model families?",
        "multi_asset_baselines_h1_paired_summary_btc_no_sent_fold_manifest.csv;"
        "multi_asset_transformers_h1_paired_summary_btc_no_sent_fold_manifest.csv",
    )

    # FIG 06: Eval window size distribution by family
    fold_frames = []
    for p in sorted(root.glob("multi_asset_baselines_h1_paired_summary_*_no_sent_fold_manifest.csv")):
        f = pd.read_csv(p)
        f["family"] = "baselines"
        fold_frames.append(f)
    for p in sorted(root.glob("multi_asset_transformers_h1_paired_summary_*_no_sent_fold_manifest.csv")):
        f = pd.read_csv(p)
        f["family"] = "transformers"
        fold_frames.append(f)
    for p in sorted(root.glob(f"{foundation_prefix}_*_fold_manifest.csv")):
        f = pd.read_csv(p)
        f["family"] = "foundation"
        fold_frames.append(f)
    fold_df = pd.concat(fold_frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    fams = ["baselines", "transformers", "foundation"]
    vals = [fold_df.loc[fold_df["family"] == fam, "eval_size"].to_numpy(dtype=float) for fam in fams]
    ax.boxplot(vals, labels=fams, showmeans=True)
    ax.set_title("Evaluation Window Size Distribution by Family")
    ax.set_ylabel("Eval window size")
    ax.grid(axis="y", alpha=0.2)
    _save_and_record(
        records,
        fig,
        out_dir / "fig_v2_06_eval_window_distribution.png",
        "fig_v2_06",
        "Evaluation window-size distribution by model family",
        "4. Model Training and Evaluation",
        "Do walk-forward folds use comparable out-of-sample window sizes across families?",
        "*_fold_manifest.csv",
    )

    # FIG 07: Purged/CPCV schematic (conceptual)
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.set_yticks([])
    ax.set_xlabel("Time index")
    ax.set_title("Purged/Embargoed Validation Schematic")
    segments = [
        (0, 48, "train", "#2563EB"),
        (48, 56, "purge", "#DC2626"),
        (56, 68, "validation", "#EA580C"),
        (68, 74, "embargo", "#F59E0B"),
        (74, 100, "future", "#94A3B8"),
    ]
    for start, end, label, color in segments:
        ax.add_patch(patches.Rectangle((start, 3), end - start, 4, facecolor=color, alpha=0.85))
        ax.text((start + end) / 2, 5, label, ha="center", va="center", color="white", fontsize=10, weight="bold")
    ax.grid(axis="x", alpha=0.15)
    _save_and_record(
        records,
        fig,
        out_dir / "fig_v2_07_purged_cv_schematic.png",
        "fig_v2_07",
        "Purged/embargoed validation schematic used for leakage control",
        "4. Model Training and Evaluation",
        "How are temporal leakage risks handled during model selection?",
        "conceptual_protocol",
    )

    # FIG 08: stationarity test heatmap on -log10(p)
    st = stationarity_tests.copy()
    st = st[st["series"] == "return_1d"].copy()
    st["asset"] = st["asset"].str.upper()
    pivot = st.pivot_table(index="asset", columns="test", values="p_value", aggfunc="mean").sort_index()
    pivot = pivot.reindex(columns=["adf", "kpss"])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    pv = np.clip(pivot.to_numpy(dtype=float), 1e-30, 1.0)
    z = -np.log10(pv)
    im = ax.imshow(z, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("ADF/KPSS Significance on Daily Returns (-log10 p-value)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if not pd.notna(val):
                txt = "-"
            elif val < 1e-16:
                txt = "<1e-16"
            elif val >= 0.0999:
                txt = ">=1e-1"
            else:
                txt = f"{val:.1e}"
            ax.text(j, i, txt, ha="center", va="center", color="white", fontsize=8)
    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("-log10(p-value)")
    _save_and_record(
        records,
        fig,
        out_dir / "fig_v2_08_stationarity_tests.png",
        "fig_v2_08",
        "Return-series stationarity test p-values across assets",
        "3.3 Stationarity Analysis",
        "Do transformed return series satisfy practical stationarity diagnostics?",
        "appendix/stationarity/*_stationarity_tests.csv",
    )

    # FIG 09: stationarity moments
    sm = stationarity_summaries.copy().sort_values("asset")
    sm["asset"] = sm["asset"].str.upper()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    metrics = [
        ("return_mean", "Return Mean"),
        ("return_std", "Return Std"),
        ("return_skew", "Return Skewness"),
        ("return_kurtosis", "Return Kurtosis"),
    ]
    for ax, (col, title) in zip(axes.ravel(), metrics):
        ax.bar(sm["asset"], sm[col], color="#475569")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    _save_and_record(
        records,
        fig,
        out_dir / "fig_v2_09_stationarity_moments.png",
        "fig_v2_09",
        "Return distribution moments used in stationarity diagnostics",
        "3.3 Stationarity Analysis",
        "How do distributional properties differ by asset after return transformation?",
        "appendix/stationarity/*_stationarity_summary.csv",
    )

    # FIG 10-18: copy appendix diagnostics (BTC/ETH/XMR)
    for asset, label, base_id in [
        ("btc", "BTC", 10),
        ("eth", "ETH", 13),
        ("xmr", "XMR", 16),
    ]:
        _copy_and_record(
            records,
            root / "appendix" / "stationarity" / f"{asset}_rolling_diagnostics.png",
            out_dir / f"fig_v2_{base_id:02d}_{asset}_rolling_diagnostics.png",
            f"fig_v2_{base_id:02d}",
            f"{label} rolling mean/variance diagnostics",
            "3.3 Stationarity Analysis",
            f"How stable are local moments over time for {label} returns?",
            f"appendix/stationarity/{asset}_rolling_diagnostics.png",
        )
        _copy_and_record(
            records,
            root / "appendix" / "stationarity" / f"{asset}_fft.png",
            out_dir / f"fig_v2_{base_id+1:02d}_{asset}_fft.png",
            f"fig_v2_{base_id+1:02d}",
            f"{label} Fourier spectrum",
            "3.4 Feature Engineering",
            f"Which dominant periodic components appear in {label} returns?",
            f"appendix/stationarity/{asset}_fft.png",
        )
        _copy_and_record(
            records,
            root / "appendix" / "stationarity" / f"{asset}_acf_pacf.png",
            out_dir / f"fig_v2_{base_id+2:02d}_{asset}_acf_pacf.png",
            f"fig_v2_{base_id+2:02d}",
            f"{label} ACF/PACF diagnostics",
            "3.4 Feature Engineering",
            f"What lag structure is visible in {label} return autocorrelation?",
            f"appendix/stationarity/{asset}_acf_pacf.png",
        )

    # FIG 19: median training curves
    h = trf_history.copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, mode in zip(axes, ["no_sentiment", "with_sentiment"]):
        sub = h[h["mode"] == mode].copy()
        grouped = (
            sub.groupby(["model", "epoch"], as_index=False)
            .agg(train_loss=("train_loss", "median"), val_loss=("val_loss", "median"))
            .sort_values(["model", "epoch"])
        )
        for model in sorted(grouped["model"].unique()):
            ms = grouped[grouped["model"] == model]
            ax.plot(ms["epoch"], ms["val_loss"], marker="o", linewidth=1.3, label=f"{model} (val)")
        ax.set_title(f"Median Val Loss by Epoch ({mode})")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=7, ncol=2)
    _save_and_record(
        records,
        fig,
        out_dir / "fig_v2_19_training_curves_median.png",
        "fig_v2_19",
        "Median transformer validation-loss curves by epoch",
        "4.2 Transformer Models",
        "How do optimization dynamics differ across transformer variants and sentiment modes?",
        "multi_asset_transformers_h1_paired_summary_history.csv",
    )

    # FIG 20: overfit example
    h2 = trf_history.copy()
    agg = h2.groupby(["asset", "mode", "model", "fold"], as_index=False).agg(
        min_val=("val_loss", "min"),
        last_val=("val_loss", "last"),
    )
    agg["overfit_gap"] = agg["last_val"] - agg["min_val"]
    pick = agg.sort_values("overfit_gap", ascending=False).iloc[0]
    ex = h2[
        (h2["asset"] == pick["asset"])
        & (h2["mode"] == pick["mode"])
        & (h2["model"] == pick["model"])
        & (h2["fold"] == pick["fold"])
    ].sort_values("epoch")
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    ax.plot(ex["epoch"], ex["train_loss"], marker="o", linewidth=1.6, label="train_loss", color="#2563EB")
    ax.plot(ex["epoch"], ex["val_loss"], marker="o", linewidth=1.6, label="val_loss", color="#DC2626")
    ax.set_title(
        f"Overfit Example: {pick['asset'].upper()} | {pick['model']} | {pick['mode']} | fold {int(pick['fold'])}"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.2)
    ax.legend()
    _save_and_record(
        records,
        fig,
        out_dir / "fig_v2_20_overfit_example.png",
        "fig_v2_20",
        "Representative overfit pattern in transformer training",
        "4.2 Transformer Models",
        "Where does validation loss diverge from train loss under the current training budget?",
        "multi_asset_transformers_h1_paired_summary_history.csv",
    )

    # FIG 21: final-epoch val loss distribution
    h3 = trf_history.copy()
    last = h3.sort_values("epoch").groupby(["asset", "mode", "model", "fold"], as_index=False).tail(1)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    order = sorted(last["model"].unique())
    vals = [last.loc[last["model"] == m, "val_loss"].to_numpy(dtype=float) for m in order]
    ax.boxplot(vals, labels=order, showmeans=True)
    ax.set_title("Final-Epoch Validation Loss Distribution")
    ax.set_ylabel("Val loss at final epoch")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2)
    _save_and_record(
        records,
        fig,
        out_dir / "fig_v2_21_final_val_distribution.png",
        "fig_v2_21",
        "Final-epoch validation loss distribution by transformer model",
        "4.2 Transformer Models",
        "Which models end training with consistently lower validation losses?",
        "multi_asset_transformers_h1_paired_summary_history.csv",
    )

    # FIG 22-24: prediction traces for BTC/ETH/XMR
    b_best = _choose_best_available_model(
        base_summary,
        pred_all,
        "btc",
        winner_policy=winner_policy,
        winner_exclude_models=winner_excluded_models,
        winner_metrics=DEFAULT_FIGURE_WINNER_METRICS,
    )
    t_best = _choose_best_available_model(
        trf_summary,
        pred_all,
        "btc",
        winner_policy=winner_policy,
        winner_exclude_models=winner_excluded_models,
        winner_metrics=DEFAULT_FIGURE_WINNER_METRICS,
    )
    f_best = _choose_best_available_model(
        chrn_summary,
        pred_all,
        "btc",
        winner_policy=winner_policy,
        winner_exclude_models=winner_excluded_models,
        winner_metrics=DEFAULT_FIGURE_WINNER_METRICS,
    )
    _prediction_trace(
        pred_all,
        "btc",
        [("baseline", b_best), ("transformer", t_best), ("foundation", f_best)],
        out_dir / "fig_v2_22_trace_btc.png",
        "BTC Return Forecast Trace (Best Baseline vs Best Transformer vs Best Foundation)",
        records,
        "fig_v2_22",
        foundation_predictions_source,
    )

    b_best = _choose_best_available_model(
        base_summary,
        pred_all,
        "eth",
        winner_policy=winner_policy,
        winner_exclude_models=winner_excluded_models,
        winner_metrics=DEFAULT_FIGURE_WINNER_METRICS,
    )
    t_best = _choose_best_available_model(
        trf_summary,
        pred_all,
        "eth",
        winner_policy=winner_policy,
        winner_exclude_models=winner_excluded_models,
        winner_metrics=DEFAULT_FIGURE_WINNER_METRICS,
    )
    f_best = _choose_best_available_model(
        chrn_summary,
        pred_all,
        "eth",
        winner_policy=winner_policy,
        winner_exclude_models=winner_excluded_models,
        winner_metrics=DEFAULT_FIGURE_WINNER_METRICS,
    )
    _prediction_trace(
        pred_all,
        "eth",
        [("baseline", b_best), ("transformer", t_best), ("foundation", f_best)],
        out_dir / "fig_v2_23_trace_eth.png",
        "ETH Return Forecast Trace (Best Baseline vs Best Transformer vs Best Foundation)",
        records,
        "fig_v2_23",
        foundation_predictions_source,
    )

    b_best = _choose_best_available_model(
        base_summary,
        pred_all,
        "xmr",
        winner_policy=winner_policy,
        winner_exclude_models=winner_excluded_models,
        winner_metrics=DEFAULT_FIGURE_WINNER_METRICS,
    )
    t_best = _choose_best_available_model(
        trf_summary,
        pred_all,
        "xmr",
        winner_policy=winner_policy,
        winner_exclude_models=winner_excluded_models,
        winner_metrics=DEFAULT_FIGURE_WINNER_METRICS,
    )
    f_best = _choose_best_available_model(
        chrn_summary,
        pred_all,
        "xmr",
        winner_policy=winner_policy,
        winner_exclude_models=winner_excluded_models,
        winner_metrics=DEFAULT_FIGURE_WINNER_METRICS,
    )
    _prediction_trace(
        pred_all,
        "xmr",
        [("baseline", b_best), ("transformer", t_best), ("foundation", f_best)],
        out_dir / "fig_v2_24_trace_xmr.png",
        "XMR Return Forecast Trace (Best Baseline vs Best Transformer vs Best Foundation)",
        records,
        "fig_v2_24",
        foundation_predictions_source,
    )

    # FIG 25: residual distributions (no sentiment)
    resid_df = pred_all[pred_all["mode"] == "no_sentiment"].copy()
    f_best_df = chrn_summary[chrn_summary["mode"] == "no_sentiment"].copy()
    if "objective_track" in f_best_df.columns:
        f_best_df = f_best_df[f_best_df["objective_track"].fillna("point") == "point"]
    foundation_best_global = f_best_df.sort_values("mae_mean", ascending=True).iloc[0]["model"]
    keep_models = ["linear_ridge", "dlinear_like", str(foundation_best_global)]
    resid_df = resid_df[resid_df["model"].isin(keep_models)].copy()
    resid_df["resid"] = resid_df["y_true"] - resid_df["y_pred"]
    fig, ax = plt.subplots(figsize=(10.2, 4.4))
    bins = np.linspace(-0.5, 0.5, 140)
    for model in keep_models:
        vals = resid_df.loc[resid_df["model"] == model, "resid"].to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=bins, density=True, histtype="step", linewidth=1.5, label=model, color=_family_color(model))
    ax.set_title("Residual Distribution Comparison (No Sentiment)")
    ax.set_xlabel("Residual (y_true - y_pred)")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.2)
    ax.legend()
    _save_and_record(
        records,
        fig,
        out_dir / "fig_v2_25_residual_distribution.png",
        "fig_v2_25",
        "Residual distribution comparison across key model families",
        "5. Results and Findings",
        "Which model family produces tighter residual distributions on the shared target scale?",
        "multi_asset_*_predictions.csv",
    )

    # FIG 26: abs error vs interval width
    q = quant_all[quant_all["mode"] == "no_sentiment"].copy()
    q = q[q["model"].isin(["linear_ridge", "dlinear_like", str(foundation_best_global)])]
    q["abs_error"] = (q["y_true"] - q["q50"]).abs()
    q["interval_width"] = q["q90"] - q["q10"]
    if len(q) > 8000:
        q = q.sample(n=8000, random_state=42)
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    for model in sorted(q["model"].unique()):
        sub = q[q["model"] == model]
        ax.scatter(
            sub["interval_width"],
            sub["abs_error"],
            s=6,
            alpha=0.25,
            label=model,
            color=_family_color(model),
        )
    ax.set_title("Absolute Error vs Predicted Interval Width")
    ax.set_xlabel("Predicted 80% interval width (q90 - q10)")
    ax.set_ylabel("Absolute error |y - q50|")
    ax.grid(alpha=0.2)
    ax.legend(markerscale=2, fontsize=8)
    _save_and_record(
        records,
        fig,
        out_dir / "fig_v2_26_error_vs_interval_width.png",
        "fig_v2_26",
        "Relationship between uncertainty width and realized absolute error",
        "5. Results and Findings",
        "Do wider prediction intervals correspond to larger realized errors?",
        "multi_asset_*_quantiles.csv",
    )

    # FIG 27: best model by asset
    bb = best_by_asset.copy().sort_values("asset")
    bb["asset"] = bb["asset"].str.upper()
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    colors = ["#2563EB" if f == "baselines" else "#7C3AED" if f == "transformers" else "#059669" for f in bb["family"]]
    bars = ax.bar(bb["asset"], bb["mae_mean"], color=colors)
    for b, m in zip(bars, bb["model"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), m, ha="center", va="bottom", rotation=90, fontsize=7)
    ax.set_title("Best Model per Asset (Composite Winner Policy)")
    ax.set_ylabel("MAE")
    ax.grid(axis="y", alpha=0.2)
    _save_and_record(
        records,
        fig,
        out_dir / "fig_v2_27_best_model_per_asset.png",
        "fig_v2_27",
        "Best model per asset under composite winner policy",
        "5.4 Final Comparative Results",
        "Which family wins per asset under composite ranking while retaining naive models as references?",
        "PAPER_H1_best_by_asset.csv",
    )

    # FIG 28: family-level performance (avg MAE and directional acc)
    fp = family_perf.copy().sort_values("avg_mae_mean", ascending=True)
    fp["label"] = fp["family"] + "\n" + fp["model"] + "\n" + fp["mode"]
    x = np.arange(len(fp))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    axes[0].bar(x, fp["avg_mae_mean"], color=[_family_color(m) for m in fp["model"]])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(fp["label"], rotation=65, ha="right", fontsize=7)
    axes[0].set_title("Average MAE by Family/Model/Mode")
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(x, fp["avg_directional_accuracy_mean"], color=[_family_color(m) for m in fp["model"]])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(fp["label"], rotation=65, ha="right", fontsize=7)
    axes[1].set_title("Average Directional Accuracy by Family/Model/Mode")
    axes[1].grid(axis="y", alpha=0.2)
    _save_and_record(
        records,
        fig,
        out_dir / "fig_v2_28_family_performance.png",
        "fig_v2_28",
        "Family/model/mode aggregate performance comparison",
        "5.4 Final Comparative Results",
        "How do ranking patterns change across error and directional metrics?",
        "PAPER_H1_family_performance.csv",
    )

    # FIG 29: sentiment delta heatmap (MAE)
    sd = sent_delta.copy()
    pivot = (
        sd.groupby(["asset", "family"], as_index=False)["delta_mae_mean_with_minus_no"]
        .mean()
        .pivot(index="asset", columns="family", values="delta_mae_mean_with_minus_no")
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="coolwarm", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([a.upper() for a in pivot.index])
    ax.set_title("Sentiment Effect Heatmap: Delta MAE (With - No)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            ax.text(j, i, f"{v:+.4f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save_and_record(
        records,
        fig,
        out_dir / "fig_v2_29_sentiment_delta_heatmap.png",
        "fig_v2_29",
        "Asset-by-family sentiment effect on MAE",
        "5.2 Financial + Sentiment Analysis",
        "Where does sentiment help or hurt forecast error most strongly?",
        "PAPER_H1_sentiment_delta.csv",
    )

    # FIG 30: cost sensitivity (Sharpe)
    bs = best_by_asset.copy()
    bs["asset"] = bs["asset"].str.upper()
    cost_cols = [c for c in ["sharpe_0bps_mean", "sharpe_5bps_mean", "sharpe_10bps_mean", "sharpe_20bps_mean", "sharpe_50bps_mean"] if c in bs.columns]
    long = bs[["asset", *cost_cols]].melt(id_vars=["asset"], var_name="cost", value_name="sharpe")
    long["cost_bps"] = long["cost"].str.extract(r"sharpe_(\d+)bps_mean")[0].astype(int)
    fig, ax = plt.subplots(figsize=(10.2, 4.5))
    for asset in sorted(long["asset"].unique()):
        sub = long[long["asset"] == asset].sort_values("cost_bps")
        ax.plot(sub["cost_bps"], sub["sharpe"], marker="o", linewidth=1.4, label=asset)
    ax.set_title("Cost Sensitivity of Best-Per-Asset Models")
    ax.set_xlabel("Transaction cost (bps)")
    ax.set_ylabel("Sharpe")
    ax.grid(alpha=0.2)
    ax.legend(ncol=4, fontsize=7)
    _save_and_record(
        records,
        fig,
        out_dir / "fig_v2_30_cost_sensitivity.png",
        "fig_v2_30",
        "Sharpe sensitivity under increasing transaction costs",
        "5.4 Final Comparative Results",
        "Are top models robust once realistic cost frictions are applied?",
        "PAPER_H1_best_by_asset.csv",
    )

    # FIG 35: ablation deltas versus financial-only mode
    ad = ablation_delta.copy()
    ad = ad[ad["mode"] != "financial_only"].copy()
    mode_agg = ad.groupby("mode", as_index=False).agg(
        avg_delta_mae=("delta_mae_vs_financial_only", "mean"),
        avg_delta_da=("delta_directional_accuracy_vs_financial_only", "mean"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
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
    _save_and_record(
        records,
        fig,
        out_dir / "fig_v2_35_ablation_mode_deltas.png",
        "fig_v2_35",
        "Feature-ablation deltas versus financial-only mode",
        "5.4 Final Comparative Results",
        "Does adding sentiment and/or regime features improve CPCV results against financial-only inputs?",
        "feature_ablation_h1_summary_mode_deltas.csv",
    )

    # FIG 36: ablation mode wins
    aw = ablation_wins.copy().sort_values("n_assets_won", ascending=False)
    fig, ax1 = plt.subplots(figsize=(8.0, 4.2))
    x = np.arange(len(aw))
    ax1.bar(x, aw["n_assets_won"], color="#0EA5A4")
    ax1.set_ylabel("Assets won")
    ax1.set_xticks(x)
    ax1.set_xticklabels(aw["mode"], rotation=20, ha="right")
    ax1.set_title("Ablation Mode Wins (CPCV)")
    ax1.grid(axis="y", alpha=0.2)
    ax2 = ax1.twinx()
    ax2.plot(x, aw["avg_winning_mae"], color="#7C3AED", marker="o", linewidth=2)
    ax2.set_ylabel("Average winning MAE")
    _save_and_record(
        records,
        fig,
        out_dir / "fig_v2_36_ablation_mode_wins.png",
        "fig_v2_36",
        "Ablation winner counts and winning MAE by mode",
        "5.4 Final Comparative Results",
        "Which feature-mode configuration wins most often across assets under CPCV?",
        "PAPER_H1_ablation_mode_wins.csv",
    )

    # FIG 31: DM p-value heatmap
    dm_rows = []
    for family_name, df in [
        ("baselines", base_summary),
        ("transformers", trf_summary),
        ("foundation", chrn_summary),
    ]:
        sub = df.copy()
        if "mode" in sub.columns:
            sub = sub[sub["mode"] == "no_sentiment"]
        if "objective_track" in sub.columns:
            sub = sub[sub["objective_track"].fillna("point") == "point"]
        for row in sub.itertuples(index=False):
            dm_rows.append(
                {
                    "asset": row.asset,
                    "family": family_name,
                    "model": row.model,
                    "p_value": getattr(row, "dm_vs_rw_pvalue", np.nan),
                    "stat": getattr(row, "dm_vs_rw_stat", np.nan),
                }
            )
    dm_df = pd.DataFrame(dm_rows)
    dm_df = dm_df.dropna(subset=["p_value"])
    dm_df["label"] = dm_df["family"] + ":" + dm_df["model"]
    pivot = dm_df.pivot_table(index="asset", columns="label", values="p_value", aggfunc="mean").sort_index()
    val = -np.log10(np.clip(pivot.to_numpy(dtype=float), 1e-12, 1.0))
    fig, ax = plt.subplots(figsize=(14, 4.8))
    im = ax.imshow(val, cmap="magma", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=70, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([a.upper() for a in pivot.index])
    ax.set_title("Diebold-Mariano Significance vs Random Walk (-log10 p-value)")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    _save_and_record(
        records,
        fig,
        out_dir / "fig_v2_31_dm_significance_heatmap.png",
        "fig_v2_31",
        "DM significance map against random-walk benchmark",
        "5.4 Final Comparative Results",
        "Which models show statistically significant forecast gains over random walk?",
        "multi_asset_*_h1_*_summary.csv",
    )

    # FIG 32: risk-return scatter
    rr_rows = []
    for family_name, df in [("baselines", base_summary), ("transformers", trf_summary), ("foundation", chrn_summary)]:
        sub = df.copy()
        if "mode" in sub.columns:
            sub = sub[sub["mode"] == "no_sentiment"]
        if "objective_track" in sub.columns:
            sub = sub[sub["objective_track"].fillna("point") == "point"]
        rr = (
            sub.groupby("model", as_index=False)
            .agg(
                sharpe=("sharpe_5bps_mean", "mean"),
                max_drawdown=("max_drawdown_5bps_mean", "mean"),
                mae=("mae_mean", "mean"),
            )
            .assign(family=family_name)
        )
        rr_rows.append(rr)
    rr_df = pd.concat(rr_rows, ignore_index=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for row in rr_df.itertuples(index=False):
        ax.scatter(
            row.max_drawdown,
            row.sharpe,
            s=max(25, 400 * (1.0 / max(row.mae, 1e-4))),
            color=_family_color(row.model),
            alpha=0.75,
            edgecolor="black",
            linewidth=0.3,
        )
        ax.text(row.max_drawdown, row.sharpe, row.model, fontsize=7)
    ax.set_xlabel("Average max drawdown (5bps)")
    ax.set_ylabel("Average Sharpe (5bps)")
    ax.set_title("Risk-Return Frontier by Model (Bubble size ~ 1/MAE)")
    ax.grid(alpha=0.2)
    _save_and_record(
        records,
        fig,
        out_dir / "fig_v2_32_risk_return_scatter.png",
        "fig_v2_32",
        "Risk-return trade-off across model families",
        "5.4 Final Comparative Results",
        "Which models offer favorable Sharpe-drawdown trade-offs while keeping errors low?",
        "multi_asset_*_h1_*_summary.csv",
    )

    # FIG 33: transfer metrics comparison
    metric_cols = ["mae", "rmse", "directional_accuracy", "sharpe_5bps", "dsr"]
    a = transfer_no_sent[["mode", *metric_cols]].copy()
    a["setup"] = "no_sentiment"
    b = transfer_with_sent[["mode", *metric_cols]].copy()
    b["setup"] = "with_sentiment"
    tdf = pd.concat([a, b], ignore_index=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.0))
    axes = axes.ravel()
    for ax, (col, title) in zip(
        axes,
        [("mae", "MAE"), ("rmse", "RMSE"), ("directional_accuracy", "Directional Accuracy"), ("dsr", "Deflated Sharpe")],
    ):
        for setup, color in [("no_sentiment", "#2563EB"), ("with_sentiment", "#DC2626")]:
            sub = tdf[tdf["setup"] == setup]
            ax.plot(sub["mode"], sub[col], marker="o", linewidth=2, label=setup, color=color)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.2)
    axes[0].legend()
    _save_and_record(
        records,
        fig,
        out_dir / "fig_v2_33_transfer_metrics.png",
        "fig_v2_33",
        "BTC→ETH transfer metric comparison (no-sentiment vs with-sentiment)",
        "5.3 Transfer Learning Analysis",
        "Does transfer remain competitive against target-only training under both feature setups?",
        "transfer_btc_eth_patchtst_no_sent.csv;transfer_btc_eth_patchtst_with_sent.csv",
    )

    # FIG 34: transfer prediction traces
    def _prep_transfer(df: pd.DataFrame, n_tail: int = 180) -> pd.DataFrame:
        out = df.copy().tail(n_tail)
        out["time"] = pd.to_datetime(out["time"], errors="coerce")
        return out

    t0 = _prep_transfer(transfer_no_sent_pred)
    t1 = _prep_transfer(transfer_with_sent_pred)
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.2), sharex=True)
    axes[0].plot(t0["time"], t0["y_true"], color="black", linewidth=1.6, label="y_true")
    axes[0].plot(t0["time"], t0["pred_source_zero_shot"], color="#2563EB", linewidth=1.2, label="source_zero_shot")
    axes[0].plot(t0["time"], t0["pred_source_finetuned"], color="#EA580C", linewidth=1.2, label="source_finetuned")
    axes[0].plot(t0["time"], t0["pred_target_only"], color="#16A34A", linewidth=1.2, label="target_only")
    axes[0].set_title("BTC→ETH Transfer Traces (No Sentiment)")
    axes[0].grid(alpha=0.2)
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].plot(t1["time"], t1["y_true"], color="black", linewidth=1.6, label="y_true")
    axes[1].plot(t1["time"], t1["pred_source_zero_shot"], color="#DC2626", linewidth=1.2, label="source_zero_shot")
    axes[1].plot(t1["time"], t1["pred_source_finetuned"], color="#7C3AED", linewidth=1.2, label="source_finetuned")
    axes[1].plot(t1["time"], t1["pred_target_only"], color="#16A34A", linewidth=1.2, label="target_only")
    axes[1].set_title("BTC→ETH Transfer Traces (With Sentiment)")
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="upper left", fontsize=8)

    _save_and_record(
        records,
        fig,
        out_dir / "fig_v2_34_transfer_prediction_traces.png",
        "fig_v2_34",
        "BTC→ETH transfer prediction traces under both feature setups",
        "5.3 Transfer Learning Analysis",
        "How do transfer variants track short-term ETH return movements in practice?",
        "transfer_btc_eth_patchtst_no_sent_predictions.csv;transfer_btc_eth_patchtst_with_sent_predictions.csv",
    )

    sent_diag_root = root / "sentiment_diagnostics"
    trf_diag_root = root / "transformer_diagnostics"

    # FIG 37: sentiment correlation heatmap by lag
    corr_path = sent_diag_root / "sentiment_correlation_by_lag.csv"
    if corr_path.exists():
        corr = pd.read_csv(corr_path)
        if "asset" in corr.columns:
            corr = corr[corr["asset"].astype(str).isin(benchmark_assets)].copy()
        if not corr.empty and {"feature", "lag", "pearson_r"}.issubset(corr.columns):
            heat = (
                corr.groupby(["feature", "lag"], as_index=False)["pearson_r"]
                .mean()
                .pivot(index="feature", columns="lag", values="pearson_r")
                .sort_index()
            )
            if not heat.empty:
                top_features = heat.abs().max(axis=1).sort_values(ascending=False).head(30).index
                heat = heat.loc[top_features]
                vmax = float(np.nanmax(np.abs(heat.to_numpy(dtype=float)))) if np.isfinite(heat.to_numpy(dtype=float)).any() else 0.1
                vmax = max(vmax, 0.05)
                fig, ax = plt.subplots(figsize=(9.5, 8.0))
                im = ax.imshow(heat.to_numpy(dtype=float), cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
                ax.set_xticks(np.arange(len(heat.columns)))
                ax.set_xticklabels([str(c) for c in heat.columns])
                ax.set_yticks(np.arange(len(heat.index)))
                ax.set_yticklabels(heat.index, fontsize=7)
                ax.set_xlabel("Lag")
                ax.set_title("Sentiment-Target Correlation Heatmap by Lag")
                plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
                _save_and_record(
                    records,
                    fig,
                    out_dir / "fig_v2_37_sentiment_corr_heatmap_by_lag.png",
                    "fig_v2_37",
                    "Sentiment-target correlation heatmap by lag",
                    "5.2 Financial + Sentiment Analysis",
                    "Are sentiment features near-zero correlated with target returns across lags?",
                    "sentiment_diagnostics/sentiment_correlation_by_lag.csv",
                )

    # FIG 38: dimensionality profile bar chart
    dim_path = sent_diag_root / "dimensionality_profile.csv"
    if dim_path.exists():
        dim = pd.read_csv(dim_path)
        if "asset" in dim.columns:
            dim = dim[dim["asset"].astype(str).isin(benchmark_assets)].copy()
        if not dim.empty and {"mode", "n_features", "p_over_n"}.issubset(dim.columns):
            d = dim.groupby("mode", as_index=False).agg(n_features=("n_features", "mean"), p_over_n=("p_over_n", "mean"))
            d = d.sort_values("n_features", ascending=True)
            fig, ax1 = plt.subplots(figsize=(10.0, 4.5))
            x = np.arange(len(d))
            ax1.bar(x, d["n_features"], color="#0F766E")
            ax1.set_xticks(x)
            ax1.set_xticklabels(d["mode"], rotation=20, ha="right")
            ax1.set_ylabel("Number of features")
            ax1.grid(axis="y", alpha=0.2)
            ax2 = ax1.twinx()
            ax2.plot(x, d["p_over_n"], color="#DC2626", marker="o", linewidth=2.0)
            ax2.set_ylabel("p / n")
            ax1.set_title("Feature Dimensionality Profile and p/n Ratios")
            _save_and_record(
                records,
                fig,
                out_dir / "fig_v2_38_dimensionality_profile.png",
                "fig_v2_38",
                "Dimensionality profile and p/n ratios by feature mode",
                "5.2 Financial + Sentiment Analysis",
                "How does sentiment expansion change p/n ratios versus financial-only mode?",
                "sentiment_diagnostics/dimensionality_profile.csv",
            )

    # FIG 39: PCA sentiment ablation line plot
    pca_path = sent_diag_root / "pca_sentiment_ablation.csv"
    if pca_path.exists():
        pca_df = pd.read_csv(pca_path)
        if "asset" in pca_df.columns:
            pca_df = pca_df[pca_df["asset"].astype(str).isin(benchmark_assets)].copy()
        if not pca_df.empty and {"mode", "n_pca_components", "mae"}.issubset(pca_df.columns):
            fig, ax = plt.subplots(figsize=(9.2, 4.8))
            pca_mode = pca_df[pca_df["mode"] == "sentiment_pca"].copy()
            if not pca_mode.empty:
                agg = (
                    pca_mode.groupby("n_pca_components")["mae"]
                    .agg(
                        median_mae="median",
                        q25=lambda s: float(np.quantile(s, 0.25)),
                        q75=lambda s: float(np.quantile(s, 0.75)),
                    )
                    .reset_index()
                    .sort_values("n_pca_components")
                )
                ax.plot(agg["n_pca_components"], agg["median_mae"], marker="o", linewidth=2, color="#1D4ED8", label="PCA sentiment (median)")
                ax.fill_between(agg["n_pca_components"], agg["q25"], agg["q75"], color="#1D4ED8", alpha=0.15, label="PCA sentiment IQR")
            x_min = float(pca_mode["n_pca_components"].min()) if not pca_mode.empty else 0.0
            x_max = float(pca_mode["n_pca_components"].max()) if not pca_mode.empty else 1.0
            for mode_name, color in [("no_sentiment", "#059669"), ("full_sentiment", "#DC2626")]:
                sub = pca_df[pca_df["mode"] == mode_name]
                if not sub.empty:
                    q25 = float(sub["mae"].quantile(0.25))
                    q50 = float(sub["mae"].quantile(0.50))
                    q75 = float(sub["mae"].quantile(0.75))
                    ax.axhline(q50, color=color, linestyle="--", linewidth=1.6, label=f"{mode_name} (median)")
                    ax.fill_between([x_min, x_max], q25, q75, color=color, alpha=0.08)
            ax.set_xlabel("Sentiment PCA components")
            ax.set_ylabel("MAE")
            ax.set_title("Sentiment PCA Ablation (Ridge, Asset-Median with IQR)")
            ax.grid(alpha=0.2)
            ax.legend()
            _save_and_record(
                records,
                fig,
                out_dir / "fig_v2_39_pca_sentiment_ablation.png",
                "fig_v2_39",
                "PCA compression ablation for sentiment block",
                "5.2 Financial + Sentiment Analysis",
                "Does compressing sentiment dimensions recover forecast quality?",
                "sentiment_diagnostics/pca_sentiment_ablation.csv",
            )

    # FIG 40: residual ACF panel
    residual_path = trf_diag_root / "residual_acf_analysis.csv"
    if residual_path.exists():
        resid = pd.read_csv(residual_path)
        if "asset" in resid.columns:
            resid = resid[resid["asset"].astype(str).isin(benchmark_assets)].copy()
        if not resid.empty and {"mode", "lag", "acf"}.issubset(resid.columns):
            resid = resid[resid["lag"] > 0].copy()
            if not resid.empty:
                fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharex=True)
                for mode in sorted(resid["mode"].dropna().unique().tolist()):
                    sub = resid[resid["mode"] == mode]
                    line = (
                        sub.groupby("lag")["acf"]
                        .agg(
                            median_acf="median",
                            q25=lambda s: float(np.quantile(s, 0.25)),
                            q75=lambda s: float(np.quantile(s, 0.75)),
                        )
                        .reset_index()
                        .sort_values("lag")
                    )
                    axes[0].plot(line["lag"], line["median_acf"], marker="o", linewidth=1.6, label=str(mode))
                    axes[0].fill_between(line["lag"], line["q25"], line["q75"], alpha=0.15)

                    if {"acf_ci_upper", "acf_ci_lower"}.issubset(sub.columns):
                        signif = (
                            sub.assign(signif=((sub["acf_ci_lower"] > 0.0) | (sub["acf_ci_upper"] < 0.0)).astype(float))
                            .groupby("lag", as_index=False)["signif"]
                            .mean()
                            .sort_values("lag")
                        )
                        axes[1].plot(signif["lag"], signif["signif"], marker="o", linewidth=1.6, label=str(mode))

                axes[0].axhline(0.0, color="black", linewidth=1)
                axes[0].set_xlabel("Lag")
                axes[0].set_ylabel("Residual ACF")
                axes[0].set_title("Residual ACF (Median with IQR)")
                axes[0].grid(alpha=0.2)
                axes[0].legend()

                axes[1].set_xlabel("Lag")
                axes[1].set_ylabel("Share Significant")
                axes[1].set_ylim(-0.02, 1.02)
                axes[1].set_title("Share of Significant Residual ACF")
                axes[1].grid(alpha=0.2)
                axes[1].legend()
                _save_and_record(
                    records,
                    fig,
                    out_dir / "fig_v2_40_residual_acf_panel.png",
                    "fig_v2_40",
                    "Residual autocorrelation panel for transformer predictions",
                    "5.2 Financial + Sentiment Analysis",
                    "Do transformer residuals show meaningful leftover serial structure?",
                    "transformer_diagnostics/residual_acf_analysis.csv",
                )

    # FIG 41: return predictability panel
    return_path = trf_diag_root / "return_predictability.csv"
    if return_path.exists():
        rp = pd.read_csv(return_path)
        if "asset" in rp.columns:
            rp = rp[rp["asset"].astype(str).isin(benchmark_assets)].copy()
        if not rp.empty and {"lag", "acf", "vr_q2", "vr_q5", "vr_q10", "vr_q20"}.issubset(rp.columns):
            acf_agg = (
                rp.groupby("lag")["acf"]
                .agg(
                    mean_acf="mean",
                    q25=lambda s: float(np.quantile(s, 0.25)),
                    q75=lambda s: float(np.quantile(s, 0.75)),
                )
                .reset_index()
                .sort_values("lag")
            )
            vr_asset = rp.groupby("asset", as_index=False)[["vr_q2", "vr_q5", "vr_q10", "vr_q20"]].first()
            vr_long = vr_asset.melt(id_vars=["asset"], var_name="vr_q", value_name="vr")
            vr_long["vr_q"] = pd.Categorical(vr_long["vr_q"], categories=["vr_q2", "vr_q5", "vr_q10", "vr_q20"], ordered=True)
            fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
            axes[0].plot(acf_agg["lag"], acf_agg["mean_acf"], marker="o", linewidth=1.7, color="#0EA5A4", label="mean")
            axes[0].fill_between(acf_agg["lag"], acf_agg["q25"], acf_agg["q75"], color="#0EA5A4", alpha=0.15, label="asset IQR")
            axes[0].axhline(0.0, color="black", linewidth=1)
            axes[0].set_title("Average Return ACF")
            axes[0].set_xlabel("Lag")
            axes[0].set_ylabel("ACF")
            axes[0].grid(alpha=0.2)
            axes[0].legend()

            q_labels = ["vr_q2", "vr_q5", "vr_q10", "vr_q20"]
            vals = [vr_long.loc[vr_long["vr_q"] == q, "vr"].to_numpy(dtype=float) for q in q_labels]
            axes[1].boxplot(vals, labels=["VR(2)", "VR(5)", "VR(10)", "VR(20)"], showmeans=True)
            axes[1].axhline(1.0, color="#DC2626", linestyle="--", linewidth=1.4)
            axes[1].set_title("Variance-Ratio Diagnostics")
            axes[1].set_ylabel("Variance ratio")
            axes[1].grid(axis="y", alpha=0.2)
            _save_and_record(
                records,
                fig,
                out_dir / "fig_v2_41_return_predictability_panel.png",
                "fig_v2_41",
                "Return predictability diagnostics (ACF and variance ratios)",
                "5.2 Financial + Sentiment Analysis",
                "Do raw returns contain exploitable serial structure at short lags?",
                "transformer_diagnostics/return_predictability.csv",
            )

    # FIG 42: training convergence curves
    if not trf_history.empty and {"epoch", "train_loss", "val_loss", "mode"}.issubset(trf_history.columns):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
        for ax, mode in zip(axes, ["no_sentiment", "with_sentiment"]):
            sub = trf_history[trf_history["mode"] == mode].copy()
            if sub.empty:
                continue
            curve = sub.groupby("epoch", as_index=False).agg(train_loss=("train_loss", "median"), val_loss=("val_loss", "median"))
            ax.plot(curve["epoch"], curve["train_loss"], marker="o", linewidth=1.6, color="#2563EB", label="train")
            ax.plot(curve["epoch"], curve["val_loss"], marker="o", linewidth=1.6, color="#DC2626", label="val")
            ax.set_title(f"Median Loss Curves ({mode})")
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        _save_and_record(
            records,
            fig,
            out_dir / "fig_v2_42_training_convergence_curves.png",
            "fig_v2_42",
            "Median train/validation convergence curves by sentiment mode",
            "5.2 Financial + Sentiment Analysis",
            "Do optimization traces suggest undertraining or stable convergence behavior?",
            "multi_asset_transformers_h1_paired_summary_history.csv",
        )

    # FIG 43: bidirectional transfer comparison
    bidir_path = root / "transfer_bidirectional_comparison.csv"
    if bidir_path.exists():
        bidir = pd.read_csv(bidir_path)
        if not bidir.empty and {"direction", "sentiment_mode", "mode", "mae"}.issubset(bidir.columns):
            fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=True)
            mode_order = ["source_zero_shot", "source_finetuned", "target_only", "random_walk_sequence"]
            for ax, sent_mode, color in [
                (axes[0], "no_sentiment", "#2563EB"),
                (axes[1], "with_sentiment", "#DC2626"),
            ]:
                sub = bidir[bidir["sentiment_mode"] == sent_mode].copy()
                if sub.empty:
                    continue
                direction_order = sorted(sub["direction"].unique().tolist())
                width = 0.38
                x = np.arange(len(mode_order), dtype=float)
                pivot = (
                    sub.pivot_table(index="mode", columns="direction", values="mae", aggfunc="mean")
                    .reindex(mode_order)
                )
                for idx, direction in enumerate(direction_order):
                    vals = pivot[direction].to_numpy(dtype=float) if direction in pivot.columns else np.full(len(mode_order), np.nan)
                    offset = (idx - (len(direction_order) - 1) / 2.0) * width
                    ax.bar(x + offset, vals, width=width, label=direction, alpha=0.9)
                ax.set_title(f"Transfer MAE ({sent_mode})")
                ax.set_xticks(x)
                ax.set_xticklabels(mode_order, rotation=20, ha="right")
                ax.grid(alpha=0.2)
            axes[0].set_ylabel("MAE")
            axes[0].legend(fontsize=8)
            _save_and_record(
                records,
                fig,
                out_dir / "fig_v2_43_bidirectional_transfer_comparison.png",
                "fig_v2_43",
                "Bidirectional transfer comparison (BTC↔ETH)",
                "5.3 Transfer Learning Analysis",
                "How symmetric are transfer outcomes across BTC→ETH and ETH→BTC directions?",
                "transfer_bidirectional_comparison.csv",
            )

    # FIG 44: skill vs zero forecast
    if not skill_vs_zero.empty and {"asset", "model", "skill_mae_vs_zero"}.issubset(skill_vs_zero.columns):
        sv = skill_vs_zero.copy()
        if "objective_track" in sv.columns:
            sv = sv[sv["objective_track"].fillna("point") == "point"]
        if "mode" in sv.columns:
            sv = sv[sv["mode"].fillna("no_sentiment") == "no_sentiment"]
        sv = sv[sv["model"].astype(str) != "zero_forecast"].copy()
        if not sv.empty:
            pivot = (
                sv.pivot_table(index="model", columns="asset", values="skill_mae_vs_zero", aggfunc="mean")
                .sort_index(axis=1)
            )
            if not pivot.empty:
                model_order = pivot.mean(axis=1).sort_values(ascending=False).index.tolist()
                pivot = pivot.loc[model_order]
                values = pivot.to_numpy(dtype=float)
                vmax = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 0.1
                vmax = max(vmax, 0.05)
                fig, ax = plt.subplots(figsize=(11.0, 6.0))
                im = ax.imshow(values, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
                ax.set_xticks(np.arange(len(pivot.columns)))
                ax.set_xticklabels([str(c).upper() for c in pivot.columns], rotation=0)
                ax.set_yticks(np.arange(len(pivot.index)))
                ax.set_yticklabels(pivot.index, fontsize=8)
                ax.set_title("Skill vs Zero-Forecast (MAE Improvement Ratio, No Sentiment)")
                for i in range(pivot.shape[0]):
                    for j in range(pivot.shape[1]):
                        v = pivot.iloc[i, j]
                        txt = "-" if not pd.notna(v) else f"{100.0 * v:+.1f}%"
                        ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="black")
                cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
                cbar.set_label("skill_mae_vs_zero")
                _save_and_record(
                    records,
                    fig,
                    out_dir / "fig_v2_44_skill_vs_zero.png",
                    "fig_v2_44",
                    "Skill relative to zero-forecast baseline (MAE)",
                    "5.4 Final Comparative Results",
                    "How much out-of-sample error skill does each model retain versus the naive zero baseline?",
                    "PAPER_H1_skill_vs_zero.csv",
                )

    _records_to_csv(records, out_dir / "FIGURES_MANIFEST_V2.csv")
    return records
