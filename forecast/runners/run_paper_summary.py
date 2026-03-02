from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forecast.pipeline.paper_validity import (
    validate_asset_universe,
    validate_inference_nonempty,
    validate_no_fallback_rows,
    validate_required_columns,
)


EXPECTED_ASSETS = {"btc", "eth", "ada", "doge", "xmr", "xrp"}
ALLOWED_MODELS = {
    "baselines": {"linear_ridge", "random_walk"},
    "transformers": {"patchtst_like", "itransformer_like", "random_walk_sequence"},
    "chronos2_zero_shot": {"chronos2_zero_shot", "random_walk_scaled"},
    "foundation": {
        "chronos2_zero_shot",
        "timesfm_zero_shot",
        "moirai_zero_shot",
        "lagllama_zero_shot",
        "random_walk_scaled",
    },
}

def _load_required(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {label}: {path}")
    return pd.read_csv(path)


def _load_optional(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _safe_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    keep = [c for c in cols if c in df.columns]
    return df[keep].copy()


def _best_by_asset(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.sort_values(["asset", "mae_mean"], ascending=[True, True])
        .groupby("asset", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return out


def _best_by_asset_mode(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["asset", "mode"] + (["objective_track"] if "objective_track" in df.columns else [])
    sort_cols = [*group_cols, "mae_mean"]
    out = df.sort_values(sort_cols, ascending=[True] * len(sort_cols)).groupby(group_cols, as_index=False).head(1).reset_index(drop=True)
    return out


def _sentiment_delta(df: pd.DataFrame, family: str) -> pd.DataFrame:
    subset = df[df["mode"].isin(["no_sentiment", "with_sentiment"])].copy()
    if subset.empty:
        return pd.DataFrame()

    if "objective_track" in subset.columns:
        subset = subset[subset["objective_track"] == "point"]

    idx_cols = ["asset", "model"]
    values = ["mae_mean", "rmse_mean", "directional_accuracy_mean", "sharpe_5bps_mean", "dsr_mean"]
    pivot = subset.pivot_table(index=idx_cols, columns="mode", values=values, aggfunc="first")
    if pivot.empty:
        return pd.DataFrame()

    pivot.columns = [f"{v}_{m}" for v, m in pivot.columns]
    out = pivot.reset_index()
    out["family"] = family
    for metric in values:
        no_col = f"{metric}_no_sentiment"
        ws_col = f"{metric}_with_sentiment"
        if no_col in out.columns and ws_col in out.columns:
            out[f"delta_{metric}_with_minus_no"] = out[ws_col] - out[no_col]
    return out


def _fmt(v: float | int | str | None) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    if isinstance(v, (int, np.integer)):
        return f"{int(v)}"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.6f}"
    return str(v)


def _table_md(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "_No rows available._"
    data = _safe_cols(df, cols)
    headers = list(data.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in data.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in headers) + " |")
    return "\n".join(lines)


def _sentiment_effect_summary(sentiment_delta: pd.DataFrame) -> pd.DataFrame:
    if sentiment_delta.empty:
        return pd.DataFrame()
    required = [
        "delta_mae_mean_with_minus_no",
        "delta_rmse_mean_with_minus_no",
        "delta_directional_accuracy_mean_with_minus_no",
        "delta_sharpe_5bps_mean_with_minus_no",
        "delta_dsr_mean_with_minus_no",
    ]
    if not set(required).issubset(set(sentiment_delta.columns)):
        return pd.DataFrame()
    return sentiment_delta.groupby("family", as_index=False).agg(
        avg_delta_mae=("delta_mae_mean_with_minus_no", "mean"),
        avg_delta_rmse=("delta_rmse_mean_with_minus_no", "mean"),
        avg_delta_directional_acc=("delta_directional_accuracy_mean_with_minus_no", "mean"),
        avg_delta_sharpe_5bps=("delta_sharpe_5bps_mean_with_minus_no", "mean"),
        avg_delta_dsr=("delta_dsr_mean_with_minus_no", "mean"),
    )


def _merge_ci_columns(best_df: pd.DataFrame, ci_df: pd.DataFrame) -> pd.DataFrame:
    if best_df.empty or ci_df.empty:
        return best_df
    merge_keys = [c for c in ["asset", "mode", "objective_track", "model"] if c in best_df.columns and c in ci_df.columns]
    if not merge_keys:
        return best_df
    ci_cols = merge_keys + [c for c in ci_df.columns if c.endswith("_ci95_low") or c.endswith("_ci95_high")]
    ci_cols = [c for c in ci_cols if c in ci_df.columns]
    if not ci_cols:
        return best_df
    return best_df.merge(ci_df[ci_cols], on=merge_keys, how="left")


def _ensure_allowed_models(df: pd.DataFrame, *, family: str) -> None:
    allowed = ALLOWED_MODELS.get(family)
    if allowed is None or "model" not in df.columns:
        return
    got = set(df["model"].astype(str).tolist())
    disallowed = sorted(got.difference(allowed))
    if disallowed:
        raise ValueError(f"Disallowed models for family={family}: {disallowed}")


def _load_inference_tables(root: Path, *, h: int, prefix: str, strict: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if strict:
        try:
            strict_tables = validate_inference_nonempty(root, prefix)
            return (
                strict_tables["_metric_ci95.csv"],
                strict_tables["_directional_binomial.csv"],
                strict_tables["_mcs.csv"],
            )
        except Exception:
            # Allow fallback to canonical PAPER_H{h} prefix in strict mode as long as files are present and non-empty.
            strict_tables = validate_inference_nonempty(root, f"PAPER_H{h}")
            return (
                strict_tables["_metric_ci95.csv"],
                strict_tables["_directional_binomial.csv"],
                strict_tables["_mcs.csv"],
            )

    ci_df = _load_optional(root / f"{prefix}_metric_ci95.csv")
    if ci_df.empty:
        ci_df = _load_optional(root / f"PAPER_H{h}_metric_ci95.csv")
    binom_df = _load_optional(root / f"{prefix}_directional_binomial.csv")
    if binom_df.empty:
        binom_df = _load_optional(root / f"PAPER_H{h}_directional_binomial.csv")
    mcs_df = _load_optional(root / f"{prefix}_mcs.csv")
    if mcs_df.empty:
        mcs_df = _load_optional(root / f"PAPER_H{h}_mcs.csv")
    return ci_df, binom_df, mcs_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-ready consolidated summaries from experiment outputs")
    parser.add_argument("--results-root", type=str, default="results/paper")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 5, 20])
    parser.add_argument("--output-prefix", type=str, default="PAPER_H1")
    parser.add_argument("--strict", dest="strict", action="store_true", default=True)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    args = parser.parse_args()

    root = Path(args.results_root)
    h = args.horizon
    prefix = args.output_prefix

    baselines_path = root / f"multi_asset_baselines_h{h}_paired_summary.csv"
    transformers_path = root / f"multi_asset_transformers_h{h}_paired_summary.csv"
    foundation_path = root / f"multi_asset_foundation_h{h}_summary.csv"
    chronos_path = root / f"multi_asset_chronos2_h{h}_summary.csv"
    ensemble_path = root / f"ensemble_h{h}_summary.csv"
    ablation_best_path = root / f"feature_ablation_h{h}_summary_best.csv"

    base = _load_required(baselines_path, "multi-asset baselines summary")
    trf = _load_required(transformers_path, "multi-asset transformers summary")
    if foundation_path.exists():
        fnd = _load_required(foundation_path, "multi-asset foundation summary")
        foundation_label = "foundation"
    else:
        fnd = _load_required(chronos_path, "multi-asset Chronos-2 summary")
        foundation_label = "chronos2_zero_shot"
    ens = _load_optional(ensemble_path)
    abl_best = _load_required(ablation_best_path, "feature ablation best table")

    if args.strict:
        validate_required_columns(trf, ["objective_track"], "multi-asset transformers summary")

    for df in (base, trf, fnd, ens):
        if df.empty:
            continue
        if "mode" not in df.columns:
            df["mode"] = "no_sentiment"
    if "objective_track" not in base.columns:
        base["objective_track"] = "point"
    if "objective_track" not in fnd.columns:
        fnd["objective_track"] = "point"
    if not ens.empty and "objective_track" not in ens.columns:
        ens["objective_track"] = "point"
    if "objective_track" not in trf.columns and not args.strict:
        trf["objective_track"] = "point"

    base["family"] = "baselines"
    trf["family"] = "transformers"
    fnd["family"] = foundation_label
    fnd["data_path"] = fnd.get("data_path", "")
    if not ens.empty:
        ens["family"] = "ensemble"

    if args.strict:
        validate_asset_universe(base, EXPECTED_ASSETS)
        validate_asset_universe(trf, EXPECTED_ASSETS)
        validate_asset_universe(fnd, EXPECTED_ASSETS)
        validate_no_fallback_rows(base, model_col="model")
        validate_no_fallback_rows(trf, model_col="model")
        validate_no_fallback_rows(fnd, model_col="model", note_col="backend_note")
        _ensure_allowed_models(base, family="baselines")
        _ensure_allowed_models(trf, family="transformers")
        _ensure_allowed_models(fnd, family=foundation_label)

    model_frames = [base, trf, fnd] + ([ens] if not ens.empty else [])
    common_cols = sorted(set().union(*[set(df.columns) for df in model_frames]))
    aligned = []
    for df in model_frames:
        d = df.copy()
        for c in common_cols:
            if c not in d.columns:
                d[c] = np.nan
        aligned.append(d[common_cols])

    all_models = pd.concat(aligned, ignore_index=True)
    all_models = all_models.sort_values(["asset", "mode", "objective_track", "family", "mae_mean"], ascending=[True, True, True, True, True])

    point_models = all_models[all_models["objective_track"] == "point"].copy()
    if point_models.empty:
        point_models = all_models.copy()

    best_asset_mode = _best_by_asset_mode(point_models)
    best_asset_overall = _best_by_asset(point_models)

    sentiment_delta = pd.concat(
        [_sentiment_delta(base, "baselines"), _sentiment_delta(trf, "transformers")],
        ignore_index=True,
    )
    if not sentiment_delta.empty:
        sentiment_delta = sentiment_delta.sort_values(["family", "asset", "model"], ascending=[True, True, True])

    family_group_cols = ["family", "mode"] + (["objective_track"] if "objective_track" in point_models.columns else []) + ["model"]
    family_perf = (
        point_models.groupby(family_group_cols, as_index=False)
        .agg(
            avg_mae_mean=("mae_mean", "mean"),
            avg_rmse_mean=("rmse_mean", "mean"),
            avg_directional_accuracy_mean=("directional_accuracy_mean", "mean"),
            avg_sharpe_5bps_mean=("sharpe_5bps_mean", "mean"),
            avg_dsr_mean=("dsr_mean", "mean"),
            n_asset_evals=("asset", "count"),
        )
        .sort_values([*family_group_cols[:-1], "avg_mae_mean"], ascending=[True] * (len(family_group_cols)))
    )

    ablation_mode_wins = (
        abl_best.sort_values(["asset", "mae_mean"], ascending=[True, True])
        .groupby("asset", as_index=False)
        .head(1)
        .groupby("mode", as_index=False)
        .agg(n_assets_won=("asset", "count"), avg_winning_mae=("mae_mean", "mean"))
        .sort_values(["n_assets_won", "avg_winning_mae"], ascending=[False, True])
    )

    ci_df, binom_df, mcs_df = _load_inference_tables(root, h=h, prefix=prefix, strict=args.strict)

    best_asset_mode = _merge_ci_columns(best_asset_mode, ci_df)
    best_asset_overall = _merge_ci_columns(best_asset_overall, ci_df)

    out_best_asset_mode = root / f"{prefix}_best_by_asset_mode.csv"
    out_best_asset = root / f"{prefix}_best_by_asset.csv"
    out_sentiment = root / f"{prefix}_sentiment_delta.csv"
    out_family = root / f"{prefix}_family_performance.csv"
    out_ablation = root / f"{prefix}_ablation_mode_wins.csv"
    out_md = root / f"{prefix}_RESULTS_SUMMARY.md"
    out_ci = root / f"{prefix}_metric_ci95.csv"
    out_binom = root / f"{prefix}_directional_binomial.csv"
    out_mcs = root / f"{prefix}_mcs.csv"

    best_asset_mode.to_csv(out_best_asset_mode, index=False)
    best_asset_overall.to_csv(out_best_asset, index=False)
    sentiment_delta.to_csv(out_sentiment, index=False)
    family_perf.to_csv(out_family, index=False)
    ablation_mode_wins.to_csv(out_ablation, index=False)
    if not ci_df.empty:
        ci_df.to_csv(out_ci, index=False)
    if not binom_df.empty:
        binom_df.to_csv(out_binom, index=False)
    if not mcs_df.empty:
        mcs_df.to_csv(out_mcs, index=False)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = []
    lines.append(f"# Paper Results Summary (H={h})")
    lines.append("")
    lines.append(f"Generated: {now}")
    lines.append("")
    lines.append(f"- Assets covered: `{best_asset_overall['asset'].nunique() if not best_asset_overall.empty else 0}`")
    lines.append(f"- Total evaluated rows: `{len(all_models)}`")
    lines.append(f"- Point-track rows: `{len(point_models)}`")
    lines.append(f"- Sentiment delta rows: `{len(sentiment_delta)}`")
    lines.append("")
    lines.append("## Best Model Per Asset (Point Track)")
    lines.append("")
    lines.append(
        _table_md(
            best_asset_overall,
            [
                "asset",
                "family",
                "mode",
                "model",
                "mae_mean",
                "mae_ci95_low",
                "mae_ci95_high",
                "rmse_mean",
                "directional_accuracy_mean",
                "sharpe_5bps_mean",
                "dsr_mean",
            ],
        )
    )
    lines.append("")
    lines.append("## Family Performance (Average Across Assets, Point Track)")
    lines.append("")
    lines.append(
        _table_md(
            family_perf,
            [
                "family",
                "mode",
                "objective_track",
                "model",
                "avg_mae_mean",
                "avg_rmse_mean",
                "avg_directional_accuracy_mean",
                "avg_sharpe_5bps_mean",
                "avg_dsr_mean",
                "n_asset_evals",
            ],
        )
    )
    lines.append("")
    lines.append("## Sentiment Effect (With - No)")
    lines.append("")
    sentiment_effect = _sentiment_effect_summary(sentiment_delta)
    if sentiment_effect.empty:
        lines.append("_No paired sentiment rows available._")
    else:
        lines.append(
            _table_md(
                sentiment_effect,
                [
                    "family",
                    "avg_delta_mae",
                    "avg_delta_rmse",
                    "avg_delta_directional_acc",
                    "avg_delta_sharpe_5bps",
                    "avg_delta_dsr",
                ],
            )
        )
    lines.append("")
    lines.append("## Ablation Mode Wins")
    lines.append("")
    lines.append(_table_md(ablation_mode_wins, ["mode", "n_assets_won", "avg_winning_mae"]))
    lines.append("")
    lines.append("## Statistical Inference")
    lines.append("")
    lines.append(f"- Metric CI rows: `{len(ci_df)}`")
    lines.append(f"- Directional binomial rows: `{len(binom_df)}`")
    lines.append(f"- MCS rows: `{len(mcs_df)}`")
    if not mcs_df.empty:
        mcs_summary = (
            mcs_df[mcs_df["mcs_member"] == True]  # noqa: E712
            .groupby(["asset", "mode", "objective_track"], as_index=False)
            .agg(mcs_models=("model", lambda s: ",".join(sorted(set(s.astype(str).tolist())))))
        )
        lines.append("")
        lines.append(_table_md(mcs_summary, ["asset", "mode", "objective_track", "mcs_models"]))

    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- `{out_best_asset_mode}`")
    lines.append(f"- `{out_best_asset}`")
    lines.append(f"- `{out_sentiment}`")
    lines.append(f"- `{out_family}`")
    lines.append(f"- `{out_ablation}`")
    if not ci_df.empty:
        lines.append(f"- `{out_ci}`")
    if not binom_df.empty:
        lines.append(f"- `{out_binom}`")
    if not mcs_df.empty:
        lines.append(f"- `{out_mcs}`")
    lines.append(f"- `{out_md}`")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved: {out_best_asset_mode}")
    print(f"Saved: {out_best_asset}")
    print(f"Saved: {out_sentiment}")
    print(f"Saved: {out_family}")
    print(f"Saved: {out_ablation}")
    if not ci_df.empty:
        print(f"Saved: {out_ci}")
    if not binom_df.empty:
        print(f"Saved: {out_binom}")
    if not mcs_df.empty:
        print(f"Saved: {out_mcs}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
