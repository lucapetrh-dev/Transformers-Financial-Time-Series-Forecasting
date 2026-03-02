from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


INFERENCE_SUFFIXES = (
    "_metric_ci95.csv",
    "_directional_binomial.csv",
    "_mcs.csv",
)


def validate_asset_universe(df: pd.DataFrame, expected_assets: set[str]) -> None:
    if "asset" not in df.columns:
        raise ValueError("Missing required column 'asset' for asset-universe validation")
    actual_assets = set(df["asset"].astype(str).str.lower().tolist())
    missing = sorted(expected_assets.difference(actual_assets))
    extra = sorted(actual_assets.difference(expected_assets))
    if missing or extra:
        raise ValueError(
            "Asset universe mismatch. "
            f"missing={missing if missing else []}; extra={extra if extra else []}"
        )


def validate_required_columns(df: pd.DataFrame, required_cols: list[str], label: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing required columns: {', '.join(missing)}")


def validate_no_fallback_rows(
    df: pd.DataFrame,
    *,
    model_col: str = "model",
    note_col: str | None = None,
) -> None:
    if model_col in df.columns:
        model_series = df[model_col].astype(str).str.lower()
        bad_model_mask = (
            model_series.str.contains("fallback", regex=False)
            | model_series.str.contains("placeholder", regex=False)
            | model_series.str.contains("mock", regex=False)
        )
        if bool(bad_model_mask.any()):
            bad_models = sorted(df.loc[bad_model_mask, model_col].astype(str).unique().tolist())
            raise ValueError(f"Disallowed fallback/placeholder model rows detected: {bad_models}")

    if "backend_status" in df.columns:
        backend = df["backend_status"].astype(str).str.lower()
        bad_backend = ~(backend.isin({"native", "nan", ""}))
        if bool(bad_backend.any()):
            bad = sorted(df.loc[bad_backend, "backend_status"].astype(str).unique().tolist())
            raise ValueError(f"Disallowed backend_status rows detected: {bad}")

    if note_col and note_col in df.columns:
        notes = df[note_col].astype(str).str.lower()
        bad_note_mask = notes.str.contains("fallback", regex=False) | notes.str.contains("placeholder", regex=False)
        if bool(bad_note_mask.any()):
            bad_notes = sorted(df.loc[bad_note_mask, note_col].astype(str).unique().tolist())
            raise ValueError(f"Disallowed fallback/placeholder notes detected: {bad_notes}")


def _resolve_declared_path(raw_value: str, metadata_json_path: Path) -> Path:
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    rel_to_meta = (metadata_json_path.parent / candidate).resolve()
    if rel_to_meta.exists():
        return rel_to_meta
    return candidate.resolve()


def validate_metadata_paths(metadata_json_path: Path) -> None:
    if not metadata_json_path.exists():
        raise ValueError(f"Missing metadata file: {metadata_json_path}")
    with metadata_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    path_fields = [
        "predictions_path",
        "quantiles_path",
        "history_path",
        "data_quality_summary_path",
        "missingness_stages_path",
    ]

    for field in path_fields:
        raw = payload.get(field)
        if raw is None or str(raw).strip() == "":
            continue
        raw_s = str(raw).strip()
        lowered = raw_s.lower()
        if "truth_" in lowered:
            raise ValueError(f"Metadata contains deprecated truth_* reference in {field}: {raw_s}")
        resolved = _resolve_declared_path(raw_s, metadata_json_path)
        if not resolved.exists():
            raise ValueError(
                "Metadata path does not exist "
                f"(field={field}, value={raw_s}, metadata={metadata_json_path})"
            )


def validate_inference_nonempty(results_root: Path, prefix: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for suffix in INFERENCE_SUFFIXES:
        path = results_root / f"{prefix}{suffix}"
        if not path.exists():
            raise FileNotFoundError(f"Missing required inference file: {path}")
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"Inference file is empty: {path}")
        out[suffix] = df
    return out

