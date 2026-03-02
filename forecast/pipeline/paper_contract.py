from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AssetConfig:
    asset: str
    data_path: str


@dataclass(frozen=True)
class PaperContract:
    version: str
    experiment_name: str
    target_space: str
    target_horizons_days: tuple[int, ...]
    feature_modes: tuple[str, ...]
    sentiment_lag: int
    assets: tuple[AssetConfig, ...]
    raw: dict[str, Any]


def load_paper_contract(path: str | Path) -> PaperContract:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    assets = tuple(
        AssetConfig(asset=str(item["asset"]).lower(), data_path=str(item["data_path"]))
        for item in raw.get("assets", [])
    )

    contract = PaperContract(
        version=str(raw.get("version", "unknown")),
        experiment_name=str(raw.get("experiment_name", "unnamed_experiment")),
        target_space=str(raw.get("target_space", "unknown")),
        target_horizons_days=tuple(int(h) for h in raw.get("target_horizons_days", [])),
        feature_modes=tuple(str(m) for m in raw.get("feature_modes", [])),
        sentiment_lag=int(raw.get("sentiment_lag", 1)),
        assets=assets,
        raw=raw,
    )
    validate_paper_contract(contract)
    return contract


def validate_paper_contract(contract: PaperContract) -> None:
    if contract.target_space != "log_return":
        raise ValueError(f"target_space must be 'log_return', got: {contract.target_space}")
    if not contract.target_horizons_days:
        raise ValueError("target_horizons_days must not be empty")
    if any(h <= 0 for h in contract.target_horizons_days):
        raise ValueError("target_horizons_days must contain only positive values")
    if not contract.assets:
        raise ValueError("assets must not be empty")
    seen_assets: set[str] = set()
    for asset in contract.assets:
        if asset.asset in seen_assets:
            raise ValueError(f"duplicate asset in contract: {asset.asset}")
        seen_assets.add(asset.asset)


def contract_asset_table(contract: PaperContract) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cfg in contract.assets:
        path = Path(cfg.data_path)
        rows.append(
            {
                "asset": cfg.asset,
                "data_path": cfg.data_path,
                "exists": path.exists(),
            }
        )
    return rows
