from __future__ import annotations

from pathlib import Path


def asset_name_from_path(path: str) -> str:
    name = Path(path).stem.lower()
    name = name.replace("_lunarcrush_timeseries_hourly", "")
    name = name.replace("_lunarcrash_timeseries_hourly", "")
    name = name.replace("_timeseries_hourly", "")
    name = name.replace("_timeseries", "")
    return name


def parse_asset_list(value: str | None) -> set[str]:
    if value is None:
        return set()
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def filter_asset_paths(
    paths: list[str],
    *,
    include_assets: set[str] | None = None,
    exclude_assets: set[str] | None = None,
) -> list[str]:
    include = include_assets or set()
    exclude = exclude_assets or set()

    out: list[str] = []
    for path in paths:
        asset = asset_name_from_path(path)
        if include and asset not in include:
            continue
        if exclude and asset in exclude:
            continue
        out.append(path)
    return out
