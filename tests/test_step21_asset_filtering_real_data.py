from __future__ import annotations

from forecast.pipeline.assets import asset_name_from_path, filter_asset_paths, parse_asset_list


def test_asset_filtering_utils_real_data() -> None:
    paths = [
        ".data/hourly/btc_lunarcrush_timeseries_hourly.csv",
        ".data/hourly/eth_lunarcrush_timeseries_hourly.csv",
        ".data/hourly/aave_lunarcrush_timeseries_hourly.csv",
    ]

    assert asset_name_from_path(paths[0]) == "btc"
    include = parse_asset_list("btc,eth")
    exclude = parse_asset_list("eth")

    filtered = filter_asset_paths(paths, include_assets=include, exclude_assets=exclude)
    assert filtered == [".data/hourly/btc_lunarcrush_timeseries_hourly.csv"]
