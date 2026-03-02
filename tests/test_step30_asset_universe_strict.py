from __future__ import annotations

import pandas as pd
import pytest

from forecast.pipeline.paper_validity import validate_asset_universe


EXPECTED = {"btc", "eth", "ada", "doge", "xmr", "xrp"}


def test_asset_universe_strict_passes_on_exact_six_assets() -> None:
    df = pd.DataFrame({"asset": sorted(EXPECTED)})
    validate_asset_universe(df, EXPECTED)


def test_asset_universe_strict_fails_on_missing_or_extra_assets() -> None:
    df = pd.DataFrame({"asset": ["btc", "eth", "ada", "doge", "xmr", "aave"]})
    with pytest.raises(ValueError, match="Asset universe mismatch"):
        validate_asset_universe(df, EXPECTED)

