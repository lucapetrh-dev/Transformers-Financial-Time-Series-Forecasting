from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forecast.analysis.stationarity_appendix import run_stationarity_appendix
from forecast.pipeline.data_pipeline import FeatureConfig, add_targets, build_feature_frame, load_market_data


def _asset_name_from_path(path: str) -> str:
    name = Path(path).stem.lower()
    name = name.replace("_lunarcrush_timeseries_hourly", "")
    name = name.replace("_lunarcrash_timeseries_hourly", "")
    name = name.replace("_timeseries_hourly", "")
    name = name.replace("_timeseries", "")
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate appendix-grade stationarity diagnostics")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--time-col", type=str, default="time")
    parser.add_argument("--output-dir", type=str, default="results/appendix/stationarity")
    args = parser.parse_args()

    asset = _asset_name_from_path(args.data_path)
    raw = load_market_data(args.data_path, time_col=args.time_col)
    features = build_feature_frame(
        raw,
        config=FeatureConfig(lookbacks=(20, 60), ewma_span=20, roc_period=5),
        time_col=args.time_col,
    )
    features = add_targets(features, horizons=(1, 5, 20))

    out_dir = Path(args.output_dir)
    paths = run_stationarity_appendix(
        frame=features,
        time_col=args.time_col,
        output_dir=out_dir,
        asset_name=asset,
    )
    print(f"Saved summary CSV: {paths.summary_csv}")
    print(f"Saved tests CSV: {paths.tests_csv}")
    print(f"Saved rolling plot: {paths.rolling_plot_png}")
    print(f"Saved ACF/PACF plot: {paths.acf_pacf_plot_png}")
    print(f"Saved FFT plot: {paths.fft_plot_png}")


if __name__ == "__main__":
    main()
