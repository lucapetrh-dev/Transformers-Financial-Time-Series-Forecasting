from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss


@dataclass
class StationarityPaths:
    summary_csv: Path
    tests_csv: Path
    rolling_plot_png: Path
    acf_pacf_plot_png: Path
    fft_plot_png: Path


def _run_adf(series: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    res = adfuller(s)
    stat = res[0]
    pval = res[1]
    crit = res[4]
    return {
        "statistic": float(stat),
        "p_value": float(pval),
        "crit_1pct": float(crit["1%"]),
        "crit_5pct": float(crit["5%"]),
        "crit_10pct": float(crit["10%"]),
    }


def _run_kpss(series: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    res = kpss(s, regression="c", nlags="auto")
    stat = res[0]
    pval = res[1]
    crit = res[3]
    return {
        "statistic": float(stat),
        "p_value": float(pval),
        "crit_1pct": float(crit["1%"]),
        "crit_5pct": float(crit["5%"]),
        "crit_10pct": float(crit["10%"]),
    }


def run_stationarity_appendix(
    *,
    frame: pd.DataFrame,
    time_col: str,
    output_dir: str | Path,
    asset_name: str,
) -> StationarityPaths:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = frame[[time_col, "log_price", "return_1d", "target_ret_1d"]].copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[time_col, "log_price", "return_1d"]).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for series_name in ["log_price", "return_1d", "target_ret_1d"]:
        s = df[series_name].dropna()
        if len(s) < 30:
            continue
        adf = _run_adf(s)
        kpss_res = _run_kpss(s)
        rows.append(
            {
                "asset": asset_name,
                "series": series_name,
                "test": "adf",
                "statistic": adf["statistic"],
                "p_value": adf["p_value"],
                "stationary_at_5pct": bool(adf["p_value"] < 0.05),
                "null_hypothesis": "unit_root_non_stationary",
            }
        )
        rows.append(
            {
                "asset": asset_name,
                "series": series_name,
                "test": "kpss",
                "statistic": kpss_res["statistic"],
                "p_value": kpss_res["p_value"],
                "stationary_at_5pct": bool(kpss_res["p_value"] > 0.05),
                "null_hypothesis": "trend_stationary",
            }
        )
    tests_df = pd.DataFrame(rows)
    tests_csv = out_dir / f"{asset_name}_stationarity_tests.csv"
    tests_df.to_csv(tests_csv, index=False)

    summary = pd.DataFrame(
        [
            {
                "asset": asset_name,
                "rows": int(len(df)),
                "time_start": str(df[time_col].min()),
                "time_end": str(df[time_col].max()),
                "return_mean": float(df["return_1d"].mean()),
                "return_std": float(df["return_1d"].std()),
                "return_skew": float(df["return_1d"].skew()),
                "return_kurtosis": float(df["return_1d"].kurtosis()),
            }
        ]
    )
    summary_csv = out_dir / f"{asset_name}_stationarity_summary.csv"
    summary.to_csv(summary_csv, index=False)

    # Rolling diagnostics plot
    roll_w = 20
    roll_mean = df["return_1d"].rolling(roll_w, min_periods=roll_w).mean()
    roll_std = df["return_1d"].rolling(roll_w, min_periods=roll_w).std()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(df[time_col], df["return_1d"], label="return_1d", color="tab:blue")
    axes[0].plot(df[time_col], roll_mean, label=f"rolling_mean_{roll_w}", color="tab:orange")
    axes[0].legend()
    axes[0].set_title(f"{asset_name.upper()} Returns with Rolling Mean")
    axes[1].plot(df[time_col], roll_std, label=f"rolling_std_{roll_w}", color="tab:red")
    axes[1].legend()
    axes[1].set_title(f"{asset_name.upper()} Rolling Std")
    plt.tight_layout()
    rolling_plot = out_dir / f"{asset_name}_rolling_diagnostics.png"
    fig.savefig(rolling_plot, dpi=160)
    plt.close(fig)

    # ACF/PACF diagnostics
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(df["return_1d"].dropna(), lags=min(40, max(5, len(df) // 8)), ax=axes[0])
    axes[0].set_title(f"{asset_name.upper()} Return ACF")
    plot_pacf(df["return_1d"].dropna(), lags=min(30, max(5, len(df) // 10)), method="ywm", ax=axes[1])
    axes[1].set_title(f"{asset_name.upper()} Return PACF")
    plt.tight_layout()
    acf_pacf_plot = out_dir / f"{asset_name}_acf_pacf.png"
    fig.savefig(acf_pacf_plot, dpi=160)
    plt.close(fig)

    # FFT diagnostics (appendix only)
    series = df["log_price"].to_numpy(dtype=float)
    amps = np.abs(np.fft.rfft(series))
    freqs = np.fft.rfftfreq(len(series), d=1.0)
    peaks, _ = find_peaks(amps)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freqs, amps, color="tab:blue", lw=1.0)
    if len(peaks) > 0:
        k = min(20, len(peaks))
        top = peaks[np.argsort(amps[peaks])[-k:]]
        ax.scatter(freqs[top], amps[top], color="tab:red", s=12, label="peak candidates")
        ax.legend()
    ax.set_title(f"{asset_name.upper()} Log-Price FFT Amplitude")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Amplitude")
    ax.set_yscale("log")
    plt.tight_layout()
    fft_plot = out_dir / f"{asset_name}_fft.png"
    fig.savefig(fft_plot, dpi=160)
    plt.close(fig)

    return StationarityPaths(
        summary_csv=summary_csv,
        tests_csv=tests_csv,
        rolling_plot_png=rolling_plot,
        acf_pacf_plot_png=acf_pacf_plot,
        fft_plot_png=fft_plot,
    )
