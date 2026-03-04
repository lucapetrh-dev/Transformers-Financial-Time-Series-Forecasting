from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _load_tagged(path: Path, *, direction: str, sentiment_mode: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing transfer result file: {path}")
    df = pd.read_csv(path).copy()
    df.insert(0, "direction", direction)
    df.insert(1, "sentiment_mode", sentiment_mode)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bidirectional BTC<->ETH transfer comparison tables")
    parser.add_argument("--btc-eth-no-sent", type=str, default="results/paper/transfer_btc_eth_patchtst_no_sent.csv")
    parser.add_argument("--btc-eth-with-sent", type=str, default="results/paper/transfer_btc_eth_patchtst_with_sent.csv")
    parser.add_argument("--eth-btc-no-sent", type=str, default="results/paper/transfer_eth_btc_patchtst_no_sent.csv")
    parser.add_argument("--eth-btc-with-sent", type=str, default="results/paper/transfer_eth_btc_patchtst_with_sent.csv")
    parser.add_argument("--output", type=str, default="results/paper/transfer_bidirectional_comparison.csv")
    parser.add_argument("--deltas-output", type=str, default="")
    args = parser.parse_args()

    frames = [
        _load_tagged(Path(args.btc_eth_no_sent), direction="btc_to_eth", sentiment_mode="no_sentiment"),
        _load_tagged(Path(args.btc_eth_with_sent), direction="btc_to_eth", sentiment_mode="with_sentiment"),
        _load_tagged(Path(args.eth_btc_no_sent), direction="eth_to_btc", sentiment_mode="no_sentiment"),
        _load_tagged(Path(args.eth_btc_with_sent), direction="eth_to_btc", sentiment_mode="with_sentiment"),
    ]

    combined = pd.concat(frames, ignore_index=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)

    metric_cols = [c for c in combined.columns if c not in {"direction", "sentiment_mode", "mode"}]
    metric_cols = [c for c in metric_cols if pd.api.types.is_numeric_dtype(combined[c])]

    deltas_rows: list[dict[str, object]] = []
    for (sentiment_mode, mode), grp in combined.groupby(["sentiment_mode", "mode"], dropna=False):
        pivot = grp.set_index("direction")
        if "btc_to_eth" not in pivot.index or "eth_to_btc" not in pivot.index:
            continue
        for metric in metric_cols:
            btc_eth = float(pivot.loc["btc_to_eth", metric])
            eth_btc = float(pivot.loc["eth_to_btc", metric])
            deltas_rows.append(
                {
                    "sentiment_mode": sentiment_mode,
                    "mode": mode,
                    "metric": metric,
                    "btc_to_eth": btc_eth,
                    "eth_to_btc": eth_btc,
                    "delta_eth_btc_minus_btc_eth": eth_btc - btc_eth,
                }
            )

    deltas = pd.DataFrame(deltas_rows)
    deltas_path = Path(args.deltas_output) if args.deltas_output else out_path.with_name(out_path.stem + "_deltas.csv")
    deltas.to_csv(deltas_path, index=False)

    print(f"Saved bidirectional comparison: {out_path}")
    print(f"Saved direction deltas: {deltas_path}")


if __name__ == "__main__":
    main()
