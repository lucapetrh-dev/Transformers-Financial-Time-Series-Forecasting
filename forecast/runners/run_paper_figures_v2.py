from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forecast.analysis.paper_figures_v2 import generate_paper_figures_v2


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis-grade v2 figure pack from rerun artifacts")
    parser.add_argument("--results-root", type=str, default="results/paper")
    parser.add_argument("--output-dir", type=str, default="results/paper/overleaf_thesis/Images/generated_v2")
    args = parser.parse_args()

    records = generate_paper_figures_v2(
        results_root=Path(args.results_root),
        output_dir=Path(args.output_dir),
    )

    manifest = Path(args.output_dir) / "FIGURES_MANIFEST_V2.csv"
    print(f"Generated {len(records)} figures")
    print(f"Manifest: {manifest}")
    for rec in records:
        print(f" - {rec.figure_id}: {rec.filename}")


if __name__ == "__main__":
    main()
